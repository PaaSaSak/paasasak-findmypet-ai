#!/usr/bin/env python3
"""
임베딩 모델(ONNX)로 갤러리 이미지 임베딩을 추출하여 FAISS 인덱스를 생성하는 스크립트.
- 입력 이미지(갤러리): dataset/embed/val/<class_id>/*.jpg
  (초기엔 검증셋을 갤러리로 사용. 운영에선 DB/스토리지의 실제 이미지로 대체)
- 사용 모델: serving/embed.onnx (export_embed.py로 생성)
- 출력:
    serving/index.faiss  : FAISS HNSW 인덱스 파일
    serving/meta.json    : 인덱스 ID → 메타정보(클래스/원본경로 등)
"""

import json, os, glob
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
import faiss

# 전처리 기본값 (학습 시 입력 크기와 맞춰야 임베딩 분포가 일관됨)
IMG_SIZE = 224

# 출력 폴더 준비 (없으면 생성)
SERV_DIR = Path("serving")
SERV_DIR.mkdir(parents=True, exist_ok=True)

# ONNX Runtime 세션 생성
# - providers에 CUDA가 있으면 GPU로, 없으면 자동으로 CPUExecutionProvider 사용
# - 서버(배포)에서는 보통 CPU만 쓰므로 build_index는 로컬(학습 머신)에서 실행 권장
sess = ort.InferenceSession(
    str(SERV_DIR / "embed.onnx"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

def embed_image(path: str) -> np.ndarray:
    """
    단일 이미지 파일 경로를 받아 전처리 → ONNX 추론 → 512D 임베딩 반환
    반환 shape: (1, 512) float32, L2 정규화 포함
    """
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img).astype('float32') / 255.0      # 0~1 정규화
    x = np.transpose(x, (2, 0, 1))[None, ...]          # (1, 3, H, W)
    emb = sess.run(None, {sess.get_inputs()[0].name: x})[0]  # (1, 512)
    faiss.normalize_L2(emb)                            # L2 정규화: 코사인 유사도와 동일효과
    return emb


# 갤러리 후보 이미지 수집:
# - 여기서는 val 폴더를 갤러리로 사용 (PoC 단계)
# - 운영에서는 Object Storage 등에 있는 실제 갤러리를 스캔하여 사용
paths = []
metas = {}  # 인덱스 ID(0..N-1) → 메타정보
val_root = Path("dataset/embed/val")

for cid in sorted(os.listdir(val_root)):
    pdir = val_root / cid
    if not pdir.is_dir():
        continue
    for p in glob.glob(str(pdir / "*.jpg")):
        idx = len(paths)
        paths.append(p)
        # 메타정보에는 최소한 'class'(또는 개체ID)와 원본 경로를 기록
        metas[str(idx)] = {"class": cid, "path": p}

# 이미지 임베딩 일괄 추출
embs = [embed_image(p)[0] for p in paths]   # [(512,), ...]
embs = np.stack(embs).astype('float32')     # (N, 512)
faiss.normalize_L2(embs)                    # 안전을 위해 재확인

# FAISS 인덱스 생성:
# - HNSW32: 그래프 기반 근사최근접검색 (정확도/속도 밸런스 양호)
# - IDMap: 내부 ID ↔ our index row 매핑 보존
index = faiss.index_factory(512, "HNSW32,IDMap")
# 빌드 품질(삽입) 관련 파라미터. 200~400 범위에서 데이터/시간에 맞춰 조정 가능
index.hnsw.efConstruction = 200

# (0..N-1) ID를 부여하며 벡터 추가
index.add_with_ids(embs, np.arange(len(embs)).astype('int64'))

# 파일로 저장
faiss.write_index(index, str(SERV_DIR / "index.faiss"))
with open(SERV_DIR / "meta.json", "w", encoding="utf-8") as f:
    json.dump({"id_to_meta": metas}, f, ensure_ascii=False)

print("[DONE] saved:")
print(f"  - {SERV_DIR / 'index.faiss'}")
print(f"  - {SERV_DIR / 'meta.json'}\n"
      "→ 서빙 시 app.py에서 이 인덱스를 로드하여 /search 엔드포인트로 검색합니다.")
