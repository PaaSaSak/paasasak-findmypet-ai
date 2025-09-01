#!/usr/bin/env python3
"""
학습된 임베딩 모델(EfficientNet-B0 + 512D 헤드)을 ONNX로 내보내는 스크립트.
- 입력 체크포인트: outputs/embed_b0_triplet.pt
- 출력 ONNX:      serving/embed.onnx
  (서버/도커 서빙 시 ONNX Runtime(CPU)로 추론하기 위해 사용)
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path


def build_model() -> nn.Module:
    """
    train_embed.py에서 사용한 것과 동일한 네트워크 아키텍처를 생성.
    - pretrained=False: 체크포인트에서 가중치를 로드할 것이므로 굳이 사전학습 안 켬
    """
    backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(backbone.num_features, 512, bias=False),
        nn.BatchNorm1d(512),
    )
    return nn.Sequential(backbone, head)


def main():
    # 경로 설정
    ckpt_path = Path("outputs/embed_b0_triplet.pt")
    out_path  = Path("serving/embed.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 모델 생성 및 가중치 로드
    net = build_model()
    # map_location='cpu'로 로드하면 GPU 없는 환경에서도 불러올 수 있습니다.
    state = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(state, strict=False)
    net.eval()

    # 더미 입력 (배치1, RGB, 224x224) — 학습 시 사용한 입력 크기와 일치시켜 주세요.
    dummy = torch.randn(1, 3, 224, 224)

    # ONNX 내보내기
    # - opset_version=17: 최신 ORT와 호환성 양호 (필요시 16~19 범위에서 조정 가능)
    # - dynamic_axes: 배치 차원을 가변으로 지정 → 한 번에 여러 장 추론 가능
    torch.onnx.export(
        net, dummy, str(out_path),
        opset_version=17,
        input_names=["input"],
        output_names=["emb"],
        dynamic_axes={"input": {0: "batch"}, "emb": {0: "batch"}},
    )
    print(f"[OK] Exported ONNX → {out_path}\n"
          f"→ 다음으로 scripts/build_index.py 실행하여 FAISS 인덱스를 만드세요.")


if __name__ == "__main__":
    main()
