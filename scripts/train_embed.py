#!/usr/bin/env python3
"""
EfficientNet-B0 기반 임베딩 학습 스크립트 (Triplet loss + BatchHardMiner)
- 목적: 이미지 → 512차원 임베딩으로 변환하는 특성추출기 학습
- 데이터 구조 (class-per-id):
    dataset/embed/train/<class_id>/*.jpg
  ※ 여기서 <class_id>는 '동일 개체' 또는 PoC 단계에선 '품종' 등으로 사용 가능
- 출력 체크포인트:
    outputs/embed_b0_triplet.pt
- 학습 후에는 export_embed.py로 ONNX로 내보낸 뒤, build_index.py로 FAISS 인덱스를 생성합니다.
"""

import argparse, os, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import BatchHardMiner


class FolderDataset(Dataset):
    """
    폴더 구조를 그대로 읽어서 (이미지, 라벨) 튜플을 반환하는 간단한 데이터셋.
    - root/train/<class_id>/*.jpg 형태를 가정합니다.
    - class_id 디렉토리 이름을 정수로 캐스팅하여 라벨로 사용합니다.
      (예: '12' 폴더 → 라벨 12)
    - 이미지 전처리: RGB 변환 → 리사이즈(기본 224) → [0,1] 정규화 → CHW 텐서
    """
    def __init__(self, root: str, img_size: int = 224):
        self.items = []
        self.img_size = img_size
        # 각 클래스 디렉토리를 순회
        for cid in sorted(os.listdir(root)):
            pdir = Path(root) / cid
            if not pdir.is_dir():
                continue
            # 해당 클래스의 모든 .jpg 파일을 수집
            for p in glob.glob(str(pdir / '*.jpg')):
                # 라벨은 폴더명(정수 추정)으로 사용
                self.items.append((p, int(cid)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        p, cid = self.items[i]
        # 이미지 로드 및 전처리
        img = Image.open(p).convert('RGB').resize((self.img_size, self.img_size))
        x = np.asarray(img).astype('float32') / 255.0  # 0~1 스케일
        x = np.transpose(x, (2, 0, 1))                 # HWC -> CHW
        # 텐서 변환 및 라벨 생성
        return torch.from_numpy(x), torch.tensor(cid, dtype=torch.long)


def get_loader(root: str, bs: int = 32, shuffle: bool = True):
    """
    주어진 경로(root)에서 FolderDataset을 로드해 DataLoader를 반환합니다.
    - num_workers는 CPU 상황에 따라 조정(Windows라면 0~2 권장)
    """
    ds = FolderDataset(root)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=4, pin_memory=True)


def main(args):
    # CUDA 사용 가능 여부에 따라 디바이스 선택
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # timm에서 EfficientNet-B0 백본 로드 (num_classes=0 → 분류헤드 제거, feature 추출 모드)
    backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

    # 백본 출력(feature_dim)을 512-D로 투영하고 L2정규화를 하기 위한 헤드 구성
    # Flatten은 timm 백본 출력이 (B, feat) 형태여도 무해하며, (B, feat, 1, 1) 형태일 때도 대응
    head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(backbone.num_features, 512, bias=False),
        nn.BatchNorm1d(512)  # 배치 정규화로 학습 안정화
    )
    net = nn.Sequential(backbone, head).to(device)

    # 메트릭 러닝 구성:
    # - BatchHardMiner: 배치 내에서 hardest positive/negative 쌍을 찾아 Triplet 형성
    # - TripletMarginLoss: 임베딩 간 마진을 두어 분리력 증가
    miner = BatchHardMiner()
    criterion = TripletMarginLoss(margin=0.3)  # 마진은 0.2~0.5 범위에서 튜닝

    # 옵티마이저: AdamW는 weight decay로 과적합 억제
    opt = optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)

    # 데이터 로더 준비 (class-per-id 폴더 구조)
    train_loader = get_loader(args.train, bs=args.batch)

    net.train()
    # AMP(fp16) 사용 시 속도↑/메모리↓ (GPU에서만 효과)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # 간단한 학습 루프 (엔터프라이즈에선 스케줄러/검증/로그를 추가)
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                emb = net(x)                       # (B, 512)
                emb = nn.functional.normalize(emb) # L2 정규화: 코사인 유사도와 상응
                hard_pairs = miner(emb, y)         # 배치 내 hardest 쌍 선택
                loss = criterion(emb, y, hard_pairs)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        print(f"epoch {epoch+1}/{args.epochs} loss={loss.item():.4f}")

    # 체크포인트 저장 (서빙 전 ONNX로 내보낼 때 사용)
    os.makedirs('outputs', exist_ok=True)
    torch.save(net.state_dict(), 'outputs/embed_b0_triplet.pt')
    print("saved to outputs/embed_b0_triplet.pt\n"
          "→ 다음으로 scripts/export_embed.py 실행하여 ONNX로 내보내세요.")


if __name__ == "__main__":
    # 커맨드라인 인자 정의
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='dataset/embed/train', help='학습용 이미지 루트(클래스별 폴더)')
    ap.add_argument('--epochs', type=int, default=15, help='에폭 수 (데이터에 따라 10~20부터 튜닝)')
    ap.add_argument('--batch', type=int, default=32, help='배치 크기 (GPU 메모리에 맞춰 조정)')
    ap.add_argument('--fp16', action='store_true', default=True, help='AMP(FP16) 사용 여부')
    args = ap.parse_args()
    main(args)
