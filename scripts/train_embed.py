import argparse, os, glob
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import BatchHardMiner

class FolderDataset(Dataset):
    def __init__(self, root, img_size=224):
        self.items=[]
        self.img_size = img_size
        for pid in sorted(os.listdir(root)):
            pdir = Path(root)/pid
            if not pdir.is_dir(): continue
            for p in glob.glob(str(pdir/'*.jpg')):
                self.items.append((p,int(pid)))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p, pid = self.items[i]
        img = Image.open(p).convert('RGB').resize((self.img_size,self.img_size))
        x = np.asarray(img).astype('float32')/255.0
        x = np.transpose(x,(2,0,1))
        return torch.from_numpy(x), torch.tensor(pid, dtype=torch.long)

def get_loader(root, bs=32, shuffle=True):
    ds = FolderDataset(root)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=4, pin_memory=True)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
    head  = nn.Sequential(nn.Flatten(), nn.Linear(backbone.num_features, 512, bias=False), nn.BatchNorm1d(512))
    net   = nn.Sequential(backbone, head).to(device)
    miner = BatchHardMiner()
    criterion = TripletMarginLoss(margin=0.3)
    opt = optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)

    train_loader = get_loader(args.train, bs=args.batch)
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    for epoch in range(args.epochs):
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                emb = net(x)
                emb = nn.functional.normalize(emb)
                hard_pairs = miner(emb, y)
                loss = criterion(emb, y, hard_pairs)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        print(f"epoch {epoch+1}/{args.epochs} loss={loss.item():.4f}")
    os.makedirs('outputs', exist_ok=True)
    torch.save(net.state_dict(), 'outputs/embed_b0_triplet.pt')
    print("saved to outputs/embed_b0_triplet.pt")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--train', default='dataset/embed/train')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--fp16', action='store_true', default=True)
    args=ap.parse_args()
    main(args)
