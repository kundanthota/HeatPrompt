import os, json, math, random, time
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ────────────────────────── 2. Dataset class ──────────────────────────
class HeatDemandDataset(Dataset):
    """
    Returns:
        image   : FloatTensor (4,C,H,W) scaled to [0,1]
        tabular : FloatTensor (4,)  [length, energy_area, num_buildings, building_area]
        target  : FloatTensor (1,)  annual heat demand
    """
    def __init__(self, data:dict, image_dir="data/images", transform=None):
        self.samples = []
        self.tfm = transform
        for key, v in data.items():
            img_p  = os.path.join(image_dir, f"image_{key}.png")
            msk_p  = os.path.join(image_dir, f"mask_{key}.png")
            if not (os.path.exists(img_p) and os.path.exists(msk_p)): continue
            tab = [v.get("Shape_Area",0.), v.get("Shape_Length",0.),
                   v.get("geb_n",0.)]
            self.samples.append((img_p, msk_p, np.array(tab, dtype="float32"), v["wb_gs"]))
        assert self.samples, "Dataset is empty!"

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        img_p, msk_p, tab, y = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        msk = Image.open(msk_p).convert("L")
        img = np.dstack([np.array(img), np.array(msk)])       # H,W,4
        img = Image.fromarray(img)
        if self.tfm: img = self.tfm(img)                      # 4,H,W
        return img, torch.tensor(tab), torch.tensor(y, dtype=torch.float32)

# ───────────── 3. Transform & dataloader helpers ─────────────
tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),                # → [0,1]
])

def make_loader(ds, idxs, bs=16, shuffle=True):
    return DataLoader(Subset(ds, idxs), batch_size=bs, shuffle=shuffle, num_workers=4,
                      pin_memory=torch.cuda.is_available())

# ───────────────────────── 4. Model ──────────────────────────
class ResNetHeatRegressor(nn.Module):
    """ ResNet-18 backbone, first conv changed to 4-channel, tabular fusion """
    def __init__(self, tab_dim=3):
        super().__init__()
        self.backbone = resnet18(weights="IMAGENET1K_V1")
        self.backbone.conv1 = nn.Conv2d(4,64,kernel_size=7,stride=2,padding=3,bias=False)
        n_feat = self.backbone.fc.in_features           # 512
        self.backbone.fc = nn.Identity()                # drop final FC
        self.reg_head = nn.Sequential(
            nn.Linear(n_feat+tab_dim,256), nn.ReLU(),
            nn.Linear(256,64),             nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self, img, tab):
        feat = self.backbone(img)           # B,512
        x = torch.cat([feat, tab], dim=1)
        return self.reg_head(x).squeeze(1)  # B

# ──────────────────────── 5. Training utils ───────────────────────
def train_epoch(model, loader, opt, loss_fn, device):
    model.train(); total_loss, total_mae = 0,0
    for img, tab, y in loader:
        img, tab, y = img.to(device), tab.to(device), y.to(device)
        opt.zero_grad()
        pred = model(img, tab)
        loss = loss_fn(pred, y)
        loss.backward(); opt.step()
        total_loss += loss.item()*len(y)
        total_mae  += mean_absolute_error(y.cpu(), pred.detach().cpu())*len(y)
    n = len(loader.dataset)
    return total_loss/n, total_mae/n

def eval_epoch(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for img, tab, y in loader:
            img, tab = img.to(device), tab.to(device)
            p = model(img, tab).cpu()
            preds.append(p); targets.append(y)
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    rmse = math.sqrt(mean_squared_error(targets, preds))
    mae  = mean_absolute_error(targets, preds)
    r2   = r2_score(targets, preds)
    return rmse, mae, r2

# ─────────────────── 6. K-fold cross-validation ──────────────────
def run_kfold(data_dict, n_splits=5, epochs=30, bs=16, lr=1e-4, seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    ds = HeatDemandDataset(data_dict, transform=tfm)
    idxs = np.arange(len(ds))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for fold,(tr_idx,val_idx) in enumerate(kf.split(idxs),1):
        print(f"\n─ Fold {fold}/{n_splits} ─")
        tr_loader = make_loader(ds, tr_idx, bs, shuffle=True)
        val_loader= make_loader(ds, val_idx, bs, shuffle=False)

        model = ResNetHeatRegressor().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        for ep in range(1,epochs+1):
            tr_loss, tr_mae = train_epoch(model, tr_loader, opt, loss_fn, device)
            if ep % 5 == 0 or ep==1 or ep==epochs:
                rmse, mae, r2 = eval_epoch(model, val_loader, device)
                print(f"  Ep {ep:3d}: train MAE {tr_mae:.2f} | val RMSE {rmse:.2f} "
                      f"MAE {mae:.2f} R² {r2:.3f}")

        rmse, mae, r2 = eval_epoch(model, val_loader, device)
        results.append((rmse, mae, r2))
        torch.save(model.state_dict(), f"models/resnet_reg_fold{fold}.pth")

    res = np.array(results)
    print("\n=== 5-Fold results (mean ± std) ===")
    print(f"RMSE {res[:,0].mean():.2f} ± {res[:,0].std():.2f}")
    print(f"MAE  {res[:,1].mean():.2f} ± {res[:,1].std():.2f}")
    print(f"R²   {res[:,2].mean():.3f} ± {res[:,2].std():.3f}")

# ───────────────────────── 7. Run ─────────────────────────
if __name__ == "__main__":
    import time
    time_start = time.time()
    with open("data/atlas_data/features_by_id.json","r") as f:   # your 1 600-tile metadata dict
        data_dict = json.load(f)
    run_kfold(data_dict, n_splits=5, epochs=2, bs=16, lr=1e-4)
    time_end = time.time()