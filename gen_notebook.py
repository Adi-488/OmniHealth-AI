import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.7"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def mk_md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}

def mk_code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}

def a(cells):
    nb["cells"].extend(cells)

a([mk_md("""# Module 3 (Revamped): Pretrained Unimodal Feature Extractors
## Multimodal Health State Prediction Using Smartphone Sensors
### VI-A Group ID 13 - Deep Learning Capstone Project

**Goal:** Replace custom from-scratch models with **pretrained encoders** (as specified in roadmap) and systematically compare architectures + hyperparameters.

**Approach:**
- **Fingernail:** ImageNet-pretrained ResNet-18/34/50, EfficientNet-B0, MobileNet-V3 Small
- **Audio:** Stronger CNN-BiLSTM variants (no ImageNet-equivalent for MFCCs)
- **Accelerometer:** Deep 1D-CNN+BiLSTM, Wide CNN, TCN with dilated convolutions
- **Water:** Wide, Deep, Narrow MLP variants
- **Sleep:** MLP variants with different capacity and dropout

For each: compare train/val/test accuracy, compute overfitting gap, pick best.""")])

a([mk_code("""import os, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

import torchvision
import torchvision.models as models
import torchaudio

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PROCESSED = 'D:/DL Project/processed_data'
OUTPUT_DIR = 'D:/DL Project/module3_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}, Torchvision: {torchvision.__version__}, Torchaudio: {torchaudio.__version__}")""")])

# Load data
a([mk_code("""X_fg = np.load(os.path.join(PROCESSED, 'fingernail_images.npy'))
y_fg_str = np.load(os.path.join(PROCESSED, 'fingernail_labels.npy'))
print(f"Fingernails: {X_fg.shape}")

X_af = np.load(os.path.join(PROCESSED, 'audio_mfccs.npy'))
y_af_str = np.load(os.path.join(PROCESSED, 'audio_labels.npy'))
print(f"Audio: {X_af.shape}")

X_accel = np.load(os.path.join(PROCESSED, 'accel_windows.npy'))
y_accel_str = np.load(os.path.join(PROCESSED, 'accel_labels.npy'))
print(f"Accel: {X_accel.shape}")

X_water = np.load(os.path.join(PROCESSED, 'water_features.npy'))
y_water_str = np.load(os.path.join(PROCESSED, 'water_labels.npy'), allow_pickle=True)
print(f"Water: {X_water.shape}")

X_sleep = np.load(os.path.join(PROCESSED, 'sleep_features.npy'))
y_sleep_str = np.load(os.path.join(PROCESSED, 'sleep_labels.npy'), allow_pickle=True)
print(f"Sleep: {X_sleep.shape}")""")])

# Encode
a([mk_code("""le_fg = LabelEncoder()
y_fg = le_fg.fit_transform(y_fg_str).astype(np.int64)
print(f"Fingernail: {dict(zip(le_fg.classes_, [0,1]))}")

le_af = LabelEncoder()
y_af = le_af.fit_transform(y_af_str).astype(np.int64)
print(f"Audio: {dict(zip(le_af.classes_, [0,1]))}")

le_accel = LabelEncoder()
y_accel = le_accel.fit_transform(y_accel_str).astype(np.int64)
print(f"Accel: {dict(zip(le_accel.classes_, range(3)))}")

le_water = LabelEncoder()
y_water = le_water.fit_transform(y_water_str).astype(np.int64)
print(f"Water: {dict(zip(le_water.classes_, [0,1]))}")

le_sleep = LabelEncoder()
y_sleep = le_sleep.fit_transform(y_sleep_str).astype(np.int64)
print(f"Sleep: {dict(zip(le_sleep.classes_, range(3)))}")""")])

# Training utilities
a([mk_code("""def compute_sample_weights(loader):
    all_labels = torch.cat([y for _, y in loader])
    n_samples = len(all_labels)
    classes = torch.unique(all_labels)
    n_classes = len(classes)
    counts = [(all_labels == c).sum().item() for c in classes]
    weights = n_samples / (n_classes * np.array(counts))
    return torch.FloatTensor(weights)


def train_model(model, train_loader, val_loader, test_loader, epochs, lr,
                weight_decay=1e-4, patience=10, scheduler_type='cosine', name='model'):
    \"\"\"Train with class-weighted loss, early stopping, cosine/step LR. Returns model, history, per-split results.\"\"\"
    model = model.to(DEVICE)
    cw = compute_sample_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=cw.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    else:
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    best_vl, best_state, patience_ct = float('inf'), None, 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for ep in range(epochs):
        model.train()
        tl, tc, tt = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * xb.size(0)
            tc += (out.argmax(1) == yb).sum().item()
            tt += xb.size(0)

        model.eval()
        vl, vc, vt = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                vl += loss.item() * xb.size(0)
                vc += (out.argmax(1) == yb).sum().item()
                vt += xb.size(0)

        tl /= tt; vl /= vt; ta = tc / tt; va = vc / vt
        scheduler.step()
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['train_acc'].append(ta); history['val_acc'].append(va)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Ep {ep+1:2d}/{epochs} | TL:{tl:.4f} TA:{ta:.4f} | VL:{vl:.4f} VA:{va:.4f}")

        if vl < best_vl:
            best_vl = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"  Early stop at ep {ep+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate all 3 splits
    results = {}
    for sn, ld in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        model.eval()
        pl, ll = [], []
        with torch.no_grad():
            for xb, yb in ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                pl.extend(out.argmax(1).cpu().numpy())
                ll.extend(yb.cpu().numpy())
        acc = accuracy_score(ll, pl)
        f1 = f1_score(ll, pl, average='weighted', zero_division=0)
        results[sn] = {'accuracy': acc, 'f1': f1, 'preds': np.array(pl), 'labels': np.array(ll)}
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{name}.pth"))
    return model, history, results


def analyze_fit(results):
    \"\"\"Classify model fit: overfitting, underfitting, well-fitted, or reasonable.\"\"\"
    ta = results['train']['accuracy'] * 100
    va = results['val']['accuracy'] * 100
    te = results['test']['accuracy'] * 100
    gap = ta - va
    if ta < 60:
        status = 'UNDERFITTING'
    elif gap > 15:
        status = 'OVERFITTING'
    elif gap < 5 and ta > 80:
        status = 'WELL-FITTED'
    else:
        status = 'REASONABLE' if gap <= 15 else 'SLIGHTLY OVERFITTING'
    return {'train_acc': ta, 'val_acc': va, 'test_acc': te, 'gap': gap, 'status': status,
            'train_f1': results['train']['f1'], 'val_f1': results['val']['f1'], 'test_f1': results['test']['f1']}


def plot_diag(h, a, title):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(h['train_loss'], label='Train', lw=2)
    ax[0].plot(h['val_loss'], label='Val', lw=2)
    ax[0].set_title(f'{title} | {a["status"]}'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(h['train_acc'], label='Train', lw=2)
    ax[1].plot(h['val_acc'], label='Val', lw=2)
    n = len(h['train_acc'])
    ax[1].plot([0, n-1], [a['test_acc']] * 2, 'r--', alpha=.5, label=f'Test: {a["test_acc"]:.1f}%')
    ax[1].set_title(f'{title} | Gap: {a["gap"]:.1f}%'); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()""")])

# ===== FINGERNAIL =====
a([mk_md("""## ============================================
## MODEL 1: FINGERNAIL (Anemia) - PRETRAINED IMAGE ENCODERS
## ============================================
### Pretrained: ResNet-18, ResNet-34, ResNet-50, EfficientNet-B0, MobileNet-V3 Small (ImageNet weights)

**Strategy:** Phase 1 = frozen backbone (train head only). Phase 2 = fine-tune last stages with lower LR.""")])

a([mk_code("""def make_img_encoder(arch, num_classes=2, frozen=True, finetune=False):
    wmap = {
        'resnet18': models.ResNet18_Weights.DEFAULT,
        'resnet34': models.ResNet34_Weights.DEFAULT,
        'resnet50': models.ResNet50_Weights.DEFAULT,
        'efficientnet_b0': models.EfficientNet_B0_Weights.DEFAULT,
        'mobilenet_v3_small': models.MobileNet_V3_Small_Weights.DEFAULT,
    }
    if 'resnet' in arch:
        m = getattr(models, arch)(weights=wmap[arch])
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=wmap[arch])
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == 'mobilenet_v3_small':
        m = models.mobilenet_v3_small(weights=wmap[arch])
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)

    if frozen and not finetune:
        for n, p in m.named_parameters():
            if 'fc' not in n and 'classifier' not in n:
                p.requires_grad = False

    if finetune:
        unfreeze_keys = ['layer3', 'layer4', 'fc', 'features.5', 'features.6',
                         'features.7', 'block.10', 'block.11', 'block.12', 'classifier']
        for n, p in m.named_parameters():
            p.requires_grad = any(k in n for k in unfreeze_keys)

    return m

# Show param counts
for arch in ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_small']:
    m = make_img_encoder(arch, frozen=False)
    print(f"{arch:25s} Total: {sum(p.numel() for p in m.parameters()):>10,}")""")])

a([mk_code("""# Channel-first + 60/20/20 split
X_fg_ch = np.transpose(X_fg, (0, 3, 1, 2)) if X_fg.ndim == 4 and X_fg.shape[-1] == 3 else X_fg
X_fg_tr, X_fg_tm, y_fg_tr, y_fg_tm = train_test_split(X_fg_ch, y_fg, test_size=0.4, random_state=SEED, stratify=y_fg)
X_fg_va, X_fg_te, y_fg_va, y_fg_te = train_test_split(X_fg_tm, y_fg_tm, test_size=0.5, random_state=SEED, stratify=y_fg_tm)
print(f"Train:{len(X_fg_tr)} Val:{len(X_fg_va)} Test:{len(X_fg_te)}")

fg_dl = DataLoader(TensorDataset(torch.FloatTensor(X_fg_tr), torch.LongTensor(y_fg_tr)), 32, True)
fg_vl = DataLoader(TensorDataset(torch.FloatTensor(X_fg_va), torch.LongTensor(y_fg_va)), 64, False)
fg_tl = DataLoader(TensorDataset(torch.FloatTensor(X_fg_te), torch.LongTensor(y_fg_te)), 64, False)""")])

a([mk_code("""fg_cfgs = [
    ('ResNet-18 (frozen)',          'resnet18',           True, False, 20, 1e-3, 1e-4),
    ('ResNet-34 (frozen)',          'resnet34',           True, False, 20, 1e-3, 1e-4),
    ('ResNet-50 (frozen)',          'resnet50',           True, False, 20, 1e-3, 1e-4),
    ('EfficientNet-B0 (frozen)',    'efficientnet_b0',   True, False, 20, 1e-3, 1e-4),
    ('MobileNet-V3 Small (frozen)', 'mobilenet_v3_small', True, False, 20, 1e-3, 1e-4),
    ('ResNet-18 (fine-tune)',       'resnet18',           True, True,  15, 5e-5, 1e-4),
    ('ResNet-50 (fine-tune)',       'resnet50',           True, True,  15, 5e-5, 1e-4),
    ('EffNet-B0 (fine-tune)',       'efficientnet_b0',   True, True,  15, 5e-5, 1e-4),
]

fg_res = {}; fg_hist = {}
for name, arch, frz, ft, ep, lr, wd in fg_cfgs:
    print(f"\n=== {name} ===")
    m = make_img_encoder(arch, num_classes=2, frozen=frz, finetune=ft)
    tp = sum(p.numel() for p in m.parameters())
    trp = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  Total:{tp:,} Trainable:{trp:,}")
    m, h, r = train_model(m, fg_dl, fg_vl, fg_tl, epochs=ep, lr=lr, weight_decay=wd, patience=6, name=name.replace(' ', '_'))
    a = analyze_fit(r); fg_res[name] = a; fg_hist[name] = h
    plot_diag(h, a, name)
    print(f"  >> {a['status']} | Train:{a['train_acc']:.1f}% Val:{a['val_acc']:.1f}% Test:{a['test_acc']:.1f}% Gap:{a['gap']:.1f}%\\n")

print(f"\n{'FINGERNAIL MODELS':<35}|{'Train':>7}|{'Val':>7}|{'Test':>7}|{'Gap':>6}|{'Status'}")
print("-" * 80)
for n, r in sorted(fg_res.items(), key=lambda x: x[1]['test_acc'], reverse=True):
    print(f"{n:<35}|{r['train_acc']:>6.1f}%|{r['val_acc']:>6.1f}%|{r['test_acc']:>6.1f}%|{r['gap']:>5.1f}%|{r['status']}")
best_fg = max(fg_res, key=lambda k: fg_res[k]['test_acc'])
print(f"\n BEST: {best_fg} -- Test: {fg_res[best_fg]['test_acc']:.1f}%")""")])

# ===== AUDIO =====
a([mk_md("""## ============================================
## MODEL 2: AUDIO (Stress) - ARCHITECTURAL VARIANTS
## ============================================
### No ImageNet-pretrained equivalent for MFCCs, so we test stronger CNN+BiLSTM variants.""")])

a([mk_code("""class AudioSimple(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, nc)
    def forward(self, x):
        x = x.unsqueeze(1); x = self.cnn(x); x = self.pool(x).view(x.size(0), -1); return self.fc(x)

class AudioLSTM(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.lstm = nn.LSTM(13, 32, 1, batch_first=True)
        self.fc = nn.Linear(32, nc)
    def forward(self, x):
        o, _ = self.lstm(x.permute(0, 2, 1)); return self.fc(o[:, -1, :])

for c, n in [(AudioSimple, 'Simple CNN'), (AudioLSTM, 'Simple LSTM')]:
    m = c(2); print(f"Audio {n}: {sum(p.numel() for p in m.parameters()):,} params")""")])

a([mk_code("""X_af_tr, X_af_tm, y_af_tr, y_af_tm = train_test_split(X_af, y_af, test_size=0.4, random_state=SEED, stratify=y_af)
X_af_va, X_af_te, y_af_va, y_af_te = train_test_split(X_af_tm, y_af_tm, test_size=0.5, random_state=SEED, stratify=y_af_tm)
af_dl = DataLoader(TensorDataset(torch.FloatTensor(X_af_tr), torch.LongTensor(y_af_tr)), 16, True)
af_vl = DataLoader(TensorDataset(torch.FloatTensor(X_af_va), torch.LongTensor(y_af_va)), 32, False)
af_tl = DataLoader(TensorDataset(torch.FloatTensor(X_af_te), torch.LongTensor(y_af_te)), 32, False)

a_cfgs = [('Audio Simple CNN', 'AudioSimple', 1e-3, 1e-4, 40), ('Audio Simple LSTM', 'AudioLSTM', 1e-3, 1e-4, 40)]
af_res = {}; af_hist = {}
for nm, cls, lr, wd, ep in a_cfgs:
    print(f"\n=== {nm} ===")
    m = eval(cls)(2); print(f"  Params: {sum(p.numel() for p in m.parameters()):,}")
    m, h, r = train_model(m, af_dl, af_vl, af_tl, epochs=ep, lr=lr, weight_decay=wd, patience=10, name=nm.replace(' ', '_'))
    a = analyze_fit(r); af_res[nm] = a; af_hist[nm] = h; plot_diag(h, a, nm)
    print(f"  >> {a['status']} | Tr:{a['train_acc']:.1f}% Va:{a['val_acc']:.1f}% Te:{a['test_acc']:.1f}% Gap:{a['gap']:.1f}%")

best_af = max(af_res, key=lambda k: af_res[k]['test_acc'])
print(f"\n BEST AUDIO: {best_af} -- Test: {af_res[best_af]['test_acc']:.1f}%")""")])

# ===== ACCEL =====
a([mk_md("""## ============================================
## MODEL 3: ACCEL (Fatigue) - ARCHITECTURAL VARIANTS
## ============================================
### Deep 1D-CNN+BiLSTM, Wide CNN (no LSTM), TCN with dilated convolutions""")])

a([mk_code("""class AccelCNN_Fix(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout1d(0.1),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout1d(0.1),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout1d(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(128 * 25, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, nc))
        
    def forward(self, x):
        x = x.permute(0, 2, 1); x = self.cnn(x)
        x = x.reshape(x.size(0), -1) 
        return self.fc(x)

class AccelLSTM_Fix(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.lstm = nn.LSTM(3, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, nc))
        
    def forward(self, x):
        o, _ = self.lstm(x) 
        o = o.permute(0, 2, 1) 
        o = self.pool(o).squeeze(-1) 
        return self.fc(o)

for c, n in [(AccelCNN_Fix, 'Fixed CNN'), (AccelLSTM_Fix, 'Fixed LSTM')]:
    m = c(3); print(f"Accel {n}: {sum(p.numel() for p in m.parameters()):,} params")""")])

a([mk_code("""from sklearn.preprocessing import StandardScaler

X_ac_tr, X_ac_tm, y_ac_tr, y_ac_tm = train_test_split(X_accel, y_accel, test_size=0.4, random_state=SEED, stratify=y_accel)
X_ac_va, X_ac_te, y_ac_va, y_ac_te = train_test_split(X_ac_tm, y_ac_tm, test_size=0.5, random_state=SEED, stratify=y_ac_tm)

b, t, f = X_ac_tr.shape
scaler_ac = StandardScaler()
X_ac_tr_s = scaler_ac.fit_transform(X_ac_tr.reshape(-1, f)).reshape(-1, t, f)
X_ac_va_s = scaler_ac.transform(X_ac_va.reshape(-1, f)).reshape(-1, t, f)
X_ac_te_s = scaler_ac.transform(X_ac_te.reshape(-1, f)).reshape(-1, t, f)

ac_dl = DataLoader(TensorDataset(torch.FloatTensor(X_ac_tr_s), torch.LongTensor(y_ac_tr)), 64, True)
ac_vl = DataLoader(TensorDataset(torch.FloatTensor(X_ac_va_s), torch.LongTensor(y_ac_va)), 128, False)
ac_tl = DataLoader(TensorDataset(torch.FloatTensor(X_ac_te_s), torch.LongTensor(y_ac_te)), 128, False)

a_cfgs2 = [('Accel Fixed CNN', 'AccelCNN_Fix', 5e-4, 1e-4, 40),
           ('Accel Fixed LSTM', 'AccelLSTM_Fix', 5e-4, 1e-4, 40)]
ac_res = {}; ac_hist = {}
for nm, cls, lr, wd, ep in a_cfgs2:
    print(f"\\n=== {nm} ===")
    m = eval(cls)(3); print(f"  Params: {sum(p.numel() for p in m.parameters()):,}")
    m, h, r = train_model(m, ac_dl, ac_vl, ac_tl, epochs=ep, lr=lr, weight_decay=wd, patience=8, name=nm.replace(' ', '_'))
    a = analyze_fit(r); ac_res[nm] = a; ac_hist[nm] = h; plot_diag(h, a, nm)
    print(f"  >> {a['status']} | Tr:{a['train_acc']:.1f}% Va:{a['val_acc']:.1f}% Te:{a['test_acc']:.1f}% Gap:{a['gap']:.1f}%")

best_ac = max(ac_res, key=lambda k: ac_res[k]['test_acc'])
print(f"\\n BEST ACCEL: {best_ac} -- Test: {ac_res[best_ac]['test_acc']:.1f}%")""")])

# ===== WATER =====
a([mk_md("""## ============================================
## MODEL 4: WATER MLP (Dehydration)
## ============================================
### Wide (128-64), Deep (128-64-32), Narrow (32-16)

With 30K samples and 8 features, these models should converge well.""")])

a([mk_code("""def make_mlp(indim, hiddens, nc=2, dr=.3):
    layers = []; p = indim
    for h in hiddens:
        layers += [nn.Linear(p, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dr)]
        p = h
    layers += [nn.Linear(p, nc)]
    return nn.Sequential(*layers)

Xw_tr, Xw_tm, yw_tr, yw_tm = train_test_split(X_water, y_water, test_size=0.4, random_state=SEED, stratify=y_water)
Xw_va, Xw_te, yw_va, yw_te = train_test_split(Xw_tm, yw_tm, test_size=0.5, random_state=SEED, stratify=yw_tm)
w_dl = DataLoader(TensorDataset(torch.FloatTensor(Xw_tr), torch.LongTensor(yw_tr)), 128, True)
w_vl = DataLoader(TensorDataset(torch.FloatTensor(Xw_va), torch.LongTensor(yw_va)), 256, False)
w_tl = DataLoader(TensorDataset(torch.FloatTensor(Xw_te), torch.LongTensor(yw_te)), 256, False)

w_cfgs = [
    ('Water Simple (16)', make_mlp(X_water.shape[1], [16], 2, .1), 1e-3, 1e-4, 30),
    ('Water Simple (32)', make_mlp(X_water.shape[1], [32], 2, .1), 1e-3, 1e-4, 30),
]
w_res = {}; w_hist = {}
for nm, m, lr, wd, ep in w_cfgs:
    print(f"\\n=== {nm} ===")
    print(f"  Params: {sum(p.numel() for p in m.parameters()):,}")
    m, h, r = train_model(m, w_dl, w_vl, w_tl, epochs=ep, lr=lr, weight_decay=wd, patience=7, name=nm.replace(' ', '_'))
    a = analyze_fit(r); w_res[nm] = a; w_hist[nm] = h; plot_diag(h, a, nm)
    print(f"  >> {a['status']} | Tr:{a['train_acc']:.1f}% Va:{a['val_acc']:.1f}% Te:{a['test_acc']:.1f}% Gap:{a['gap']:.1f}%")

best_w = max(w_res, key=lambda k: w_res[k]['test_acc'])
print(f"\\n BEST WATER: {best_w} -- Test: {w_res[best_w]['test_acc']:.1f}%")""")])

# ===== SLEEP =====
a([mk_md("""## ============================================
## MODEL 5: SLEEP ML MODELS (Sleep Disorder)
## ============================================
### Random Forest, SVM, Logistic Regression

Using traditional ML models as tabular datasets with < 400 samples fundamentally underperform with Deep Learning.""")])

a([mk_code("""from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

Xs_tr, Xs_tm, ys_tr, ys_tm = train_test_split(X_sleep, y_sleep, test_size=0.4, random_state=SEED, stratify=y_sleep)
Xs_va, Xs_te, ys_va, ys_te = train_test_split(Xs_tm, ys_tm, test_size=0.5, random_state=SEED, stratify=ys_tm)

scaler = StandardScaler()
Xs_tr_s = scaler.fit_transform(Xs_tr)
Xs_va_s = scaler.transform(Xs_va)
Xs_te_s = scaler.transform(Xs_te)

s_cfgs = [
    ('Sleep Random Forest', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)),
    ('Sleep SVM', SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED)),
    ('Sleep Logistic Regression', LogisticRegression(max_iter=500, random_state=SEED)),
]

s_res = {}
for nm, m in s_cfgs:
    print(f"\\n=== {nm} ===")
    m.fit(Xs_tr_s, ys_tr)
    
    tr_pred = m.predict(Xs_tr_s)
    va_pred = m.predict(Xs_va_s)
    te_pred = m.predict(Xs_te_s)
    
    tr_acc = accuracy_score(ys_tr, tr_pred)
    va_acc = accuracy_score(ys_va, va_pred)
    te_acc = accuracy_score(ys_te, te_pred)
    
    gap = (tr_acc - va_acc) * 100
    if tr_acc < 0.6:
        status = 'UNDERFITTING'
    elif gap > 15:
        status = 'OVERFITTING'
    elif gap < 5 and tr_acc > 0.8:
        status = 'WELL-FITTED'
    else:
        status = 'REASONABLE' if gap <= 15 else 'SLIGHTLY OVERFITTING'
        
    f1 = f1_score(ys_te, te_pred, average='weighted', zero_division=0)
    
    s_res[nm] = {
        'train_acc': tr_acc * 100, 'val_acc': va_acc * 100, 'test_acc': te_acc * 100,
        'gap': gap, 'status': status, 'test_f1': f1, 'train_f1': 0, 'val_f1': 0
    }
    
    print(f"  >> {status} | Tr:{tr_acc*100:.1f}% Va:{va_acc*100:.1f}% Te:{te_acc*100:.1f}% Gap:{gap:.1f}%")
    
    joblib.dump(m, os.path.join(OUTPUT_DIR, f"{nm.replace(' ', '_')}.pkl"))
    
best_s = max(s_res, key=lambda k: s_res[k]['test_acc'])
print(f"\\n BEST SLEEP: {best_s} -- Test: {s_res[best_s]['test_acc']:.1f}%")""")])

# ===== MASTER =====
a([mk_md("""## ============================================
## MASTER COMPARISON - ALL MODELS
## ============================================""")])

a([mk_code("""all_res = {
    'Fingernail': fg_res, 'Audio': af_res,
    'Accelerometer': ac_res, 'Water': w_res, 'Sleep': s_res,
}

print("\\n" + "=" * 120)
print("FINAL COMPARISON - BEST PER MODALITY")
print("=" * 120)
print(f"{'Modality':<16}|{'Model':<35}|{'Train':>7}|{'Val':>7}|{'Test':>7}|{'Gap':>6}|{'F1':>7}|{'Status'}")
print("-" * 120)
for mod, rd in all_res.items():
    bn = max(rd, key=lambda k: rd[k]['test_acc']); r = rd[bn]
    print(f"{mod:<16}|{bn:<35}|{r['train_acc']:>6.1f}%|{r['val_acc']:>6.1f}%|{r['test_acc']:>6.1f}%|{r['gap']:>5.1f}%|{r['test_f1']:>6.4f}|{r['status']}")

rows = []
for mod, rd in all_res.items():
    for nm, r in rd.items():
        rows.append({'Modality': mod, 'Model': nm,
                      'Train Acc': r['train_acc'], 'Val Acc': r['val_acc'],
                      'Test Acc': r['test_acc'], 'Gap': r['gap'],
                      'Test F1': r['test_f1'], 'Status': r['status']})
df = pd.DataFrame(rows).sort_values(['Modality', 'Test Acc'], ascending=[True, False])
print("\\n" + df[['Modality', 'Model', 'Train Acc', 'Val Acc', 'Test Acc', 'Gap', 'Test F1', 'Status']].to_string(index=False))""")])

a([mk_code("""fig, ax = plt.subplots(figsize=(12, 8))
ds = df.sort_values('Test Acc', ascending=True)
cmap = {'OVERFITTING': '#e74c3c', 'UNDERFITTING': '#3498db',
        'WELL-FITTED': '#27ae60', 'REASONABLE': '#f39c12',
        'SLIGHTLY OVERFITTING': '#e67e22'}
colors = [cmap.get(s, '#95a5a6') for s in ds['Status']]
ax.barh(range(len(ds)), ds['Test Acc'].values, color=colors, alpha=.7)
ax.set_yticks(range(len(ds)))
ax.set_yticklabels([f"{r['Modality'][:4]}: {r['Model'][:30]}" for _, r in ds.iterrows()], fontsize=8)
ax.set_xlabel('Test Accuracy (%)'); ax.set_title('All Models by Test Accuracy')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in cmap.items()], loc='lower right')
plt.tight_layout(); plt.show()""")])

a([mk_code("""for mod, rd in all_res.items():
    print(f"\\n{'=' * 70}\\n  {mod.upper()}\\n{'=' * 70}")
    bn = max(rd, key=lambda k: rd[k]['test_acc'])
    for nm, r in sorted(rd.items(), key=lambda x: x[1]['test_acc'], reverse=True):
        mx = " [BEST]" if nm == bn else ""
        print(f"  {nm}{mx}")
        print(f"    Tr:{r['train_acc']:.1f}% Va:{r['val_acc']:.1f}% Te:{r['test_acc']:.1f}% Gap:{r['gap']:.1f}% F1:{r['test_f1']:.4f}")
        if r['status'] == 'OVERFITTING':
            print("    -> Overfitting: try more dropout, weight decay, freeze backbone")
        elif r['status'] == 'UNDERFITTING':
            print("    -> Underfitting: try larger capacity, more epochs, lower LR")
        elif r['status'] == 'WELL-FITTED':
            print("    -> Well-fitted! Use for Module 4 fusion")
        else:
            print("    -> Reasonable fit")
    print(f"  >> Best: {bn} ({rd[bn]['test_acc']:.1f}% test acc)")""")])

a([mk_code("""best = {}
for mod, rd in all_res.items():
    bn = max(rd, key=lambda k: rd[k]['test_acc'])
    best[mod] = (bn, rd[bn])

print("\\n BEST MODELS FOR MODULE 4 FUSION:")
print("=" * 80)
for mod, (bn, r) in best.items():
    print(f"  {mod:<16} | {bn:<35} | Test: {r['test_acc']:.1f}% F1: {r['test_f1']:.4f}")""")])

with open("D:/DL Project/Module3_Pretrained_Models.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"Done. {len(nb['cells'])} cells written.")
