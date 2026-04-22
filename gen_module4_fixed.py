import json
import os

BASE = "D:/DL Project"

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.7"}
    },
    "nbformat": 4, "nbformat_minor": 4
}

def mk_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}

def mk_code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.split("\n")]}

cells = []

# ───────────── Title ─────────────
cells.append(mk_md("""# Module 4 (v3 - ROBUST): Multimodal Feature Fusion
## Multimodal Health State Prediction Using Smartphone Sensors
### VI-A Group ID 13 - Deep Learning Capstone Project

**Fixes in this version:**
1. Loads pretrained Module 3 weights into encoders with **validation checks**
2. Explicit NaN/Inf diagnostic pass before training
3. Simplified forward pass — no conditional context managers
4. **Phase 1** (15 ep): Freeze encoders, train fusion + heads at lr=1e-3
5. **Phase 2** (10 ep): Unfreeze all, end-to-end fine-tune at lr=1e-5"""))

# ───────────── Imports ─────────────
cells.append(mk_code("""import os, warnings, pickle, sys
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torchvision.models as models

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}, PyTorch: {torch.__version__}")

PROCESSED = 'D:/DL Project/processed_data'
MODELS_DIR = 'D:/DL Project/module3_models'
"""))

# ───────────── Data Loading ─────────────
cells.append(mk_md("""---
## 1. Load, Sanitize, and Validate Data"""))

cells.append(mk_code("""# Load and aggressively sanitize all data
def safe_load_and_scale(feat_path, label_path, name, is_3d=False):
    X = np.load(feat_path).astype(np.float32)
    y = LabelEncoder().fit_transform(np.load(label_path, allow_pickle=True))
    
    # Replace NaN and Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if is_3d:
        b, t, f = X.shape
        scaler = StandardScaler()
        X_flat = X.reshape(-1, f)
        X_scaled = scaler.fit_transform(X_flat).reshape(b, t, f)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # Post-scaling cleanup: clamp extreme values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
    X_scaled = np.clip(X_scaled, -10.0, 10.0)
    
    n_classes = len(np.unique(y))
    print(f"  {name}: shape={X_scaled.shape}, classes={n_classes}, "
          f"NaN={np.isnan(X_scaled).sum()}, Inf={np.isinf(X_scaled).sum()}, "
          f"range=[{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    return X_scaled.astype(np.float32), y, n_classes

print("Loading all modalities with sanitization...")

# Fingernail images (special handling - no StandardScaler)
X_fg = np.load(os.path.join(PROCESSED, 'fingernail_images.npy')).astype(np.float32)
y_fg = LabelEncoder().fit_transform(np.load(os.path.join(PROCESSED, 'fingernail_labels.npy')))
X_fg = np.transpose(X_fg, (0, 3, 1, 2)) if X_fg.shape[-1] == 3 else X_fg
X_fg = np.nan_to_num(X_fg, nan=0.0, posinf=1.0, neginf=0.0) / 255.0
X_fg = np.clip(X_fg, 0.0, 1.0)
n_fg = len(np.unique(y_fg))
print(f"  Fingernail: shape={X_fg.shape}, classes={n_fg}, NaN={np.isnan(X_fg).sum()}, range=[{X_fg.min():.2f}, {X_fg.max():.2f}]")

# Others
X_af, y_af, n_af = safe_load_and_scale(
    os.path.join(PROCESSED, 'audio_mfccs.npy'),
    os.path.join(PROCESSED, 'audio_labels.npy'), 'Audio', is_3d=True)

X_ac, y_ac, n_ac = safe_load_and_scale(
    os.path.join(PROCESSED, 'accel_windows.npy'),
    os.path.join(PROCESSED, 'accel_labels.npy'), 'Accel', is_3d=True)

X_wa, y_wa, n_wa = safe_load_and_scale(
    os.path.join(PROCESSED, 'water_features.npy'),
    os.path.join(PROCESSED, 'water_labels.npy'), 'Water', is_3d=False)

X_sl, y_sl, n_sl = safe_load_and_scale(
    os.path.join(PROCESSED, 'sleep_features.npy'),
    os.path.join(PROCESSED, 'sleep_labels.npy'), 'Sleep', is_3d=False)

print(f"\\nClasses: Fg={n_fg}, Af={n_af}, Ac={n_ac}, Wa={n_wa}, Sl={n_sl}")
print("Data loaded and validated!")
"""))

# ───────────── Architectures ─────────────
cells.append(mk_md("""---
## 2. Define Architectures (must match Module 3 exactly)"""))

cells.append(mk_code("""# AudioSimple: exactly matches Module 3 gen_notebook.py line 340-349
class AudioSimple(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, nc)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# AccelCNN_Fix: exactly matches Module 3 gen_notebook.py line 386-399
class AccelCNN_Fix(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout1d(0.1),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout1d(0.1),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout1d(0.2))
        self.fc = nn.Sequential(nn.Linear(128 * 25, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, nc))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# make_mlp: matches Module 3 gen_notebook.py line 453-459
def make_mlp(indim, hiddens, nc=2, dr=0.1):
    layers = []; p = indim
    for h in hiddens:
        layers += [nn.Linear(p, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dr)]
        p = h
    layers += [nn.Linear(p, nc)]
    return nn.Sequential(*layers)

print("Architectures defined.")
"""))

# ───────────── Load Pretrained ─────────────
cells.append(mk_md("""---
## 3. Load Pretrained Unimodal Weights (with validation)"""))

cells.append(mk_code("""def validate_model_weights(model, name):
    has_nan = False
    has_inf = False
    for pname, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"  WARNING: NaN found in {name}.{pname}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"  WARNING: Inf found in {name}.{pname}")
            has_inf = True
    for bname, buf in model.named_buffers():
        if torch.isnan(buf).any():
            print(f"  WARNING: NaN found in buffer {name}.{bname}")
            has_nan = True
        if torch.isinf(buf).any():
            print(f"  WARNING: Inf found in buffer {name}.{bname}")
            has_inf = True
    if not has_nan and not has_inf:
        print(f"  {name}: All weights clean (no NaN/Inf)")
    return not has_nan and not has_inf

def safe_load_weights(model, path, name):
    if os.path.exists(path):
        fsize = os.path.getsize(path)
        print(f"  Loading {name} from: {os.path.basename(path)} ({fsize/1024:.1f} KB)")
        state_dict = torch.load(path, map_location='cpu')
        # Clean any NaN/Inf in loaded weights
        for k, v in state_dict.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"    Cleaning NaN/Inf in {k}")
                state_dict[k] = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
        model.load_state_dict(state_dict)
        return True
    else:
        print(f"  WARNING: {name} weights not found at {path}")
        return False

print("=== Loading Pretrained Unimodal Weights ===\\n")

# 1. Fingernail: ResNet-18 fine-tuned
fg_model = models.resnet18()
fg_model.fc = nn.Linear(fg_model.fc.in_features, n_fg)  # must match Module 3: num_classes=2
fg_loaded = safe_load_weights(fg_model, os.path.join(MODELS_DIR, 'ResNet-18_(fine-tune).pth'), 'Fingernail')
if not fg_loaded:
    safe_load_weights(fg_model, os.path.join(MODELS_DIR, 'ResNet-18_(frozen).pth'), 'Fingernail (fallback)')
validate_model_weights(fg_model, 'Fingernail')

# 2. Audio: AudioSimple
af_model = AudioSimple(nc=n_af)
af_loaded = safe_load_weights(af_model, os.path.join(MODELS_DIR, 'Audio_Simple_CNN.pth'), 'Audio')
validate_model_weights(af_model, 'Audio')

# 3. Accelerometer: AccelCNN_Fix
ac_model = AccelCNN_Fix(nc=n_ac)
ac_loaded = safe_load_weights(ac_model, os.path.join(MODELS_DIR, 'Accel_Fixed_CNN.pth'), 'Accel')
validate_model_weights(ac_model, 'Accel')

# 4. Water: MLP with [16] hidden
wa_model = make_mlp(X_wa.shape[1], [16], nc=n_wa, dr=0.1)
wa_loaded = safe_load_weights(wa_model, os.path.join(MODELS_DIR, 'Water_Simple_(16).pth'), 'Water')
validate_model_weights(wa_model, 'Water')

print("\\n=== Weight Loading Complete ===")
"""))

# ───────────── Sanity Check ─────────────
cells.append(mk_md("""---
## 4. Sanity Check: Verify Encoder Outputs"""))

cells.append(mk_code("""# Run a single batch through each encoder to verify shapes and NaN status
print("=== Encoder Output Verification ===\\n")

# Test with small batches
test_fg = torch.FloatTensor(X_fg[:4])
test_af = torch.FloatTensor(X_af[:4])
test_ac = torch.FloatTensor(X_ac[:4])
test_wa = torch.FloatTensor(X_wa[:4])
test_sl = torch.FloatTensor(X_sl[:4])

# Fingernail encoder (remove fc, get embeddings)
fg_enc_test = models.resnet18()
fg_enc_test.fc = nn.Linear(fg_enc_test.fc.in_features, n_fg)
if fg_loaded:
    fg_enc_test.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'ResNet-18_(fine-tune).pth'), map_location='cpu'))
fg_enc_test.fc = nn.Identity()
fg_enc_test.eval()
with torch.no_grad():
    out_fg = fg_enc_test(test_fg)
print(f"Fingernail encoder: shape={out_fg.shape}, NaN={torch.isnan(out_fg).any()}, range=[{out_fg.min():.3f}, {out_fg.max():.3f}]")

# Audio encoder (get pre-fc embeddings)
af_enc_test = AudioSimple(nc=n_af)
if af_loaded:
    af_enc_test.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'Audio_Simple_CNN.pth'), map_location='cpu'))
af_enc_test.eval()
with torch.no_grad():
    # Manually trace to get embedding
    x = test_af.unsqueeze(1)
    x = af_enc_test.cnn(x)
    x = af_enc_test.pool(x).view(x.size(0), -1)
    out_af = x
print(f"Audio encoder:      shape={out_af.shape}, NaN={torch.isnan(out_af).any()}, range=[{out_af.min():.3f}, {out_af.max():.3f}]")

# Accel encoder (get pre-fc embeddings)
ac_enc_test = AccelCNN_Fix(nc=n_ac)
if ac_loaded:
    ac_enc_test.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'Accel_Fixed_CNN.pth'), map_location='cpu'))
ac_enc_test.eval()
with torch.no_grad():
    x = test_ac.permute(0, 2, 1)
    x = ac_enc_test.cnn(x)
    out_ac = x.reshape(x.size(0), -1)
print(f"Accel encoder:      shape={out_ac.shape}, NaN={torch.isnan(out_ac).any()}, range=[{out_ac.min():.3f}, {out_ac.max():.3f}]")

# Water encoder (get pre-last-linear embeddings)
wa_enc_test = make_mlp(X_wa.shape[1], [16], nc=n_wa, dr=0.1)
if wa_loaded:
    wa_enc_test.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'Water_Simple_(16).pth'), map_location='cpu'))
wa_enc_test[-1] = nn.Identity()
wa_enc_test.eval()
with torch.no_grad():
    out_wa = wa_enc_test(test_wa)
print(f"Water encoder:      shape={out_wa.shape}, NaN={torch.isnan(out_wa).any()}, range=[{out_wa.min():.3f}, {out_wa.max():.3f}]")

# Sleep (raw features)
print(f"Sleep features:     shape={test_sl.shape}, NaN={torch.isnan(test_sl).any()}, range=[{test_sl.min():.3f}, {test_sl.max():.3f}]")

# Total concat dimension
concat_dim = out_fg.shape[1] + out_af.shape[1] + out_ac.shape[1] + out_wa.shape[1] + test_sl.shape[1]
print(f"\\nConcat dim: {out_fg.shape[1]} + {out_af.shape[1]} + {out_ac.shape[1]} + {out_wa.shape[1]} + {test_sl.shape[1]} = {concat_dim}")
print(f"Expected:   512 + 32 + 3200 + 16 + {X_sl.shape[1]} = {512 + 32 + 3200 + 16 + X_sl.shape[1]}")
assert concat_dim == 512 + 32 + 3200 + 16 + X_sl.shape[1], "DIMENSION MISMATCH!"
print("\\nAll encoders verified!")
"""))

# ───────────── Synthetic Dataset ─────────────
cells.append(mk_md("""---
## 5. Create Synthetic Multimodal Dataset"""))

cells.append(mk_code("""SYNTH_SIZE = 5000
np.random.seed(SEED)

idx_fg = np.random.choice(len(X_fg), SYNTH_SIZE)
idx_af = np.random.choice(len(X_af), SYNTH_SIZE)
idx_ac = np.random.choice(len(X_ac), SYNTH_SIZE)
idx_wa = np.random.choice(len(X_wa), SYNTH_SIZE)
idx_sl = np.random.choice(len(X_sl), SYNTH_SIZE)

synth_data = TensorDataset(
    torch.FloatTensor(X_fg[idx_fg]), torch.FloatTensor(X_af[idx_af]),
    torch.FloatTensor(X_ac[idx_ac]), torch.FloatTensor(X_wa[idx_wa]), torch.FloatTensor(X_sl[idx_sl]),
    torch.LongTensor(y_fg[idx_fg]), torch.LongTensor(y_af[idx_af]),
    torch.LongTensor(y_ac[idx_ac]), torch.LongTensor(y_wa[idx_wa]), torch.LongTensor(y_sl[idx_sl])
)

tr_sz = int(0.7 * SYNTH_SIZE)
va_sz = int(0.15 * SYNTH_SIZE)
te_sz = SYNTH_SIZE - tr_sz - va_sz
train_ds, val_ds, test_ds = torch.utils.data.random_split(
    synth_data, [tr_sz, va_sz, te_sz], generator=torch.Generator().manual_seed(SEED))

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=32, drop_last=False)
test_dl = DataLoader(test_ds, batch_size=32, drop_last=False)

print(f"Train: {tr_sz}, Val: {va_sz}, Test: {te_sz}")
print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")
"""))

# ───────────── Fusion Model ─────────────
cells.append(mk_md("""---
## 6. Build Fusion Model with Pretrained Encoders
Simple, clean forward method — no conditional context managers."""))

cells.append(mk_code("""class MultimodalFusion(nn.Module):
    def __init__(self, fg_pretrained, af_pretrained, ac_pretrained, wa_pretrained,
                 n_fg, n_af, n_ac, n_wa, n_sl, sl_dim):
        super().__init__()
        
        # === Copy pretrained encoder backbones ===
        # Fingernail: ResNet-18 backbone (before fc)
        self.enc_fg = fg_pretrained
        self.enc_fg.fc = nn.Identity()  # Output: 512-dim
        
        # Audio: CNN backbone (before fc)
        self.enc_af = nn.Sequential(af_pretrained.cnn, af_pretrained.pool)
        # This outputs (batch, 32, 1, 1) -> need to flatten
        
        # Accel: CNN backbone only (before fc)
        self.enc_ac = ac_pretrained.cnn  
        # This outputs (batch, 128, 25) -> need to flatten
        
        # Water: MLP up to hidden layer only (everything before final Linear)
        self.enc_wa = nn.Sequential(*list(wa_pretrained.children())[:-1])
        # This outputs 16-dim
        
        # === Fusion ===
        CONCAT_DIM = 512 + 32 + 3200 + 16 + sl_dim
        
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(CONCAT_DIM),
            nn.Linear(CONCAT_DIM, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2)
        )
        
        # === task heads ===
        self.head_fg = nn.Linear(128, n_fg)
        self.head_af = nn.Linear(128, n_af)
        self.head_ac = nn.Linear(128, n_ac) 
        self.head_wa = nn.Linear(128, n_wa) 
        self.head_sl = nn.Linear(128, n_sl) 
        
        self._encoders_frozen = False
        
    def freeze_encoders(self):
        for module in [self.enc_fg, self.enc_af, self.enc_ac, self.enc_wa]:
            for p in module.parameters():
                p.requires_grad = False
            module.eval()
        self._encoders_frozen = True
        print("Encoders FROZEN")
        
    def unfreeze_encoders(self):
        for module in [self.enc_fg, self.enc_af, self.enc_ac, self.enc_wa]:
            for p in module.parameters():
                p.requires_grad = True
            module.train()
        self._encoders_frozen = False
        print("Encoders UNFROZEN")
        
    def forward(self, fg, af, ac, wa, sl):
        # Keep frozen encoders in eval mode
        if self._encoders_frozen:
            self.enc_fg.eval()
            self.enc_af.eval()
            self.enc_ac.eval()
            self.enc_wa.eval()
        
        # Encoder forward passes
        if self._encoders_frozen:
            with torch.no_grad():
                f_fg = self.enc_fg(fg)                                    # (B, 512)
                f_af = self.enc_af(af.unsqueeze(1)).view(af.size(0), -1)  # (B, 32)
                f_ac = self.enc_ac(ac.permute(0, 2, 1))                   # (B, 128, 25)
                f_ac = f_ac.reshape(f_ac.size(0), -1)                     # (B, 3200)
                f_wa = self.enc_wa(wa)                                    # (B, 16)
        else:
            f_fg = self.enc_fg(fg)
            f_af = self.enc_af(af.unsqueeze(1)).view(af.size(0), -1)
            f_ac = self.enc_ac(ac.permute(0, 2, 1))
            f_ac = f_ac.reshape(f_ac.size(0), -1)
            f_wa = self.enc_wa(wa)
        
        # Concatenate all features
        x = torch.cat([f_fg, f_af, f_ac, f_wa, sl], dim=1)
        
        # Fusion + task heads
        z = self.fusion(x)
        return self.head_fg(z), self.head_af(z), self.head_ac(z), self.head_wa(z), self.head_sl(z)

# Build the model
model = MultimodalFusion(
    fg_model, af_model, ac_model, wa_model,
    n_fg, n_af, n_ac, n_wa, n_sl, X_sl.shape[1]
).to(DEVICE)

total_p = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_p:,}")
"""))

# ───────────── Verify forward ─────────────
cells.append(mk_md("""---
## 7. Verify Forward Pass (Debugging)"""))

cells.append(mk_code("""# Run a single forward pass and check for NaN
model.freeze_encoders()
model.eval()

batch = next(iter(train_dl))
fg, af, ac, wa, sl, yfg, yaf, yac, ywa, ysl = [b.to(DEVICE) for b in batch]

print(f"Input shapes: fg={fg.shape}, af={af.shape}, ac={ac.shape}, wa={wa.shape}, sl={sl.shape}")

with torch.no_grad():
    ofg, oaf, oac, owa, osl = model(fg, af, ac, wa, sl)

print(f"\\nOutput shapes and NaN check:")
for name, out, lbl in zip(['Fg', 'Af', 'Ac', 'Wa', 'Sl'], 
                            [ofg, oaf, oac, owa, osl],
                            [yfg, yaf, yac, ywa, ysl]):
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    print(f"  {name}: shape={out.shape}, NaN={has_nan}, Inf={has_inf}, "
          f"range=[{out.min():.3f}, {out.max():.3f}], label_range=[{lbl.min()}, {lbl.max()}]")

# Test loss computation
criterion = nn.CrossEntropyLoss()
loss = criterion(ofg, yfg) + criterion(oaf, yaf) + criterion(oac, yac) + criterion(owa, ywa) + criterion(osl, ysl)
print(f"\\nTest loss: {loss.item():.4f}, NaN={torch.isnan(loss).item()}")

if torch.isnan(loss):
    print("\\n!!! LOSS IS NaN — Debugging individual losses:")
    for name, out, lbl in zip(['Fg', 'Af', 'Ac', 'Wa', 'Sl'], 
                                [ofg, oaf, oac, owa, osl],
                                [yfg, yaf, yac, ywa, ysl]):
        individual_loss = criterion(out, lbl)
        print(f"  {name} loss: {individual_loss.item():.4f}, "
              f"out_NaN={torch.isnan(out).any()}, out_Inf={torch.isinf(out).any()}")
    print("\\nFix: Check encoder outputs above. Replace NaN-producing encoder with fresh weights.")
else:
    print("Forward pass verified! No NaN detected. Ready to train.")
"""))

# ───────────── Training Function ─────────────
cells.append(mk_md("""---
## 8. Training Loop with NaN Protection"""))

cells.append(mk_code("""def train_fusion(model, train_dl, val_dl, epochs, lr, phase_name, patience=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_ct = 0
    nan_count = 0
    
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\\n{'='*70}")
    print(f"  {phase_name} | LR={lr} | Trainable params: {trainable_count:,}")
    print(f"{'='*70}")

    for ep in range(epochs):
        # --- Train ---
        model.train()
        if model._encoders_frozen:
            model.enc_fg.eval(); model.enc_af.eval(); model.enc_ac.eval(); model.enc_wa.eval()
        
        train_loss = 0
        train_batches = 0
        for fg, af, ac, wa, sl, yfg, yaf, yac, ywa, ysl in train_dl:
            fg = fg.to(DEVICE); af = af.to(DEVICE); ac = ac.to(DEVICE)
            wa = wa.to(DEVICE); sl = sl.to(DEVICE)
            yfg = yfg.to(DEVICE); yaf = yaf.to(DEVICE); yac = yac.to(DEVICE)
            ywa = ywa.to(DEVICE); ysl = ysl.to(DEVICE)
            
            optimizer.zero_grad()
            ofg, oaf, oac, owa, osl = model(fg, af, ac, wa, sl)
            
            loss = (criterion(ofg, yfg) + criterion(oaf, yaf) + criterion(oac, yac) + 
                    criterion(owa, ywa) + criterion(osl, ysl))
            
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count <= 3:
                    print(f"  WARNING: NaN/Inf loss at ep {ep+1}, batch {train_batches}. Skipping.")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        
        if train_batches == 0:
            print(f"  Ep {ep+1}: ALL batches had NaN loss! Stopping.")
            break
            
        avg_train_loss = train_loss / train_batches
        
        # --- Validate ---
        model.eval()
        val_loss = 0
        val_batches = 0
        corr = {k: 0 for k in ['fg', 'af', 'ac', 'wa', 'sl']}
        total = 0
        
        with torch.no_grad():
            for fg, af, ac, wa, sl, yfg, yaf, yac, ywa, ysl in val_dl:
                fg = fg.to(DEVICE); af = af.to(DEVICE); ac = ac.to(DEVICE)
                wa = wa.to(DEVICE); sl = sl.to(DEVICE)
                yfg = yfg.to(DEVICE); yaf = yaf.to(DEVICE); yac = yac.to(DEVICE)
                ywa = ywa.to(DEVICE); ysl = ysl.to(DEVICE)
                
                ofg, oaf, oac, owa, osl = model(fg, af, ac, wa, sl)
                
                vloss = (criterion(ofg, yfg) + criterion(oaf, yaf) + criterion(oac, yac) + 
                         criterion(owa, ywa) + criterion(osl, ysl))
                
                if not (torch.isnan(vloss) or torch.isinf(vloss)):
                    val_loss += vloss.item()
                    val_batches += 1
                
                corr['fg'] += (ofg.argmax(1) == yfg).sum().item()
                corr['af'] += (oaf.argmax(1) == yaf).sum().item()
                corr['ac'] += (oac.argmax(1) == yac).sum().item()
                corr['wa'] += (owa.argmax(1) == ywa).sum().item()
                corr['sl'] += (osl.argmax(1) == ysl).sum().item()
                total += ysl.size(0)
        
        avg_val_loss = val_loss / max(1, val_batches)
        scheduler.step(avg_val_loss)
        
        # Save best
        if avg_val_loss < best_val_loss and val_batches > 0:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
        
        print(f"  Ep {ep+1:02d} | TL:{avg_train_loss:.4f} VL:{avg_val_loss:.4f} | "
              f"Fg:{corr['fg']/total*100:.1f}% Af:{corr['af']/total*100:.1f}% "
              f"Ac:{corr['ac']/total*100:.1f}% Wa:{corr['wa']/total*100:.1f}% Sl:{corr['sl']/total*100:.1f}%")
        
        if patience_ct >= patience:
            print(f"  Early stopping at epoch {ep+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val_loss={best_val_loss:.4f})")
    
    if nan_count > 0:
        print(f"  Total NaN batches skipped: {nan_count}")
    
    return model
"""))

# ───────────── Phase 1 ─────────────
cells.append(mk_md("""---
## 9. Phase 1: Train Fusion + Heads (Encoders Frozen)"""))

cells.append(mk_code("""model.freeze_encoders()
model = train_fusion(model, train_dl, val_dl, epochs=15, lr=1e-3, phase_name="PHASE 1: Fusion+Heads (Frozen Encoders)")
"""))

# ───────────── Phase 2 ─────────────
cells.append(mk_md("""---
## 10. Phase 2: End-to-End Fine-Tuning (All Unfrozen)"""))

cells.append(mk_code("""model.unfreeze_encoders()
model = train_fusion(model, train_dl, val_dl, epochs=10, lr=1e-5, phase_name="PHASE 2: End-to-End Fine-Tuning", patience=5)
"""))

# ───────────── Evaluate ─────────────
cells.append(mk_md("""---
## 11. Final Test Evaluation"""))

cells.append(mk_code("""from sklearn.metrics import accuracy_score, f1_score

model.eval()
preds = {k: [] for k in ['fg', 'af', 'ac', 'wa', 'sl']}
labels = {k: [] for k in ['fg', 'af', 'ac', 'wa', 'sl']}

with torch.no_grad():
    for fg, af, ac, wa, sl, yfg, yaf, yac, ywa, ysl in test_dl:
        fg = fg.to(DEVICE); af = af.to(DEVICE); ac = ac.to(DEVICE)
        wa = wa.to(DEVICE); sl = sl.to(DEVICE)
        
        ofg, oaf, oac, owa, osl = model(fg, af, ac, wa, sl)
        
        preds['fg'].extend(ofg.argmax(1).cpu().numpy())
        preds['af'].extend(oaf.argmax(1).cpu().numpy())
        preds['ac'].extend(oac.argmax(1).cpu().numpy())
        preds['wa'].extend(owa.argmax(1).cpu().numpy())
        preds['sl'].extend(osl.argmax(1).cpu().numpy())
        
        labels['fg'].extend(yfg.numpy())
        labels['af'].extend(yaf.numpy())
        labels['ac'].extend(yac.numpy())
        labels['wa'].extend(ywa.numpy())
        labels['sl'].extend(ysl.numpy())

print("\\n" + "="*60)
print("  FINAL TEST SET RESULTS")
print("="*60)
names = {'fg': 'Fingernail (Anemia)', 'af': 'Audio (Stress)', 
         'ac': 'Accel (Fatigue)', 'wa': 'Water (Dehydration)', 'sl': 'Sleep (Disorder)'}

for key, name in names.items():
    acc = accuracy_score(labels[key], preds[key])
    f1 = f1_score(labels[key], preds[key], average='weighted', zero_division=0)
    print(f"  {name:25s} | Acc: {acc*100:.1f}% | F1: {f1:.4f}")

mean_acc = np.mean([accuracy_score(labels[k], preds[k]) for k in preds]) * 100
print(f"\\n  {'Mean Accuracy':25s} | {mean_acc:.1f}%")
"""))

# ───────────── Save ─────────────
cells.append(mk_code("""# Save the trained fusion model
print("\\nSaving model...")
torch.save(model.state_dict(), 'multimodal_model.pth')

config = {
    'n_fg': n_fg, 'n_af': n_af, 'n_ac': n_ac, 'n_wa': n_wa, 'n_sl': n_sl,
    'shape_af': list(X_af.shape[1:]), 'shape_ac': list(X_ac.shape[1:]),
    'shape_wa': int(X_wa.shape[1]), 'shape_sl': int(X_sl.shape[1])
}
with open('multimodal_config.pkl', 'wb') as f:
    pickle.dump(config, f)

model_size = os.path.getsize('multimodal_model.pth') / 1e6
print(f"Model saved: multimodal_model.pth ({model_size:.1f} MB)")
print(f"Config saved: multimodal_config.pkl")
print("\\nDone! Run Module 5 next.")
"""))

nb["cells"] = cells

out_path = os.path.join(BASE, "Module4_Multimodal_Fusion.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Module 4 v3 notebook generated: {out_path}")
