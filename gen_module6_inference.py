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
cells.append(mk_md("""# Module 6: Model Inference & Deployment Preparation
## Multimodal Health State Prediction Using Smartphone Sensors
### VI-A Group ID 13 — Deep Learning Capstone Project

**Goal:**
1. Export the trained model to TorchScript format for deployment
2. Save all preprocessing artifacts (LabelEncoders, StandardScalers) with `joblib`
3. Build a standalone inference pipeline
4. Measure end-to-end inference latency (target < 1 second)
5. Validate the exported model produces identical outputs"""))

# ───────────── Imports ─────────────
cells.append(mk_code("""import os, warnings, pickle, time, json
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE = 'D:/DL Project'
PROCESSED = os.path.join(BASE, 'processed_data')
EXPORT_DIR = os.path.join(BASE, 'exported_model')
os.makedirs(EXPORT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Export directory: {EXPORT_DIR}")
print("Inference Environment Ready ✓")"""))

# ───────────── Load Data ─────────────
cells.append(mk_md("""---
## 1. Load Data & Preprocessing Artifacts
Load the same data used in training to rebuild the model and save preprocessing objects."""))

cells.append(mk_code("""# Load processed data
X_fg = np.load(os.path.join(PROCESSED, 'fingernail_images.npy'))
y_fg_raw = np.load(os.path.join(PROCESSED, 'fingernail_labels.npy'))
le_fg = LabelEncoder().fit(y_fg_raw)
y_fg = le_fg.transform(y_fg_raw)
X_fg = np.transpose(X_fg, (0, 3, 1, 2)) if X_fg.shape[-1] == 3 else X_fg
X_fg = np.nan_to_num(X_fg).astype(np.float32) / 255.0

X_af = np.load(os.path.join(PROCESSED, 'audio_mfccs.npy'))
y_af_raw = np.load(os.path.join(PROCESSED, 'audio_labels.npy'))
le_af = LabelEncoder().fit(y_af_raw)
y_af = le_af.transform(y_af_raw)
b_af, t_af, f_af = X_af.shape
X_af = np.nan_to_num(X_af)
scaler_af = StandardScaler().fit(X_af.reshape(-1, f_af))
X_af = scaler_af.transform(X_af.reshape(-1, f_af)).reshape(b_af, t_af, f_af)

X_ac = np.load(os.path.join(PROCESSED, 'accel_windows.npy'))
y_ac_raw = np.load(os.path.join(PROCESSED, 'accel_labels.npy'))
le_ac = LabelEncoder().fit(y_ac_raw)
y_ac = le_ac.transform(y_ac_raw)
b_ac, t_ac, f_ac = X_ac.shape
X_ac = np.nan_to_num(X_ac)
scaler_ac = StandardScaler().fit(X_ac.reshape(-1, f_ac))
X_ac = scaler_ac.transform(X_ac.reshape(-1, f_ac)).reshape(b_ac, t_ac, f_ac)

X_wa = np.load(os.path.join(PROCESSED, 'water_features.npy'))
y_wa_raw = np.load(os.path.join(PROCESSED, 'water_labels.npy'), allow_pickle=True)
le_wa = LabelEncoder().fit(y_wa_raw)
y_wa = le_wa.transform(y_wa_raw)
scaler_wa = StandardScaler().fit(np.nan_to_num(X_wa))
X_wa = scaler_wa.transform(np.nan_to_num(X_wa))

X_sl = np.load(os.path.join(PROCESSED, 'sleep_features.npy'))
y_sl_raw = np.load(os.path.join(PROCESSED, 'sleep_labels.npy'), allow_pickle=True)
le_sl = LabelEncoder().fit(y_sl_raw)
y_sl = le_sl.transform(y_sl_raw)
scaler_sl = StandardScaler().fit(np.nan_to_num(X_sl))
X_sl = scaler_sl.transform(np.nan_to_num(X_sl))

n_fg = len(np.unique(y_fg))
n_af = len(np.unique(y_af))
n_ac = len(np.unique(y_ac))
n_wa = len(np.unique(y_wa))
n_sl = len(np.unique(y_sl))

print(f"Classes → Fg:{n_fg}, Af:{n_af}, Ac:{n_ac}, Wa:{n_wa}, Sl:{n_sl}")
print("Data & encoders loaded ✓")"""))

# ───────────── Save Preprocessing ─────────────
cells.append(mk_md("""---
## 2. Save Preprocessing Artifacts
Save LabelEncoders and StandardScalers using `joblib` for consistent inference."""))

cells.append(mk_code("""# Save all preprocessing artifacts
artifacts = {
    'le_fg': le_fg, 'le_af': le_af, 'le_ac': le_ac, 'le_wa': le_wa, 'le_sl': le_sl,
    'scaler_af': scaler_af, 'scaler_ac': scaler_ac, 'scaler_wa': scaler_wa, 'scaler_sl': scaler_sl
}

for name, obj in artifacts.items():
    path = os.path.join(EXPORT_DIR, f'{name}.joblib')
    joblib.dump(obj, path)
    print(f"  Saved: {name}.joblib")

# Save model config
config = {
    'n_fg': n_fg, 'n_af': n_af, 'n_ac': n_ac, 'n_wa': n_wa, 'n_sl': n_sl,
    'shape_af': list(X_af.shape[1:]), 'shape_ac': list(X_ac.shape[1:]),
    'shape_wa': int(X_wa.shape[1]), 'shape_sl': int(X_sl.shape[1]),
    'fg_classes': le_fg.classes_.tolist(),
    'af_classes': le_af.classes_.tolist(),
    'ac_classes': le_ac.classes_.tolist(),
    'wa_classes': le_wa.classes_.tolist(),
    'sl_classes': le_sl.classes_.tolist()
}

with open(os.path.join(EXPORT_DIR, 'model_config.json'), 'w') as f:
    json.dump(config, f, indent=2, default=str)
print("  Saved: model_config.json")
print("\\nAll preprocessing artifacts saved ✓")"""))

# ───────────── Architecture ─────────────
cells.append(mk_md("""---
## 3. Rebuild & Load Model"""))

cells.append(mk_code("""# --- Architecture MUST match Module 4 v3 exactly ---
class AudioSimple(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, nc)

class AccelCNN_Fix(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout1d(0.1),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout1d(0.1),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout1d(0.2))
        self.fc = nn.Sequential(nn.Linear(128 * 25, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, nc))

def make_mlp(indim, hiddens, nc=2, dr=0.1):
    layers = []; p = indim
    for h in hiddens:
        layers += [nn.Linear(p, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dr)]
        p = h
    layers += [nn.Linear(p, nc)]
    return nn.Sequential(*layers)

class MultimodalFusion(nn.Module):
    def __init__(self, n_fg, n_af, n_ac, n_wa, n_sl, sl_dim, wa_dim):
        super().__init__()
        self.enc_fg = models.resnet18()
        self.enc_fg.fc = nn.Identity()
        
        _af = AudioSimple(nc=n_af)
        self.enc_af = nn.Sequential(_af.cnn, _af.pool)
        
        _ac = AccelCNN_Fix(nc=n_ac)
        self.enc_ac = _ac.cnn
        
        _wa = make_mlp(wa_dim, [16], nc=n_wa, dr=0.1)
        self.enc_wa = nn.Sequential(*list(_wa.children())[:-1])
        
        CONCAT_DIM = 512 + 32 + 3200 + 16 + sl_dim
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(CONCAT_DIM),
            nn.Linear(CONCAT_DIM, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        
        self.head_fg = nn.Linear(128, n_fg)
        self.head_af = nn.Linear(128, n_af)
        self.head_ac = nn.Linear(128, n_ac)
        self.head_wa = nn.Linear(128, n_wa)
        self.head_sl = nn.Linear(128, n_sl)
        self._encoders_frozen = True
        
    def forward(self, fg, af, ac, wa, sl):
        self.enc_fg.eval(); self.enc_af.eval(); self.enc_ac.eval(); self.enc_wa.eval()
        with torch.no_grad():
            f_fg = self.enc_fg(fg)
            f_af = self.enc_af(af.unsqueeze(1)).view(af.size(0), -1)
            f_ac = self.enc_ac(ac.permute(0, 2, 1))
            f_ac = f_ac.reshape(f_ac.size(0), -1)
            f_wa = self.enc_wa(wa)
        x = torch.cat([f_fg, f_af, f_ac, f_wa, sl], dim=1)
        z = self.fusion(x)
        return self.head_fg(z), self.head_af(z), self.head_ac(z), self.head_wa(z), self.head_sl(z)

# Load trained weights
model = MultimodalFusion(n_fg, n_af, n_ac, n_wa, n_sl, X_sl.shape[1], X_wa.shape[1]).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(BASE, 'multimodal_model.pth'), map_location=DEVICE))
model.eval()
print("Model loaded ✓")"""))

# ───────────── TorchScript Export ─────────────
cells.append(mk_md("""---
## 4. Export to TorchScript
TorchScript serializes the model into a platform-independent format that runs without Python class definitions."""))

cells.append(mk_code("""# Export via torch.jit.trace (more reliable for complex models than torch.jit.script)
model.eval()
model_cpu = model.to('cpu')

# Create dummy inputs matching expected shapes
dummy_fg = torch.randn(1, 3, 224, 224)
dummy_af = torch.randn(1, t_af, f_af)
dummy_ac = torch.randn(1, t_ac, f_ac)
dummy_wa = torch.randn(1, X_wa.shape[1])
dummy_sl = torch.randn(1, X_sl.shape[1])

try:
    # Try tracing
    traced_model = torch.jit.trace(model_cpu, (dummy_fg, dummy_af, dummy_ac, dummy_wa, dummy_sl))
    
    script_path = os.path.join(EXPORT_DIR, 'multimodal_model_scripted.pt')
    traced_model.save(script_path)
    
    file_size = os.path.getsize(script_path) / 1e6
    print(f"TorchScript model saved: {script_path}")
    print(f"File size: {file_size:.1f} MB")
    print("TorchScript export ✓")
except Exception as e:
    print(f"TorchScript tracing failed: {e}")
    print("Falling back to state_dict export...")
    torch.save(model.state_dict(), os.path.join(EXPORT_DIR, 'multimodal_model_fallback.pth'))
    print("State dict exported as fallback ✓")

# Move model back to original device
model = model.to(DEVICE)"""))

# ───────────── Validate Export ─────────────
cells.append(mk_md("""---
## 5. Validate Exported Model
Ensure the exported TorchScript model produces identical outputs to the original."""))

cells.append(mk_code("""# Validate: compare original vs exported model outputs
script_path = os.path.join(EXPORT_DIR, 'multimodal_model_scripted.pt')

if os.path.exists(script_path):
    loaded_model = torch.jit.load(script_path)
    loaded_model.eval()
    
    # Run both models on same input
    test_fg = torch.FloatTensor(X_fg[:5])
    test_af = torch.FloatTensor(X_af[:5])
    test_ac = torch.FloatTensor(X_ac[:5])
    test_wa = torch.FloatTensor(X_wa[:5])
    test_sl = torch.FloatTensor(X_sl[:5])
    
    model_cpu = model.to('cpu')
    model_cpu.eval()
    
    with torch.no_grad():
        orig_out = model_cpu(test_fg, test_af, test_ac, test_wa, test_sl)
        script_out = loaded_model(test_fg, test_af, test_ac, test_wa, test_sl)
    
    # Compare outputs
    all_match = True
    for i, (o, s) in enumerate(zip(orig_out, script_out)):
        max_diff = (o - s).abs().max().item()
        match = max_diff < 1e-5
        names = ['Fingernail', 'Audio', 'Accel', 'Water', 'Sleep']
        print(f"  {names[i]}: max_diff = {max_diff:.8f} {'✓' if match else '✗'}")
        if not match:
            all_match = False
    
    print(f"\\nExport validation: {'✓ PASSED — outputs match' if all_match else '⚠ MISMATCH detected'}")
    
    model = model.to(DEVICE)
else:
    print("TorchScript model not found, skipping validation")"""))

# ───────────── Inference Pipeline ─────────────
cells.append(mk_md("""---
## 6. Standalone Inference Pipeline
A complete end-to-end inference function that can be used in the webapp."""))

cells.append(mk_code("""# Complete inference pipeline
def predict_health_state(model, fg_image, af_mfcc, ac_window, wa_features, sl_features, device='cpu'):
    # End-to-end prediction for a single patient.
    # Args: model, fg_image (3,224,224), af_mfcc (130,40), ac_window (200,3), wa/sl features
    # Returns: dict with predictions and confidence scores
    model.eval()
    
    # Ensure correct shapes and add batch dimension
    if fg_image.shape[-1] == 3:  # HWC → CHW
        fg_image = np.transpose(fg_image, (2, 0, 1))
    fg_t = torch.FloatTensor(fg_image).unsqueeze(0).to(device)
    af_t = torch.FloatTensor(af_mfcc).unsqueeze(0).to(device)
    ac_t = torch.FloatTensor(ac_window).unsqueeze(0).to(device)
    wa_t = torch.FloatTensor(wa_features).unsqueeze(0).to(device)
    sl_t = torch.FloatTensor(sl_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ofg, oaf, oac, owa, osl = model(fg_t, af_t, ac_t, wa_t, sl_t)
    
    results = {}
    outputs = [ofg, oaf, oac, owa, osl]
    names = ['anemia', 'stress', 'fatigue', 'dehydration', 'sleep_disorder']
    label_maps = {
        'anemia': {0: 'NonAnemic', 1: 'Anemic'},
        'stress': {0: 'Not Stressed', 1: 'Stressed'},
        'fatigue': {i: f'Activity_{i}' for i in range(n_ac)},
        'dehydration': {0: 'Good Hydration', 1: 'Poor Hydration'},
        'sleep_disorder': {0: 'None', 1: 'Sleep Apnea', 2: 'Insomnia'}
    }
    
    for name, out in zip(names, outputs):
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_class = int(out.argmax(dim=1).item())
        confidence = float(probs[pred_class])
        
        results[name] = {
            'prediction': label_maps[name].get(pred_class, f'Class_{pred_class}'),
            'class_index': pred_class,
            'confidence': round(confidence, 4),
            'all_probabilities': {label_maps[name].get(i, f'Class_{i}'): round(float(p), 4) 
                                  for i, p in enumerate(probs)}
        }
    
    return results

# Test the pipeline
print("Testing inference pipeline with sample data...")
sample_result = predict_health_state(
    model.to('cpu'), X_fg[0], X_af[0], X_ac[0], X_wa[0], X_sl[0], device='cpu'
)
model = model.to(DEVICE)

for condition, result in sample_result.items():
    print(f"\\n  {condition.upper()}:")
    print(f"    Prediction: {result['prediction']}")
    print(f"    Confidence: {result['confidence']*100:.1f}%")
    print(f"    Probabilities: {result['all_probabilities']}")

print("\\nInference pipeline test ✓")"""))

# ───────────── Latency ─────────────
cells.append(mk_md("""---
## 7. End-to-End Latency Benchmark
Measure full preprocessing + inference latency for 100 calls."""))

cells.append(mk_code("""# Comprehensive latency benchmark
print("Running latency benchmark (100 inference calls)...")
print("="*50)

model_cpu = model.to('cpu')
model_cpu.eval()

# Warmup
for _ in range(5):
    predict_health_state(model_cpu, X_fg[0], X_af[0], X_ac[0], X_wa[0], X_sl[0])

# Benchmark
latencies = []
for i in range(100):
    start = time.perf_counter()
    _ = predict_health_state(model_cpu, X_fg[i % len(X_fg)], X_af[i % len(X_af)], 
                              X_ac[i % len(X_ac)], X_wa[i % len(X_wa)], X_sl[i % len(X_sl)])
    elapsed = (time.perf_counter() - start) * 1000
    latencies.append(elapsed)

model = model.to(DEVICE)

print(f"\\n  Mean latency:   {np.mean(latencies):.2f} ms")
print(f"  Median latency: {np.median(latencies):.2f} ms")
print(f"  Std latency:    {np.std(latencies):.2f} ms")
print(f"  Min latency:    {np.min(latencies):.2f} ms")
print(f"  Max latency:    {np.max(latencies):.2f} ms")
print(f"  P95 latency:    {np.percentile(latencies, 95):.2f} ms")
print(f"  P99 latency:    {np.percentile(latencies, 99):.2f} ms")

target_met = np.mean(latencies) < 1000
print(f"\\n  Target < 1 second: {'✓ MET' if target_met else '✗ NOT MET'}")
print(f"  Throughput: ~{1000/np.mean(latencies):.1f} predictions/second")"""))

# ───────────── Model Card ─────────────
cells.append(mk_md("""---
## 8. Export Summary & Model Card"""))

cells.append(mk_code("""# Generate model card
print("="*60)
print("MODEL CARD — Multimodal Health State Prediction")
print("="*60)
print(f"Model Name:       MultimodalFusion v1")
print(f"Architecture:     ResNet18 + AudioCNN + AccelCNN + MLP x2 -> Concat Fusion -> MTL Heads")
print(f"Training Data:    5000 synthetic patients (70/15/15 split)")
print(f"Framework:        PyTorch {torch.__version__}")
print(f"")
print(f"Input Requirements:")
print(f"  - Fingernail Image: 3x224x224 (RGB, normalized 0-1)")
print(f"  - Audio MFCC:       {t_af}x{f_af} (StandardScaled)")
print(f"  - Accelerometer:    {t_ac}x{f_ac} (StandardScaled)")
print(f"  - Water Features:   {X_wa.shape[1]} features (StandardScaled)")
print(f"  - Sleep Features:   {X_sl.shape[1]} features (StandardScaled)")
print(f"")
print(f"Output: 5 classification heads")
print(f"  - Anemia:        {n_fg} classes")
print(f"  - Stress:        {n_af} classes")
print(f"  - Fatigue:       {n_ac} classes")
print(f"  - Dehydration:   {n_wa} classes")
print(f"  - Sleep Disorder:{n_sl} classes")
print(f"")
print("Exported Files:")

export_files = os.listdir(EXPORT_DIR)
for f in sorted(export_files):
    fpath = os.path.join(EXPORT_DIR, f)
    size = os.path.getsize(fpath) / 1024
    unit = 'KB' if size < 1024 else 'MB'
    size_val = size if size < 1024 else size / 1024
    print(f"  {f}: {size_val:.1f} {unit}")

print(f"\\nTotal export size: {sum(os.path.getsize(os.path.join(EXPORT_DIR, f)) for f in export_files)/1e6:.1f} MB")"""))

cells.append(mk_md("""---
### ✅ Module 6 Complete — Inference Pipeline Ready

**Exported artifacts in `exported_model/`:**
- `multimodal_model_scripted.pt` — TorchScript model (no Python needed)
- `le_*.joblib` — LabelEncoders for each modality
- `scaler_*.joblib` — StandardScalers for feature normalization
- `model_config.json` — Architecture configuration

**Next:** Run the webapp (`webapp/app.py`) using these exported artifacts.

---
### Version Log
| Version | Date | Notes |
|---------|------|-------|
| v1.0 | 2026-04-19 | Initial export from Module 4 training |"""))

nb["cells"] = cells

out_path = os.path.join(BASE, "Module6_Inference.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Module 6 Inference notebook generated: {out_path}")
