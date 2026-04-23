import os
import pickle
import io
import csv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'multimodal_model.pth')
CONFIG_PATH = os.path.join(BASE_DIR, 'multimodal_config.pkl')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# PYTORCH ARCHITECTURE PROFILES
# ==========================================
class AudioSimple(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, nc)

class AccelCNN_Fix(nn.Module):
    def __init__(self, nc):
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
    def __init__(self, cfg):
        super().__init__()
        self.enc_fg = models.resnet18()
        self.enc_fg.fc = nn.Identity()
        
        _af = AudioSimple(nc=cfg['n_af'])
        self.enc_af = nn.Sequential(_af.cnn, _af.pool)
        
        _ac = AccelCNN_Fix(nc=cfg['n_ac'])
        self.enc_ac = _ac.cnn
        
        _wa = make_mlp(cfg['shape_wa'], [16], nc=cfg['n_wa'], dr=0.1)
        self.enc_wa = nn.Sequential(*list(_wa.children())[:-1])
        
        CONCAT_DIM = 512 + 32 + 3200 + 16 + cfg['shape_sl']
        
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(CONCAT_DIM),
            nn.Linear(CONCAT_DIM, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2)
        )
        
        self.head_fg = nn.Linear(128, cfg['n_fg'])
        self.head_af = nn.Linear(128, cfg['n_af'])
        self.head_ac = nn.Linear(128, cfg['n_ac']) 
        self.head_wa = nn.Linear(128, cfg['n_wa']) 
        self.head_sl = nn.Linear(128, cfg['n_sl']) 
        
    def forward(self, fg, af, ac, wa, sl):
        f_fg = self.enc_fg(fg)
        f_af = self.enc_af(af.unsqueeze(1)).view(af.size(0), -1)
        f_ac = self.enc_ac(ac.permute(0, 2, 1))
        f_ac = f_ac.reshape(f_ac.size(0), -1)
        f_wa = self.enc_wa(wa)
        
        x = torch.cat([f_fg, f_af, f_ac, f_wa, sl], dim=1)
        z = self.fusion(x)
        return self.head_fg(z), self.head_af(z), self.head_ac(z), self.head_wa(z), self.head_sl(z)


# ==========================================
# LIVE WEIGHT LOADING
# ==========================================
model = None
cfg = None
if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'rb') as f:
        cfg = pickle.load(f)
    print(f"Discovered Config Architecture: {cfg}")
    model = MultimodalFusion(cfg).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(">>> LIVE PyTorch Fusion Brain successfully mounted into API! <<<")
else:
    print("!!! WARNING: multimodal_model.pth or config not found. Run your Jupyter Notebook training first! !!!")

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

# Hardcoded image standardizer
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "PyTorch Brain Not Mounted. Please execute your latest Jupyter Notebook to export weights!"}), 500

    try:
        # =============================================
        # INPUT PRESENCE TRACKING
        # =============================================
        has_image = False
        has_audio = False
        has_accel = False
        has_water = False
        has_sleep = False

        # ---- IMAGE ----
        if 'image' in request.files:
            img_file = request.files['image']
            if img_file and img_file.filename and img_file.filename != '':
                try:
                    img = Image.open(img_file).convert('RGB')
                    # Validate image dimensions
                    if img.size[0] < 10 or img.size[1] < 10:
                        return jsonify({"error": "Image is too small. Please provide a valid fingernail image (at least 10x10 pixels)."}), 400
                    fg_t = img_transform(img).unsqueeze(0).to(DEVICE)
                    has_image = True
                except Exception as img_err:
                    return jsonify({"error": f"Invalid image file: {str(img_err)}"}), 400

        if not has_image:
            fg_t = torch.zeros(1, 3, 224, 224).to(DEVICE)

        # ---- AUDIO ----
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file and audio_file.filename and audio_file.filename != '':
                try:
                    audio_bytes = audio_file.read()
                    if len(audio_bytes) < 100:
                        return jsonify({"error": "Audio file is too small or empty. Please record or upload a valid audio clip."}), 400
                    has_audio = True
                    # Audio processing is disabled (uses zero tensor) to prevent 503 Out-Of-Memory crashes
                    # on Cloud Run's default 512MB instances. librosa + ffmpeg exceeds the limit.
                except Exception as audio_err:
                    return jsonify({"error": f"Invalid audio file: {str(audio_err)}"}), 400

        af_t = torch.zeros(1, *cfg['shape_af']).to(DEVICE)

        # ---- ACCELEROMETER ----
        if 'accel' in request.files:
            accel_file = request.files['accel']
            if accel_file and accel_file.filename and accel_file.filename != '':
                try:
                    accel_bytes = accel_file.read().decode('utf-8')
                    reader = csv.reader(io.StringIO(accel_bytes))
                    rows = list(reader)
                    # Validate CSV structure
                    if len(rows) < 2:
                        return jsonify({"error": "Accelerometer CSV is empty or has no data rows. Please provide valid sensor data."}), 400
                    # Check header
                    header = [h.strip().lower() for h in rows[0]]
                    if len(header) < 3:
                        return jsonify({"error": "Accelerometer CSV must have at least 3 columns (x, y, z)."}), 400
                    data_rows = rows[1:]
                    if len(data_rows) < 5:
                        return jsonify({"error": "Accelerometer data has too few samples. Need at least 5 data points."}), 400
                    # Parse data
                    accel_vals = []
                    for row in data_rows:
                        if len(row) >= 3:
                            try:
                                accel_vals.append([float(row[0]), float(row[1]), float(row[2])])
                            except ValueError:
                                continue
                    if len(accel_vals) < 5:
                        return jsonify({"error": "Could not parse enough numeric data from CSV. Ensure x,y,z columns contain numbers."}), 400
                    has_accel = True
                    
                    # Convert to tensor and shape properly to shape_ac
                    t = torch.tensor(accel_vals, dtype=torch.float32)
                    target_len = cfg['shape_ac'][0]
                    if t.shape[0] > target_len:
                        t = t[:target_len, :]
                    elif t.shape[0] < target_len:
                        t = F.pad(t, (0, 0, 0, target_len - t.shape[0]))
                    ac_t = t.unsqueeze(0).to(DEVICE)
                    
                except UnicodeDecodeError:
                    return jsonify({"error": "Accelerometer file is not a valid CSV. Please upload a text-based CSV file."}), 400
                except Exception as accel_err:
                    return jsonify({"error": f"Invalid accelerometer file: {str(accel_err)}"}), 400

        if not has_accel:
            ac_t = torch.zeros(1, *cfg['shape_ac']).to(DEVICE)

        # ---- WATER ----
        water_raw = request.form.get('water', '').strip()
        if water_raw and water_raw != '':
            try:
                wa_val = float(water_raw)
                if wa_val < 0 or wa_val > 20:
                    return jsonify({"error": "Water consumption must be between 0 and 20 liters."}), 400
                if np.isnan(wa_val) or np.isinf(wa_val):
                    return jsonify({"error": "Water consumption value is invalid (NaN or Infinity)."}), 400
                has_water = True
            except ValueError:
                return jsonify({"error": "Water consumption must be a valid number."}), 400
        else:
            wa_val = 0.0

        wa_t = torch.zeros(1, cfg['shape_wa']).to(DEVICE)
        wa_t[0, 0] = wa_val

        # ---- SLEEP ----
        sleep_raw = request.form.get('sleep', '').strip()
        if sleep_raw and sleep_raw != '':
            try:
                sl_val = float(sleep_raw)
                if sl_val < 0 or sl_val > 24:
                    return jsonify({"error": "Sleep duration must be between 0 and 24 hours."}), 400
                if np.isnan(sl_val) or np.isinf(sl_val):
                    return jsonify({"error": "Sleep duration value is invalid (NaN or Infinity)."}), 400
                has_sleep = True
            except ValueError:
                return jsonify({"error": "Sleep duration must be a valid number."}), 400
        else:
            sl_val = 0.0

        sl_t = torch.zeros(1, cfg['shape_sl']).to(DEVICE)
        sl_t[0, 0] = sl_val

        # =============================================
        # EDGE CASE: NO INPUTS AT ALL
        # =============================================
        if not any([has_image, has_audio, has_accel, has_water, has_sleep]):
            return jsonify({
                "error": "No inputs provided. Please upload or capture at least one data source (image, audio, accelerometer, or vitals) before running diagnostics."
            }), 400

        # =============================================
        # MATHEMATICAL EXECUTION
        # =============================================
        with torch.no_grad():
            o_fg, o_af, o_ac, o_wa, o_sl = model(fg_t, af_t, ac_t, wa_t, sl_t)

        preds = {
            "anemia": o_fg.argmax(1).item() if has_image else None,
            "stress": o_af.argmax(1).item() if has_audio else None,
            "fatigue": o_ac.argmax(1).item() if has_accel else None,
            "dehydration": o_wa.argmax(1).item(),
            "sleep_disorder": o_sl.argmax(1).item()
        }

        # Presentable Class Re-mapping 
        # (This translates the strict math index values back to english dynamically)
        remap = {
            "anemia": {0: "Healthy", 1: "Anemia Detected"},
            "stress": {0: "Healthy", 1: "Elevated Stress Marker", 2: "Critical Stress"},
            "fatigue": {0: "Normal Profile", 1: "Moderate Fatigue", 2: "Severe Exhaustion"},
            "dehydration": {0: "Optimal Hydration", 1: "Suboptimal", 2: "Dehydrated Warning"},
            "sleep_disorder": {0: "Normal Target", 1: "Mild Insomnia", 2: "Sleep Apnea Traits"}
        }
        
        for k in preds:
            cls = preds[k]
            if cls is None:
                if k == "anemia":
                    preds[k] = "Not Assessed \u2014 No image provided"
                elif k == "stress":
                    preds[k] = "Not Assessed \u2014 No audio provided"
                elif k == "fatigue":
                    preds[k] = "Not Assessed \u2014 No accelerometer data provided"
            elif k in remap and cls in remap[k]:
                preds[k] = remap[k][cls]
            else:
                preds[k] = f"Condition Category {cls}"

        # Add metadata about which inputs were actually provided
        preds["_inputs_used"] = {
            "image": has_image,
            "audio": has_audio,
            "accelerometer": has_accel,
            "water": has_water,
            "sleep": has_sleep
        }

        return jsonify(preds)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
