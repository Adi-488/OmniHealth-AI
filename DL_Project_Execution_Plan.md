# Multimodal Health State Prediction — Complete Execution Plan
**Project:** Multimodal Health State Prediction Using Smartphone Sensors  
**Team:** Aditya Borkar | Anshul Bagal | Arushi Jain | VI-A, Group ID 13  
**Target Conditions:** Anemia (Fingernail), Stress (Audio/RAVDESS), Fatigue (Accelerometer/WISDM), Dehydration (Water Intake CSV), Sleep Disorder (Sleep Health CSV)

---

> **HOW TO USE THIS PLAN:**  
> This plan is written for all 5 datasets/modalities in a unified pipeline. Every step covers all modalities at once — you never repeat a step per dataset. Feed this plan section-by-section to the AI code generator. Each section = one notebook or script.

---

## STEP 1 — Data Understanding (EDA)

**Notebook:** `Module1_EDA.ipynb`

**What to do:**
- Load all 5 data sources:
  - `Fingernails/` → image folder (anemia labels from folder names)
  - `audio_speech_actors_01-24/` → RAVDESS audio files (.wav), emotion labels from filenames
  - `WISDM_ar_latest/` → accelerometer `.txt` file, activity labels (walking, jogging, sitting, etc. → map to fatigue/non-fatigue)
  - `Daily_Water_Intake.csv` → tabular, target column = `Hydration Level` (Good/Poor)
  - `Sleep_health_and_lifestyle_dataset.csv` → tabular, target column = `Sleep Disorder` (None / Sleep Apnea / Insomnia)
- For each modality, print:
  - Shape, class distribution, sample counts per class
  - Missing value count
  - For images: display 5 sample images per class
  - For audio: plot waveform + spectrogram of 2 samples
  - For accelerometer: plot time-series of x/y/z axes per activity
  - For tabular: plot histograms of key features, correlation heatmap, class balance bar chart

**Why:** EDA tells you what you are dealing with before writing a single preprocessing line. Class imbalance discovered here will directly inform your loss function choice in Step 6. This is mandatory for any research paper's Dataset section.

---

## STEP 2 — Data Cleaning & Preprocessing

**Notebook:** `Module2_Preprocessing.ipynb`  
**Output:** Save processed arrays to `processed_data/` folder as `.npy` files

**2A — Fingernail Images (Anemia)**
- Load all images from folder, read label from subfolder name
- Resize all to 224×224
- Convert to RGB (drop alpha channel if any)
- Normalize pixel values to [0, 1] by dividing by 255
- Encode labels with LabelEncoder
- Save: `fingernail_images.npy`, `fingernail_labels.npy`

**Why 224×224:** Standard input size for ImageNet pretrained models (ResNet, EfficientNet). You get transfer learning for free at this resolution.

**2B — Audio Files (Stress/RAVDESS)**
- Load each .wav file with librosa at 16kHz
- Extract 40 MFCC features per frame using a 25ms window, 10ms hop
- Pad/truncate all MFCCs to fixed length T=130 timesteps → shape (N, 130, 40)
- Map RAVDESS emotion labels: calm/happy/neutral → not stressed; angry/fearful/sad/disgust → stressed (2-class) OR keep all 8 emotions as-is (multi-class, your current notebooks do 2-class)
- Encode labels, StandardScaler on features
- Save: `audio_mfccs.npy`, `audio_labels.npy`

**Why MFCCs:** MFCCs capture the timbral texture of speech — how strained, tense, or calm someone sounds. They are the standard feature for speech emotion recognition and proven in hundreds of papers.

**2C — Accelerometer (Fatigue/WISDM)**
- Load WISDM `.txt`, parse columns: user, activity, timestamp, x, y, z
- Map activities: walking/jogging/upstairs/downstairs → active (not fatigued); sitting/standing → sedentary (fatigued proxy)
- Apply sliding window: 200 samples per window, 50% overlap → shape (N, 200, 3)
- Compute extra derived features per window: signal magnitude area, mean, std of each axis
- StandardScaler on features
- Encode labels
- Save: `accel_windows.npy`, `accel_labels.npy`

**Why sliding window:** Accelerometer data is time-series. A single reading means nothing. A 2-second window (at 100Hz = 200 samples) captures a full motion pattern. Overlap prevents losing events at boundaries.

**2D — Daily Water Intake CSV (Dehydration)**
- Load CSV, select features: Age, Gender (encode), Weight, Daily Water Intake, Physical Activity Level (encode), Weather (encode)
- Target: `Hydration Level` (Good=0, Poor=1)
- Handle class imbalance: check counts, oversample minority with SMOTE or use class weights
- StandardScaler on all numerical features
- Save: `water_features.npy`, `water_labels.npy`

**2E — Sleep Health CSV (Sleep Disorder)**
- Load CSV, drop `Person ID`
- Parse `Blood Pressure` string → split into systolic, diastolic integers
- Encode categorical: Gender, Occupation, BMI Category
- Target: `Sleep Disorder` (None, Sleep Apnea, Insomnia → 3 classes)
- StandardScaler on numerical features
- Save: `sleep_features.npy`, `sleep_labels.npy`

---

## STEP 3 — Feature Engineering / Feature Selection

**Done inside Module2_Preprocessing.ipynb (no separate notebook needed)**

**For Accelerometer:**
- Already computed: magnitude = sqrt(x²+y²+z²), frequency domain features via FFT (mean frequency, dominant frequency)
- These derived features are concatenated to the raw window before saving

**For Tabular (Water + Sleep):**
- Check feature importance using a quick RandomForest fit (5 lines of code)
- Drop features with importance < 0.01
- Log-transform skewed continuous features if needed (check with skewness > 1.0 threshold)

**For Audio:**
- 40 MFCCs already capture the most relevant frequency bands. Delta and Delta-Delta coefficients can be added if model accuracy is low (adds context on rate-of-change of MFCCs).

**For Images:**
- No manual feature engineering needed — the CNN backbone (ResNet18) learns its own features via convolutional filters.

**Why:** For images, CNNs do feature engineering automatically. For time-series and tabular, manual engineering bridges the gap between raw data and what the model can meaningfully learn.

---

## STEP 4 — Data Splitting (Train / Validation / Test)

**Done inside Module2_Preprocessing.ipynb or at the start of each training notebook**

- Split ratio: **70% Train / 15% Validation / 15% Test**
- Use `train_test_split` with `stratify=y` for all modalities to preserve class distribution in each split
- Fix random seed = 42 for all splits for reproducibility
- Save split indices or split arrays separately so the same split is reused across all modules

**Why stratified split:** With imbalanced datasets (e.g., Water CSV has more "Good" than "Poor"), a random split might put all "Poor" samples in test. Stratified split ensures each class appears proportionally in every split.

**Why 70/15/15:** Standard in research. 15% validation gives enough samples to reliably estimate generalization during training. 15% test is held completely unseen until final evaluation.

---

## STEP 5 — Model Selection

**Notebook:** `Module3_Unimodal_Models.ipynb` (already partially done)

**One model per modality, trained independently first:**

| Modality | Model | Why |
|---|---|---|
| Fingernail Images | ResNet18 (pretrained ImageNet) | Lightweight, fast, strong visual features; ResNet50 if accuracy < 70% |
| Audio MFCCs | 2D-CNN on spectrogram (or 1D-CNN on MFCC sequences) | MFCCs as 2D image → CNN can learn frequency-time patterns |
| Accelerometer | 1D-CNN + LSTM | 1D-CNN extracts local motion patterns; LSTM captures temporal dependency across the window |
| Water Intake | MLP (3 layers: 32→16→2) | Small tabular dataset, MLP is fast and sufficient |
| Sleep Health | MLP (3 layers: 64→32→3) | Same reasoning; 3-class output |

**Why ResNet18 and not ResNet50:** Your fingernail dataset is small (few hundred images at most). ResNet50 would overfit. ResNet18 with frozen early layers + fine-tuned last 2 blocks is the right call. Less parameters = faster convergence.

**Why not Transformers for audio:** Vision Transformers and Audio Spectrogram Transformers need large datasets. RAVDESS has ~2400 samples. A CNN is better here.

---

## STEP 6 — Model Building / Training

**Notebook:** `Module3_Unimodal_Models.ipynb` (unimodals) + `Module4_Multimodal_Fusion.ipynb` (fusion)

### Phase A: Unimodal Training (run once per modality)

For each modality:
- Build model, freeze pretrained backbone (for ResNet18), add custom classification head
- Loss: CrossEntropyLoss with class weights (compute weights as `N_total / (N_classes × N_samples_per_class)`)
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Early stopping: patience=5 on validation loss
- Train for max 30 epochs
- Save best model weights per modality: `best_fg_model.pth`, `best_af_model.pth`, etc.

### Phase B: Fusion Training (Module4)

- Load the 5 pretrained unimodal encoders (freeze their weights)
- Extract feature embeddings (penultimate layer output) from each encoder
- Concatenate all embeddings → fusion layer
- Fusion architecture: BatchNorm → Linear(CONCAT_DIM, 512) → ReLU → Dropout(0.3) → Linear(512, 128) → ReLU → Dropout(0.2)
- 5 task-specific heads on top (one per condition): Linear(128, n_classes)
- This is a **Multi-Task Learning** setup — one forward pass, 5 losses summed
- Total loss = sum of 5 CrossEntropy losses (equal weights to start)
- Train only the fusion layer + heads, encoders remain frozen
- After 10 epochs: unfreeze encoders and fine-tune end-to-end with lr=1e-5

**Why Multi-Task Learning:** You are detecting 5 health conditions, all from the same person's sensor data. MTL lets the model share representations — knowledge about fatigue helps understand stress. This is a key research contribution.

**Why freeze encoders first:** If you unfreeze everything from epoch 1, the randomly initialized fusion layer will corrupt the pretrained encoder weights with large gradients. Freeze first → train fusion → then fine-tune together.

---

## STEP 7 — Hyperparameter Tuning

**Not a separate notebook — do this inside Module3 and Module4**

**For unimodal models:**
- Learning rate: try [1e-3, 5e-4, 1e-4] → pick best val accuracy
- Batch size: 32 for images, 64 for tabular/audio/accel
- Dropout: 0.2, 0.3, 0.5 → try if model is overfitting

**For fusion model:**
- Fusion hidden dim: try [256, 512] → 512 works well (current code uses this)
- Number of fusion layers: 2 is sufficient (currently implemented correctly)
- Task loss weights: if one modality consistently underperforms, give its loss a weight of 1.5

**Keep it simple:** Don't run a full grid search. Use the following rule: if val accuracy is not improving after 5 epochs, halve the LR. If train acc >> val acc (gap > 15%), increase dropout. That covers 90% of tuning scenarios.

**Why AdamW over Adam:** AdamW properly decouples weight decay from the adaptive gradient, preventing the optimizer from ignoring regularization on parameters with large gradient history. This is especially important in multi-task settings.

---

## STEP 8 — Model Evaluation

**Notebook:** `Module5_Evaluation.ipynb`

**For each modality and for the fused model, report:**
- Accuracy
- Precision, Recall, F1-Score (per class, weighted average)
- Confusion Matrix (plot as heatmap using seaborn)
- ROC-AUC (one-vs-rest for multi-class using sklearn)

**Comparison table to build for your paper:**

| Model | Fg Acc | Af Acc | Ac Acc | Wa Acc | Sl Acc | Mean Acc |
|---|---|---|---|---|---|---|
| Unimodal only | — | — | — | — | — | — |
| Early Fusion | — | — | — | — | — | — |
| Late Fusion (concat) | — | — | — | — | — | — |
| **MTL Fusion (ours)** | — | — | — | — | — | **—** |

**This ablation table is your main result. It proves your method is better than simpler baselines.**

**Why ROC-AUC:** Accuracy alone is misleading on imbalanced classes. AUC measures the model's ability to rank positive samples higher than negative ones regardless of threshold — far more informative for health prediction where "Poor" or "Sick" samples are rare.

---

## STEP 9 — Model Optimization

**Done during and after training**

**Regularization (already in place, confirm):**
- Dropout layers in fusion head (0.2–0.3) — prevents co-adaptation of neurons
- Weight decay in AdamW (1e-4) — L2 penalty, prevents large weights
- BatchNorm in every block — stabilizes training, reduces internal covariate shift
- Gradient clipping (max_norm=1.0) — already in your Module4 code, keep it

**If fingernail/audio accuracy is stuck below 60%:**
- Apply data augmentation:
  - Images: random horizontal flip, random brightness ±20%, random rotation ±15°
  - Audio: add Gaussian noise (SNR=20dB), time-stretch ±10%
- Use mixup augmentation for tabular data

**If model is too large for deployment:**
- Apply post-training quantization: convert float32 weights to int8 using PyTorch's `torch.quantization`
- This reduces model size by ~4x with <1% accuracy drop

**Why BatchNorm:** Normalizes activations per batch, reducing internal covariate shift. It acts as a mild regularizer and allows higher learning rates. Without it, deep networks train much more slowly and unstably.

---

## STEP 10 — Testing on Unseen Data

**Done inside Module5_Evaluation.ipynb**

- Load the saved `multimodal_model.pth` weights
- Load the held-out test split (the 15% set aside in Step 4, never touched during training or tuning)
- Run inference on the full test set
- Report all metrics from Step 8 on this test set
- Compare test metrics vs validation metrics — large gap = overfitting, go back to Step 9

**Synthetic patient test:** Since your datasets don't share patient IDs, create 100 synthetic test patients (same approach as Module4 — random sampling across modalities) and report mean test accuracy across all 5 tasks.

**Why separate test set matters:** Validation accuracy is used to make decisions (LR, dropout, early stopping). So it is technically not "unseen." The true test set was never used in any decision — it gives the honest estimate of real-world performance. This number goes in your paper.

---

## STEP 11 — Model Deployment

**Notebook/Script:** `Module6_Inference.ipynb` + `webapp/app.py`

**Steps:**
1. Export the trained model to TorchScript: `torch.jit.script(model)` → saves as `multimodal_model_scripted.pt`
2. Build a FastAPI inference endpoint OR a Streamlit/Gradio web app (webapp folder already exists in your Drive)
3. Input: upload image + audio file + CSV row for accel/water/sleep data
4. Preprocessing: apply the exact same transforms from Step 2 (use saved scalers)
5. Output: predicted class + confidence score for each of the 5 health conditions
6. Measure latency: run 100 inference calls, report mean latency (target < 1 second)

**Important:** Save your LabelEncoders and StandardScalers using `joblib.dump()` during preprocessing. Load them during inference. Without this, your feature scaling during inference won't match training.

**Why TorchScript:** TorchScript serializes the model into a platform-independent format. It runs without needing your Python class definitions — critical for deployment to servers or mobile. ONNX is another option but TorchScript is simpler for PyTorch-native deployment.

---

## STEP 12 — Monitoring & Maintenance

**Conceptual (write in paper's Conclusion section, implement if time permits)**

**Data Drift Detection:**
- Compare incoming inference data distribution vs training distribution monthly
- Use Population Stability Index (PSI) for tabular features
- For audio/images: track mean model confidence score — if it drops below 0.6 consistently, trigger retraining

**Retraining Trigger:**
- Collect new labeled samples every 3 months (self-reported or clinical)
- Retrain only the fusion layer + heads with new data (keep encoders frozen unless accuracy drops > 5%)
- Use Continual Learning strategy: mix 20% old data with 100% new data to prevent catastrophic forgetting

**Model Versioning:**
- Save each model with timestamp: `multimodal_model_v2_20260419.pth`
- Keep a JSON log of version, training date, dataset size, val accuracy

**Why monitor drift:** Health data changes with seasons, demographics, device hardware. A model trained on 2025 data may degrade in 2027 if newer phone sensors output different accelerometer scales. Monitoring catches this before patients get wrong predictions.

---

## STEP 13 — Iteration / Improvement Cycle

**After your first end-to-end run, improve in this order of priority:**

1. **If fingernail accuracy < 65%:** Add more augmentation → try EfficientNet-B0 instead of ResNet18 → collect more images
2. **If audio accuracy < 60%:** Try Log-Mel Spectrogram as input instead of MFCC → treat audio as image input to ResNet18
3. **If fusion does not beat unimodal by at least 3%:** Add cross-modal attention (simple dot-product attention between modality pairs) instead of simple concatenation
4. **For paper strength:** Run ablation study — train fusion with each modality removed one at a time and report accuracy drop. This shows which modality contributes most.
5. **Final upgrade (optional):** Replace the concat fusion with a Transformer encoder — treat each modality's embedding as a token, use multi-head self-attention to learn inter-modality relationships. This is the state-of-the-art approach in multimodal health papers (2024–2025).

---

## Summary: Notebook Execution Order

| Order | Notebook / Script | What it does |
|---|---|---|
| 1 | `Module1_EDA.ipynb` | Understand all 5 datasets |
| 2 | `Module2_Preprocessing.ipynb` | Clean + process all 5 → save .npy files |
| 3 | `Module3_Unimodal_Models.ipynb` | Train 5 independent models, get baselines |
| 4 | `Module4_Multimodal_Fusion.ipynb` | Train fusion model (already done, improve it) |
| 5 | `Module5_Evaluation.ipynb` | Full metrics + ablation table |
| 6 | `Module6_Inference.ipynb` | TorchScript export + latency test |
| 7 | `webapp/app.py` | Streamlit or FastAPI app |

---

## Current State of Your Project (What's Already Done)

- **Module2:** Preprocessing done — all .npy files exist in `processed_data/`
- **Module3:** Unimodal models trained — encoders available
- **Module4:** Multimodal fusion trained — `multimodal_model.pth` saved (53MB), `multimodal_config.pkl` saved
- **Current results (from Module4 outputs):** Fg: ~57%, Af: ~55%, Ac: ~91.7%, Wa: ~79%, Sl: ~82%
- **What's pending:** Module1 EDA (for paper), Module5 full evaluation metrics, Module6 inference + webapp

**Key observation:** Fingernail (anemia) and Audio (stress) accuracy are low (~55–57%). This is the main area to improve. Prioritize Step 13 items 1 and 2 first.

---

*Plan authored for Antigravity code generation. Each section is self-contained and can be fed as a prompt with the section title as context.*
