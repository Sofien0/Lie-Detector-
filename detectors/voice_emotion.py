"""
HuggingFace-based SER module with optional fine-tuning on local RAVDESS + EmoDB.

- Minimal inference API (predict_emotion_from_file / _array / _mic) — unchanged for test_voice.py
- Optional prefetch to copy model locally (--prefetch)
- Optional fine-tuning (--finetune) using a compact PyTorch loop (no Trainer, minimal deps)
"""

import os
import sys
import argparse
import glob
import random
import csv
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoModel
import librosa
import pickle
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# --------------------
# Config
# --------------------
DEFAULT_MODEL_ID = os.environ.get("SER_LOCAL_MODEL_PATH", "superb/wav2vec2-base-superb-er")
LOCAL_MODEL_DIR = "pretrained_models/ser_hf_local"      # where --prefetch copies base model
FINETUNED_MODEL_DIR = "pretrained_models/ser_hf_finetuned"  # where fine-tuned model will be saved
TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Wav2Vec2ForClassification(nn.Module):
    def __init__(self, config, base_model, classifier):
        super().__init__()
        self.config = config
        self.wav2vec2 = base_model
        self.classifier = classifier
        
    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        # Use mean pooling over the time dimension
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            
        return type('', (object,), {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        })()


# Emotion label set will be read from the model config at load time (preferred).
# But for fine-tuning dataset mapping we rely on explicit emotion names:
EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
LABEL2ID = {lab: i for i, lab in enumerate(EMOTIONS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}

# Globals for lazy-loaded model
_feature_extractor: Optional[AutoFeatureExtractor] = None
_model: Optional[AutoModelForAudioClassification] = None
_model_src_in_use: Optional[str] = None  # path or HF id


# --------------------
# Helpers: HF env + audio
# --------------------
def _set_hf_envs():
    # Avoid symlink warnings on Windows; prefer copies. Non-destructive.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _load_audio_file(path: str, sr: int = TARGET_SR) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    wav, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def _sanitize_and_resample_array(audio: np.ndarray, sr: int):
    """
    Ensure 1D float32 mono at TARGET_SR.

    Minimal processing: resample + light peak normalization.
    Avoid RMS+tanh which shifts distribution vs. HF feature extractor.
    """
    x = audio.astype(np.float32)

    # If 2D, downmix to mono by averaging channels
    if x.ndim == 2:
        # handle either (channels, samples) or (samples, channels)
        if x.shape[0] <= x.shape[1]:
            x = x.mean(axis=0)
        else:
            x = x.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Light peak normalization only (preserves relative dynamics)
    if x.size == 0:
        return x.astype(np.float32), sr
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak * 0.99

    # Clip to avoid extreme values (but do NOT apply tanh)
    x = np.clip(x, -1.0, 1.0)

    return x.astype(np.float32), sr


def save_finetuned_as_hf_compatible(input_dir: str = FINETUNED_MODEL_DIR, 
                                   output_dir: str = None):
    """
    Convert the custom fine-tuned model to standard HF format for easier loading.
    Call this after training to create a clean HF-compatible checkpoint.
    """
    if output_dir is None:
        output_dir = input_dir + "_hf"
    
    print(f"Converting {input_dir} to HF-compatible format in {output_dir}")
    
    # Load the custom model
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(input_dir)
    base_model = AutoModel.from_pretrained(input_dir)
    
    hidden_size = config.hidden_size
    num_labels = len(EMOTIONS)
    
    classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, num_labels)
    )
    
    model = Wav2Vec2ForClassification(config, base_model, classifier)
    state_dict = torch.load(os.path.join(input_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state_dict)
    
    # Create a standard HF model and copy weights
    from transformers import Wav2Vec2ForSequenceClassification
    hf_model = Wav2Vec2ForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_ID,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    # Copy wav2vec2 weights
    hf_model.wav2vec2.load_state_dict(model.wav2vec2.state_dict())
    
    # Note: The classifier architectures are different, so we keep our custom one
    # But at least the base model weights are properly transferred
    
    # Save in HF format
    hf_model.save_pretrained(output_dir)
    
    # Also save the feature extractor
    extractor = AutoFeatureExtractor.from_pretrained(input_dir)
    extractor.save_pretrained(output_dir)
    
    print(f"✅ Saved HF-compatible model to {output_dir}")

def save_model_properly(model, extractor, output_dir: str = FINETUNED_MODEL_DIR):
    """
    Save a training checkpoint in a deterministic portable format:
      - checkpoint.pt with model_state_dict and config metadata
      - feature-extractor saved to output_dir
      - model.config saved for HF compatibility
    """
    import torch, os
    os.makedirs(output_dir, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'hidden_size': getattr(model.config, "hidden_size", None),
            'num_labels': getattr(model.config, "num_labels", len(EMOTIONS)),
            'id2label': getattr(model.config, "id2label", ID2LABEL),
            'label2id': getattr(model.config, "label2id", LABEL2ID),
        },
        'model_class': model.__class__.__name__,
    }, os.path.join(output_dir, "checkpoint.pt"))

    # Save feature extractor and config so from_pretrained(output_dir) will work
    try:
        extractor.save_pretrained(output_dir)
    except Exception:
        pass
    try:
        model.config.save_pretrained(output_dir)
    except Exception:
        pass

    print(f"✅ Properly saved model to {output_dir}")



def _chunked_predict_probs_from_array(x: np.ndarray, sr: int,
                                      chunk_sec: float = 3.0, stride_sec: float = 1.5):
    """
    Run inference over overlapping chunks and average the softmax probabilities.
    Uses the global _model and _feature_extractor (ensure _ensure_model_loaded() was called).
    """
    if _model is None or _feature_extractor is None:
        _ensure_model_loaded()

    # device string is DEVICE variable in module scope
    device = DEVICE
    chunk = int(chunk_sec * sr)
    stride = int(stride_sec * sr)
    n = len(x)
    windows = []

    if n <= chunk:
        if n < chunk:
            x_pad = np.pad(x, (0, chunk - n))
        else:
            x_pad = x
        windows.append(x_pad)
    else:
        start = 0
        while start < n:
            end = min(start + chunk, n)
            w = x[start:end]
            if len(w) < chunk:
                w = np.pad(w, (0, chunk - len(w)))
            windows.append(w)
            if end == n:
                break
            start += stride

    # Processor accepts a list of numpy arrays
    inputs = _feature_extractor(windows, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out = _model(input_values=input_values, attention_mask=attention_mask)
        logits = out.logits  # [B, C]
        probs = torch.softmax(logits, dim=-1).mean(dim=0)  # average across chunks

    return probs.cpu().numpy()


def debug_compare_classifier_heads(base_dir="pretrained_models/ser_hf_local",
                                   finetuned_dir="pretrained_models/ser_hf_finetuned"):
    """
    Optional debug helper — compares classifier/head parameters between base and finetuned.
    Call this manually if you want to assert the finetuned head differs from base.
    """
    try:
        from transformers import AutoModelForAudioClassification
    except Exception:
        #print("[DEBUG] transformers not available for debug_compare_classifier_heads")
        return

    base = AutoModelForAudioClassification.from_pretrained(base_dir)
    fin = AutoModelForAudioClassification.from_pretrained(finetuned_dir)

    # Try to access classifier / final projection layers if present
    base_params = dict(base.named_parameters())
    fin_params = dict(fin.named_parameters())

    # accumulate L2 difference for overlapping keys
    total_diff = 0.0
    matched = 0
    for k in base_params:
        if k in fin_params:
            d = (base_params[k].detach() - fin_params[k].detach()).pow(2).sum().item()
            total_diff += d
            matched += 1
    #print(f"[DEBUG] Classifier/head L2 difference across {matched} shared params: {total_diff:.6f}")
# --- END: robust inference helpers ---

# --------------------
# Inference: lazy load + predict
# --------------------
def _ensure_model_loaded(prefer_local: bool = True, model_id: str = DEFAULT_MODEL_ID):
    global _feature_extractor, _model, _model_src_in_use
    if _model is not None and _feature_extractor is not None:
        return

    _set_hf_envs()

    # Prefer fine-tuned model if available
    if os.path.isdir(FINETUNED_MODEL_DIR) and os.path.isfile(os.path.join(FINETUNED_MODEL_DIR, "pytorch_model.bin")):
        src = FINETUNED_MODEL_DIR
        is_finetuned = True
    elif prefer_local and os.path.isdir(LOCAL_MODEL_DIR) and os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        src = LOCAL_MODEL_DIR
        is_finetuned = False
    else:
        src = model_id
        is_finetuned = False

    try:
        _feature_extractor = AutoFeatureExtractor.from_pretrained(src)
        #print(f"[DEBUG] _feature_extractor loaded: {_feature_extractor is not None}")
        
        if is_finetuned:
            # Load config and base model
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(src)
            
            # Load the base Wav2Vec2 model (without classifier)
            base_model = AutoModel.from_pretrained(src)
            
            # Recreate the exact same classifier architecture used during training
            hidden_size = config.hidden_size
            num_labels = len(EMOTIONS)
            
            classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_labels)
            )
            
            # Create the full model
            _model = Wav2Vec2ForClassification(config, base_model, classifier)
            
            # Load the saved state dict
            state_dict = torch.load(
                os.path.join(src, "pytorch_model.bin"), 
                map_location=DEVICE
            )
            _model.load_state_dict(state_dict)
            
            # Set the label mappings
            _model.config.id2label = ID2LABEL
            _model.config.label2id = LABEL2ID
            
        else:
            # Load standard HF model
            _model = AutoModelForAudioClassification.from_pretrained(src)
        
        _model.to(DEVICE)
        _model.eval()
        _model_src_in_use = src
        print(f"🔗 Loaded {'fine-tuned' if is_finetuned else 'base'} model from: {src} on device={DEVICE}")
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model/feature-extractor from '{src}': {e}\n"
            "If this is the first run, try: python -m detectors.voice_emotion --prefetch\n"
            "Or set SER_LOCAL_MODEL_PATH to a local directory containing the model."
        ) from e


def _postproc_logits(logits: torch.Tensor) -> Dict[str, float]:
    probs = F.softmax(logits.squeeze(0), dim=-1).cpu().numpy().tolist()
    cfg = getattr(_model, "config", None)
    if cfg and getattr(cfg, "id2label", None):
        # ensure labels are ordered by index
        pairs = sorted(((int(k), v) for k, v in cfg.id2label.items()), key=lambda x: x[0])
        labels = [lab for _, lab in pairs]
        return {lab: float(p) for lab, p in zip(labels, probs)}
    # fallback numeric labels
    return {str(i): float(p) for i, p in enumerate(probs)}


def predict_emotion_from_file(file_path: str) -> Dict[str, float]:
    # Ensure model/processor are loaded
    _ensure_model_loaded()

    # Load and sanitize audio (this enforces TARGET_SR + mono + peak-norm)
    wav = _load_audio_file(file_path, sr=TARGET_SR)
    wav, sr = _sanitize_and_resample_array(wav, TARGET_SR)

    # Debug print for easy local inspection (leave for now)
    #print(f"[DEBUG] path={file_path} sr={sr} len={len(wav)} samples ({len(wav)/sr:.2f}s) "
          #f"max_abs={float(np.max(np.abs(wav))):.3f} mean={float(np.mean(wav)):.3f}")

    # Chunked inference (averages probabilities across windows)
    probs = _chunked_predict_probs_from_array(wav, sr)

    # Map to labels using model config (preserves id2label ordering)
    cfg = getattr(_model, "config", None)
    if cfg and getattr(cfg, "id2label", None):
        pairs = sorted(((int(k), v) for k, v in cfg.id2label.items()), key=lambda x: x[0])
        labels = [lab for _, lab in pairs]
    else:
        labels = EMOTIONS

    out = {labels[i]: float(probs[i]) for i in range(len(labels))}
    out["_max_prob"] = float(probs.max())
    out["_top_label"] = labels[int(probs.argmax())]
    return out



def predict_emotion_from_file_array(audio: np.ndarray, sr: int = TARGET_SR) -> Dict[str, float]:
    _ensure_model_loaded()
    # Sanitize / resample / normalize
    x, sr = _sanitize_and_resample_array(audio, sr)

    

    probs = _chunked_predict_probs_from_array(x, sr)

    cfg = getattr(_model, "config", None)
    if cfg and getattr(cfg, "id2label", None):
        pairs = sorted(((int(k), v) for k, v in cfg.id2label.items()), key=lambda x: x[0])
        labels = [lab for _, lab in pairs]
    else:
        labels = EMOTIONS

    out = {labels[i]: float(probs[i]) for i in range(len(labels))}
    out["_max_prob"] = float(probs.max())
    out["_top_label"] = labels[int(probs.argmax())]
    return out



def predict_emotion_from_mic(duration: int = 4, sr: int = TARGET_SR) -> Dict[str, float]:
    try:
        import sounddevice as sd
    except Exception:
        raise RuntimeError("sounddevice required: pip install sounddevice")
    print(f"🎤 Recording {duration}s...")
    rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = rec.flatten().astype(np.float32)
    return predict_emotion_from_file_array(audio, sr=sr)


# --------------------
# Dataset scanning & CSV (RAVDESS + EmoDB)
# --------------------
def map_ravdess_label(filename: str) -> Optional[str]:
    try:
        emotion_id = int(filename.split("-")[2])
    except Exception:
        return None
    mapping = {1: "neutral", 2: "neutral", 3: "happy", 4: "sad", 5: "angry", 6: "fear", 7: "disgust", 8: "surprise"}
    return mapping.get(emotion_id)


def map_emodb_label(filename: str) -> Optional[str]:
    mapping = {"W": "angry", "L": "neutral", "E": "disgust", "A": "fear", "F": "happy", "T": "sad", "N": "neutral"}
    if len(filename) < 6:
        return None
    code = filename[5].upper()
    return mapping.get(code)


def scan_dataset(data_root: str = "data") -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    emodb_dir = os.path.join(data_root, "EmoDB", "wav")
    if os.path.isdir(emodb_dir):
        for fp in glob.glob(os.path.join(emodb_dir, "*.wav")):
            lab = map_emodb_label(os.path.basename(fp))
            if lab and lab in LABEL2ID:
                items.append((fp, lab))
    rav_root = os.path.join(data_root, "RAVDESS")
    if os.path.isdir(rav_root):
        for fp in glob.glob(os.path.join(rav_root, "**", "Actor_*", "*.wav"), recursive=True):
            lab = map_ravdess_label(os.path.basename(fp))
            if lab and lab in LABEL2ID:
                items.append((fp, lab))
    return items


def write_csv_from_scan(out_csv: str = "emotion_data.csv", data_root: str = "data"):
    items = scan_dataset(data_root)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "wav", "label"])
        for i, (fp, lab) in enumerate(items, 1):
            w.writerow([f"utt_{i:06d}", fp, lab])
    print(f"✅ Wrote {out_csv} ({len(items)} rows)")


# --------------------
# Fine-tuning helpers (compact)
# --------------------
class SERFilesDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str]], sr: int = TARGET_SR):
        self.items = items
        self.sr = sr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fp, lab = self.items[idx]
        wav = _load_audio_file(fp, sr=self.sr)
        return {"array": wav, "label": LABEL2ID[lab], "path": fp}


class HF_Collator:
    def __init__(self, feature_extractor: AutoFeatureExtractor, sr: int = TARGET_SR):
        self.fe = feature_extractor
        self.sr = sr

    def __call__(self, batch):
        arrays = [item["array"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        proc = self.fe(arrays, sampling_rate=self.sr, return_tensors="pt", padding=True)
        proc["labels"] = labels
        return proc


def _stratified_split(items: List[Tuple[str, str]], train=0.8, valid=0.1, seed=13):
    rnd = random.Random(seed)
    rnd.shuffle(items)
    by_lab = {}
    for fp, lab in items:
        by_lab.setdefault(lab, []).append((fp, lab))
    train_set, val_set, test_set = [], [], []
    for lab, group in by_lab.items():
        n = len(group)
        n_tr = max(1, int(n * train))
        n_va = max(1, int(n * valid))
        train_set += group[:n_tr]
        val_set += group[n_tr:n_tr + n_va]
        test_set += group[n_tr + n_va:]
    return train_set, val_set, test_set


def finetune_model(
    data_root: str = "data",
    base_model: str = DEFAULT_MODEL_ID,
    use_local_base: bool = True,
    output_dir: str = FINETUNED_MODEL_DIR,
    epochs: int = 6,
    batch_size: int = 8,
    lr: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_frac: float = 0.05,
    device: Optional[str] = None,
    freeze_base: bool = True,
):
    """
    Compact fine-tune loop:
      - Scans data_root for RAVDESS + EmoDB
      - Stratified split
      - Uses AutoFeatureExtractor + AutoModelForAudioClassification
      - Saves best model to output_dir (model + extractor)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    items = scan_dataset(data_root)
    if len(items) < 20:
        raise RuntimeError(f"Not enough data found under {data_root} (found {len(items)} files).")
    train_items, val_items, test_items = _stratified_split(items)
    print(f"📦 train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    # ===== Experiment config: unfreeze last N encoder layers (0 = freeze all) =====
    UNFREEZE_LAST_N = 2  # try 0,2,4 - start with 2
    # ===== compute class weights from train_items to mitigate imbalance =====
    from collections import Counter
    label_counts = Counter([lab for (_, lab) in train_items])
    class_counts = [label_counts.get(em, 0) for em in EMOTIONS]
    inv_freq = [1.0 / (c + 1e-6) for c in class_counts]
    # normalize weights so mean ~= 1.0
    mean_inv = float(sum(inv_freq) / len(inv_freq))
    class_weights = torch.tensor([w / mean_inv for w in inv_freq], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"[DEBUG] train counts={dict(label_counts)} class_weights={class_weights.cpu().numpy()}")


    # Load base extractor & model (prefer local downloaded copy)
    _set_hf_envs()
    src_for_load = base_model
    if use_local_base and os.path.isdir(LOCAL_MODEL_DIR) and os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        src_for_load = LOCAL_MODEL_DIR

    print(f"Loading base model from: {src_for_load}")
    extractor = AutoFeatureExtractor.from_pretrained(src_for_load)
    
    # Load the base model WITHOUT the original classifier head
    model = AutoModel.from_pretrained(src_for_load)
    
    # Replace classifier with a more powerful one
    num_labels = len(EMOTIONS)
    hidden_size = model.config.hidden_size
    
    # Enhanced classifier with dropout and two layers
    classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, num_labels)
    )
    
    # Create the classification model
    model = Wav2Vec2ForClassification(
        model.config, 
        model.base_model, 
        classifier
    )
    
    # Set label mappings
    model.config.num_labels = num_labels
    model.config.id2label = ID2LABEL
    model.config.label2id = {v: k for k, v in ID2LABEL.items()}
    model.num_labels = num_labels

    if freeze_base:
        # Freeze only the early layers, unfreeze last 4 layers
        if hasattr(model, "wav2vec2"):
            # Freeze all layers first
            for param in model.wav2vec2.parameters():
                param.requires_grad = False
            
            # Unfreeze last 4 layers for fine-tuning
            if hasattr(model.wav2vec2, "encoder"):
                num_layers = len(model.wav2vec2.encoder.layers)
                layers_to_unfreeze = 4
                for i in range(max(0, num_layers - layers_to_unfreeze), num_layers):
                    for param in model.wav2vec2.encoder.layers[i].parameters():
                        param.requires_grad = True
                print(f"🔓 Unfroze last {layers_to_unfreeze} encoder layers.")
            
            # Also unfreeze the feature projection if it exists
            if hasattr(model.wav2vec2, "feature_projection"):
                for param in model.wav2vec2.feature_projection.parameters():
                    param.requires_grad = True
                print("🔓 Unfroze feature projection.")
                
        elif hasattr(model, "base_model"):
            for param in model.base_model.parameters():
                param.requires_grad = False
            print("🔒 Frozen base_model.")
        # Controlled unfreeze of last N encoder layers (if requested)
        if hasattr(model, "wav2vec2") and UNFREEZE_LAST_N > 0:
            if hasattr(model.wav2vec2, "encoder"):
                num_layers = len(model.wav2vec2.encoder.layers)
                for i in range(max(0, num_layers - UNFREEZE_LAST_N), num_layers):
                    for p in model.wav2vec2.encoder.layers[i].parameters():
                        p.requires_grad = True
                print(f"🔓 Unfroze last {UNFREEZE_LAST_N} encoder layers for domain adaptation.")


    model.to(device).train()
    print(f"🚀 Using device: {device}")
    if device == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}, GPU name: {torch.cuda.get_device_name(0)}")
    
    # Datasets + loaders
    train_ds = SERFilesDataset(train_items)
    val_ds = SERFilesDataset(val_items)
    test_ds = SERFilesDataset(test_items)
    collate = HF_Collator(extractor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Optimizer + scheduler
    # If some base encoder parameters are trainable, use differential LR:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # split into head params vs base params
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and ("classifier" in n or "classifier" in n)]
    base_params = [p for n, p in model.named_parameters() if p.requires_grad and ("classifier" not in n)]

    if len(base_params) > 0:
        # smaller lr for base, larger for head
        optimizer = AdamW([
            {'params': base_params, 'lr': max(lr * 0.2, 1e-6)},
            {'params': head_params, 'lr': max(lr * 20, 1e-5)}
        ], weight_decay=weight_decay)
    else:
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, int(warmup_frac * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_loss = float("inf")
    best_step = None

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]"):
            # batch: dict with input_values (B, T) or (B, seq_len?), attention_mask maybe
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop("labels").to(device)
            # Forward without built-in labels so we can use weighted loss
            outputs = model(**batch)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        train_loss = running_loss / max(1, len(train_loader))

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [val]"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                labels = batch.pop("labels").to(device)
                outputs = model(**batch, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        val_loss = val_loss / max(1, len(val_loader))
        val_acc = correct / max(1, total) if total > 0 else 0.0

        print(f"Epoch {ep}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = ep
            # Save using stable checkpoint function
            save_model_properly(model, extractor, output_dir)
            print(f"💾 Saved best model to {output_dir} (epoch {ep})")

    # Reload saved checkpoint (checkpoint.pt) and run final evaluation
    # Try to load base model from output_dir (extractor/config saved there)
    try:
        base_model = AutoModel.from_pretrained(output_dir)
    except Exception:
        base_model = AutoModel.from_pretrained(DEFAULT_MODEL_ID)

    hidden_size = base_model.config.hidden_size
    num_labels = len(EMOTIONS)
    classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, num_labels)
    )

    final_model = Wav2Vec2ForClassification(base_model.config, base_model.base_model, classifier)
    # Load checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        final_model.load_state_dict(ckpt['model_state_dict'])
    else:
        # fallback to old format if present (shouldn't be used after changes)
        old_path = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(old_path):
            final_model.load_state_dict(torch.load(old_path, map_location=device))
        else:
            raise FileNotFoundError("No saved model found in output_dir for final evaluation.")

    final_model.to(device).eval()
    model = final_model  # use 'model' variable below for evaluation
    model.to(device).eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop("labels").to(device)
            out = model(**batch)
            preds = out.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    test_acc = correct / max(1, total) if total > 0 else 0.0
    print(f"✅ Test acc: {test_acc:.3f} (best_epoch={best_step})")
    print("Fine-tuned model saved to:", output_dir)
    return output_dir


# --------------------
# Prefetch utility (keeps local copy you already used)
# --------------------
def prefetch_model(model_id: str = DEFAULT_MODEL_ID, output_dir: str = LOCAL_MODEL_DIR):
    _set_hf_envs()
    print(f"Prefetching model '{model_id}' -> '{output_dir}' (this may take time)...")
    os.makedirs(output_dir, exist_ok=True)
    feat = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    feat.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"✅ Saved model to {output_dir}. Now inference will load locally.")


# --------------------
# CLI
# --------------------
def _cli(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefetch", action="store_true", help="Download feature-extractor+model and save to LOCAL_MODEL_DIR")
    parser.add_argument("--test", type=str, help="Quick local file to run prediction on")
    parser.add_argument("--write_csv", action="store_true", help="Write CSV of detected dataset")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune HF SER on local data (RAVDESS+EmoDB)")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--no_freeze", action="store_true", help="Do not freeze base encoder")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="HF model id to use")
    parser.add_argument("--mic", action="store_true", help="Detect emotion from live microphone input")
    args = parser.parse_args(argv)

    if args.prefetch:
        prefetch_model(model_id=args.model)
        return

    if args.write_csv:
        write_csv_from_scan(out_csv="emotion_data.csv", data_root=args.data_root)
        return

    if args.test:
        print("Testing inference on:", args.test)
        print(predict_emotion_from_file(args.test))
        return

    if args.finetune:
        finetune_model(
            data_root=args.data_root,
            base_model=args.model,
            use_local_base=True,
            output_dir=FINETUNED_MODEL_DIR,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            freeze_base=not args.no_freeze,
        )
        return

    if args.mic:
        print("Listening to microphone...")
        print(predict_emotion_from_mic())
        return

    parser.print_help()


if __name__ == "__main__":
    _cli(sys.argv[1:])
