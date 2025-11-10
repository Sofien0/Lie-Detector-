"""
Diagnostic script to verify voice emotion model inference.
Run this after training to check if the model works correctly.
"""
import logging
logging.getLogger().setLevel(logging.ERROR)
import numpy as np
import torch
from detectors import voice_emotion

def generate_test_signals():
    """Generate synthetic test signals with known characteristics."""
    duration = 2.0  # seconds
    samples = int(duration * voice_emotion.TARGET_SR)
    
    # Different synthetic patterns that should produce different features
    signals = {
        "sine_low": np.sin(2 * np.pi * 200 * np.linspace(0, duration, samples)),  # Low frequency
        "sine_high": np.sin(2 * np.pi * 800 * np.linspace(0, duration, samples)),  # High frequency
        "noise": np.random.randn(samples) * 0.1,  # White noise
        "chirp": np.sin(2 * np.pi * np.linspace(100, 1000, samples) * np.linspace(0, duration, samples)),  # Chirp
        "pulse": np.concatenate([np.ones(1000), np.zeros(samples - 1000)]) * 0.5,  # Pulse
    }
    
    return signals

def check_model_weights():
    """Check if model weights are properly loaded and not random."""
    voice_emotion._ensure_model_loaded()
    
    if voice_emotion._feature_extractor is None:
        raise RuntimeError("Feature extractor was not loaded. Check model loading logic.")
    
    print("\n=== Model Weight Statistics ===")
    
    # Check classifier weights
    if hasattr(voice_emotion._model, 'classifier'):
        classifier = voice_emotion._model.classifier
        if isinstance(classifier, torch.nn.Sequential):
            for i, layer in enumerate(classifier):
                if hasattr(layer, 'weight'):
                    w = layer.weight.data
                    print(f"Classifier Layer {i}: mean={w.mean():.4f}, std={w.std():.4f}, "
                          f"min={w.min():.4f}, max={w.max():.4f}")
    
    # Check if model produces different outputs for different inputs
    print("\n=== Testing Model Discrimination ===")
    signals = generate_test_signals()
    
    results = {}
    for name, signal in signals.items():
        # Process through feature extractor
        inputs = voice_emotion._feature_extractor(
            [signal.astype(np.float32)], 
            sampling_rate=voice_emotion.TARGET_SR, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            input_values = inputs["input_values"].to(voice_emotion.DEVICE)
            outputs = voice_emotion._model(input_values)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
        results[name] = probs
        top_emotion = voice_emotion.EMOTIONS[np.argmax(probs)]
        confidence = np.max(probs)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        print(f"\n{name}:")
        print(f"  Top: {top_emotion} ({confidence:.3f})")
        print(f"  Entropy: {entropy:.3f}")
        print(f"  Distribution: {dict(zip(voice_emotion.EMOTIONS, probs.round(3)))}")
    
    # Check if all outputs are identical (bad sign)
    all_probs = list(results.values())
    max_diff = np.max([np.max(np.abs(all_probs[0] - p)) for p in all_probs[1:]])
    print(f"\nMax difference between predictions: {max_diff:.4f}")
    
    if max_diff < 0.01:
        print("⚠️ WARNING: Model produces nearly identical outputs for different inputs!")
        print("   This suggests the model weights may not be loaded correctly.")
    else:
        print("✅ Model produces different outputs for different inputs.")
    
    return results

def test_real_audio(file_path: str):
    """Test on a real audio file with detailed diagnostics."""
    print(f"\n=== Testing Real Audio: {file_path} ===")
    
    # Load raw audio
    import librosa
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    print(f"Original: {len(audio)} samples @ {sr}Hz, duration={len(audio)/sr:.2f}s")
    print(f"Stats: mean={audio.mean():.4f}, std={audio.std():.4f}, "
          f"min={audio.min():.4f}, max={audio.max():.4f}")
    
    # Test with different preprocessing approaches
    results = {}
    
    # 1. Standard prediction
    result_standard = voice_emotion.predict_emotion_from_file(file_path)
    results["standard"] = result_standard
    print(f"\nStandard prediction: {result_standard.get('_top_label')} "
          f"({result_standard.get('_max_prob'):.3f})")
    
    # 2. Test without normalization
    from detectors.voice_emotion import _load_audio_file, _chunked_predict_probs_from_array
    voice_emotion._ensure_model_loaded()
    
    audio_raw = _load_audio_file(file_path, sr=voice_emotion.TARGET_SR)
    probs_raw = _chunked_predict_probs_from_array(audio_raw, voice_emotion.TARGET_SR)
    results["no_norm"] = dict(zip(voice_emotion.EMOTIONS, probs_raw))
    print(f"\nWithout normalization: {voice_emotion.EMOTIONS[np.argmax(probs_raw)]} ({np.max(probs_raw):.3f})")
    
    # 3. Test with different chunk sizes
    for chunk_sec in [1.0, 2.0, 5.0]:
        probs = _chunked_predict_probs_from_array(
            audio_raw, voice_emotion.TARGET_SR, chunk_sec=chunk_sec, stride_sec=chunk_sec/2
        )
        results[f"chunk_{chunk_sec}s"] = dict(zip(voice_emotion.EMOTIONS, probs))
        print(f"Chunk {chunk_sec}s: {voice_emotion.EMOTIONS[np.argmax(probs)]} ({np.max(probs):.3f})")
    
    return results

def verify_training_data():
    """Verify that the model works on training data samples."""
    from detectors.voice_emotion import scan_dataset
    
    print("\n=== Testing on Training Data Samples ===")
    items = scan_dataset("data")
    
    if len(items) > 0:
        # Test a few random samples
        import random
        test_samples = random.sample(items, min(5, len(items)))
        
        correct = 0
        for file_path, true_label in test_samples:
            result = voice_emotion.predict_emotion_from_file(file_path)
            pred_label = result.get("_top_label")
            confidence = result.get("_max_prob")
            
            is_correct = pred_label == true_label
            if is_correct:
                correct += 1
            
            print(f"File: {file_path}")
            print(f"  True: {true_label}, Pred: {pred_label} ({confidence:.3f}) "
                  f"{'✅' if is_correct else '❌'}")
        
        accuracy = correct / len(test_samples)
        print(f"\nAccuracy on training samples: {accuracy:.1%}")
        
        if accuracy < 0.5:
            print("⚠️ WARNING: Poor accuracy on training data suggests model loading issue!")
    else:
        print("No training data found in 'data' directory")


def inspect_saved_weights():
    import torch
    import os

    path = "pretrained_models/ser_hf_finetuned/checkpoint.pt"
    if not os.path.exists(path):
        print(f"❌ No checkpoint found at {path}")
        return
    
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)  # support both formats

    print("Saved state dict keys:")
    for key in sorted(state_dict.keys()):
        if "classifier" in key or "projector" in key or "head" in key:
            tensor = state_dict[key]
            print(f"  {key}: shape={tuple(tensor.shape)}, "
                  f"mean={tensor.mean():.4f}, std={tensor.std():.4f}")

    has_encoder = any("encoder" in k for k in state_dict.keys())
    has_classifier = any("classifier" in k for k in state_dict.keys())
    print(f"\nHas encoder weights: {has_encoder}")
    print(f"Has classifier weights: {has_classifier}")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Voice Emotion Model Diagnostic")
    print("=" * 60)
    
    # Run diagnostics
    check_model_weights()
    
    # Test on training data if available
    verify_training_data()
    
    print("\n=== Inspecting Saved Weights ===")
    inspect_saved_weights()
    # Test on user-provided audio file
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        test_real_audio(test_file)
    else:
        print("\nTip: Run with an audio file path to test real audio")
        print("Example: python diagnostic.py test_audio.wav")