"""
Simple rule-based lie detector that combines face and voice emotion scores
into a small explainable "lie probability" and label.

This is intentionally lightweight and deterministic: it uses configurable
weights and a handful of human-readable rules (modality conflict, fear/anger
elevation, drop in neutral) to compute a score in [0,1].

Contract (functions):
- evaluate_lie(face_scores, voice_scores, fused_scores) -> dict
  - returns: {
      'lie_prob': float (0..1),
      'lie_label': 'lie'|'no_lie'|'uncertain',
      'reasons': [str],
      'details': {"face_max":(emo,score), "voice_max":(...), "fused_max":(...)}
    }

This module is deterministic and easy to tune. It does NOT attempt to be a
real forensic lie detector — it's a demo-level fusion and explainability layer.
"""
from typing import Dict, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class LieDetectorConfig:
    # how much modality conflict contributes to lie probability
    conflict_weight: float = 0.4
    # how much fear/anger elevation contributes
    threat_weight: float = 0.35
    # how much drop in neutral contributes
    neutral_drop_weight: float = 0.25

    # thresholds
    conflict_delta: float = 0.35  # absolute diff between modalities to count as conflict
    high_fear_threshold: float = 0.45
    high_anger_threshold: float = 0.5
    neutral_low_threshold: float = 0.25

    # labeling cutoffs
    lie_prob_cutoff: float = 0.6
    uncertain_cutoff: float = 0.35
    # temporal smoothing / persistence
    temporal_alpha: float = 0.6  # EWMA weight for recent frames
    window_len: int = 12         # sliding window length (frames)
    window_threshold_count: int = 4  # how many high-prob frames in window triggers boost
    temporal_boost: float = 0.18      # boost added when window threshold exceeded


def _max_item(scores: Dict[str, float]) -> Tuple[str, float]:
    if not scores:
        return ("", 0.0)
    k = max(scores.items(), key=lambda kv: kv[1])
    return k


def evaluate_lie(face_scores: Dict[str, float], voice_scores: Dict[str, float], fused_scores: Dict[str, float], cfg: LieDetectorConfig = LieDetectorConfig()) -> Dict:
    """Compute a lie probability and explainable reasons.

    Inputs are dictionaries mapping emotion names to floats (not necessarily normalized).
    The function will tolerate missing keys by treating them as 0.0.
    """
    # ensure keys exist
    face = {k: float(v) for k, v in (face_scores or {}).items()}
    voice = {k: float(v) for k, v in (voice_scores or {}).items()}
    fused = {k: float(v) for k, v in (fused_scores or {}).items()}

    face_max_emo, face_max_val = _max_item(face)
    voice_max_emo, voice_max_val = _max_item(voice)
    fused_max_emo, fused_max_val = _max_item(fused)

    reasons: List[str] = []

    # 1) modality conflict: different dominant emotions with large delta
    conflict_score = 0.0
    # compute confidences (how strongly each modality favors its top emotion)
    face_conf = face_max_val
    voice_conf = voice_max_val
    if face_max_emo and voice_max_emo and face_max_emo != voice_max_emo:
        # larger delta AND higher confidences => stronger conflict
        delta = abs(face_max_val - voice_max_val)
        if delta >= cfg.conflict_delta:
            base = min(1.0, (delta - cfg.conflict_delta) / (1.0 - cfg.conflict_delta))
            # weight by geometric mean of confidences to reduce noisy low-confidence conflicts
            conf_weight = math.sqrt(face_conf * voice_conf)
            conflict_score = base * conf_weight
            reasons.append(f"modality_conflict: face={face_max_emo}({face_max_val:.2f}) vs voice={voice_max_emo}({voice_max_val:.2f}), delta={delta:.2f}")

    # 2) threat emotions (fear / angry) in fused signal
    threat_score = 0.0
    fear_val = fused.get('fear', 0.0)
    angry_val = fused.get('angry', 0.0)
    if fear_val >= cfg.high_fear_threshold:
        threat_score = max(threat_score, (fear_val - cfg.high_fear_threshold) / (1.0 - cfg.high_fear_threshold))
        reasons.append(f"elevated_fear: {fear_val:.2f}")
    if angry_val >= cfg.high_anger_threshold:
        threat_score = max(threat_score, (angry_val - cfg.high_anger_threshold) / (1.0 - cfg.high_anger_threshold))
        reasons.append(f"elevated_anger: {angry_val:.2f}")

    # 3) drop in neutral compared to an absolute low threshold
    neutral_val = fused.get('neutral', 0.0)
    neutral_score = 0.0
    if neutral_val <= cfg.neutral_low_threshold:
        neutral_score = (cfg.neutral_low_threshold - neutral_val) / max(1e-6, cfg.neutral_low_threshold)
        reasons.append(f"low_neutral: {neutral_val:.2f}")

    # combine weighted
    lie_prob = (cfg.conflict_weight * conflict_score +
                cfg.threat_weight * threat_score +
                cfg.neutral_drop_weight * neutral_score)

    # clamp
    lie_prob = max(0.0, min(1.0, float(lie_prob)))

    # decide label
    if lie_prob >= cfg.lie_prob_cutoff:
        label = 'lie'
    elif lie_prob >= cfg.uncertain_cutoff:
        label = 'uncertain'
    else:
        label = 'no_lie'

    details = {
        'face_max': (face_max_emo, round(face_max_val, 3)),
        'voice_max': (voice_max_emo, round(voice_max_val, 3)),
        'fused_max': (fused_max_emo, round(fused_max_val, 3)),
        'conflict_score': round(conflict_score, 3),
        'threat_score': round(threat_score, 3),
        'neutral_score': round(neutral_score, 3),
    }

    return {
        'lie_prob': lie_prob,
        'lie_label': label,
        'reasons': reasons,
        'details': details,
    }


from collections import deque


class LieDetector:
    """Stateful lie detector with temporal smoothing.

    Usage:
      det = LieDetector(cfg=LieDetectorConfig(...))
      out = det.update(face_scores, voice_scores, fused_scores)

    The returned dict mirrors evaluate_lie but includes temporal indications
    and uses EWMA + sliding window boosting to surface sustained suspicious
    patterns into higher probabilities.
    """
    def __init__(self, cfg: LieDetectorConfig = LieDetectorConfig()):
        self.cfg = cfg
        self.ewma = 0.0
        self.window = deque(maxlen=cfg.window_len)

    def reset(self):
        self.ewma = 0.0
        self.window.clear()

    def update(self, face_scores: Dict[str, float], voice_scores: Dict[str, float], fused_scores: Dict[str, float]) -> Dict:
        instant = evaluate_lie(face_scores, voice_scores, fused_scores, cfg=self.cfg)
        inst_prob = float(instant.get('lie_prob', 0.0))

        # EWMA update
        a = self.cfg.temporal_alpha
        self.ewma = a * inst_prob + (1.0 - a) * self.ewma

        # sliding window of frame probabilities
        self.window.append(inst_prob)
        high_count = sum(1 for p in self.window if p >= self.cfg.uncertain_cutoff)

        # combined probability: prefer sustained evidence (ewma) but allow spikes
        combined = max(inst_prob, self.ewma)

        # apply boost if window shows repeated suspicious frames
        boost = 0.0
        if high_count >= self.cfg.window_threshold_count and len(self.window) > 0:
            boost = self.cfg.temporal_boost * (high_count / len(self.window))
            combined = min(1.0, combined + boost)

        # decide label using same cutoffs
        if combined >= self.cfg.lie_prob_cutoff:
            label = 'lie'
        elif combined >= self.cfg.uncertain_cutoff:
            label = 'uncertain'
        else:
            label = 'no_lie'

        out = {
            'lie_prob': combined,
            'lie_label': label,
            'reasons': list(instant.get('reasons', [])),
            'details': {
                **instant.get('details', {}),
                'ewma': round(self.ewma, 3),
                'window_high_count': int(high_count),
                'window_len': len(self.window),
                'boost': round(boost, 3),
            }
        }

        # Append temporal reason if boosted
        if boost > 0:
            out['reasons'].append(f"temporal_boost: {high_count}/{len(self.window)}")

        return out


if __name__ == '__main__':
    # quick smoke test
    f = {'neutral': 0.1, 'fear': 0.6}
    v = {'neutral': 0.05, 'fear': 0.55}
    fused = {'neutral': 0.08, 'fear': 0.57}
    out = evaluate_lie(f, v, fused)
    print(out)
