import pandas as pd
from datetime import datetime
from collections import deque
import numpy as np

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

class AdaptiveBaseline:
    def __init__(self, emotions=EMOTIONS, maxlen=200, alpha=0.7):
        # alpha: how strongly to subtract the running baseline (0..1)
        self.buffers = {e: deque(maxlen=maxlen) for e in emotions}
        self.emotions = emotions
        self.alpha = alpha

    def update(self, scores: dict):
        for e in self.emotions:
            self.buffers[e].append(scores.get(e, 0.0))

    def debias(self, scores: dict) -> dict:
        debiased = {}
        for e in self.emotions:
            s = scores.get(e, 0.0)
            baseline = np.mean(self.buffers[e]) if len(self.buffers[e]) > 0 else 0.0
            debiased[e] = max(0.0, s - self.alpha * baseline)

        total = sum(debiased.values())
        if total <= 1e-8:
            # Avoid division by zero; fall back to original scores
            return {e: scores.get(e, 0.0) for e in self.emotions}
        return {e: debiased[e] / total for e in self.emotions}


emotion_log = []
rolling_buffer = {emotion: deque(maxlen=10) for emotion in EMOTIONS}



def log_emotion(emotion_scores):
    timestamp = datetime.now().isoformat()
    row = {"timestamp": timestamp, **emotion_scores}
    emotion_log.append(row)

def save_log(filepath="data/emotion_logs.csv"):
    df = pd.DataFrame(emotion_log)
    df.to_csv(filepath, index=False)

def smooth_emotions(current_scores):
    for emotion, score in current_scores.items():
        rolling_buffer[emotion].append(score)
    return {emotion: np.mean(scores) for emotion, scores in rolling_buffer.items()}