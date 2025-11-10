import cv2
from detectors.face_emotion import detect_emotion
from utils.helpers import draw_emotion_bars
from logger.emotion_logger import log_emotion, save_log, smooth_emotions, AdaptiveBaseline, EMOTIONS
cap = cv2.VideoCapture(0)
EMOTION_THRESHOLD = 0.05  # Ignore scores below this
RELATIVE_CUTOFF = 0.6       # Keep emotions >= 60% of the frame max
TOPK = 3                    # Display top-3 after smoothing

calibrator = AdaptiveBaseline(EMOTIONS, maxlen=200, alpha=0.7)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emotion_result = detect_emotion(frame)
    if emotion_result:
        raw_emotions = emotion_result['emotions']
        # 1) Update the baseline and compute debiased scores
        calibrator.update(raw_emotions)
        debiased = calibrator.debias(raw_emotions)

        # 2) Build a full vector for smoothing (zeros if missing)
        full = {e: debiased.get(e, 0.0) for e in EMOTIONS}

        # 3) Smooth (your rolling mean)
        smoothed = smooth_emotions(full)

        # 4) Filter: keep only emotions above max(relative, absolute) cutoff
        m = max(smoothed.values()) if smoothed else 0.0 
        cut = max(EMOTION_THRESHOLD, RELATIVE_CUTOFF * m) if m > 0 else EMOTION_THRESHOLD
        filtered = {k: v for k, v in smoothed.items() if v >= cut}

        # 5) Sort and take top-K (fallback to smoothed if filter nuked everything)
        source = filtered if filtered else smoothed
        sorted_emotions = dict(sorted(source.items(), key=lambda x: x[1], reverse=True))
        topk = dict(list(sorted_emotions.items())[:TOPK])

        # Always show all emotions, sorted by value
        sorted_emotions = dict(sorted(smoothed.items(), key=lambda x: x[1], reverse=True))
        log_emotion(sorted_emotions)
        frame = draw_emotion_bars(frame, sorted_emotions)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
save_log()
cap.release()
cv2.destroyAllWindows()
