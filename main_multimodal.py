import cv2
import threading
import time
from detectors.face_emotion import detect_emotion
from detectors import voice_emotion
from detectors.lie_detector import evaluate_lie, LieDetectorConfig, LieDetector

# Tweakable lie detector config for the app. Feel free to change these values
# to make detection more or less sensitive. These defaults are slightly more
# sensitive than the module defaults so you see more "uncertain" / "lie"
# candidates during testing.
LIE_CONFIG = LieDetectorConfig(
    conflict_weight=0.45,
    threat_weight=0.35,
    neutral_drop_weight=0.20,
    conflict_delta=0.20,
    high_fear_threshold=0.35,
    high_anger_threshold=0.45,
    neutral_low_threshold=0.28,
    lie_prob_cutoff=0.5,
    uncertain_cutoff=0.25,
)

# Stateful detector instance used by the app so temporal patterns are considered
LIE_DETECTOR = LieDetector(cfg=LIE_CONFIG)
from utils.helpers import draw_emotion_bars
from logger.emotion_logger import log_emotion, save_log, smooth_emotions, AdaptiveBaseline, EMOTIONS

# settings
VOICE_REFRESH = 4  # seconds between mic predictions
FACE_WEIGHT = 0.6
VOICE_WEIGHT = 0.4

voice_scores = {e: 0.0 for e in EMOTIONS}
stop_flag = False

def mic_loop():
    global voice_scores, stop_flag
    while not stop_flag:
        try:
            res = voice_emotion.predict_emotion_from_mic(duration=VOICE_REFRESH)
            voice_scores = {e: res.get(e, 0.0) for e in EMOTIONS}
        except Exception as e:
            print("[Voice error]", e)
        time.sleep(0.1)  # tiny pause

def fuse(face, voice):
    # normalize
    fs = sum(face.values()) or 1.0
    vs = sum(voice.values()) or 1.0
    fused = {}
    for e in EMOTIONS:
        fused[e] = FACE_WEIGHT * (face.get(e,0.0)/fs) + VOICE_WEIGHT * (voice.get(e,0.0)/vs)
    return fused

def run_webcam():
    global stop_flag
    cap = cv2.VideoCapture(0)
    mic_thread = threading.Thread(target=mic_loop, daemon=True)
    mic_thread.start()
    calibrator = AdaptiveBaseline(EMOTIONS)

    while True:
        ret, frame = cap.read()
        if not ret: break
        face_scores = {e:0.0 for e in EMOTIONS}
        det = detect_emotion(frame)
        if det:
            raw = det['emotions']
            calibrator.update(raw)
            face_scores = calibrator.debias(raw)

        fused = fuse(face_scores, voice_scores)
        smoothed = smooth_emotions(fused)

        # Evaluate lie detector (stateful; considers temporal evidence)
        try:
            lie_out = LIE_DETECTOR.update(face_scores, voice_scores, smoothed)
        except Exception as e:
            lie_out = {'lie_prob': 0.0, 'lie_label': 'error', 'reasons': [str(e)], 'details': {}}

        # Merge lie outputs into logged row
        to_log = dict(smoothed)
        to_log['lie_prob'] = float(lie_out.get('lie_prob', 0.0))
        to_log['lie_label'] = str(lie_out.get('lie_label', 'unknown'))
        log_emotion(to_log)

        # Draw emotions and overlay lie result in a readable panel (top-right)
        frame = draw_emotion_bars(frame, smoothed)
        try:
            # Panel size and position
            h, w = frame.shape[:2]
            pad = 8
            box_w = 260
            box_h = 70
            x1 = w - box_w - pad
            y1 = pad
            x2 = w - pad
            y2 = pad + box_h

            # Background (semi-transparent)
            overlay = frame.copy()
            bg_color = (20, 20, 20)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Choose color by label
            label = to_log.get('lie_label', 'unknown')
            prob = to_log.get('lie_prob', 0.0)
            if label == 'lie':
                col = (0, 0, 200)  # red-ish
            elif label == 'uncertain':
                col = (0, 200, 200)  # yellow-ish
            elif label == 'no_lie':
                col = (0, 180, 0)  # green-ish
            else:
                col = (200, 200, 200)

            # Main text
            title = f"Lie: {label}"
            prob_text = f"Prob: {prob:.2f}"
            cv2.putText(frame, title, (x1 + 10, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.putText(frame, prob_text, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

            # Show primary reason (if available) on the panel
            reasons = lie_out.get('reasons', [])
            if reasons:
                reason_text = reasons[0][:40]  # truncate
                cv2.putText(frame, reason_text, (x1 + 140, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        except Exception:
            pass
        cv2.imshow("Multimodal Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_flag = True
    save_log("data/emotion_logs_multimodal.csv")
    cap.release()
    cv2.destroyAllWindows()

def run_video(path):
    cap = cv2.VideoCapture(path)
    # one-shot voice prediction from full audio
    try:
        res = voice_emotion.predict_emotion_from_file(path)
        v_scores = {e: res.get(e,0.0) for e in EMOTIONS}
    except Exception as e:
        print("[Voice error]", e)
        v_scores = {e:0.0 for e in EMOTIONS}

    calibrator = AdaptiveBaseline(EMOTIONS)
    while True:
        ret, frame = cap.read()
        if not ret: break
        face_scores = {e:0.0 for e in EMOTIONS}
        det = detect_emotion(frame)
        if det:
            raw = det['emotions']
            calibrator.update(raw)
            face_scores = calibrator.debias(raw)

        fused = fuse(face_scores, v_scores)
        smoothed = smooth_emotions(fused)


        # Evaluate lie detector (stateful; considers temporal evidence)
        try:
            lie_out = LIE_DETECTOR.update(face_scores, v_scores, smoothed)
        except Exception as e:
            lie_out = {'lie_prob': 0.0, 'lie_label': 'error', 'reasons': [str(e)], 'details': {}}

        to_log = dict(smoothed)
        to_log['lie_prob'] = float(lie_out.get('lie_prob', 0.0))
        to_log['lie_label'] = str(lie_out.get('lie_label', 'unknown'))
        log_emotion(to_log)

        frame = draw_emotion_bars(frame, smoothed)
        try:
            # panel top-right (reuse same layout as webcam)
            h, w = frame.shape[:2]
            pad = 8
            box_w = 260
            box_h = 70
            x1 = w - box_w - pad
            y1 = pad
            x2 = w - pad
            y2 = pad + box_h
            overlay = frame.copy()
            bg_color = (20, 20, 20)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            label = to_log.get('lie_label', 'unknown')
            prob = to_log.get('lie_prob', 0.0)
            if label == 'lie':
                col = (0, 0, 200)
            elif label == 'uncertain':
                col = (0, 200, 200)
            elif label == 'no_lie':
                col = (0, 180, 0)
            else:
                col = (200, 200, 200)

            title = f"Lie: {label}"
            prob_text = f"Prob: {prob:.2f}"
            cv2.putText(frame, title, (x1 + 10, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.putText(frame, prob_text, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)
            reasons = lie_out.get('reasons', [])
            if reasons:
                reason_text = reasons[0][:40]
                cv2.putText(frame, reason_text, (x1 + 140, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        except Exception:
            pass
        cv2.imshow("Multimodal Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    save_log("data/emotion_logs_multimodal.csv")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["webcam","video"], required=True)
    p.add_argument("--video_path", default="")
    args = p.parse_args()
    if args.source=="webcam":
        run_webcam()
    else:
        run_video(args.video_path)
