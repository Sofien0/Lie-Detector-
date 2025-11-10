from fer import FER
import cv2

detector = FER(mtcnn=True)  # Use MTCNN for better face detection

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Convert back to 3-channel (FER expects color)
    processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return processed


def detect_emotion(frame):
    # Use preprocessing (CLAHE) before detection
    processed = preprocess_frame(frame)

    detections = detector.detect_emotions(processed)
    if not detections:
        return None

    # Pick the largest face (more stable than “first”)
    detections.sort(key=lambda d: d.get('box', [0,0,0,0])[2] * d.get('box', [0,0,0,0])[3], reverse=True)
    return detections[0]

