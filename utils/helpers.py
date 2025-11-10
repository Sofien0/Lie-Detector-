import cv2

def draw_emotion_bars(frame, emotion_scores):
    bar_height = 20
    spacing = 35  # more vertical space
    max_score = max(emotion_scores.values()) if emotion_scores else 1
    for idx, (emotion, score) in enumerate(emotion_scores.items()):
        y = 30 + idx * spacing
        # Normalize relative to max score
        bar_len = int((score / max_score) * 300)

        # Black background box for each line
        cv2.rectangle(frame, (10, y - 15), (320, y + bar_height), (0, 0, 0), -1)

        # Red emotion bar
        cv2.rectangle(frame, (10, y), (10 + bar_len, y + bar_height), (0, 0, 255), -1)

        # White text on top
        cv2.putText(frame, f"{emotion}: {score:.2f}",
                    (15, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    return frame

