import argparse
import time
import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh

LEFT = [33, 160, 158, 133, 153, 144]
RIGHT = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def ear(points):
    a = dist(points[1], points[5])
    b = dist(points[2], points[4])
    c = dist(points[0], points[3])
    return (a + b) / (2 * c)

def mar(points):
    return dist(points[0], points[1]) / dist(points[2], points[3])

def run_drowsiness_detection(source="sample_input.mp4", output_video="output_drowsiness.mp4", screenshot_path="output_screenshot.jpg"):
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    closed_start = None
    yawn_start = None
    frame_count = 0
    saved_screenshot = False

    with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face:
        print(f"Processing driver drowsiness detection on '{source}'...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Flip only if webcam
            if isinstance(source, int):
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face.process(rgb)

            status = "ALERT"

            if result.multi_face_landmarks:
                h, w, _ = frame.shape
                lm = result.multi_face_landmarks[0].landmark

                pts = [(int(p.x * w), int(p.y * h)) for p in lm]

                le = ear([pts[i] for i in LEFT])
                re = ear([pts[i] for i in RIGHT])
                eye_ratio = (le + re) / 2

                mouth = [pts[i] for i in MOUTH]
                mouth_ratio = mar(mouth)

                now = time.time()

                if eye_ratio < 0.21:
                    if closed_start is None:
                        closed_start = now
                    if now - closed_start > 1.5:
                        status = "DROWSY"
                else:
                    closed_start = None

                if mouth_ratio > 0.65:
                    if yawn_start is None:
                        yawn_start = now
                    if now - yawn_start > 0.8:
                        status = "YAWNING"
                else:
                    yawn_start = None

                for i in LEFT + RIGHT + MOUTH:
                    cv2.circle(frame, pts[i], 3, (0, 255, 0), -1)

                cv2.putText(
                    frame,
                    f"EAR: {eye_ratio:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"MAR: {mouth_ratio:.2f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            status_color = (0, 255, 0) if status == "ALERT" else (0, 0, 255)
            cv2.putText(
                frame,
                f"STATUS: {status}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2
            )

            out.write(frame)

            # Save screenshot when face landmarks are detected
            if not saved_screenshot and result.multi_face_landmarks and frame_count > 10:
                cv2.imwrite(screenshot_path, frame)
                saved_screenshot = True

    cap.release()
    out.release()
    print(f"Finished processing {frame_count} frames.")
    print(f"Saved output video to: {output_video}")
    print(f"Saved output screenshot to: {screenshot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Drowsiness Monitoring System")
    parser.add_argument("--source", type=str, default="sample_input.mp4", help="Video file path or webcam ID (0)")
    parser.add_argument("--output", type=str, default="output_drowsiness.mp4", help="Output video file path")
    parser.add_argument("--screenshot", type=str, default="output_screenshot.jpg", help="Output screenshot path")
    args = parser.parse_args()

    run_drowsiness_detection(source=args.source, output_video=args.output, screenshot_path=args.screenshot)
