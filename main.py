import cv2
import mediapipe as mp
import numpy as np
import time

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

cap = cv2.VideoCapture(0)

closed_start = None
yawn_start = None

with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face.process(rgb)

        status = "ALERT"

        if result.multi_face_landmarks:
            h, w, _ = frame.shape
            lm = result.multi_face_landmarks[0].landmark

            pts = [
                (int(p.x * w), int(p.y * h))
                for p in lm
            ]

            le = ear([pts[i] for i in LEFT])
            re = ear([pts[i] for i in RIGHT])

            eye_ratio = (le + re) / 2

            mouth = [pts[i] for i in MOUTH]
            mouth_ratio = mar(mouth)

            now = time.time()

            if eye_ratio < 0.21:
                if closed_start is None:
                    closed_start = now

                if now - closed_start > 2:
                    status = "DROWSY"
            else:
                closed_start = None

            if mouth_ratio > 0.65:
                if yawn_start is None:
                    yawn_start = now

                if now - yawn_start > 1:
                    status = "YAWNING"
            else:
                yawn_start = None

            for i in LEFT + RIGHT + MOUTH:
                cv2.circle(
                    frame,
                    pts[i],
                    2,
                    (0, 255, 0),
                    -1
                )

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

        cv2.putText(
            frame,
            f"STATUS: {status}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255) if status != "ALERT"
            else (0, 255, 0),
            2
        )

        cv2.imshow("Driver Monitoring System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
