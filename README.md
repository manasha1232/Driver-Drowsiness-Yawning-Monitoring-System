# Driver Drowsiness & Yawning Monitoring System

An AI-driven Computer Vision application designed to monitor driver attentiveness in real-time. Built using **MediaPipe Face Mesh**, **OpenCV**, and **NumPy**, this system tracks facial landmarks to compute the **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** to detect drowsiness and yawning.

---

## 🌟 Key Features

- 👁️ **Eye Aspect Ratio (EAR) Calculation**: Calculates eye openness using 6 landmark points per eye. Triggers `DROWSY` when EAR drops below `0.21` for over 1.5 seconds.
- 😮 **Mouth Aspect Ratio (MAR) Calculation**: Calculates mouth vertical/horizontal opening ratio using 4 inner mouth points. Triggers `YAWNING` when MAR exceeds `0.65` for over 0.8 seconds.
- 🚨 **Real-Time Driver Status Overlay**:
  - `ALERT` (Green): Normal driver state.
  - `DROWSY` (Red): Eye closure detected for prolonged period.
  - `YAWNING` (Red): Mouth opened wide indicating fatigue.
- 📍 **Facial Landmark Visualization**: Draws green landmark indicators over key eye and mouth contour locations.
- 🎥 **Dual Mode Support**:
  - **Live Webcam Mode**: Interactive feed with real-time HUD metrics (`main.py`).
  - **Video File / Headless Mode**: Batch process pre-recorded driver footage and export `.mp4` video & `.jpg` screenshots (`process_video.py`).

---

## 📁 Project Structure

```
driver_drowsiness_monitoring/
├── main.py                   # Live webcam driver monitoring script
├── process_video.py          # Video file / stream processing script (with video & screenshot export)
├── README.md                 # System documentation & setup guide
├── sample_input.mp4          # Sample input video of a driver's face
├── output_screenshot.jpg     # Exported screenshot frame showing landmarks and metrics
└── output_drowsiness.mp4     # Exported output video with real-time status overlay
```

---

## ⚙️ Requirements & Environment Setup

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- Webcam (optional, required for live webcam mode)

### 2. Installation
Open PowerShell or Terminal inside the project directory:

```powershell
# Navigate to the project directory
cd C:\Users\manas\.gemini\antigravity\scratch\driver_drowsiness_monitoring

# Create a virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install mediapipe opencv-python numpy
```

---

## 🚀 How to Run

### Option 1: Live Webcam Mode (`main.py`)
Run real-time monitoring through your computer's webcam:

```powershell
python main.py
```
> Press <kbd>q</kbd> while focused on the window to exit.

---

### Option 2: Process Video File (`process_video.py`)
Process a pre-recorded driver video file and export output results:

```powershell
# Process default sample video
python process_video.py --source sample_input.mp4 --output output_drowsiness.mp4 --screenshot output_screenshot.jpg

# Process custom video file
python process_video.py --source "path/to/driver_video.mp4"
```

---

## 📐 Mathematical Formulation

### 1. Eye Aspect Ratio (EAR)
Defined by the 6 landmark points around each eye:
$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

- $p_1, p_4$: Outer and inner corners of the eye.
- $p_2, p_3, p_5, p_6$: Upper and lower eyelid boundary points.

### 2. Mouth Aspect Ratio (MAR)
Defined by 4 key points around the lips:
$$MAR = \frac{||p_{top} - p_{bottom}||}{||p_{left} - p_{right}||}$$

---

## 📊 Sample Output & Screenshot

- **Output Screenshot**: [`output_screenshot.jpg`](file:///C:/Users/manas/.gemini/antigravity/scratch/driver_drowsiness_monitoring/output_screenshot.jpg)
- **Output Video**: [`output_drowsiness.mp4`](file:///C:/Users/manas/.gemini/antigravity/scratch/driver_drowsiness_monitoring/output_drowsiness.mp4)

---

## 💻 Source Code Overview (`main.py`)

```python
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
```
