# AI-Based Multi-Hazard Detection System

An AI-powered computer vision system developed to detect **road accidents**, **human falls**, and **street light faults** in real time. The system integrates multiple AI models with a WebSocket-based alert mechanism to simulate an intelligent emergency monitoring platform.

---

## Features

- Road Accident Detection using YOLOv8
- Human Fall Detection using MediaPipe Pose
- Street Light Fault Detection using sensor simulation
- Real-time monitoring using OpenCV
- WebSocket-based emergency alert system
- SQLite database integration
- Simulation-based hybrid monitoring system

---

## Technologies Used

- Python
- OpenCV
- YOLOv8
- MediaPipe
- TensorFlow
- SQLite
- WebSockets

---

## Project Structure

```
AI-Based-Multi-Hazard-Detection-System/
│
├── accident_detection.py
├── human_collapse_detection.py
├── street_light_fault_module.py
├── live_hybrid_system.py
├── hybrid_menu_system.py
├── ws_server.py
├── final_smart_street_system.py
├── best.pt
├── pose_landmarker_lite.task
├── users.db
└── README.md
```

---

## System Modules

### Road Accident Detection

- Uses YOLOv8 object detection
- Detects road accidents in images and live camera feeds
- Generates emergency alerts when an accident is detected

### Human Fall Detection

- Uses MediaPipe Pose Landmarker
- Detects human posture and identifies fall events
- Reduces false positives using movement-based validation

### Street Light Fault Detection

- Simulates LDR and current sensor values
- Detects street light OFF and flickering conditions
- Generates maintenance alerts

### Alert System

- Uses WebSockets for real-time alert communication
- Stores alerts using SQLite
- Designed for integration with a web dashboard

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/souravsct1028723/AI-Based-Multi-Hazard-Detection-System.git
```

2. Install the required libraries

```bash
pip install opencv-python ultralytics mediapipe tensorflow websockets
```

3. Run the application

```bash
python final_smart_street_system.py
```

---

## Future Improvements

- Mobile application integration using Flutter
- GPS-based emergency location sharing
- Cloud database integration
- Real-time notification system
- False alert reduction using advanced AI models

---

## Team Project

This project was developed as a **B.Tech Mini Project**.

### My Contributions

- Implemented road accident detection using YOLOv8
- Developed human fall detection using MediaPipe Pose
- Integrated the WebSocket-based alert system
- Implemented AI module integration and testing

---

## License

This project is developed for educational and research purposes.
