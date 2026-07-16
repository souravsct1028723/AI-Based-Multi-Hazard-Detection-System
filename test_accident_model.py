from ultralytics import YOLO

# load your trained model
model = YOLO("best.pt")

# run detection
model.predict(
    source="pic2.jpeg",
    show=True,
    save=True
)