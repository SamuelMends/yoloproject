from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(data="mlpath.yaml", epochs=30, device=0)
    model.val()
