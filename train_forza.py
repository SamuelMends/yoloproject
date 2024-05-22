from ultralytics import YOLO # type: ignore

def main():
    model = YOLO("yolov8n.pt")
    model.train(data="mlpath.yaml", epochs=30)
    model.val()

if __name__== '__main__':
    main()
