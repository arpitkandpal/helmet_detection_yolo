from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (you can use 'yolov8n.pt' for small, fast training)
model = YOLO('yolov8n.pt')  # You can use yolov8s.pt for slightly better results

# Train the model
model.train(
    data='dataset/data.yaml',  # Path to your dataset's YAML file
    epochs=20,                 # Number of training epochs
    imgsz=640,                 # Image size (same as Roboflow resize)
    batch=8,                   # Adjust based on your system memory
    name='helmet_detector'     # Folder name where weights will be saved
)

