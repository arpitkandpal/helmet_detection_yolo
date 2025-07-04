import cv2
from roboflow import Roboflow
import gradio as gr
import time

# Roboflow setup
rf = Roboflow(api_key="TIH0Wbioyq8qETaGnFEZ")
project = rf.workspace("cap-detection-4722t").project("helmet-detection-ccbxp")
model = project.version(1).model

def detect_and_capture(frame):
    result = model.predict(frame, confidence=40, overlap=30).json()
    predictions = result.get("predictions", [])
    
    no_helmet_detected = any(pred["class"] == "NoHelmet" for pred in predictions)
    
    if no_helmet_detected:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"no_helmet_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
    
    return frame

gr.Interface(
    fn=detect_and_capture,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs="image",
    live=True,
    title="Helmet Detection App"
).launch()
