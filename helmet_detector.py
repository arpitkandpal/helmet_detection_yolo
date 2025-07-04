import cv2
import uuid
import os
import time
import gradio as gr
from roboflow import Roboflow

# ====== CONFIGURATION ======
API_KEY = "TIH0Wbioyq8qETaGnFEZ"
WORKSPACE = "cap-detection-4722t"
PROJECT_ID = "helmet-detection-ccbxp"
MODEL_VERSION = 2
SAVE_DIR = "no_helmet_photos"
os.makedirs(SAVE_DIR, exist_ok=True)

# ====== Load Roboflow Model ======
print("[INFO] Loading Roboflow model...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_ID)
model = project.version(MODEL_VERSION).model

# ====== Helmet Detection Function ======
def predict_frame():
    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save current frame temporarily
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, frame)

        # Predict from Roboflow
        prediction = model.predict(temp_path, confidence=40, overlap=30).json()

        # Draw boxes
        no_helmet_found = False
        for obj in prediction['predictions']:
            x, y, w, h = int(obj['x']), int(obj['y']), int(obj['width']), int(obj['height'])
            label = obj['class']
            color = (0, 255, 0) if label.lower() == "helmet" else (0, 0, 255)

            if label.lower() == "nohelmet":
                no_helmet_found = True

            # Draw label and box
            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 2)
            cv2.putText(frame, label, (x - w//2, y - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save frame if "nohelmet" detected
        if no_helmet_found:
            filename = f"{SAVE_DIR}/nohelmet_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[ALERT] No helmet detected. Saved to: {filename}")
            time.sleep(2)  # Pause to avoid rapid-fire saving

        # Send to browser via Gradio
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()

# ====== Launch Gradio App ======
print("[INFO] Launching Gradio app...")
gr.Interface(
    fn=predict_frame,
    inputs=[],
    outputs=gr.Image(type="numpy", label="Live Helmet Detection"),
    live=True,
    title="Helmet Detection via Webcam (Roboflow + Gradio)",
).launch(share=True)



                    
