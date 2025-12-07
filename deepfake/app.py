from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
import json
import tensorflow as tf
import tempfile
import os

app = FastAPI()

# Load model
MODEL_PATH = "model.keras"   # CHANGE this to your real filename
model = tf.keras.models.load_model(MODEL_PATH)

# Load index map
with open("classifier_indices.json", "r") as f:
    index_map = json.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(await file.read())
    temp.close()

    # Read video
    cap = cv2.VideoCapture(temp.name)
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        pred = model.predict(frame)[0]
        predictions.append(pred)

    cap.release()
    os.remove(temp.name)

    # Average prediction across frames
    avg_pred = np.mean(predictions, axis=0)

    # Convert to JSON
    results = {
        "class_probabilities": {index_map[str(i)]: float(avg_pred[i]) for i in range(len(avg_pred))},
        "predicted_class": index_map[str(np.argmax(avg_pred))]
    }

    return results

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
