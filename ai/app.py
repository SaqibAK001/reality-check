from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# Load Model
MODEL_PATH = "ai_image_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))   # adjust to your model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    score = float(prediction[0][0])

    return {
        "ai_generated_probability": score,
        "is_ai_generated": score > 0.5
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
