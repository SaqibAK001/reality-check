from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("ai_image_classifier.keras")

# Load classifier indices JSON
with open("classifier_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping (0→FAKE, 1→REAL)
index_to_class = {v: k for k, v in class_indices.items()}


def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            try:
                img = Image.open(file.stream).convert("RGB")
                x = preprocess(img)

                pred = model.predict(x)[0]
                class_id = int(np.argmax(pred))
                confidence = float(np.max(pred)) * 100

                result = f"{index_to_class[class_id]} ({confidence:.2f}% confidence)"

            except Exception as e:
                result = f"Error: {str(e)}"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
