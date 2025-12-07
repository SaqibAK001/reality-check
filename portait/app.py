from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# -------------------------------------------------------
# Load TFLite model
# -------------------------------------------------------
interpreter = tf.lite.Interpreter(model_path="portrait_detector.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SHAPE = input_details[0]["shape"][1:3]  # height, width

# Preprocess image
def preprocess_image(img):
    img = img.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# -------------------------------------------------------
# Homepage UI
# -------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------------------------------
# Predict endpoint
# -------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    image = Image.open(filepath).convert("RGB")
    input_data = preprocess_image(image)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0][0]  # first value

    ai_prob = float(output)                    # 0–1
    human_prob = float(1 - output)             # 0–1

    verdict = "AI Generated" if ai_prob > 0.5 else "Real Human"

    return jsonify({
        "filename": file.filename,
        "ai_probability": round(ai_prob * 100, 2),
        "human_probability": round(human_prob * 100, 2),
        "verdict": verdict
    })


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=8000, debug=True)
