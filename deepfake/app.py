from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import json
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# Load Class Labels
# ------------------------------------------------------------

CLASSES_PATH = os.path.join(os.path.dirname(__file__), "classes.json")

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)


# ------------------------------------------------------------
# Load MobileNetV2 Model (✓ matches your best_model.pth)
# ------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")

model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, len(classes))

state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

# ------------------------------------------------------------
# Image Preprocessing (matches MobileNetV2 training)
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image found"}), 400

    file = request.files["image"]

    try:
        image = Image.open(file).convert("RGB")
    except:
        return jsonify({"error": "Invalid image"}), 400

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

    return jsonify({"prediction": label})


# ------------------------------------------------------------
# Run App
# ------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
