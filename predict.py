import os
import sys
import cv2
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "solar_model.pth")
THRESHOLD = 0.4


# Create the same model structure used in main.py
model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(16 * 32 * 32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)


# Load the saved trained weights
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print("Could not read image:", img_path)
        return

    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img)
        probability = torch.sigmoid(prediction).item()

    if probability >= THRESHOLD:
        result = "DEFECT"
    else:
        result = "GOOD"

    print("Image:", img_path)
    print("Prediction:", result)
    print("Defect probability:", round(probability, 4))

if len(sys.argv) < 2:
    print("Please provide an image path.")
    print("Example:")
    print("python predict.py data/good/1.jpg")
else:
    image_path = sys.argv[1]

    if not os.path.isabs(image_path):
        image_path = os.path.join(BASE_DIR, image_path)

    predict_image(image_path)

