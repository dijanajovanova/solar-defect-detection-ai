import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 0.4


def load_images(folder, label):
    images = []
    labels = []
    folder_path = os.path.join(BASE_DIR, folder)

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        img = cv2.imread(path)

        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            images.append(img)
            labels.append(label)

    return images, labels


# Load data
good_images, good_labels = load_images("data/good", 0)
defect_images, defect_labels = load_images("data/defect", 1)

X = good_images + defect_images
y = good_labels + defect_labels

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X, y = shuffle(X, y, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# Model
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

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


# Train
for epoch in range(50):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Save trained model
model_path = os.path.join(BASE_DIR, "solar_model.pth")
torch.save(model.state_dict(), model_path)
print("Model saved:", model_path)

# Test accuracy
model.eval()

with torch.no_grad():
    test_logits = model(X_test)
    test_probs = torch.sigmoid(test_logits)
    preds = (test_probs >= THRESHOLD).float()

    correct = (preds == y_test).sum().item()
    total = y_test.size(0)
    accuracy = correct / total

    print("\nTest Accuracy:", accuracy)
    print("Predicted defect count:", int(preds.sum().item()))
    print("Actual defect count:", int(y_test.sum().item()))

    y_true = y_test.numpy()
    y_pred = preds.numpy()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["GOOD", "DEFECT"]))


def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return None, None

    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        prediction = model(img)
        pred_value = torch.sigmoid(prediction).item()

    if pred_value >= THRESHOLD:
        result = "DEFECT"
    else:
        result = "GOOD"

    return result, pred_value


# Test multiple GOOD images
print("\n--- Testing GOOD images ---")

good_folder = os.path.join(BASE_DIR, "data/good")

for file in os.listdir(good_folder)[:5]:
    img_path = os.path.join(good_folder, file)
    result, pred_value = predict_image(img_path)

    if result is not None:
        print(file, "->", result, f"({pred_value:.2f})")


# Test multiple DEFECT images
print("\n--- Testing DEFECT images ---")

defect_folder = os.path.join(BASE_DIR, "data/defect")

for file in os.listdir(defect_folder)[:5]:
    img_path = os.path.join(defect_folder, file)
    result, pred_value = predict_image(img_path)

    if result is not None:
        print(file, "->", result, f"({pred_value:.2f})")

