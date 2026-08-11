import numpy as np
import torch
from model import load_model

images = np.load("./data/kanji_images_64.npy")
labels = np.load("./data/kanji_labels.npy")
class_list = np.load("./data/class_list.npy", allow_pickle=True)

cnn_model = load_model("./models/kanji_model.pth", device="cpu")

indices = [0, 5, 32, 305, 402]
for idx in indices:
    sample_img = images[idx].astype(np.float32) / 255.0
    true_label_idx = int(labels[idx])
    true_char = class_list[true_label_idx]

    tensor = torch.from_numpy(sample_img).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
    with torch.no_grad():
        logits = cnn_model(tensor)
        pred_idx = torch.argmax(logits, dim=1).item()
    pred_char = class_list[pred_idx]

    print(f"True label: {true_char}   Predicted: {pred_char}   Match: {true_char == pred_char}")