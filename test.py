import os
import numpy as np
from PIL import Image
import torch
from model import load_model

file_path = './test_data'
image_files = sorted([
    os.path.join(file_path, fname)
    for fname in os.listdir(file_path)
    if fname.lower().endswith((".png", ".jpg", ".jpeg"))
])
class_list = np.load("./data/class_list.npy", allow_pickle=True)
true_labels = ["あ", "ぺ", "雨", "下", "高"]

cnn_model = load_model("./models/kanji_model.pth", device="cpu")
cnn_model.eval()

for image_file in image_files:

    # preprocessing
    img = Image.open(image_file).convert("L")
    img = img.resize((64, 64), Image.BILINEAR)

    image = np.array(img).astype(np.float32) / 255.0

    # add channel and batch dimensions:
    # (64, 64) -> (1, 1, 64, 64)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    # run model
    with torch.no_grad():
        logits = cnn_model(tensor)
        pred_idx = torch.argmax(logits, dim=1).item()

    pred_char = class_list[pred_idx]
    pred_char = str(pred_char).replace("0x", "")
    pred_char = chr(int(pred_char, 16))

    print(
        f"{os.path.basename(image_file)} -> "
        f"{pred_char} (class {pred_idx})"
    )