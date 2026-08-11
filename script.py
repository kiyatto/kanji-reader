from model import load_model
import numpy as np

class_list = np.load("class_list.npy", allow_pickle=True)  # your idx -> character mapping
model = load_model("kanji_model.pth", device="cpu")

# later, at inference time:
# pred_idx = torch.argmax(model(tensor), dim=1).item()
# predicted_char = class_list[pred_idx]