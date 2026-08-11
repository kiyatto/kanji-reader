import time
import cv2
from model import load_model
import numpy as np
import mediapipe as mp
import torch
from PIL import Image, ImageDraw, ImageFont

# load japanese font
font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
font = ImageFont.truetype(font_path, size=64, index=0)

# load model structure + weights
class_list = np.load("./data/class_list.npy", allow_pickle=True)
cnn_model = load_model("./models/kanji_model.pth", device="cpu")
hand_model_path = './models/hand_landmarker.task'

print("Checkpoint 0")

# create landmarker
 
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
 
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)
landmarker = HandLandmarker.create_from_options(options)

print("Created landmarker")
# create white canvas
CANVAS_SIZE = 512
canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), 255, dtype=np.uint8)

THUMB_TIP = 4
INDEX_TIP = 8
PINCH_THRESHOLD = 0.05
STROKE_TIMEOUT = 1.3

last_pt = None
last_pen_down_time = time.time()
has_stroke = False
print("Created canvas")

# capture from webcam
cap = cv2.VideoCapture(0)
print("Checkpoint 1!")
if not cap.isOpened():
    print("Cannot open camera")
    exit()



def draw_char_on_canvas(canvas_uint8, char, font_path, size=20, bottom_margin=40):
    pil_img = Image.fromarray(canvas_uint8).convert("L")
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, size=size, index=0)

    canvas_w, canvas_h = pil_img.size

    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (canvas_w - text_w) // 2 - bbox[0]
    y = canvas_h - bottom_margin - text_h - bbox[1]

    # black lines
    draw.text((x, y), char, fill=0, font=font)
    return np.array(pil_img)



# crops to 64x64 and normalizes input image
def preprocess_frame(canvas_uint8):
    ys, xs = np.where(canvas_uint8 < 255)
    if len(xs) == 0:
        return None
 
    # 1. boundaries of the drawn strokes
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    
    char_w = max_x - min_x
    char_h = max_y - min_y
    
    # 2. calculate & apply dynamic padding (25%)
    pad = int(max(char_w, char_h) * 0.25)
    x0 = max(min_x - pad, 0)
    x1 = min(max_x + pad, canvas_uint8.shape[1])
    y0 = max(min_y - pad, 0)
    y1 = min(max_y + pad, canvas_uint8.shape[0])
    
    cropped = canvas_uint8[y0:y1, x0:x1]
 
    # 3. square & resize
    h, w = cropped.shape
    size = max(h, w)
    square = np.full((size, size), 255, dtype=np.uint8)
    
    y_off, x_off = (size - h) // 2, (size - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped
 
    resized = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized



def run_model(img):
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = cnn_model(tensor)
        pred_idx = torch.argmax(logits, dim=1).item()
    return class_list[pred_idx]

# loop

# debug stuff

# test_img = cv2.imread("./tester.png")

# sim_stroke = 255 - test_img

# # paste it into a canvas-sized array, centered, same as a real drawn character would sit
# sim_canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
# h, w = sim_stroke.shape
# y0, x0 = (CANVAS_SIZE - h) // 2, (CANVAS_SIZE - w) // 2
# sim_canvas[y0:y0 + h, x0:x0 + w] = sim_stroke

# # now push it through your REAL pipeline, exactly as the live loop does
# processed = preprocess_frame(sim_canvas)
# cv2.imwrite("debug_roundtrip.png", (processed * 255).astype(np.uint8))
# prediction = run_model(processed)
# print(f"Round-trip prediction: {prediction}  (expected: {target_char})")

while True:
    # read latest frame from camera
    ret, frame = cap.read()

    # frame read correctly?
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.flip(frame, 1)  # mirror canvas
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]

        pinch_dist = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
        print(f"Pinch distance: {pinch_dist}")
        pen_down = pinch_dist < PINCH_THRESHOLD

        px = int(index.x * CANVAS_SIZE)
        py = int(index.y * CANVAS_SIZE)

        if pen_down:
            if last_pt is not None:
                cv2.line(canvas, last_pt, (px, py), (128, 128, 128), thickness=10, lineType=cv2.LINE_AA)
            last_pt = (px, py)
            last_pen_down_time = time.time()
            has_stroke = True
        else:
            last_pt = None
    else:
        last_pt = None

    # send character for inference if past stroke timeout
    if has_stroke and time.time() - last_pen_down_time > STROKE_TIMEOUT:
        img = preprocess_frame(canvas)
        if img is not None:
            # cv2.imwrite("debug_model_input.png", (img * 255).astype(np.uint8))
            prediction = run_model(img)
            code_str = str(prediction).replace("0x", "")
            decoded_prediction = chr(int(code_str, 16))
            text = f"Predicted character: {decoded_prediction}"
            canvas = draw_char_on_canvas(canvas, text, font_path)
            print(f"Predicted character: {decoded_prediction}.") # console
            cv2.imshow("canvas", canvas)
            cv2.waitKey(5000)
        canvas[:] = 255
        has_stroke = False
        last_pt = None

    cv2.imshow("canvas", canvas)
    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
        break


# release video capture & landmarker
cap.release()
cv2.destroyAllWindows()
landmarker.close()