import time
import cv2
from model import load_model
import numpy as np
import mediapipe as mp
import torch
from PIL import Image, ImageDraw, ImageFont



latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# load japanese font
font_path = "./assets/NotoSansJP-Regular.ttf"
font = ImageFont.truetype(font_path, size=64, index=0)

# load model structure + weights
class_list = np.load("./data/class_list.npy", allow_pickle=True)
cnn_model = load_model("./models/kanji_model.pth", device="cpu")
hand_model_path = './models/hand_landmarker.task'

print("Loaded necessary files")

# create landmarker
 
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
 
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
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
STROKE_TIMEOUT = 2

last_pt = None
frame_last_pt = None
last_pen_down_time = time.time()
last_pen_down_time = time.time()
has_stroke = False
print("Created canvas")

# capture from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()



def draw_char_on_canvas(canvas_uint8, char, font_path, size=20, bottom_margin=40):
    is_color = len(canvas_uint8.shape) == 3
    
    # preserve color channels for webcam, use grayscale for canvas
    if is_color:
        pil_img = Image.fromarray(cv2.cvtColor(canvas_uint8, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(canvas_uint8).convert("L")
        
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, size=size, index=0)

    canvas_w, canvas_h = pil_img.size

    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (canvas_w - text_w) // 2 - bbox[0]
    y = canvas_h - bottom_margin - text_h - bbox[1]

    fill_color = (0, 255, 0) if is_color else 0
    draw.text((x, y), char, fill=fill_color, font=font)
    
    result_array = np.array(pil_img)
    
    if is_color:
        return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
    else:
        return result_array


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
    MIN_SIZE = 100 
    if char_w < MIN_SIZE:
        center_x = min_x + char_w // 2
        min_x = max(center_x - MIN_SIZE // 2, 0)
        max_x = min(center_x + MIN_SIZE // 2, canvas_uint8.shape[1])
        char_w = max_x - min_x
        
    if char_h < MIN_SIZE:
        center_y = min_y + char_h // 2
        min_y = max(center_y - MIN_SIZE // 2, 0)
        max_y = min(center_y + MIN_SIZE // 2, canvas_uint8.shape[0])
        char_h = max_y - min_y


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
    # simulate ETL images
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    normalized = blurred.astype(np.float32) / 255.0
    
    return normalized



def run_model(img):
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = cnn_model(tensor)
        pred_idx = torch.argmax(logits, dim=1).item()
    return class_list[pred_idx]



# apply stabilization for pen strokes

SMOOTHING_FACTOR = 0.3 
smoothed_pt = None
frame_smoothed_pt = None
frame_overlay = None

# loop

while True:

    # read latest frame from camera
    ret, frame = cap.read()
    h, w, c = frame.shape

    # frame read correctly?
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.flip(frame, 1)  # mirror canvas
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


    timestamp_ms = time.time_ns() // 1_000_000
    landmarker.detect_async(mp_image, timestamp_ms)


    if latest_result and latest_result.hand_landmarks:
        landmarks = latest_result.hand_landmarks[0]
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]

        # webcam frame coordinates for thumb and index
        thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
        index_x, index_y = int(index.x * w), int(index.y * h)

        pad = 30
        x_min = min(thumb_x, index_x) - pad
        x_max = max(thumb_x, index_x) + pad
        y_min = min(thumb_y, index_y) - pad
        y_max = max(thumb_y, index_y) + pad

        # clamp box to frame dimensions
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        pinch_dist = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
        
        pen_down = pinch_dist < PINCH_THRESHOLD
        
        # canvas coords
        px = int(index.x * CANVAS_SIZE)
        py = int(index.y * CANVAS_SIZE)

        # live webcam coords
        frame_px = int(index.x * w)
        frame_py = int(index.y * h)

        if pen_down:
            if smoothed_pt is None:
                smoothed_pt = (float(px), float(py))
                last_pt = (px, py)
                frame_smoothed_pt = (float(frame_px), float(frame_py))
                frame_last_pt = (frame_px, frame_py)
            else:
                # Canvas smoothing
                sx = SMOOTHING_FACTOR * px + (1 - SMOOTHING_FACTOR) * smoothed_pt[0]
                sy = SMOOTHING_FACTOR * py + (1 - SMOOTHING_FACTOR) * smoothed_pt[1]
                smoothed_pt = (sx, sy)
                current_pt_int = (int(sx), int(sy))
                
                # Webcam frame smoothing
                fsx = SMOOTHING_FACTOR * frame_px + (1 - SMOOTHING_FACTOR) * frame_smoothed_pt[0]
                fsy = SMOOTHING_FACTOR * frame_py + (1 - SMOOTHING_FACTOR) * frame_smoothed_pt[1]
                frame_smoothed_pt = (fsx, fsy)
                frame_current_pt_int = (int(fsx), int(fsy))
                
                # Create the overlay if it doesn't exist yet
                if frame_overlay is None:
                    frame_overlay = np.zeros((h, w, 3), dtype=np.uint8)

                # Draw lines on canvas and the PERSISTENT overlay
                if last_pt is not None:
                    cv2.line(canvas, last_pt, current_pt_int, (150, 150, 150), thickness=10, lineType=cv2.LINE_AA)
                if frame_last_pt is not None:
                    cv2.line(frame_overlay, frame_last_pt, frame_current_pt_int, (255, 255, 255), thickness=10, lineType=cv2.LINE_AA)
                
                last_pt = current_pt_int
                frame_last_pt = frame_current_pt_int

            last_pen_down_time = time.time()
            has_stroke = True
        else:
            last_pt = None
            smoothed_pt = None
            frame_last_pt = None
            frame_smoothed_pt = None
    else:
        last_pt = None
        frame_last_pt = None

    # send character for inference if past stroke timeout
    if has_stroke and time.time() - last_pen_down_time > STROKE_TIMEOUT:
        img = preprocess_frame(canvas)
        if img is not None:
            prediction = run_model(img)
            code_str = str(prediction).replace("0x", "")
            decoded_prediction = chr(int(code_str, 16))
            text = f"Predicted character: {decoded_prediction}"
            
            canvas = draw_char_on_canvas(canvas, text, font_path)
            frame = draw_char_on_canvas(frame, text, font_path)
            
            print(f"Predicted character: {decoded_prediction}.") 
            cv2.imshow("canvas", canvas)
            cv2.waitKey(3000)
            
        # reset canvas and webcam overlay
        canvas[:] = 255
        if frame_overlay is not None:
            frame_overlay[:] = 0
            
        has_stroke = False
        last_pt = None
        frame_last_pt = None


    if frame_overlay is not None:
        mask = frame_overlay > 0
        frame[mask] = frame_overlay[mask]

    cv2.imshow("canvas", canvas)
    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
        break


# release video capture & landmarker
cap.release()
cv2.destroyAllWindows()
landmarker.close()