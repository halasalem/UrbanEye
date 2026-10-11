from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import base64
import os
import re
import time
import torch
from datetime import datetime

torch.set_grad_enabled(False)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INCIDENT_MODEL_PATH = os.path.join(BASE_DIR, "models", "incident_best.pt")
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "plate_best.pt")

API_VERSION = "ocr_v2_full_plate_voting_stacked_2026_06_02"

# ---------------------------------------------------------------------------
# Tuning values
# ---------------------------------------------------------------------------
INCIDENT_CONFIDENCE = 0.50
PLATE_CONFIDENCE = 0.35
OCR_MIN_CONFIDENCE = 60.0      # below this, return "unreadable" instead of a wrong plate
OCR_MIN_CHARS = 3              # ignore OCR results shorter than this
FULL_PLATE_MIN_CONFIDENCE = 45.0  # full 2+5 digit plate can beat a high-confidence partial read
OCR_RESIZE_HEIGHT = 96         # plate crop is resized to this height before OCR
# Debug image saving is OFF for website/demo speed.
SAVE_OCR_DEBUG = False
DEBUG_DIR = os.path.join(BASE_DIR, "debug_plates")

# ---------------------------------------------------------------------------
# GPU / CPU setup
# ---------------------------------------------------------------------------
# On an NVIDIA laptop, this will use GPU only if CUDA-enabled PyTorch is installed.
# On AMD / CPU-only laptops, it safely falls back to CPU.
USE_CUDA = torch.cuda.is_available()
YOLO_DEVICE = 0 if USE_CUDA else "cpu"   # Ultralytics accepts GPU index 0 or "cpu"
EASYOCR_GPU = USE_CUDA

print("=" * 70)
print("UrbanEye CV API starting...")
print(f"API version: {API_VERSION}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {USE_CUDA}")
if USE_CUDA:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU device: CPU mode")
print("=" * 70)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
print("Loading YOLO models...")
incident_model = YOLO(INCIDENT_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

# Move YOLO models to GPU when CUDA is available. Prediction also passes device.
if USE_CUDA:
    try:
        incident_model.to("cuda")
        plate_model.to("cuda")
        print("YOLO models moved to CUDA GPU.")
    except Exception as e:
        print(f"[GPU WARNING] Could not move YOLO models to CUDA: {e}")
        print("YOLO will continue using the device passed during prediction.")
else:
    print("YOLO running on CPU.")

print("Loading OCR...")
try:
    reader = easyocr.Reader(["en"], gpu=EASYOCR_GPU)
    print(f"EasyOCR GPU enabled: {EASYOCR_GPU}")
except Exception as e:
    # If GPU OCR fails for any reason, do not crash the API.
    print(f"[OCR WARNING] EasyOCR GPU loading failed: {e}")
    print("Falling back to EasyOCR CPU mode.")
    EASYOCR_GPU = False
    reader = easyocr.Reader(["en"], gpu=False)

print("UrbanEye CV API ready.")


def clean_plate_text(text):
    """Convert OCR output into a Jordan-style numeric plate string.

    Jordan demo plates are treated as numeric-only. If OCR outputs a letter,
    map it to the closest digit shape before cleaning.
    """
    text = (text or "").upper().strip()

    # Convert separators first.
    separator_replacements = {
        " ": "-",
        "_": "-",
        ".": "",
        ",": "",
        ":": "",
        ";": "",
        "/": "-",
        "\\": "-",
        "|": "1",
    }
    for old, new in separator_replacements.items():
        text = text.replace(old, new)

    # Numeric-only plate mapping. This is intentionally used only for OCR text,
    # not for class names or other API fields.
    letter_to_digit = {
        "A": "4", "B": "8", "C": "0", "D": "0", "E": "6",
        "F": "7", "G": "6", "H": "4", "I": "1", "J": "3",
        "K": "4", "L": "1", "M": "0", "N": "0", "O": "0",
        "P": "9", "Q": "0", "R": "2", "S": "5", "T": "1",
        "U": "0", "V": "7", "W": "0", "X": "8", "Y": "7", "Z": "2",
    }
    text = "".join(letter_to_digit.get(ch, ch) for ch in text)

    # Keep only digits and hyphens.
    text = re.sub(r"[^0-9\-]", "", text)
    text = re.sub(r"-+", "-", text).strip("-")

    # If EasyOCR reads a Jordan plate without the separator, format 7 digits
    # like 22-34251 / 44-87523.
    digits_only = re.sub(r"\D", "", text)
    if len(digits_only) == 7:
        return f"{digits_only[:2]}-{digits_only[2:]}"

    return text


def expand_box(x1, y1, x2, y2, img_w, img_h, pad_x_ratio=0.35, pad_y_ratio=1.10, min_pad=10):
    """Add padding around a detected plate box so OCR sees the full plate.

    Wide plates keep nearly the same padding as the good v1 version.
    Square/two-row plates get extra vertical padding to catch the top row.
    """
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(bh)

    # If the detected plate is closer to square, it may be a two-row plate.
    if aspect < 2.3:
        pad_x_ratio = max(pad_x_ratio, 0.45)
        pad_y_ratio = max(pad_y_ratio, 1.80)
    else:
        pad_y_ratio = max(pad_y_ratio, 1.10)

    pad_x = max(min_pad, int(bw * pad_x_ratio))
    pad_y = max(min_pad, int(bh * pad_y_ratio))

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(img_w - 1, x2 + pad_x)
    py2 = min(img_h - 1, y2 + pad_y)

    return px1, py1, px2, py2

def resize_keep_aspect(image, target_height=OCR_RESIZE_HEIGHT):
    """Resize a crop to a readable height without stretching the plate shape."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image

    scale = target_height / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)


def sharpen_image(gray):
    """Light sharpening for blurry plate crops."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)


def make_ocr_variants(plate_crop):
    """Create several processed versions of the same crop and let EasyOCR pick the best."""
    variants = []

    resized = resize_keep_aspect(plate_crop)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 1) Normal grayscale
    variants.append(("gray", gray))

    # 2) Contrast-enhanced grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    variants.append(("contrast", contrast))

    # 3) Sharpened contrast image
    sharp = sharpen_image(contrast)
    variants.append(("sharp", sharp))

    # 4) Otsu threshold, useful when text/background contrast is clear
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    # 5) Inverted threshold, sometimes plate text is easier this way
    variants.append(("otsu_inverted", cv2.bitwise_not(otsu)))

    # 6) Adaptive threshold for uneven lighting
    adaptive = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )
    variants.append(("adaptive", adaptive))

    return variants


def save_debug_images(frame_id, plate_index, raw_crop, variants):
    """Debug image saving disabled for website/demo speed."""
    return


def plate_digits(text):
    """Return only digits from an OCR plate candidate."""
    return re.sub(r"\D", "", text or "")


def make_candidate(text, raw_text, confidence, method, source="single"):
    """Create a normalized OCR candidate with useful metadata."""
    cleaned = clean_plate_text(text)
    digits = plate_digits(cleaned)
    return {
        "text": cleaned,
        "raw_text": str(raw_text),
        "confidence": round(float(confidence), 2),
        "method": method,
        "source": source,
        "digits_count": len(digits),
        "is_full_plate": len(digits) == 7,
        "is_partial_plate": len(digits) == 5,
    }


def read_ocr_candidates_from_image(image, method_prefix, min_chars=OCR_MIN_CHARS, source="single"):
    """Run EasyOCR on one processed image and return cleaned candidates."""
    candidates = []
    try:
        ocr_results = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        )
    except Exception as e:
        print(f"[OCR WARNING] OCR failed on {method_prefix}: {e}")
        return candidates

    for item in ocr_results:
        # EasyOCR returns: [box, text, confidence]
        try:
            raw_text = str(item[1])
            conf = float(item[2]) * 100.0
        except Exception:
            continue

        cleaned = clean_plate_text(raw_text)
        if len(plate_digits(cleaned)) < min_chars:
            continue

        candidates.append(make_candidate(
            text=cleaned,
            raw_text=raw_text,
            confidence=conf,
            method=method_prefix,
            source=source
        ))

    return candidates


def get_standard_ocr_candidates(plate_crop):
    """Run the original v1 multi-method OCR over the full crop."""
    candidates = []
    variants = make_ocr_variants(plate_crop)
    for method_name, image in variants:
        candidates.extend(read_ocr_candidates_from_image(image, method_name, min_chars=OCR_MIN_CHARS, source="single"))
    return candidates


def get_stacked_plate_candidates(plate_crop):
    """Try to read square/two-row Jordan plates as top 2 digits + bottom 5 digits.

    This only adds candidates. It does not force a result. If the top row is not
    actually detected, the API will still return unreadable/partial instead of
    inventing the missing digits.
    """
    if plate_crop is None or plate_crop.size == 0:
        return []

    h, w = plate_crop.shape[:2]
    if h <= 0 or w <= 0:
        return []

    aspect = w / float(h)
    # Only try the extra stacked logic on near-square / tall plate crops.
    # Normal wide plates already work better with the standard v1 logic.
    if aspect > 1.35:
        return []

    top_crop = plate_crop[0:max(1, int(h * 0.55)), :]
    bottom_crop = plate_crop[max(0, int(h * 0.32)):h, :]

    top_candidates = []
    bottom_candidates = []

    for name, img in make_ocr_variants(top_crop):
        top_candidates.extend(read_ocr_candidates_from_image(img, f"stacked_top_{name}", min_chars=1, source="stacked_top"))

    for name, img in make_ocr_variants(bottom_crop):
        bottom_candidates.extend(read_ocr_candidates_from_image(img, f"stacked_bottom_{name}", min_chars=3, source="stacked_bottom"))

    top_two = []
    for c in top_candidates:
        d = plate_digits(c["text"])
        # Accept exact 2 digits. If OCR reads extra noise, do not guess from it.
        if len(d) == 2 and c["confidence"] >= 20.0:
            top_two.append((d, c))

    bottom_five = []
    for c in bottom_candidates:
        d = plate_digits(c["text"])
        if len(d) == 5 and c["confidence"] >= 35.0:
            bottom_five.append((d, c))

    combined = []
    for top_digits, top in top_two:
        for bottom_digits, bottom in bottom_five:
            avg_conf = (top["confidence"] + bottom["confidence"]) / 2.0
            text = f"{top_digits}-{bottom_digits}"
            combined.append({
                "text": text,
                "raw_text": f"{top['raw_text']} / {bottom['raw_text']}",
                "confidence": round(avg_conf, 2),
                "method": f"stacked:{top['method']}+{bottom['method']}",
                "source": "stacked_combined",
                "digits_count": 7,
                "is_full_plate": True,
                "is_partial_plate": False,
            })

    return combined


def choose_best_plate_candidate(candidates):
    """Choose the safest OCR result.

    Rules:
    1) A complete Jordan-style 7-digit plate beats a partial 5-digit read.
    2) If several full-plate candidates exist, prefer the plate text that appears
       in more than one OCR method. This prevents one high-confidence odd read
       like 44-87533 from beating repeated reads of 44-87523.
    3) If there is no repeated full plate, use the highest-confidence full plate.
    4) Only fall back to partial/other reads when no valid full plate exists.
    """
    if not candidates:
        return None, "no_readable_text"

    # Full plate = 7 digits, formatted like 44-87523.
    full_candidates = [
        c for c in candidates
        if c.get("is_full_plate") and c.get("confidence", 0.0) >= FULL_PLATE_MIN_CONFIDENCE
    ]

    if full_candidates:
        grouped = {}
        for c in full_candidates:
            key = c.get("text", "")
            grouped.setdefault(key, []).append(c)

        group_summaries = []
        for text, group in grouped.items():
            max_conf = max(c["confidence"] for c in group)
            avg_conf = sum(c["confidence"] for c in group) / len(group)
            support = len(group)
            # Prefer repeated agreement first, then confidence.
            group_summaries.append((support, max_conf, avg_conf, text, group))

        support, max_conf, avg_conf, text, group = max(group_summaries, key=lambda x: (x[0], x[1], x[2]))
        best_full = max(group, key=lambda c: c["confidence"]).copy()
        best_full["support_count"] = support
        best_full["methods_agreeing"] = sorted({c["method"] for c in group})
        best_full["best_group_confidence"] = round(max_conf, 2)
        best_full["avg_group_confidence"] = round(avg_conf, 2)
        return best_full, "read"

    # Fallback: choose the highest confidence candidate, same as v1.
    best = max(candidates, key=lambda c: c["confidence"])
    if best["confidence"] < OCR_MIN_CONFIDENCE:
        return best, "low_confidence"

    # Five digits are useful but incomplete for Jordan plate display.
    if best.get("is_partial_plate"):
        return best, "partial_read"

    return best, "read"


def run_plate_ocr(plate_crop, frame_id="frame", plate_index=0):
    """Run OCR on multiple crop versions and return the best safe result."""
    if plate_crop is None or plate_crop.size == 0:
        return {
            "plate_number": "unreadable",
            "ocr_confidence": 0.0,
            "ocr_status": "empty_crop",
            "ocr_method": None,
            "raw_ocr_text": "",
            "ocr_candidates": []
        }

    save_debug_images(frame_id, plate_index, plate_crop, [])

    candidates = get_standard_ocr_candidates(plate_crop)

    # Extra stacked/two-row attempt only if the standard OCR did not already find
    # a full plate. This avoids slowing down the good wide-plate cases.
    if not any(c.get("is_full_plate") and c["confidence"] >= FULL_PLATE_MIN_CONFIDENCE for c in candidates):
        candidates.extend(get_stacked_plate_candidates(plate_crop))

    # Sort output candidates by confidence so the JSON is easier to inspect.
    candidates_for_output = sorted(candidates, key=lambda c: c["confidence"], reverse=True)[:8]

    best, status = choose_best_plate_candidate(candidates)

    if best is None:
        return {
            "plate_number": "unreadable",
            "ocr_confidence": 0.0,
            "ocr_status": "no_readable_text",
            "ocr_method": None,
            "raw_ocr_text": "",
            "ocr_candidates": []
        }

    if status == "low_confidence":
        return {
            "plate_number": "unreadable",
            "ocr_confidence": best["confidence"],
            "ocr_status": "low_confidence",
            "ocr_method": best["method"],
            "raw_ocr_text": best["raw_text"],
            "ocr_candidates": candidates_for_output
        }

    return {
        "plate_number": best["text"],
        "ocr_confidence": best["confidence"],
        "ocr_status": status,
        "ocr_method": best["method"],
        "raw_ocr_text": best["raw_text"],
        "support_count": best.get("support_count", 1),
        "methods_agreeing": best.get("methods_agreeing", [best["method"]]),
        "ocr_candidates": candidates_for_output
    }


def compact_cv_result(results):
    """Return only the fields the dashboard/history needs.

    The detection/OCR decision logic is unchanged. This only removes bulky debug
    text such as OCR candidate lists from the HTTP response.
    """
    compact = {
        "incidents": results.get("incidents", []),
        "license_plates": [],
        "summary": results.get("summary", {}),
    }

    for plate in results.get("license_plates", []):
        compact["license_plates"].append({
            "plate_number": plate.get("plate_number", "unreadable"),
            "detection_confidence": plate.get("detection_confidence", 0.0),
            "ocr_confidence": plate.get("ocr_confidence", 0.0),
            "ocr_status": plate.get("ocr_status", "unknown"),
            "ocr_method": plate.get("ocr_method"),
            "support_count": plate.get("support_count", 1),
            "box": plate.get("box", []),
            "crop_size": plate.get("crop_size", {}),
        })

    return compact


def analyze_frame_array(frame, skip_plate_if_no_incident=False, compact=False):
    start_time = time.time()
    frame_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results = {}

    # -----------------------------
    # 1. Incident detection model
    # -----------------------------
    with torch.inference_mode():
        incident_results = incident_model(
            frame,
            conf=INCIDENT_CONFIDENCE,
            verbose=False,
            device=YOLO_DEVICE
        )

    incidents_found = []

    for result in incident_results:
        if result.boxes is not None:
            for box_obj in result.boxes:
                class_id = int(box_obj.cls[0])
                class_name = incident_model.names[class_id]
                confidence = float(box_obj.conf[0])
                box = box_obj.xyxy[0].tolist()

                incidents_found.append({
                    "type": class_name,
                    "confidence": round(confidence * 100, 2),
                    "box": [round(x) for x in box]
                })

    results["incidents"] = incidents_found

    if skip_plate_if_no_incident and not incidents_found:
        results["license_plates"] = []
        results["summary"] = {
            "api_version": API_VERSION,
            "violation_detected": False,
            "incidents_detected": 0,
            "plates_detected": 0,
            "readable_plates_detected": 0,
            "full_plates_detected": 0,
            "partial_plates_detected": 0,
            "incident_types": [],
            "plate_numbers": [],
            "full_plate_numbers": [],
            "partial_plate_numbers": [],
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(time.time() - start_time, 3),
            "device": "cuda" if USE_CUDA else "cpu",
            "easyocr_gpu": EASYOCR_GPU,
            "fast_scan": True,
            "plate_stage_skipped": True
        }
        return compact_cv_result(results) if compact else results

    # -----------------------------
    # 2. License plate detection + safer OCR
    # -----------------------------
    with torch.inference_mode():
        plate_results = plate_model(
            frame,
            conf=PLATE_CONFIDENCE,
            verbose=False,
            device=YOLO_DEVICE
        )

    plates_found = []
    h, w = frame.shape[:2]
    plate_index = 0

    for result in plate_results:
        if result.boxes is not None:
            for box_obj in result.boxes:
                plate_index += 1
                detection_confidence = float(box_obj.conf[0])

                x1, y1, x2, y2 = map(int, box_obj.xyxy[0])

                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                px1, py1, px2, py2 = expand_box(x1, y1, x2, y2, w, h)
                plate_crop = frame[py1:py2, px1:px2]

                ocr_result = run_plate_ocr(plate_crop, frame_id=frame_id, plate_index=plate_index)

                plates_found.append({
                    "plate_number": ocr_result["plate_number"],
                    "detection_confidence": round(detection_confidence * 100, 2),
                    "ocr_confidence": ocr_result["ocr_confidence"],
                    "ocr_status": ocr_result["ocr_status"],
                    "ocr_method": ocr_result["ocr_method"],
                    "raw_ocr_text": ocr_result["raw_ocr_text"],
                    "support_count": ocr_result.get("support_count", 1),
                    "methods_agreeing": ocr_result.get("methods_agreeing", []),
                    "ocr_candidates": ocr_result["ocr_candidates"],
                    "box": [x1, y1, x2, y2],
                    "padded_box": [px1, py1, px2, py2],
                    "crop_size": {
                        "width": int(max(0, px2 - px1)),
                        "height": int(max(0, py2 - py1))
                    }
                })

    results["license_plates"] = plates_found

    full_plate_numbers = [
        p["plate_number"] for p in plates_found
        if p["plate_number"] != "unreadable" and p.get("ocr_status") == "read"
    ]
    partial_plate_numbers = [
        p["plate_number"] for p in plates_found
        if p["plate_number"] != "unreadable" and p.get("ocr_status") == "partial_read"
    ]
    readable_plate_numbers = full_plate_numbers + partial_plate_numbers

    results["summary"] = {
        "api_version": API_VERSION,
        "violation_detected": len(incidents_found) > 0,
        "incidents_detected": len(incidents_found),
        "plates_detected": len(plates_found),
        "readable_plates_detected": len(readable_plate_numbers),
        "full_plates_detected": len(full_plate_numbers),
        "partial_plates_detected": len(partial_plate_numbers),
        "incident_types": sorted(list(set([i["type"] for i in incidents_found]))),
        "plate_numbers": readable_plate_numbers,
        "full_plate_numbers": full_plate_numbers,
        "partial_plate_numbers": partial_plate_numbers,
        "timestamp": datetime.now().isoformat(),
        "processing_time_seconds": round(time.time() - start_time, 3),
        "device": "cuda" if USE_CUDA else "cpu",
        "easyocr_gpu": EASYOCR_GPU,
        "ocr_min_confidence": OCR_MIN_CONFIDENCE,
        "full_plate_min_confidence": FULL_PLATE_MIN_CONFIDENCE,
        "debug_plates_saved": SAVE_OCR_DEBUG,
        "ocr_numeric_mapping": True,
        "full_plate_priority": True,
        "full_plate_voting": True,
        "stacked_plate_support": True,
        "fast_scan": bool(skip_plate_if_no_incident),
        "plate_stage_skipped": False
    }

    return compact_cv_result(results) if compact else results


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "api_version": API_VERSION,
        "status": "running",
        "incident_model_loaded": os.path.exists(INCIDENT_MODEL_PATH),
        "plate_model_loaded": os.path.exists(PLATE_MODEL_PATH),
        "torch_version": torch.__version__,
        "cuda_available": USE_CUDA,
        "cuda_version": torch.version.cuda if USE_CUDA else None,
        "gpu_name": torch.cuda.get_device_name(0) if USE_CUDA else None,
        "yolo_device": "cuda:0" if USE_CUDA else "cpu",
        "easyocr_gpu": EASYOCR_GPU,
        "ocr_min_confidence": OCR_MIN_CONFIDENCE,
        "full_plate_min_confidence": FULL_PLATE_MIN_CONFIDENCE,
        "save_ocr_debug": SAVE_OCR_DEBUG,
        "debug_dir": DEBUG_DIR if SAVE_OCR_DEBUG else None,
        "ocr_numeric_mapping": True,
        "full_plate_priority": True,
        "full_plate_voting": True,
        "stacked_plate_support": True
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Receives a base64 image from Unreal / website / Python code.
    Expected JSON:
    {
        "frame": "base64_encoded_image_here"
    }
    """
    try:
        data = request.get_json()

        if data is None or "frame" not in data:
            return jsonify({"error": "Missing 'frame' field"}), 400

        image_data = base64.b64decode(data["frame"])
        np_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        result = analyze_frame_array(frame)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze_file", methods=["POST"])
def analyze_file():
    """
    Receives an uploaded image file. Easier for testing with Postman/Python.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "Missing image file"}), 400

        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image file"}), 400

        result = analyze_frame_array(frame)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze_file_fast", methods=["POST"])
def analyze_file_fast():
    """
    Fast dashboard endpoint.

    It first runs incident detection. If there is no violation, it skips the
    license-plate/OCR stage because the dashboard only saves history when a
    violation exists. If a violation is found, it runs the same plate detection
    and OCR logic as /analyze_file, then returns a compact JSON response.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "Missing image file"}), 400

        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image file"}), 400

        result = analyze_frame_array(frame, skip_plate_if_no_incident=True, compact=True)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze_fast", methods=["POST"])
def analyze_fast():
    """Base64 version of /analyze_file_fast for future use."""
    try:
        data = request.get_json()
        if data is None or "frame" not in data:
            return jsonify({"error": "Missing 'frame' field"}), 400

        image_data = base64.b64decode(data["frame"])
        np_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        result = analyze_frame_array(frame, skip_plate_if_no_incident=True, compact=True)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
