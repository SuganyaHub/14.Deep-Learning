import io
import inspect
import pickle
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Paths / constants (unchanged project layout)
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "runs" / "detect" / "fashion_yolo_fast-2" / "weights" / "best.pt"
CACHE_FILE = PROJECT_DIR / "fashion_crop_cache.pkl"

# Process 1: detection categories (all 10)
DETECT_CATEGORIES = ("dress", "bag", "shoe", "shirt", "pants", "shorts", "skirt", "jacket", "hat", "sunglass")
# Process 2: recommendation categories (only these 3)
REC_CATEGORIES = ("dress", "bag", "shoe")

CLOTHING = {"dress", "shirt", "pants", "shorts", "skirt", "jacket"}
SMALL_ITEMS = {"hat", "sunglass", "shoe", "bag"}   # small objects get a slightly lower threshold
SMALL_CONF_FACTOR = 0.75

# Different spellings the trained model might use -> canonical names
ALIASES = {
    "dresses": "dress", "bags": "bag", "handbag": "bag", "shoes": "shoe", "footwear": "shoe",
    "shirts": "shirt", "pant": "pants", "trousers": "pants", "jeans": "pants",
    "short": "shorts", "skirts": "skirt", "jackets": "jacket", "coat": "jacket",
    "hats": "hat", "sunglasses": "sunglass", "glasses": "sunglass",
}

MAX_IMAGE_SIDE = 1600
SAME_CLASS_IOU = 0.45
SAME_CLASS_CONTAIN = 0.85       # smaller same-class box inside a bigger one -> fragment
CLOTHING_CROSS_IOU = 0.75       # e.g. dress vs skirt on the same region
OTHER_CROSS_IOU = 0.92          # bag vs dress etc. are only merged when almost identical
CNN_DIM = 1280                  # MobileNetV2 avg-pooled output
COLOUR_DIM = 18 * 8

BOX_COLOURS = {
    "dress": "#8A3FFC", "bag": "#007C91", "shoe": "#D97706", "shirt": "#2563EB", "pants": "#16A34A",
    "shorts": "#DB2777", "skirt": "#DC2626", "jacket": "#65A30D", "hat": "#0891B2", "sunglass": "#CA8A04",
}


def _version_tuple(v):
    parts = []
    for p in str(v).split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# Newer Streamlit replaced use_container_width with width="stretch"
STRETCH = {"width": "stretch"} if _version_tuple(st.__version__) >= (1, 50) else {"use_container_width": True}


def normalise_label(name):
    key = str(name).strip().lower().replace("_", " ").replace("-", " ")
    return ALIASES.get(key, key)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Colorful Fashion Recommendation", page_icon="👗", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1400px}
.hero{padding:1.2rem 1.4rem;border-radius:16px;background:linear-gradient(120deg,#f7e8f0,#e9f2ff);margin-bottom:1rem}
.hero h1{margin:0;color:#25233a}.hero p{margin:.4rem 0 0;color:#55536a}
div[data-testid="stMetric"]{background:#faf9fc;padding:.8rem;border-radius:12px}
</style>
<div class="hero"><h1>👗 Colorful Fashion Recommendation</h1>
<p>Detect fashion items and discover visually similar styles for dresses, bags and shoes.</p></div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached resources (loaded once per server session, warmed up immediately)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained YOLO model not found:\n{MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    # Warm-up so the first real image is not slowed by lazy initialisation
    model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)
    return model


@st.cache_resource(show_spinner=False)
def load_feature_model():
    from tensorflow.keras.applications import MobileNetV2
    model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)  # warm-up
    return model


@st.cache_resource(show_spinner=False)
def load_gallery_cache():
    if not CACHE_FILE.is_file():
        raise FileNotFoundError(f"Feature cache not found:\n{CACHE_FILE}")
    with CACHE_FILE.open("rb") as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict) or not {"meta", "cnn", "colour"}.issubset(cache):
        raise ValueError("The saved feature cache has an unexpected format.")
    meta, cnn, colour = cache["meta"], cache["cnn"], cache["colour"]

    valid, problems = {}, []
    for cat in REC_CATEGORIES:
        if cat not in meta or cat not in cnn or cat not in colour:
            problems.append(f"'{cat}' is missing from the feature cache.")
            continue
        c = np.asarray(cnn[cat])
        k = np.asarray(colour[cat])
        n = len(meta[cat])
        if n == 0:
            problems.append(f"'{cat}' has no gallery items.")
        elif c.ndim != 2 or k.ndim != 2 or c.shape[0] != n or k.shape[0] != n:
            problems.append(f"'{cat}' cache sizes do not match (meta={n}, cnn={c.shape}, colour={k.shape}).")
        elif c.shape[1] != CNN_DIM or k.shape[1] != COLOUR_DIM:
            problems.append(f"'{cat}' feature dimension mismatch (cnn={c.shape[1]}, colour={k.shape[1]}).")
        else:
            valid[cat] = {"meta": meta[cat], "cnn": c.astype(np.float32), "colour": k.astype(np.float32)}
    return valid, problems


@st.cache_data(show_spinner=False, max_entries=512)
def load_gallery_crop(source, box):
    path = Path(str(source).replace("\\", "/"))
    if not path.is_absolute():
        path = PROJECT_DIR / path
    image = cv2.imread(str(path))
    if image is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# Load everything up-front so the first upload is fast
try:
    with st.spinner("Loading models and fashion gallery (first start only)…"):
        YOLO_MODEL = load_yolo_model()
        FEATURE_MODEL = load_feature_model()
        GALLERY, GALLERY_PROBLEMS = load_gallery_cache()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Check the trained model path and fashion_crop_cache.pkl path. No retraining is needed for this error.")
    st.stop()
except Exception as exc:
    st.error("Startup failed.")
    st.exception(exc)
    st.stop()

# Class mapping read straight from the trained model
MODEL_CLASSES = {int(i): normalise_label(n) for i, n in YOLO_MODEL.names.items()}
AVAILABLE = [c for c in DETECT_CATEGORIES if c in MODEL_CLASSES.values()]
MISSING = [c for c in DETECT_CATEGORIES if c not in MODEL_CLASSES.values()]
UNKNOWN = sorted(set(MODEL_CLASSES.values()) - set(DETECT_CATEGORIES))

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    selected_categories = st.multiselect("Item types to detect", AVAILABLE, default=AVAILABLE)
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.30, 0.05,
                           help="Small items (hat, sunglass, shoe, bag) use a slightly lower threshold automatically.")
    imgsz = st.select_slider("Detection resolution", options=[320, 416, 480, 640], value=320,
                             help="The model was trained at 320. Keep 320 for best accuracy; try 416/480 if small items (hat, sunglass) are missed.")
    tta = st.checkbox("Test-time augmentation (slower, more complete boxes)", value=False)
    top_k = st.slider("Recommendations per item", 3, 5, 5)
    shape_weight = st.slider("Shape vs colour weight", 0.0, 1.0, 0.70, 0.05,
                             help="0.70 means 70% visual feature similarity and 30% colour similarity.")
    st.caption("Models and gallery data are loaded once and cached for this server session.")
    if MISSING:
        st.warning("The trained model has no class for: " + ", ".join(MISSING)
                   + ". These can only be detected after retraining with labelled data.")
    if UNKNOWN:
        st.caption("Model classes not in the 10-category list (ignored): " + ", ".join(UNKNOWN))
    for problem in GALLERY_PROBLEMS:
        st.warning("Gallery: " + problem)


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------
def prepare_image(image_bytes):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return image


def extract_features(crops, model):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    batch = np.stack([preprocess_input(np.asarray(c.resize((224, 224)).convert("RGB"), dtype=np.float32))
                      for c in crops])
    return model.predict(batch, verbose=0)


def colour_feature(rgb):
    h, w = rgb.shape[:2]
    cy, cx = int(h * .2), int(w * .2)
    centre = rgb[cy:h - cy, cx:w - cx]
    if centre.size == 0:
        centre = rgb
    hsv = cv2.cvtColor(centre, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).flatten()
    hist /= hist.sum() + 1e-8
    return np.sqrt(hist)


def box_overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    iou = inter / (area_a + area_b - inter + 1e-8)
    containment = inter / (min(area_a, area_b) + 1e-8)
    return iou, containment


def filter_duplicates(detections):
    kept = []
    for item in sorted(detections, key=lambda x: x["confidence"], reverse=True):
        duplicate = False
        for prev in kept:
            iou, contain = box_overlap(item["box"], prev["box"])
            if item["label"] == prev["label"]:
                if iou >= SAME_CLASS_IOU or contain >= SAME_CLASS_CONTAIN:
                    duplicate = True
            elif item["label"] in CLOTHING and prev["label"] in CLOTHING:
                if iou >= CLOTHING_CROSS_IOU:      # same garment labelled twice (dress/skirt...)
                    duplicate = True
            elif iou >= OTHER_CROSS_IOU:           # bag/hat/shoe next to clothes are kept
                duplicate = True
            if duplicate:
                break
        if not duplicate:
            kept.append(item)
    order = {c: i for i, c in enumerate(DETECT_CATEGORIES)}
    return sorted(kept, key=lambda x: (order.get(x["label"], 99), -x["confidence"]))


def load_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_detections(image, detections):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    scale = max(image.size) / 900
    font = load_font(max(14, int(16 * scale)))
    line_w = max(3, int(3 * scale))
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        colour = BOX_COLOURS.get(item["label"], "#444444")
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=line_w)
        label = f'{item["label"].title()} {item["confidence"]:.0%}'
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_h = bbox[3] - bbox[1]
        label_y = max(0, y1 - text_h - 8)
        draw.rectangle((x1, label_y, x1 + (bbox[2] - bbox[0]) + 12, label_y + text_h + 8), fill=colour)
        draw.text((x1 + 6, label_y + 4), label, fill="white", font=font)
    return canvas


# ----------------------------------------------------------------------------
# Process 1: detect ALL selected categories
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def analyse_image(image_bytes, conf, wanted, size, use_tta, _yolo, _feature_model):
    image = prepare_image(image_bytes)
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    low_conf = max(0.05, conf * SMALL_CONF_FACTOR)   # collect candidates, then apply per-class thresholds
    result = _yolo.predict(source=bgr, conf=low_conf, iou=0.60, imgsz=int(size), max_det=60,
                           augment=bool(use_tta), save=False, verbose=False)[0]
    width, height = image.size
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            label = normalise_label(result.names[int(box.cls[0])])
            if label not in wanted:
                continue
            score = float(box.conf[0])
            threshold = conf * SMALL_CONF_FACTOR if label in SMALL_ITEMS else conf
            if score < threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            detections.append({"box": (x1, y1, x2, y2), "label": label, "confidence": score})
    detections = filter_duplicates(detections)
    items = [{**d, "image": image.crop(d["box"])} for d in detections]

    # CNN features are only needed for the recommendation categories
    rec_items = [i for i in items if i["label"] in REC_CATEGORIES]
    if rec_items:
        features = extract_features([i["image"] for i in rec_items], _feature_model)
        for item, feature in zip(rec_items, features):
            item["features"] = feature
    return draw_detections(image, detections), items


# ----------------------------------------------------------------------------
# Process 2: recommendations (dress / bag / shoe only)
# ----------------------------------------------------------------------------
def group_items(rec_items):
    """Two similar-sized shoes side by side = one pair (one recommendation set)."""
    shoes = [i for i in rec_items if i["label"] == "shoe"]
    groups = [[i] for i in rec_items if i["label"] != "shoe"]
    paired = False
    if len(shoes) == 2:
        a, b = shoes
        ha, hb = a["box"][3] - a["box"][1], b["box"][3] - b["box"][1]
        cya, cyb = (a["box"][1] + a["box"][3]) / 2, (b["box"][1] + b["box"][3]) / 2
        if min(ha, hb) / max(ha, hb, 1) >= 0.6 and abs(cya - cyb) <= 0.8 * (ha + hb) / 2:
            groups.append([a, b])
            paired = True
    if not paired:
        groups += [[i] for i in shoes]
    return groups


def get_recommendations(members, uploaded_name, k, weight):
    category = members[0]["label"]
    if category not in REC_CATEGORIES or category not in GALLERY or any("features" not in m for m in members):
        return []
    data = GALLERY[category]
    feature = np.mean([np.asarray(m["features"], dtype=np.float32) for m in members], axis=0).reshape(1, -1)
    if feature.shape[1] != data["cnn"].shape[1]:
        return []
    shape = cosine_similarity(feature, data["cnn"])[0]
    query_colour = np.mean([colour_feature(np.asarray(m["image"])) for m in members], axis=0).reshape(1, -1)
    colour = cosine_similarity(query_colour, data["colour"])[0]
    scores = weight * shape + (1 - weight) * colour
    selected, used = [], set()
    uploaded_basename = Path(uploaded_name).name.lower()
    for idx in np.argsort(scores)[::-1]:
        source = str(data["meta"][idx]["source"])
        source_basename = Path(source.replace("\\", "/")).name.lower()
        if source_basename == uploaded_basename or source in used:
            continue
        used.add(source)
        selected.append({"idx": int(idx), "score": float(scores[idx]),
                         "shape": float(shape[idx]), "colour": float(colour[idx])})
        if len(selected) >= k:
            break
    return selected


def render_results(image_bytes, filename):
    with st.spinner("Detecting items…"):
        annotated, items = analyse_image(image_bytes, float(confidence), tuple(sorted(selected_categories)),
                                         int(imgsz), bool(tta), YOLO_MODEL, FEATURE_MODEL)
    st.subheader("Detection Results")
    st.image(annotated, caption="All detected items — boxes match the item cards below.", **STRETCH)
    if not items:
        st.warning("No selected items detected. Try a lower confidence (0.20–0.30), a higher resolution, "
                   "and a clear, well-lit image.")
        return

    st.subheader("Detected Fashion Items")
    counts = {}
    for i in items:
        counts[i["label"]] = counts.get(i["label"], 0) + 1
    metrics = st.columns(3)
    metrics[0].metric("Items detected", len(items))
    metrics[1].metric("Categories found", len(counts))
    metrics[2].metric("Recommendations per item", top_k)
    st.caption("Found: " + " · ".join(f"{c.title()} ×{n}" for c, n in counts.items()))

    cols = st.columns(min(4, len(items)))
    for index, item in enumerate(items):
        with cols[index % len(cols)]:
            st.image(item["image"], **STRETCH)
            st.markdown(f'**{item["label"].title()}**')
            st.caption(f'Confidence: {item["confidence"]:.0%}')

    rec_items = [i for i in items if i["label"] in REC_CATEGORIES]
    st.subheader("Similar Style Recommendations")
    st.caption("Recommendations are shown only for Dress, Bag and Shoe, matched within the same category.")
    if not rec_items:
        st.info("No Dress, Bag or Shoe detected, so there are no recommendations for this image.")
        return

    for number, members in enumerate(group_items(rec_items), start=1):
        category = members[0]["label"]
        recs = get_recommendations(members, filename, top_k, shape_weight)
        with st.container(border=True):
            title = f"{number}. {category.title()}" + (" (pair)" if len(members) == 2 else "")
            st.markdown(f"### {title}")
            columns = st.columns(top_k + 1)
            with columns[0]:
                for m in members:
                    st.image(m["image"], **STRETCH)
                st.caption("Detected pair" if len(members) == 2 else "Detected item")
            if not recs:
                st.info(f"No saved {category} recommendations available.")
                continue
            for col, rec in zip(columns[1:], recs):
                meta = GALLERY[category]["meta"][rec["idx"]]
                crop = load_gallery_crop(meta["source"], tuple(meta["box"]))
                with col:
                    if crop is None:
                        st.caption("Image unavailable")
                    else:
                        st.image(crop, **STRETCH)
                        st.caption(f'Match {rec["score"]:.0%}')
                        st.caption(f'Shape {rec["shape"]:.2f} · Colour {rec["colour"]:.2f}')
    st.success("Recommendation process completed.")


# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------
uploaded = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])
if not selected_categories:
    st.info("Select at least one item type in Settings.")
elif uploaded is None:
    st.info("Upload a clear fashion image to begin.")
else:
    file_bytes = uploaded.getvalue()
    try:
        preview = prepare_image(file_bytes)
        left, right = st.columns([1, 1.8], gap="large")
        with left:
            st.subheader("Input Image")
            st.image(preview, **STRETCH)
        with right:
            st.subheader("Ready to analyse?")
            st.write("Detect all selected items, then recommend similar styles for dresses, bags and shoes.")
            clicked = st.button("✨ Detect & Recommend", type="primary", **STRETCH)
        file_id = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("file_id") != file_id:
            st.session_state["file_id"] = file_id
            st.session_state["show_results"] = False
        if clicked:
            st.session_state["show_results"] = True
        if st.session_state.get("show_results", False):
            try:
                render_results(file_bytes, uploaded.name)
            except FileNotFoundError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error("The app could not complete this image.")
                st.exception(exc)
    except Exception as exc:
        st.error(f"Could not open this image: {exc}")
