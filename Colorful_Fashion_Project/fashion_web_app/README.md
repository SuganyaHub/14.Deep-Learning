# Colorful Fashion Recommendation

A Streamlit web app that detects selected fashion categories (Dress, Bag, Shoe) with the existing trained YOLO model and recommends visually similar items from the saved feature gallery.

## Features
- Upload JPG, JPEG, or PNG images.
- Detect Dress, Bag, and Shoe.
- Filter strongly overlapping detections to reduce duplicate boxes.
- Show a clean detection image matching the detected-item cards.
- Recommend category-matched images using MobileNetV2 visual features and colour histograms.
- Adjust confidence, number of recommendations, and shape/colour weighting.
- Cache models, gallery data, crops, and repeated image analysis during the running session.

## Expected project structure
```text
Colorful_Fashion_Project/
├── fashion_web_app/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── fashion_crop_cache.pkl
└── runs/detect/fashion_yolo_fast-2/weights/best.pt
```
Keep the existing trained model and `fashion_crop_cache.pkl` in their current locations. This app does not retrain or overwrite the model.

## Run on Windows
```bat
conda activate aidl
cd /d "D:\Hope\AI Course Tamil\16. Week-11-Deep Learning Module\4.Assignment\Colorful_Fashion_Project\fashion_web_app"
python -m streamlit run app.py
```
Or use the exact environment interpreter:
```bat
"C:\Anaconda3\envs\aidl\python.exe" -m streamlit run app.py
```
Streamlit normally opens `http://localhost:8501`. Keep the terminal open while using the app.

## Settings
- **Detection confidence:** start around 0.30–0.35. Lower values may add false detections; higher values may miss items.
- **Recommendations per item:** 3–5.
- **Shape vs colour weight:** 0.70 gives 70% weight to MobileNetV2 visual similarity and 30% to colour similarity.

## Speed
The first load can take longer because TensorFlow and the models are initialized. Model and gallery loaders and repeated image analysis are cached for the current Streamlit process. Restarting Streamlit clears in-memory caches, so the first analysis after a restart can be slower.

## Detection limitations
The extra overlap filter reduces repeated boxes, but cannot guarantee perfect classification. If the trained model labels a garment part as a bag or confuses a sock with a shoe, that is a model prediction error and may require better training examples and model evaluation. This app does not change the trained weights.

## Troubleshooting
- **Model not found:** verify `runs/detect/fashion_yolo_fast-2/weights/best.pt`.
- **Feature cache not found:** verify `fashion_crop_cache.pkl` in the project root.
- **No detections:** try a clear image and reduce confidence slightly.
- **Slow first run:** wait for model loading to finish; keep the terminal open.
- **Port in use:** stop the previous Streamlit process before launching another one.

## Technologies
Python, Streamlit, Ultralytics YOLO, TensorFlow/Keras MobileNetV2, OpenCV, Pillow, NumPy, scikit-learn.
