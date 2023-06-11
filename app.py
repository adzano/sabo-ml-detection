from flask import Flask
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

app = Flask(__name__)

@app.route("/")
def hello():
    return "Success!!"

@app.route("/predict")
def predict():
    model = YOLO("result1.pt")
    results = model.predict(source="0", show=True)
    return(results)