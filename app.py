from flask import Flask, render_template, request, Response
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import cv2
from rembg import remove
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model TFLite 1
interpreter1 = tf.lite.Interpreter(model_path="model/model_mangga.tflite")
interpreter1.allocate_tensors()

# Load model TFLite 2
interpreter2 = tf.lite.Interpreter(model_path="model/model_mangotype.tflite")
interpreter2.allocate_tensors()

label_model1 = ["Bukan Mangga", "Mangga"]
label_model2 = ["Amplem Sari", "Angus", "Bila", "Dodol", "Gedang", "Golek", "Lainnya", "Sanih", "Wini"]

def preprocess_image_cv2(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_image_upload(image_path):
    # Buka gambar dengan PIL
    img = Image.open(image_path).convert('RGB')

    # Convert PIL image ke bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    # Hapus background (rembg hanya terima bytes)
    img_no_bg_bytes = remove(img_bytes)

    # Convert hasilnya kembali ke PIL Image
    img_rgb = Image.open(io.BytesIO(img_no_bg_bytes)).convert('RGB')

    # Resize dan normalisasi
    img_resized = img_rgb.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)



def predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Halaman upload dan prediksi gambar
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            input_data = preprocess_image_upload(filepath)

            result1 = predict(interpreter1, input_data)
            idx1 = int(np.argmax(result1))
            label1 = label_model1[idx1]
            confidence1 = float(np.max(result1))

            label2 = None
            confidence2 = None
            top3_labels = []
            top3_confidences = []

            if label1 == "Mangga":
                result2 = predict(interpreter2, input_data)[0]
                top3_indices = np.argsort(result2)[-3:][::-1]
                top3_labels = [label_model2[i] for i in top3_indices]
                top3_confidences = [float(result2[i]) for i in top3_indices]

                label2 = top3_labels[0]
                confidence2 = top3_confidences[0]

            top3 = list(zip(top3_labels, top3_confidences))
            return render_template("result.html", 
                       img_path=filepath,
                       label1=label1, confidence1=confidence1,
                       label2=label2, confidence2=confidence2,
                       top3=top3)

    return render_template("index.html")

# Generator video streaming webcam
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        success, frame = cap.read()
        if not success:
            break

        input_data = preprocess_image_cv2(frame)
        result1 = predict(interpreter1, input_data)
        idx1 = int(np.argmax(result1))
        label1 = label_model1[idx1]
        confidence1 = float(np.max(result1))

        label2 = ""
        confidence2 = 0

        if label1 == "Mangga":
            result2 = predict(interpreter2, input_data)[0]
            top3_indices = np.argsort(result2)[-3:][::-1]
            top3_labels = [label_model2[i] for i in top3_indices]
            top3_confidences = [result2[i] for i in top3_indices]

            label2 = top3_labels[0]
            confidence2 = top3_confidences[0]

        text1 = f"{label1} ({confidence1*100:.1f}%)"
        cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if label1 == "Mangga":
            text2 = f"{label2} ({confidence2*100:.1f}%)"
            cv2.putText(frame, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Route baru untuk halaman streaming webcam
@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

# Route streaming video webcam
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
