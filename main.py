from flask import Flask, render_template, request, redirect, url_for, jsonify
import mysql.connector
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import cv2
import os

app = Flask(__name__)
CORS(app) 

model = tf.keras.models.load_model('C:\\work_based\\leukemia_classification.h5')
class_names = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']

def load_and_preprocess_image(img, target_size=(224, 224)):
    if isinstance(img, str):
        img = image.load_img(img, target_size=target_size)
    else:
        img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def decode_base64_image(base64_string):
    image_data = BytesIO(base64.b64decode(base64_string))
    img = Image.open(image_data)
    return img

template_image_path = "E:/delta univeristy/Level 2 . semester 2/Work-based Professional Project in Compute Science (1)/DATA/Blood cell Cancer [ALL]/[Malignant] Pre-B/Snap_0s05.jpg"
template_image = cv2.imread(template_image_path)
template_image_resized = cv2.resize(template_image, (224, 224))

def is_similar_image(input_image_path, template_image_resized, threshold=0.6):
    input_image = cv2.imread(input_image_path)
    input_image_resized = cv2.resize(input_image, (224, 224))
    def calculate_hist(image, channel):
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        return hist
    
    similarity = 0
    for channel in range(3):
        hist_template = calculate_hist(template_image_resized, channel)
        hist_input = calculate_hist(input_image_resized, channel)
        similarity += cv2.compareHist(hist_template, hist_input, cv2.HISTCMP_CORREL)
    
    overall_similarity = similarity / 3
    return overall_similarity >= threshold
    #####################
    hist_template_r = cv2.calcHist([template_image_resized], [2], None, [256], [0, 256])
    hist_input_r = cv2.calcHist([input_image_resized], [2], None, [256], [0, 256])
    
    hist_template_g = cv2.calcHist([template_image_resized], [1], None, [256], [0, 256])
    hist_input_g = cv2.calcHist([input_image_resized], [1], None, [256], [0, 256])
    
    hist_template_b = cv2.calcHist([template_image_resized], [0], None, [256], [0, 256])
    hist_input_b = cv2.calcHist([input_image_resized], [0], None, [256], [0, 256])
    
    cv2.normalize(hist_template_r, hist_template_r)
    cv2.normalize(hist_input_r, hist_input_r)
    
    cv2.normalize(hist_template_g, hist_template_g)
    cv2.normalize(hist_input_g, hist_input_g)
    
    cv2.normalize(hist_template_b, hist_template_b)
    cv2.normalize(hist_input_b, hist_input_b)
    
    similarity_r = cv2.compareHist(hist_template_r, hist_input_r, cv2.HISTCMP_CORREL)
    similarity_g = cv2.compareHist(hist_template_g, hist_input_g, cv2.HISTCMP_CORREL)
    similarity_b = cv2.compareHist(hist_template_b, hist_input_b, cv2.HISTCMP_CORREL)
    
    overall_similarity = (similarity_r + similarity_g + similarity_b) / 3
    
    return overall_similarity >= threshold
    #########################
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    base64_string = data['image']
    img = decode_base64_image(base64_string)
    img_array = load_and_preprocess_image(img)
    
    temp_image_path = 'temp_image.jpg'
    img.save(temp_image_path)
    
    if is_similar_image(temp_image_path, template_image_resized):
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        class_name = class_names[predicted_class]
        response = {
            "class_name": class_name,
            "predictions": predictions.tolist()
        }
    else:
        response = {
            "error": "this image does not match the template."
        }
    
    os.remove(temp_image_path)
    
    return jsonify(response)

db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="AHMED@4221320",
  database="HospitalDB" 
)
cursor = db.cursor()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_info', methods=['POST'])
def submit_info():
    if request.method == 'POST':
        patient_name = request.form['patient_name']
        gender = request.form['gender']
        age = request.form['age']
        patient_address = request.form['patient_address']
        doctor_name = request.form['doctor_name']
        contact_number = request.form['contact_number']
        final_result = request.form['final_result']

        sql = "INSERT INTO Info (patient_name, gender, age, patient_address, doctor_name, contact_number, final_result) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        values = (patient_name, gender, age, patient_address, doctor_name, contact_number, final_result)
        cursor.execute(sql, values)
        db.commit()

        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
    
