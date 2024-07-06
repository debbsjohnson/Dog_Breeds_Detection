from flask import Flask, request, render_template
import os
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__, template_folder="/Users/mac/PycharmProjects/DogBreed_CV/templates")

class_labels = ['beagle', 'bulldog', 'dalmatian', 'german-shepherd', 'husky', 'labrador-retriever', 'poodle', 'rottweiler']

def predict_and_display(img):
    model = tf.keras.models.load_model('models/fine_tuned_inception.h5')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence_level = np.max(predictions)
    predicted_class_name = class_labels[predicted_class]

    return predicted_class_name, confidence_level

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            os.makedirs('static', exist_ok=True)
            file.save(filepath)
            img = image.load_img(filepath, target_size=(224, 224))

            predicted_class, confidence = predict_and_display(img)
            alert_message = f"Detected: {predicted_class} with confidence: {confidence:.2f}%"
            return render_template('index.html', alert_message=alert_message, image_path=filepath, actual_label=predicted_class,
                                   predicted_label=predicted_class, confidence=confidence)
    return render_template('index.html', message='Upload an image')

if __name__ == '__main__':
    app.run(debug=True)
