from flask import Flask, request, jsonify, render_template
import pandas as pd
import pefile
import joblib
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model2.pkl')

# Load the model for predicting malware from images
model_image = joblib.load('best_model_RDM.pkl')

# Load the model for predicting malware types
malware_type_model = joblib.load('malware_classifier_model.pkl')


# Define function to extract features from images
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features, hog_image = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return hog_features

# Define the features used by the model
selected_features = ['ImageBase', 'Characteristics', 'SizeOfStackReserve',
                     'VersionInformationSize', 'ResourcesMinSize', 'MinorImageVersion',
                     'DllCharacteristics', 'MajorOperatingSystemVersion', 'ResourcesNb',
                     'ExportNb', 'SectionsMaxEntropy', 'ResourcesMinEntropy', 'Subsystem',
                     'SizeOfInitializedData', 'BaseOfData']

# Set the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'exe'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(filepath):
    try:
        pe = pefile.PE(filepath)

        version_info_size = 0
        if hasattr(pe, 'FileInfo') and pe.FileInfo:
            for file_info in pe.FileInfo:
                if hasattr(file_info, 'StringTable') and file_info.StringTable:
                    version_info_size += sum(
                        entry.struct.Version for entry in file_info.StringTable[0].entries.values())

        resources_min_size = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') and hasattr(pe.DIRECTORY_ENTRY_RESOURCE, 'entries'):
            resource_sizes = []
            for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                if hasattr(resource_type, 'directory'):
                    for resource_id in resource_type.directory.entries:
                        if hasattr(resource_id, 'directory'):
                            for resource_lang in resource_id.directory.entries:
                                if hasattr(resource_lang, 'data'):
                                    resource_sizes.append(resource_lang.data.struct.Size)
            if resource_sizes:
                resources_min_size = min(resource_sizes)

        resources_nb = len(pe.DIRECTORY_ENTRY_RESOURCE.entries) if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') and hasattr(
            pe.DIRECTORY_ENTRY_RESOURCE, 'entries') else 0
        export_nb = len(pe.DIRECTORY_ENTRY_EXPORT.symbols) if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT') and hasattr(
            pe.DIRECTORY_ENTRY_EXPORT, 'symbols') else 0

        sections_max_entropy = max([section.get_entropy() for section in pe.sections]) if pe.sections else 0
        sections_min_entropy = min([section.get_entropy() for section in pe.sections]) if pe.sections else 0

        features = {
            'ImageBase': pe.OPTIONAL_HEADER.ImageBase,
            'Characteristics': pe.FILE_HEADER.Characteristics,
            'SizeOfStackReserve': pe.OPTIONAL_HEADER.SizeOfStackReserve,
            'VersionInformationSize': version_info_size,
            'ResourcesMinSize': resources_min_size,
            'MinorImageVersion': pe.OPTIONAL_HEADER.MinorImageVersion,
            'DllCharacteristics': pe.OPTIONAL_HEADER.DllCharacteristics,
            'MajorOperatingSystemVersion': pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
            'ResourcesNb': resources_nb,
            'ExportNb': export_nb,
            'SectionsMaxEntropy': sections_max_entropy,
            'ResourcesMinEntropy': sections_min_entropy,
            'Subsystem': pe.OPTIONAL_HEADER.Subsystem,
            'SizeOfInitializedData': pe.OPTIONAL_HEADER.SizeOfInitializedData,
            'BaseOfData': getattr(pe.OPTIONAL_HEADER, 'BaseOfData', 0)
        }
        return features

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/file')
def file_page():
    return render_template('file.html')

@app.route('/image')
def image_page():
    return render_template('image.html')

@app.route('/typemalware')
def typemalware_page():
    return render_template('typemalware.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(os.path.basename(file.filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({'error': f"Failed to save file: {str(e)}"}), 500

        features = extract_features(filepath)
        if not features:
            return jsonify({'error': 'Failed to extract features from the uploaded file.'}), 500

        df = pd.DataFrame([features])
        df = df[selected_features]

        predictions = model.predict(df)
        result = 'Malware' if predictions[0] == 1 else 'Legitimate'

        return jsonify({'result': result})

    return jsonify({'error': 'Invalid file type.'}), 400

# Route for uploading image
@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return 'No image part'
    image = request.files['image']
    if image.filename == '':
        return 'No selected image'
    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        
        # Charger l'image et la convertir en niveaux de gris
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (42, 42))  # Adapter selon les besoins du modèle
        img = img.flatten().reshape(1, -1)
        
        pred = model_image.predict(img)
        result = 'Malware' if pred[0] == 1 else 'Not Malware'
        return f'Prediction for image {filename}: {result}'

    
@app.route('/typemalware', methods=['POST'])
def predict_typemalware():
    if 'image' not in request.files:
        return 'No image part'
    image = request.files['image']
    if image.filename == '':
        return 'No selected image'
    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        
        # Charger l'image et la convertir en niveaux de gris
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))  # Adapter selon les besoins du modèle
        img = img.flatten().reshape(1, -1)
        
        pred = malware_type_model.predict(img)
        result = f'Type of Malware: {pred[0]}'
        return jsonify({'result': result})


    return jsonify({'error': 'Invalid image type.'}), 400   
   


if __name__ == '__main__':
    app.run(debug=True)
