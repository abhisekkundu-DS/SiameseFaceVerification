
from flask import Flask, request, jsonify
from keras.models import load_model
import keras
from keras.layers import Layer
import tensorflow as tf
import numpy as np
import cv2
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration - FIXED INPUT SIZE
MODEL_PATH = r"C:\Users\Abhisek kundu\Downloads\face_verification_model_updated_2_1.h5"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
SIMILARITY_THRESHOLD = 0.8
INPUT_SIZE = (100, 100)  #  CHANGED TO MATCH YOUR MODEL (100x100)

#  Define the custom L2Normalize layer with axis parameter
class L2Normalize(Layer):
    def __init__(self, axis=1, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        config = super(L2Normalize, self).get_config()
        config.update({'axis': self.axis})
        return config

#  Global model variable
model = None

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_face_verification_model() -> bool:
    """Load the face verification model with custom layer handling"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return False
            
        logger.info(" Loading model...")
        keras.config.enable_unsafe_deserialization()
        
        # Load model with custom objects
        model = load_model(
            MODEL_PATH, 
            custom_objects={'L2Normalize': L2Normalize},
            safe_mode=False
        )
        
        logger.info(" Model loaded successfully!")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        logger.info(f"Number of model inputs: {len(model.inputs)}")
        
        return True
    except Exception as e:
        logger.error(f" Error loading model: {e}")
        return False

def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """Preprocess image for model inference"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image from {image_path}")
            return None
            
        # Convert BGR to RGB (most models expect RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size - NOW 100x100
        img = cv2.resize(img, INPUT_SIZE)
        
        # Normalize pixel values
        img = img.astype("float32") / 255.0
        
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

def get_similarity_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Get similarity score directly from the model"""
    try:
        # Add batch dimension
        input1 = np.expand_dims(img1, axis=0)
        input2 = np.expand_dims(img2, axis=0)
        
        # The model expects 2 inputs and returns similarity directly
        if len(model.inputs) == 2:
            # Pass both images to the model
            similarity = model.predict([input1, input2], verbose=0)
        else:
            # Fallback - try single input (though unlikely based on your error)
            similarity = model.predict(input1, verbose=0)
        
        # Extract the similarity score from the prediction
        if isinstance(similarity, (list, tuple)):
            similarity = similarity[0]
        
        # Handle different output formats
        if similarity.ndim > 1:
            similarity_score = float(similarity[0][0])
        else:
            similarity_score = float(similarity[0])
        
        logger.info(f"Model raw output: {similarity}, Extracted score: {similarity_score}")
        return similarity_score
        
    except Exception as e:
        logger.error(f"Error getting similarity from model: {e}")
        return 0.0

#  Load model when the app starts
if not load_face_verification_model():
    logger.error("Failed to load model. API will not function properly.")

@app.route("/verify", methods=["POST"])
def verify_faces():
    """Verify if two faces belong to the same person"""
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    # Check if files are present
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Please upload both image1 and image2 files"}), 400

    file1 = request.files["image1"]
    file2 = request.files["image2"]

    # Check if files are selected
    if file1.filename == '' or file2.filename == '':
        return jsonify({"error": "No files selected. Please select two image files."}), 400

    # Check file extensions
    if not allowed_file(file1.filename):
        return jsonify({"error": f"Invalid file type for image1. Allowed types: {ALLOWED_EXTENSIONS}"}), 400
    
    if not allowed_file(file2.filename):
        return jsonify({"error": f"Invalid file type for image2. Allowed types: {ALLOWED_EXTENSIONS}"}), 400

    # Save uploaded files temporarily
    image1_path = "temp1.jpg"
    image2_path = "temp2.jpg"
    
    try:
        file1.save(image1_path)
        file2.save(image2_path)

        # Preprocess images
        img1 = preprocess_image(image1_path)
        img2 = preprocess_image(image2_path)

        if img1 is None:
            return jsonify({"error": "Error processing image1. Please ensure it's a valid image file."}), 400
        
        if img2 is None:
            return jsonify({"error": "Error processing image2. Please ensure it's a valid image file."}), 400

        # Get similarity score directly from model
        logger.info("Getting similarity score from model...")
        similarity_score = get_similarity_score(img1, img2)
        
        # Determine match (if similarity > threshold)
        is_match = similarity_score > SIMILARITY_THRESHOLD

        logger.info(f"Final similarity score: {similarity_score:.4f}, Match: {is_match}")

        return jsonify({
            "similarity_score": round(similarity_score, 4),
            "match": is_match,
            "threshold": SIMILARITY_THRESHOLD,
            "status": "success",
            "message": "Faces verified successfully",
            "model_output_used": True
        })

    except Exception as e:
        logger.error(f"Error during face verification: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    finally:
        # Clean up temporary files
        for temp_file in [image1_path, image2_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {temp_file}: {e}")

@app.route("/")
def home():
    return jsonify({
        "message": "Face Verification API is running!",
        "model_loaded": model is not None,
        "model_type": "dual_input_similarity" if model and len(model.inputs) == 2 else "unknown",
        "input_size": "100x100 pixels",
        "endpoints": {
            "verify": "POST /verify with image1 and image2 files"
        },
        "allowed_file_types": list(ALLOWED_EXTENSIONS)
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_inputs": len(model.inputs) if model else 0,
        "input_size": "100x100 pixels",
        "timestamp": np.datetime64('now').astype(str)
    })

if __name__ == "__main__":
    # Disable oneDNN warnings if needed
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    logger.info("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)