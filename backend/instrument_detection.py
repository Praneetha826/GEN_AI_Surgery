# backend/instrument_detection.py
from flask import Blueprint, request, jsonify
from pymongo import MongoClient
import os
import torch
from datetime import datetime
from werkzeug.utils import secure_filename
import jwt as pyjwt
from torchvision import transforms
from PIL import Image
# Assuming 'mvit.py' is in a location accessible by the backend process
# If not, you may need to adjust the import path or ensure mvit.py is copied.
try:
    from mvit import MViTForMultiLabel
except ImportError:
    # Fallback/Error handling if mvit is not directly accessible
    # You might need to adjust sys.path or the package structure
    print("Warning: Could not import MViTForMultiLabel. Ensure 'mvit.py' is in the Python path.")
    # Define a dummy class to prevent runtime errors if MViT is not critical for immediate testing
    class MViTForMultiLabel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("MViTForMultiLabel is a dummy class now.")
        def forward(self, x):
            return torch.zeros(x.size(0), 4) # Mock output for 4 classes


instrument_detection_bp = Blueprint('instrument_detection', __name__)

# --- CONFIG ---
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/surgical_ai_db')
client = MongoClient(MONGO_URI)
db = client['surgical_ai_db']
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
# Updated to allow image extensions
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'webp'} 

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ["Bipolar", "NeedleDriver", "Monopolar", "Suction"]
# Per-class thresholds
THRESH = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.4} 

# Inference Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- MODEL LOADING ---
def load_model():
    # Model architecture definition must match the trained model

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model = MViTForMultiLabel(
        img_size=224, 
        patch_size=4, # Changed from 8 to 4 to match your training script
        in_chans=3,
        num_classes=len(CLASS_NAMES),
        embed_dims=[64, 128, 256, 384],
        num_blocks=[2, 2, 8, 4],
        num_heads=[1, 2, 4, 8],
        mlp_ratio=2.0,
        drop_rate=0.1, 
        attn_drop_rate=0.1,
        drop_path_rate=0.2, 
        use_aux_head=True
    )
    # Ensure this path is correct for your deployed environment
    # ckpt_path = os.path.join('backend', 'models', 'instrument_detection_weights.pth')
    ckpt_path = os.path.join(base_dir, 'models', 'instrument_detection_weights.pth')
    
    try:
        # Load the checkpoint
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        # Extract state_dict, accommodating for different ways it might be saved
        state_dict = ckpt.get('model_state_dict', ckpt) 
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {ckpt_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {ckpt_path} - returning uninitialized model")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)} - returning uninitialized model")
        
    model.eval()
    return model.to(DEVICE)

# Load model once globally for efficiency
# This should be done carefully in a production environment (e.g., within the application factory)
# For simplicity here, we'll keep the function and rely on the environment's handling.
# However, for a real Flask app, initialize it outside the request-handling function.
# GLOBAL_MODEL = load_model()

def get_instrument_detection_result(frame_rgb_pil: Image.Image, model: torch.nn.Module, device: torch.device):
    """
    Core logic to run detection on a single PIL Image.
    :param frame_rgb_pil: A PIL Image object (RGB mode).
    :param model: The loaded MViT model.
    :param device: The computation device.
    :return: (list of detected instrument names, dictionary of probabilities)
    """
    input_tensor = transform(frame_rgb_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    detected = []
    prob_dict = {}
    for i, p in enumerate(probs):
        prob_dict[CLASS_NAMES[i]] = float(p)
        if p > THRESH.get(i, 0.5):
            detected.append(CLASS_NAMES[i])
            
    return detected, prob_dict
# --- PREDICTION ROUTE ---
@instrument_detection_bp.route('/detect_image_instruments', methods=['POST', 'OPTIONS']) # New, descriptive path
def predict_image_route():
    print("Entered predict_image_route function for /detect_image_instruments")
    if request.method == 'OPTIONS':
        return '', 200
    
    # --- Authentication ---
    auth = request.headers.get('Authorization')
    if not auth:
        return jsonify({'error': 'Missing token'}), 401
    try:
        token = auth.split()[1]
        # Replace with your actual JWT key and algorithm
        payload = pyjwt.decode(token, os.getenv('JWT_SECRET_KEY', 'your_secret_key_here'), algorithms=['HS256']) 
        user_id = payload['user_id']
    except Exception as e:
        return jsonify({'error': f'Invalid token: {str(e)}'}), 401

    # --- File Handling (Image) ---
    if 'image' not in request.files: # Expect 'image' in the POST data
        return jsonify({'error': 'No image file part'}), 400
    file = request.files['image']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXT)}'}), 400

    fname = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, fname)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(image_path)
    print(f"Saved image to {image_path}")

    # --- Inference ---
    model = load_model() # Load model per-request (less efficient but safer), consider global loading
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return jsonify({'error': f'Could not open or process image file: {str(e)}'}), 500

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # Apply thresholds
    detected = []
    print(f"Image Probabilities: {probs}")
    for i, p in enumerate(probs):
        if p > THRESH.get(i, 0.5):
            detected.append(CLASS_NAMES[i])
            
    print(f"Detected instruments: {detected}")

    # --- Database Logging ---
    rel_path = f"Uploads/{fname}"
    entry = {
        'user_id': user_id,
        'file_path': rel_path, # Changed from video_path to file_path
        'model': 'instrument_detection_image',
        'result': detected,
        'timestamp': datetime.utcnow()
    }
    db.history.insert_one(entry)
    print(f"Saved to history: {entry}")

    # --- Response ---
    return jsonify({
        'instruments': detected,
        'image_path': rel_path,
        'probabilities': {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
    })