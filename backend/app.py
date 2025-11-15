#backend/app.py
from flask import Flask, send_from_directory, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import mimetypes 

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key')

mimetypes.add_type('video/mp4', '.mp4')

CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"], "supports_credentials": True}},
     methods=["GET", "POST", "OPTIONS", "DELETE"],
     allow_headers=["Authorization", "Content-Type", "Origin"])

from auth import auth_bp, init_auth
init_auth(app)

try:
    from instrument_detection import instrument_detection_bp
    print("Successfully imported instrument_detection_bp")
except ImportError as e:
    print(f"Failed to import instrument_detection_bp: {e}")
from instrument_segmentation import instrument_segmentation_bp
from atomic_actions import atomic_actions_bp
from phase_step import phase_step_bp
try:
    from combined_inference import combined_bp
    print("Successfully imported combined_bp")
except ImportError as e:
    print(f"Failed to import combined_bp: {e}")

app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(instrument_detection_bp, url_prefix='/predict')
app.register_blueprint(instrument_segmentation_bp, url_prefix='/predict')
app.register_blueprint(atomic_actions_bp, url_prefix='/predict')
app.register_blueprint(phase_step_bp, url_prefix='/predict')
app.register_blueprint(combined_bp, url_prefix='/predict')

@app.route('/predict/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    print(f"Handling OPTIONS for /predict/{path}")
    return '', 200

@app.route('/Uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory('Uploads', filename)

@app.route('/predict/uploads/<path:filename>')
def serve_uploads(filename):
    file_path = os.path.join('Uploads', filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return "File not found", 404

    mime_type, _ = mimetypes.guess_type(filename)
    if filename.lower().endswith(('.mp4', '.mov', '.avi')):
        mime_type = 'video/mp4'

    print(f"Serving: {filename} | MIME: {mime_type}")
    return send_from_directory('Uploads', filename, mimetype=mime_type)


@app.before_request
def log_request():
    print(f"Received {request.method} request for {request.path}")

if __name__ == '__main__':
    os.makedirs('Uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    app.run(host="0.0.0.0", debug=True, port=5000)