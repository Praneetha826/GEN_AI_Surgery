from flask import Blueprint, request, jsonify, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask_mail import Mail, Message
import jwt as pyjwt
from datetime import datetime, timedelta
from twilio.rest import Client
from werkzeug.security import generate_password_hash, check_password_hash
import os
import secrets

auth_bp = Blueprint('auth', __name__)

# Load config from env
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/surgical_ai_db')
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
GOOGLE_REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:5000/auth/google/callback')
MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True') == 'True'
MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
RESET_TOKEN_EXPIRATION_SECONDS = int(os.environ.get('RESET_TOKEN_EXPIRATION_SECONDS', 3600))
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_VERIFY_SERVICE_SID = os.environ.get('TWILIO_VERIFY_SERVICE_SID')
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:3000')

# Initialize global variables
mongo_client = None
db = None
mail = None
oauth = None
google = None
twilio_client = None

def init_auth(app):
    global mongo_client, db, mail, oauth, google, twilio_client
    required_vars = {
        'GOOGLE_CLIENT_ID': GOOGLE_CLIENT_ID,
        'GOOGLE_CLIENT_SECRET': GOOGLE_CLIENT_SECRET,
        'MAIL_USERNAME': MAIL_USERNAME,
        'MAIL_PASSWORD': MAIL_PASSWORD,
        'TWILIO_ACCOUNT_SID': TWILIO_ACCOUNT_SID,
        'TWILIO_AUTH_TOKEN': TWILIO_AUTH_TOKEN,
        'TWILIO_VERIFY_SERVICE_SID': TWILIO_VERIFY_SERVICE_SID,
        'JWT_SECRET_KEY': JWT_SECRET_KEY
    }
    for var_name, var_value in required_vars.items():
        if not var_value:
            raise ValueError(f"Missing required environment variable: {var_name}")

    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client['surgical_ai_db']
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MongoDB at {MONGO_URI}: {str(e)}")

    app.config['MAIL_SERVER'] = MAIL_SERVER
    app.config['MAIL_PORT'] = MAIL_PORT
    app.config['MAIL_USE_TLS'] = MAIL_USE_TLS
    app.config['MAIL_USERNAME'] = MAIL_USERNAME
    app.config['MAIL_PASSWORD'] = MAIL_PASSWORD
    global mail
    mail = Mail(app)

    global oauth, google
    oauth = OAuth(app)
    google = oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )

    global twilio_client
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def generate_token(user_id):
    payload = {
        'user_id': str(user_id),
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return pyjwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')

def create_reset_token(user_id):
    payload = {
        'user_id': str(user_id),
        'exp': datetime.utcnow() + timedelta(seconds=RESET_TOKEN_EXPIRATION_SECONDS)
    }
    return pyjwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    phone = data.get('phone')

    if not name or not email or not password:
        return jsonify({'error': 'Name, email, and password required'}), 400

    if db.users.find_one({'email': email}):
        return jsonify({'error': 'Email already exists'}), 400

    hashed_password = generate_password_hash(password)
    user = {
        'name': name,
        'email': email,
        'password': hashed_password,
        'phone': phone
    }
    result = db.users.insert_one(user)
    user_id = str(result.inserted_id)
    token = generate_token(user_id)
    user_data = {'id': user_id, 'name': name, 'email': email, 'phone': phone}
    return jsonify({'message': 'Signup successful', 'token': token, 'user': user_data})

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    user = db.users.find_one({'email': email})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 400

    user_id = str(user['_id'])
    token = generate_token(user_id)
    user_data = {'id': user_id, 'email': email, 'name': user.get('name'), 'phone': user.get('phone')}
    return jsonify({'message': 'Login successful', 'token': token, 'user': user_data})

@auth_bp.route('/google')
def google_login():
    nonce = secrets.token_urlsafe(16)
    session['nonce'] = nonce
    redirect_uri = url_for('auth.google_callback', _external=True)
    return google.authorize_redirect(redirect_uri, nonce=nonce)

@auth_bp.route('/google/callback')
def google_callback():
    try:
        token = google.authorize_access_token()
        nonce = session.pop('nonce', None)
        if not nonce:
            return jsonify({'error': 'Missing nonce in session'}), 400

        user_info = google.parse_id_token(token, nonce=nonce)
        email = user_info.get('email')
        name = user_info.get('name')

        user = db.users.find_one({'email': email})
        if not user:
            user_data = {
                'name': name,
                'email': email,
                'google_id': user_info.get('sub')
            }
            result = db.users.insert_one(user_data)
            user_id = result.inserted_id
        else:
            user_id = user['_id']

        token = generate_token(user_id)
        user_id_str = str(user_id)
        user_data = {'id': user_id_str, 'name': name, 'email': email}
        return redirect(f'{FRONTEND_URL}/?token={token}')
    except Exception as e:
        return jsonify({'error': f'Google OAuth failed: {str(e)}'}), 500

@auth_bp.route('/send-otp', methods=['POST'])
def send_otp():
    data = request.json
    phone = data.get('phone')

    if not phone:
        return jsonify({'error': 'Phone number required'}), 400

    verification = twilio_client.verify.v2.services(TWILIO_VERIFY_SERVICE_SID).verifications.create(to=phone, channel='sms')
    return jsonify({'message': 'OTP sent'})

@auth_bp.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.json
    phone = data.get('phone')
    code = data.get('code')

    if not phone or not code:
        return jsonify({'error': 'Phone and code required'}), 400

    verification_check = twilio_client.verify.v2.services(TWILIO_VERIFY_SERVICE_SID).verification_checks.create(to=phone, code=code)

    if verification_check.status == 'approved':
        user = db.users.find_one({'phone': phone})
        if not user:
            user = {'phone': phone}
            result = db.users.insert_one(user)
            user['_id'] = result.inserted_id

        token = generate_token(user['_id'])
        user_id = str(user['_id'])
        user_data = {'id': user_id, 'phone': phone, 'name': user.get('name')}
        return jsonify({'message': 'Verified', 'token': token, 'user': user_data})
    else:
        return jsonify({'error': 'Invalid code'}), 400

@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.json
    email = data.get('email')

    user = db.users.find_one({'email': email})
    if user:
        token = create_reset_token(user['_id'])
        reset_url = f"{FRONTEND_URL}/reset-password/{token}"
        msg = Message('Password Reset Request', sender=MAIL_USERNAME, recipients=[email])
        msg.body = f'To reset your password, visit: {reset_url}'
        mail.send(msg)

    return jsonify({'message': 'Reset link sent if email exists'}), 200

@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.json
    token = data.get('token')
    new_password = data.get('password')

    try:
        decoded = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        user_id = decoded['user_id']
        hashed_password = generate_password_hash(new_password)
        db.users.update_one({'_id': ObjectId(user_id)}, {'$set': {'password': hashed_password}})
        return jsonify({'message': 'Password reset successful'})
    except pyjwt.ExpiredSignatureError:
        return jsonify({'error': 'Token expired'}), 400
    except pyjwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 400

@auth_bp.route('/me', methods=['GET'])
def me():
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'Missing token'}), 401
    try:
        token = auth_header.split(' ')[1]
        decoded = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        user = db.users.find_one({'_id': ObjectId(decoded['user_id'])})
        if user:
            user['_id'] = str(user['_id'])
            if 'password' in user:
                del user['password']
            return jsonify({'user': user})
        return jsonify({'error': 'User not found'}), 404
    except pyjwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401

@auth_bp.route('/history', methods=['GET'])
def get_history():
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'Missing token'}), 401
    try:
        token = auth_header.split(' ')[1]
        decoded = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        user_id = decoded['user_id']
        history = list(db.history.find({'user_id': user_id}))
        for entry in history:
            entry['_id'] = str(entry['_id'])
            entry['timestamp'] = entry['timestamp'].isoformat()
        return jsonify({'history': history})
    except pyjwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401

@auth_bp.route('/history/<history_id>', methods=['DELETE'])
def delete_history(history_id):
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'Missing token'}), 401
    try:
        token = auth_header.split(' ')[1]
        decoded = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        user_id = decoded['user_id']
    except pyjwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401

    result = db.history.delete_one({'_id': ObjectId(history_id), 'user_id': user_id})
    if result.deleted_count == 1:
        return jsonify({'message': 'History entry deleted'})
    else:
        return jsonify({'error': 'History entry not found or not authorized'}), 404

# Feedback/Contact Form Routes
@auth_bp.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Handle feedback/contact form submissions"""
    try:
        data = request.json
        
        # Validate required fields
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        message = data.get('message', '').strip()
        
        if not name or not email or not message:
            return jsonify({'error': 'Name, email, and message are required'}), 400
        
        # Optional fields
        phone = data.get('phone', '').strip()
        rating = data.get('rating', 0)  # Ensure rating is stored
        category = data.get('category', 'contact')
        
        # Create feedback document
        feedback_doc = {
            'name': name,
            'email': email,
            'phone': phone,
            'message': message,
            'rating': rating,  # Rating is stored in the database
            'category': category,
            'timestamp': datetime.utcnow(),
            'status': 'new'
        }
        
        # Insert into database
        result = db.feedback.insert_one(feedback_doc)
        
        # Optional: Send email notification to admin
        try:
            admin_email = MAIL_USERNAME
            msg = Message(
                subject=f'New Feedback from {name}',
                sender=MAIL_USERNAME,
                recipients=[admin_email]
            )
            msg.body = f"""
New feedback received:

Name: {name}
Email: {email}
Phone: {phone or 'Not provided'}
Rating: {rating}/5
Category: {category}

Message:
{message}

Submitted at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
            """
            mail.send(msg)
        except Exception as email_error:
            print(f"Failed to send notification email: {email_error}")
        
        return jsonify({
            'message': 'Thank you for your feedback! We will get back to you soon.',
            'feedback_id': str(result.inserted_id),
            'rating': rating,  # Return the submitted rating for confirmation
        }), 201
        
    except Exception as e:
        print(f"Error submitting feedback: {str(e)}")
        return jsonify({'error': 'Failed to submit feedback. Please try again later.'}), 500

@auth_bp.route('/api/feedback', methods=['GET'])
def get_all_feedback():
    """Get all feedback (admin only)"""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'Missing token'}), 401
    
    try:
        token = auth_header.split(' ')[1]
        decoded = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        
        # Get all feedback
        feedback_list = list(db.feedback.find().sort('timestamp', -1))
        
        # Convert ObjectId and datetime to strings
        for feedback in feedback_list:
            feedback['_id'] = str(feedback['_id'])
            if 'timestamp' in feedback:
                feedback['timestamp'] = feedback['timestamp'].isoformat()
        
        return jsonify({'feedback': feedback_list}), 200
        
    except pyjwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        print(f"Error fetching feedback: {str(e)}")
        return jsonify({'error': 'Failed to fetch feedback'}), 500