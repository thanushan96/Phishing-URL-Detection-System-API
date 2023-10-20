from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from database import db, User
import re  # Import the regex module

auth_bp = Blueprint("auth", __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        # Check for duplicate username
        existing_username = User.query.filter_by(username=username).first()
        if existing_username:
            return jsonify(message='Username already taken. Please choose a different username.'), 400

        # Check for duplicate email
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify(message='Email already in use. Please use a different email address.'), 400

        # Validation using regex for email
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify(message='Invalid email address.'), 400

     


        # Your password complexity checks
        error_messages = []
        
        if not re.search(r"(?=.*[a-z])", password):
            error_messages.append('one lowercase letter')
        
        if not re.search(r"(?=.*[A-Z])", password):
            error_messages.append('uppercase letter')
        
        if not re.search(r"(?=.*\d)", password):
            error_messages.append('one digit')
        
        if not re.search(r"(?=.*\W)", password):
            error_messages.append('one special symbol')
        
        if len(password) < 8:
            error_messages.append('8 characters')
        
        if error_messages:
            error_messages = ['Password should contain at least ' + ' & '.join(error_messages)]
            return jsonify(message=' '.join(error_messages)), 400



        # If all checks pass, proceed with registration
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return jsonify(message='Registration successful. Please log in.')

@auth_bp.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            return jsonify(message='Login successful!')
        else:
            return jsonify(message='Login failed. Please check your credentials and try again.'), 401

@auth_bp.route('/logout', methods=['POST'])
def logout():
    return jsonify(message='Logout successful!')



