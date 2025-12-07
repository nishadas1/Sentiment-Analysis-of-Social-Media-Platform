"""
Sentiment Prediction Web Application
Flask application with login functionality and sentiment analysis
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pickle
import re
import os
import sqlite3
from functools import wraps

# Try to import NLTK components
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Download NLTK data with timeout handling
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"⚠ NLTK download failed (might already be installed): {e}")
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠ NLTK not available, using basic text processing")

app = Flask(__name__)
app.secret_key = 'sentiment_analysis_secret_key_2025'

# Get the current directory (where the app.py is located)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(APP_DIR, 'users.db')

# Database helper functions
def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with users table"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT
        )
    ''')
    # Add default users if table is empty
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        default_users = [
            ('admin', 'admin123', 'Administrator', 'admin@example.com'),
            ('nisha', 'nisha123', 'Nisha', 'nisha@example.com'),
            ('user', 'user123', 'User', 'user@example.com')
        ]
        cursor.executemany('INSERT INTO users (username, password, name, email) VALUES (?, ?, ?, ?)', default_users)
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Load the trained model and vectorizer
try:
    model_path = os.path.join(APP_DIR, 'best_sentiment_model.pkl')
    vectorizer_path = os.path.join(APP_DIR, 'tfidf_vectorizer.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✓ Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"⚠ Model files not found: {e}")
    print(f"  Looking in: {APP_DIR}")
    model = None
    vectorizer = None

# Initialize NLP components
if NLTK_AVAILABLE:
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
    except Exception as e:
        print(f"⚠ Error initializing NLTK components: {e}")
        NLTK_AVAILABLE = False
        lemmatizer = None
        stop_words = set()
else:
    lemmatizer = None
    stop_words = set()

def clean_text(text):
    """Clean and preprocess text data"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def preprocess_text(text):
    """Advanced text preprocessing with tokenization and lemmatization"""
    # Clean text
    text = clean_text(text)
    
    if NLTK_AVAILABLE and lemmatizer:
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens 
                  if token not in stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    else:
        # Basic tokenization if NLTK is not available
        tokens = text.split()
        tokens = [token for token in tokens if len(token) > 2]
        return ' '.join(tokens)

def login_required(f):
    """Decorator to require login for certain routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    """Home page - redirect to login or dashboard"""
    if 'logged_in' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['logged_in'] = True
            session['username'] = user['username']
            session['name'] = user['name']
            session['email'] = user['email'] or ''
            flash(f'Welcome back, {user["name"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
        elif len(password) < 4:
            flash('Password must be at least 4 characters.', 'error')
        elif not username or not name:
            flash('Please fill in all fields.', 'error')
        else:
            conn = get_db()
            cursor = conn.cursor()
            try:
                cursor.execute('INSERT INTO users (username, password, name, email) VALUES (?, ?, ?, ?)',
                             (username, password, name, email))
                conn.commit()
                flash('Registration successful! Please login.', 'success')
                conn.close()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username already exists. Please choose another.', 'error')
            conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html', name=session.get('name', 'User'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Predict sentiment for given text (AJAX endpoint)"""
    if model is None or vectorizer is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Please check server configuration.'})
    
    # Get JSON data
    data = request.get_json()
    text = data.get('text', '').strip() if data else ''
    
    if not text:
        return jsonify({'success': False, 'error': 'Please enter some text to analyze.'})
    
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text:
        return jsonify({'success': False, 'error': 'Text could not be processed. Please enter valid text.'})
    
    # Transform and predict
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    
    # Get probability scores if available
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    
    try:
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = float(max(probabilities))
    except:
        # For models without predict_proba (like LinearSVC)
        decision_scores = model.decision_function(text_tfidf)[0]
        # Convert decision scores to pseudo-probabilities using softmax
        import numpy as np
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        confidence = float(np.max(exp_scores / exp_scores.sum()))
    
    sentiment = sentiment_labels[prediction]
    
    return jsonify({
        'success': True,
        'sentiment': sentiment,
        'confidence': confidence,
        'original_text': text
    })

@app.route('/about')
@login_required
def about():
    """About page"""
    return render_template('about.html', name=session.get('name', 'User'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
