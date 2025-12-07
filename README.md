# Sentiment Prediction Web Application

A Flask-based web application for analyzing text sentiment using machine learning. The application provides a user-friendly interface to predict whether text has positive, negative, or neutral sentiment.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green.svg)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)

## Features

- **User Authentication**: Secure login and registration system with SQLite database
- **Sentiment Analysis**: Real-time text sentiment prediction (Positive, Negative, Neutral)
- **Confidence Scores**: Displays prediction confidence with visual indicators
- **Modern UI**: Clean, responsive design with Bootstrap 5 and custom CSS
- **Text Preprocessing**: Advanced NLP preprocessing using NLTK
  - Tokenization
  - Lemmatization
  - Stopword removal
  - URL and special character cleaning

## Tech Stack

- **Backend**: Flask (Python)
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **ML/NLP**: 
  - Scikit-learn (TF-IDF Vectorizer, ML Model)
  - NLTK (Natural Language Toolkit)
- **Icons**: Font Awesome 6

## Project Structure

```
├── app.py                 # Main Flask application
├── best_sentiment_model.pkl    # Trained ML model
├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
├── users.db               # SQLite database (auto-generated)
├── templates/
│   ├── base.html          # Base template with navbar and styling
│   ├── login.html         # User login page
│   ├── register.html      # User registration page
│   ├── dashboard.html     # Main sentiment analysis dashboard
│   ├── about.html         # About page
│   └── history.html       # Analysis history page
└── README.md
```

## Installation

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Setup

1. **Clone or download the project**
   ```bash
   cd "Nisha_Sentiment Prediction WebApp 2"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install flask nltk scikit-learn numpy
   ```

4. **Download NLTK data** (automatically done on first run, or manually):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt_tab')
   ```

5. **Ensure model files are present**
   - `best_sentiment_model.pkl` - Trained sentiment classification model
   - `tfidf_vectorizer.pkl` - TF-IDF vectorizer for text transformation

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   
   Open your browser and navigate to: `http://localhost:5002`

3. **Default login credentials**
   | Username | Password | Name |
   |----------|----------|------|
   | admin | admin123 | Administrator |
   | nisha | nisha123 | Nisha |
   | user | user123 | User |

4. **Analyze sentiment**
   - Login with your credentials
   - Enter text in the dashboard textarea
   - Click "Analyze Sentiment"
   - View the prediction result with confidence score

## API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Redirects to login or dashboard |
| `/login` | GET, POST | User login page |
| `/register` | GET, POST | User registration page |
| `/logout` | GET | Logout and clear session |
| `/dashboard` | GET | Main analysis dashboard (protected) |
| `/predict` | POST | Sentiment prediction API (JSON) |
| `/about` | GET | About page (protected) |

### Prediction API Example

**Request:**
```json
POST /predict
Content-Type: application/json

{
  "text": "I love this product! It's amazing!"
}
```

**Response:**
```json
{
  "success": true,
  "sentiment": "Positive",
  "confidence": 0.92,
  "original_text": "I love this product! It's amazing!"
}
```

## Configuration

The application can be configured by modifying the following in `app.py`:

- **Secret Key**: `app.secret_key` - Change for production
- **Port**: Default is `5002`, modify in `app.run()`
- **Debug Mode**: Set `debug=False` for production
- **Host**: Default is `0.0.0.0` (accessible on network)

## Screenshots

### Login Page
Clean and modern login interface with gradient styling.

### Dashboard
Interactive sentiment analysis with real-time results and confidence visualization.

### About Page
Information about the application, technologies used, and features.

## Development

To run in development mode with auto-reload:
```bash
FLASK_ENV=development python app.py
```

## Security Notes

⚠️ **For Production Use:**
- Change the default `secret_key`
- Use password hashing (e.g., bcrypt, werkzeug.security)
- Enable HTTPS
- Use a production WSGI server (Gunicorn, uWSGI)
- Consider using environment variables for sensitive data

## License

This project is for educational purposes.

## Author

Nisha

---

*Built with ❤️ using Flask and Machine Learning*
