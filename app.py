import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, render_template, request
import joblib
import pytesseract
from PIL import Image
import os

# 🔹 Path to Tesseract OCR (Windows default path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🔹 Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 🔹 Function to highlight suspicious words in news
def highlight_suspicious(text):
    suspicious_words = ["shocking", "fake", "miracle", "hoax", "unbelievable", "breaking"]
    found = [word for word in suspicious_words if word.lower() in text.lower()]
    return found if found else ["No suspicious words detected"]

# 🔹 Home Page
@app.route('/')
def home():
    return render_template('index.html')

# 🔹 Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form.get('news', '')

    # Check if an image was uploaded
    if 'news_image' in request.files and request.files['news_image'].filename != '':
        image = request.files['news_image']
        path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(path)

        # Extract text from the image
        extracted_text = pytesseract.image_to_string(Image.open(path))
        news_text += " " + extracted_text

    if not news_text.strip():
        return render_template('result.html', prediction="No input provided.", confidence="0%", words=[])

    # Transform text and predict
    vectorized_news = vectorizer.transform([news_text])
    prediction = model.predict(vectorized_news)[0]
    confidence = model.predict_proba(vectorized_news)[0][prediction] * 100

    result = "Real News ✅" if prediction == 1 else "Fake News ❌"
    suspicious_words = highlight_suspicious(news_text)

    return render_template('result.html',
                           prediction=result,
                           confidence=f"{confidence:.2f}%",
                           words=suspicious_words,
                           extracted_text=news_text)

# 🔹 Run Flask App
if __name__ == "_main_":
    app.run(debug=True)

