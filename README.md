# 🔍 Fake News Classifier - Streamlit App

A machine learning-powered web application to detect fake news using Logistic Regression and TF-IDF vectorization.

## 📋 Requirements

- Python 3.8+
- Required packages in `requirements.txt`

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Model Files
Run the Jupyter notebook (`appfakenews.ipynb`) to:
- Train the Logistic Regression model
- Create `vectorizer.jb` file
- Create `lr_model.jb` file

Or run this in Python:
```python
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained models
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

## Fonts (offline-safe)

The header image uses a TrueType font. To avoid any internet dependency, the app now looks for fonts in the repository first:

- Preferred (bundled):
   - assets/fonts/DejaVuSans.ttf
   - assets/fonts/DejaVuSans-Bold.ttf

If these files are not present, the app will try common system locations (e.g., Arial on Windows, DejaVu on Linux). As a last resort, it falls back to the default PIL font, so the app still runs.

You can place DejaVu fonts in the paths above (they are freely available) or rely on your system fonts.

The app will open at `http://localhost:8501` in your browser.

## 📁 File Structure
```
fake news classifier/
├── appfakenews.ipynb          # Main training notebook
├── app.py                      # Streamlit web app
├── requirements.txt            # Python dependencies
├── vectorizer.jb              # Saved TF-IDF vectorizer
├── lr_model.jb                # Saved Logistic Regression model
├── Fake.csv                   # Fake news dataset
└── True.csv                   # Real news dataset
```

## 🎯 Features

✅ **Real-time Prediction** - Instantly classify news articles  
✅ **Confidence Scoring** - Get probability scores for predictions  
✅ **Text Preprocessing** - Automatic cleaning and normalization  
✅ **Lemmatization** - Convert words to base forms  
✅ **Stop Word Removal** - Remove common English words  
✅ **Probability Distribution** - See confidence for both classes  
✅ **Interactive UI** - User-friendly tabs and controls  

## 🔧 How It Works

1. **Input:** User enters a news article or text
2. **Preprocessing:**
   - Remove punctuation
   - Remove stop words
   - Lemmatization
3. **Vectorization:** Convert text to TF-IDF features
4. **Classification:** Logistic Regression predicts Real/Fake
5. **Output:** Display prediction with confidence score

## 📊 Model Performance

- **Algorithm:** Logistic Regression
- **Vectorizer:** TF-IDF
- **Accuracy:** ~95%
- **Dataset:** 45,000+ news articles

## 🎨 App Sections

### 🏠 Predict Tab
- Input text manually or use example
- Get instant prediction
- View confidence scores
- See processed text

### 📊 About Tab
- Model information
- Feature list
- How detection works

### ⚙️ Settings Tab
- Model file info
- System information

## 💾 Model Files

Make sure these files exist in the same directory as `app.py`:
- `vectorizer.jb` - TF-IDF Vectorizer (required)
- `lr_model.jb` - Logistic Regression Model (required)

## ⚠️ Troubleshooting

**Error: "FileNotFoundError: vectorizer.jb not found"**
- Run the notebook first to generate model files
- Ensure files are in the same directory as `app.py`

**Error: "NLTK data not found"**
- The app auto-downloads required NLTK data on first run

**Slow predictions**
- First load will cache models (~2 seconds)
- Subsequent predictions are instant

## 📝 Example Usage

1. Launch the app: `streamlit run app.py`
2. Go to "Predict" tab
3. Paste a news article
4. Click "Predict" button
5. View results!

## 🔗 Dataset Source

[Kaggle Fake & Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## 👨‍💻 Author

Machine Learning Project - Fake News Classification

## 📄 License

Open source - Feel free to use and modify!
# Fake-News-Detection
