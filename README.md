# ❤️ Heart Disease Prediction Web Service

A complete machine learning web application for predicting heart disease risk based on patient health parameters. This project includes both a **Streamlit app** (recommended for quick deployment) and a **Flask app** with custom HTML/CSS/JS frontend.

## 🎯 Features

- **Random Forest Model**: Classifier trained on 13 patient health parameters, with SMOTE for class imbalance
- **Modern FastAPI Backend**: High performance, automatic API documentation, type validation
- **User-Friendly Interface**: Beautiful, responsive UI for easy data entry
- **Real-time Predictions**: Instant heart disease risk assessment
- **Interactive API Docs**: Built-in Swagger UI and ReDoc documentation
- **Probability Scores**: Shows confidence levels for predictions
- **Multiple Deployment Options**: Choose between FastAPI (recommended), Streamlit, or Flask
- **Free Deployment**: Deploy on Render, Railway, or Streamlit Cloud for free

## 📊 Model Details

- **Algorithm**: Random Forest Classifier with SMOTE for handling class imbalance
- **Features**: 13 patient health parameters
- **Output**: Predicted class with probability score

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- The trained model file (`heart_model.pkl`)

### Local Installation

1. **Clone or navigate to the project directory**:
   ```bash
   git clone https://github.com/ade-mic/heart-disease-classifier.git
   cd heart-disease-classifier
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the model file is present**:
   - Make sure `heart_model.pkl` is in the root directory
   - If not, run the notebook to generate it or copy it from Google Drive

### Run with FastAPI (Recommended ⭐)

```bash
cd fastapi_app
uvicorn app:app --reload
```

**Benefits:**
- 🚀 High performance (async support)
- 📚 Automatic interactive API docs at `http://localhost:8000/docs`
- ✅ Automatic request validation with Pydantic
- 📖 Alternative docs at `http://localhost:8000/redoc`
- 🔧 Type hints and better IDE support

### Run with Streamlit (Easiest for Non-Technical Users)

```bash
streamlit run app.py
```

- Simple deployment to Streamlit Cloud
- No frontend coding needed
- Opens at `http://localhost:8501`

### Run with Flask (Traditional Approach)

```bash
cd flask_app
python app.py
```

- Traditional REST API
- Opens at `http://localhost:5000`

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (FREE & Easiest)

1. **Prepare your repository**:
   ```bash
   # Initialize git if not already done
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Push to GitHub**:
   ```bash
   # Create a new repository on GitHub first, then:
   git remote add origin https://github.com/yourusername/heart-disease-predictor.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"!

**Note**: Make sure `heart_model.pkl` is committed to your repository (it should be < 100MB for GitHub).

### Option 2: Render (FREE)

1. **Create `render.yaml`** (already included):
   ```yaml
   services:
     - type: web
       name: heart-disease-predictor
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

2. **Deploy**:
   - Push code to GitHub
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect settings
   - Click "Create Web Service"

### Option 3: Railway (FREE)

1. **Deploy**:
   - Go to [railway.app](https://railway.app)
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Connect your repository
   - Railway will auto-deploy!

## 🔧 Flask App (Alternative)

If you prefer a traditional Flask backend with custom frontend:

### Run Locally

```bash
# Make sure you're in the virtual environment
cd flask_app
python app.py
```

Open `http://localhost:5000` in your browser.

### Deploy Flask App

For **Render** or **Railway**, use the `Procfile`:
```
web: gunicorn flask_app.app:app
```

The platform will automatically detect and use this file.

## 📁 Project Structure

```
heart_disease_web/
├── app.py                          # Streamlit application
├── heart_model.pkl                 # Trained ML model
├── requirements.txt                # Python dependencies
├── Procfile                        # For deployment (FastAPI)
├── runtime.txt                     # Python version specification
├── .gitignore                     # Git ignore file
├── README.md                       # This file
│
├── fastapi_app/                    # FastAPI (Recommended ⭐)
│   ├── app.py                      # FastAPI backend with auto docs
│   ├── templates/
│   │   └── index.html             # Frontend HTML
│   └── static/
│       ├── css/
│       │   └── style.css          # Styling
│       └── js/
│           └── script.js          # Frontend logic
│
├── flask_app/                      # Flask alternative
│   ├── app.py                      # Flask backend
│   ├── templates/
│   │   └── index.html             # Frontend HTML
│   └── static/
│       ├── css/
│       │   └── style.css          # Styling
│       └── js/
│           └── script.js          # Frontend logic
│
└── Group_Assignment__1.ipynb      # Model training notebook
```

## 💡 Input Features

The application accepts the following patient parameters:

1. **Age**: Patient's age in years
2. **Sex**: Male (1) or Female (0)
3. **Chest Pain Type**: 0-3 (Typical Angina, Atypical Angina, Non-anginal, Asymptomatic)
4. **Resting Blood Pressure**: In mm Hg
5. **Serum Cholesterol**: In mg/dl
6. **Fasting Blood Sugar**: > 120 mg/dl (Yes/No)
7. **Resting ECG Results**: 0-2 (Normal, ST-T abnormality, LV hypertrophy)
8. **Maximum Heart Rate**: Achieved during exercise
9. **Exercise Induced Angina**: Yes (1) or No (0)
10. **ST Depression**: Induced by exercise relative to rest
11. **Slope**: Of peak exercise ST segment (0-2)
12. **Number of Major Vessels**: Colored by fluoroscopy (0-3)
13. **Thal**: Thalium stress test result (0-3)

## ⚠️ Disclaimer

This application is for **educational purposes only** and should NOT replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## 🛠️ Troubleshooting

### Model File Not Found
If you see "Model file not found":
1. Make sure `heart_model.pkl` is in the root directory
2. If missing, run the Jupyter notebook to generate it
3. Or copy it from your Google Drive location

### Dependencies Issues
If you encounter package conflicts:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Port Already in Use
If port 8501 (Streamlit) or 5000 (Flask) is busy:
```bash
# For Streamlit
streamlit run app.py --server.port 8502

# For Flask
PORT=5001 python flask_app/app.py
```

## 📝 API Endpoints (FastAPI)

FastAPI provides **automatic interactive documentation** at `/docs` and `/redoc`!

### GET `/`
Home page with prediction form

### GET `/health`
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Model loaded and ready"
}
```

### POST `/predict`
Make a prediction (automatically validated by Pydantic)
```json
{
  "age": 50,
  "sex": 1,
  "cp": 0,
  "trestbps": 120,
  "chol": 200,
  "fbs": 0,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 0.0,
  "slope": 0,
  "ca": 0,
  "thal": 0
}
```

Response:
```json
{
  "success": true,
  "prediction": 0,
  "prediction_label": "No Heart Disease",
  "probabilities": {
    "no_disease": 95.5,
    "disease": 4.5
  },
  "risk_level": "LOW",
  "confidence": 95.5
}
```

### GET `/model-info`
Get model information
```json
{
  "success": true,
  "model_type": "Random Forest Classifier",
  "features": [...],
  "performance": {...},
  "description": "Random Forest model trained with SMOTE"
}
```

### GET `/docs`
Interactive Swagger UI documentation (try it out live!)

### GET `/redoc`
Beautiful ReDoc alternative documentation

## 🎯 Why FastAPI?

✅ **Performance**: 2-3x faster than Flask  
✅ **Auto Docs**: Interactive API documentation out of the box  
✅ **Validation**: Automatic request/response validation with Pydantic  
✅ **Type Safety**: Full type hints support  
✅ **Async**: Native async/await support  
✅ **Modern**: Built on modern Python standards  
✅ **Production Ready**: Used by companies like Uber, Netflix, Microsoft

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created as part of Machine Learning coursework.

---

**Happy Predicting! ❤️**
