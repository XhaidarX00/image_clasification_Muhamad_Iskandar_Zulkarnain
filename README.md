# 🐱🐶 Cat vs Dog Image Classification

A deep learning web application for classifying images of cats and dogs using MobileNetV2 transfer learning with a FastAPI backend.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)

---

## ✨ Features

- **Image Classification**: Upload an image to classify it as a cat or dog
- **Model Retraining**: Upload your own dataset to retrain the model
- **Prediction History**: Track all predictions with correction capability
- **Modern UI**: Responsive web interface with real-time feedback
- **Two-Phase Training**: Feature extraction + fine-tuning for better accuracy

---

## 🏗️ Architecture

```
├── app/
│   ├── api/          # FastAPI endpoints
│   ├── core/         # Configuration & AI model
│   ├── models/       # Pydantic schemas
│   ├── repositories/ # Data persistence
│   └── services/     # Prediction & training logic
├── model/            # Trained model files
├── static/uploads/   # Uploaded images
├── templates/        # HTML templates
└── data/             # JSON data files
```

### Model Details
- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Input Size**: 160x160 pixels
- **Output**: Binary classification (Cat: 0, Dog: 1)
- **Training**: Two-phase (10 epochs frozen + 15 epochs fine-tuning)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone repository
git clone <repository-url>
cd miniProject

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
python -m app.main
```

Open http://localhost:8000 in your browser.

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker

```bash
# Build image
docker build -t cat-dog-classifier .

# Run container
docker run -d -p 8000:8000 --name cat-dog-classifier cat-dog-classifier
```

---

## 📁 Dataset Format

For training, upload a ZIP file with this structure:

```
dataset.zip
├── cats/
│   ├── cat.001.jpg
│   ├── cat.002.jpg
│   └── ...
└── dogs/
    ├── dog.001.jpg
    ├── dog.002.jpg
    └── ...
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `POST` | `/api/predict` | Classify an image |
| `GET` | `/api/history` | Get prediction history |
| `DELETE` | `/api/history` | Clear all history |
| `POST` | `/api/train` | Start model training |
| `GET` | `/api/training-status` | Get training progress |
| `GET` | `/api/health` | Health check |

---

## 📊 Performance

After training with the provided dataset:
- **Validation Accuracy**: ~98-99%
- **Confidence Scores**: Typically >95% for clear images

---

## 🛠️ Tech Stack

- **Backend**: FastAPI + Uvicorn
- **ML Framework**: TensorFlow/Keras
- **Model**: MobileNetV2 (Transfer Learning)
- **Frontend**: HTML + CSS + JavaScript
- **Storage**: JSON file-based

---

## 📝 License

MIT License

---

## 👥 Contributors

Developed as a mini project for Computer Vision course.
