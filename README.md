# 🐱🐶 Cat vs Dog Image Classification

A deep learning web application for classifying images of cats and dogs using MobileNetV2 transfer learning with a FastAPI backend.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)

**🌐 Live Demo**: [https://image-clasification-muhamad-iskandar.onrender.com](https://image-clasification-muhamad-iskandar.onrender.com)

---

## ✨ Features

- **Image Classification**: Upload an image to classify it as a cat or dog
- **Model Retraining**: Upload your own dataset to retrain the model
- **Prediction History**: Track all predictions with correction capability
- **📊 Presentation Materials**: Comprehensive case study presentation with visualizations
- **Modern UI**: Responsive web interface with real-time feedback
- **Two-Phase Training**: Feature extraction + fine-tuning for better accuracy
- **Auto-Generated Visualizations**: Training plots, confusion matrix, and metrics

---

## 🆕 New Features (Latest Update)

### 📊 Presentation Page
A dedicated presentation page for Computer Vision case study with 7 comprehensive sections:
- **Problem Statement**: Project overview and objectives
- **Dataset Analysis**: Dynamic statistics with class distribution
- **Model Architecture**: MobileNetV2 details and hyperparameters
- **Training Results**: Accuracy/Loss curves and final metrics
- **Model Evaluation**: Confusion matrix and performance metrics
- **Demo & Deployment**: Tech stack and API documentation
- **Limitations & Future Work**: Current constraints and improvement roadmap

**Access**: Click "📊 Presentation" tab in the main app or visit `/presentation`

### 🎨 Auto-Generated Training Materials
Training now automatically generates:
- **Training History Plot**: Accuracy and Loss curves (both phases)
- **Confusion Matrix**: Visual performance breakdown
- **Training Metrics JSON**: Precision, Recall, F1-Score, Accuracy
- **Dataset Info JSON**: Class distribution and split ratios

All materials saved to `static/presentation/` directory.

### 🔧 Enhanced Services

**Training Service** (`app/services/training.py`):
- ✅ Automatic visualization generation after training
- ✅ Combined history from 2-phase training
- ✅ Comprehensive metrics calculation (Precision, Recall, F1)
- ✅ Dataset statistics tracking
- ✅ Progress tracking with presentation material generation

**Prediction Service** (`app/services/prediction.py`):
- ✅ Aggregated statistics API
- ✅ Top/bottom confidence predictions tracking
- ✅ Average confidence calculation
- ✅ Class distribution analysis

---

## 🏗️ Architecture

```
├── app/
│   ├── api/          # FastAPI endpoints + presentation APIs
│   ├── core/         # Configuration & AI model
│   ├── models/       # Pydantic schemas
│   ├── repositories/ # Data persistence
│   └── services/     # Prediction & training logic (enhanced)
├── model/            # Trained model files
├── static/
│   ├── uploads/      # Uploaded images
│   └── presentation/ # Auto-generated materials (NEW)
├── templates/
│   ├── index.html    # Main application
│   └── presentation.html # Presentation page (NEW)
└── data/             # JSON data files
```

### Model Details
- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Input Size**: 160x160 pixels
- **Output**: Binary classification (Cat: 0, Dog: 1)
- **Training**: Two-phase (10 epochs frozen + 15 epochs fine-tuning)
- **Optimizer**: Adam (lr=1e-5 for fine-tuning)
- **Loss Function**: Binary Crossentropy

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

### Main Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/presentation` | Presentation page (NEW) |
| `POST` | `/api/predict` | Classify an image |
| `GET` | `/api/history` | Get prediction history |
| `DELETE` | `/api/history` | Clear all history |
| `POST` | `/api/train` | Start model training |
| `GET` | `/api/training-status` | Get training progress |
| `GET` | `/api/health` | Health check |

### Presentation Endpoints (NEW)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/presentation/materials` | List available materials |
| `GET` | `/api/presentation/metrics` | Get training metrics JSON |
| `GET` | `/api/presentation/dataset-info` | Get dataset information |
| `GET` | `/api/presentation/prediction-stats` | Get prediction statistics |

---

## 📊 Performance

After training with the provided dataset:
- **Validation Accuracy**: ~98-99%
- **Precision**: ~98%
- **Recall**: ~98%
- **F1-Score**: ~98%
- **Confidence Scores**: Typically >95% for clear images

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI + Uvicorn
- **ML Framework**: TensorFlow/Keras
- **Model**: MobileNetV2 (Transfer Learning)
- **Visualization**: Matplotlib, Seaborn (NEW)
- **Metrics**: scikit-learn (NEW)

### Frontend
- **UI**: HTML + CSS + JavaScript
- **Charts**: Dynamic data visualization
- **Design**: Responsive, modern glassmorphism style

### Storage
- **Database**: JSON file-based
- **Images**: Local file storage

---

## 📸 Screenshots

### Main Application
- **Classify Tab**: Upload and get instant predictions
- **History Tab**: Track all predictions with correction
- **Training Tab**: Retrain model with custom dataset

### Presentation Page (NEW)
- **7 Comprehensive Sections**: Problem → Dataset → Model → Training → Evaluation → Demo → Limitations
- **Auto-Populated Data**: Fetches from training materials
- **Professional Visuals**: Training plots, confusion matrix, metrics tables

---

## 🌐 Live Deployment

**Production URL**: [https://image-clasification-muhamad-iskandar.onrender.com](https://image-clasification-muhamad-iskandar.onrender.com)

**Note**: Training feature is disabled on Render (512MB RAM limitation). For local training with full presentation materials generation, run the application locally.

---

## 🎓 Academic Use

This project is ideal for:
- **Computer Vision case studies**
- **Transfer Learning demonstrations**
- **ML model deployment tutorials**
- **FastAPI backend examples**

The presentation page (`/presentation`) provides ready-made materials for:
- Class presentations
- Project reports
- Portfolio showcases
- Technical documentation

---

## 🔧 Development

### Project Structure
```bash
miniProject/
├── app/                    # Application code
│   ├── api/               # API routes
│   │   ├── endpoints.py   # Main + presentation endpoints
│   │   └── views.py       # HTML template routes
│   ├── core/              # Core modules
│   │   ├── ai_model.py    # Model singleton
│   │   └── config.py      # Configuration
│   ├── services/          # Business logic
│   │   ├── training.py    # Enhanced with viz generation
│   │   └── prediction.py  # Enhanced with statistics
│   └── models/            # Pydantic schemas
├── templates/             # HTML templates
│   ├── index.html         # Main app
│   └── presentation.html  # Presentation page (NEW)
├── static/                # Static files
│   ├── uploads/           # User uploads
│   └── presentation/      # Generated materials (NEW)
└── requirements.txt       # Dependencies
```

### Adding New Features
1. **Backend**: Add endpoints in `app/api/endpoints.py`
2. **Frontend**: Update `templates/index.html` or create new template
3. **Services**: Extend logic in `app/services/`
4. **Presentation**: Materials auto-generate after training

---

## 📝 License

MIT License

---

## 👥 Contributors

Developed as a mini project for Computer Vision course by **Muhamad Iskandar Zulkarnain**.

**Features Timeline**:
- ✅ Basic classification (v1.0)
- ✅ Model retraining (v1.1)
- ✅ Prediction history (v1.2)
- ✅ Presentation materials & auto-visualization (v2.0 - Latest)

---

## 🔗 Links

- **Live Application**: [https://image-clasification-muhamad-iskandar.onrender.com](https://image-clasification-muhamad-iskandar.onrender.com)
- **Presentation Page**: [/presentation](https://image-clasification-muhamad-iskandar.onrender.com/presentation)
- **GitHub Repository**: [https://github.com/XhaidarX00/image_clasification_Muhamad_Iskandar_Zulkarnain.git](https://github.com/XhaidarX00/image_clasification_Muhamad_Iskandar_Zulkarnain.git)

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the developer.

---

**Happy Classifying! 🐱🐶**
