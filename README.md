# Fruit Grading System

An automated fruit quality classification system using deep learning and multi-angle imaging. The system captures fruit images from multiple camera angles, extracts features using a ShuffleNetV2 CNN, and classifies fruit into **Premium**, **Standard**, or **Market** grades via a trained fully-connected classifier.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [ML Pipeline](#ml-pipeline)
- [Authentication & Roles](#authentication--roles)
- [Running Tests](#running-tests)

---

## Overview

The Fruit Grading System provides an end-to-end solution for automated fruit quality inspection on a production line. It supports:

- **Multi-angle image capture** — 4 cameras (Front, Right, Back, Left) for comprehensive coverage
- **Deep learning classification** — ShuffleNetV2 feature extraction with a 2-layer fully connected classifier
- **Real-time pipeline monitoring** — live logs and step-by-step progress tracking
- **Role-based dashboards** — separate views for administrators and operators
- **Historical analytics** — per-class metrics, confusion matrices, and training history
- **CSV export** — classification results exportable for downstream analysis

---

## Architecture

```
┌─────────────────────────────────────────┐
│            React Frontend               │
│  (Vite · React Router · Axios)          │
│  Port 3000                              │
└────────────────┬────────────────────────┘
                 │ REST API
┌────────────────▼────────────────────────┐
│           Flask Backend                 │
│  (Flask · Flask-CORS)                   │
│  Port 5000                              │
│                                         │
│  ┌─────────────┐   ┌─────────────────┐  │
│  │ ML Pipeline │   │  API Routes     │  │
│  │ ShuffleNetV2│   │  user/admin/    │  │
│  │ FC Classifier│  │  pipeline/      │  │
│  │ PCA · CLAHE │   │  results/...    │  │
│  └─────────────┘   └─────────────────┘  │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│              MongoDB                    │
│  images · dashboard_metadata            │
└─────────────────────────────────────────┘
```

---

## Tech Stack

### Backend
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Runtime |
| Flask | 3.1.0 | Web framework |
| Flask-CORS | 5.0.0 | Cross-origin requests |
| PyTorch | 2.5.1 | Neural network training |
| TorchVision | 0.20.1 | ShuffleNetV2 pre-trained model |
| Scikit-learn | latest | PCA, evaluation metrics |
| OpenCV | 4.10.0 | Image preprocessing |
| Pillow | 11.0.0 | Image I/O |
| pymongo | 4.10.1 | MongoDB driver |
| Matplotlib / Seaborn | 3.9.2 / 0.13.2 | Confusion matrix visualization |
| pytest / pytest-cov | 8.3.4 / 6.0.0 | Testing |

### Frontend
| Technology | Version | Purpose |
|---|---|---|
| React | 18.2.0 | UI framework |
| Vite | 7.3.1 | Build tool & dev server |
| React Router DOM | 6.20.0 | Client-side routing |
| Axios | 1.13.6 | HTTP client |
| React Icons | 4.12.0 | Icon library |

---

## Project Structure

```
final_project/
├── Backend/
│   ├── app.py                        # Flask application entry point
│   ├── db_config.py                  # MongoDB connection & configuration
│   ├── cnn/
│   │   ├── pre_trained_feature_map.py    # ShuffleNetV2 feature extractor
│   │   ├── fine_tune_classifier.py       # Fine-tuned ShuffleNetV2
│   │   ├── fully_connected_layer.py      # 2-layer FC classifier
│   │   ├── feature_map_insertion.py      # Feature storage utilities
│   │   └── activation_functions.py       # ReLU, softmax implementations
│   ├── preprocessing/
│   │   ├── preprocessing_from_db.py      # Load & preprocess from MongoDB
│   │   └── preprocessing_insertion.py    # Data augmentation pipeline
│   ├── processes/
│   │   ├── build_model.py                # 6-step training orchestration
│   │   └── data_insertion.py             # New fruit image processing
│   ├── routes/
│   │   ├── user_dashboard.py             # /api/user/...
│   │   ├── admin_dashboard.py            # /api/admin/...
│   │   ├── camera_monitor.py             # /api/cameras/...
│   │   ├── processing.py                 # /api/pipeline/...
│   │   ├── results.py                    # /api/results/...
│   │   ├── settings.py                   # /api/settings
│   │   └── add_fruit.py                  # /api/fruit/...
│   ├── utils/
│   │   ├── utils.py                      # DB helpers, error handling
│   │   ├── model_metadata.py             # Dashboard metadata persistence
│   │   └── shared_state.py               # Pipeline state management
│   ├── visuals/
│   │   └── confusion_matrix.py           # Confusion matrix generation
│   ├── Streamers/
│   │   ├── database_creation.py          # MongoDB schema setup
│   │   └── database_insertion.py         # Batch data insertion
│   ├── Tests/                            # Backend test suite
│   ├── saved_models/                     # Trained model weights & metadata
│   └── utils/requirements.txt           # Python dependencies
│
├── fruit-grading-ui/
│   ├── src/
│   │   ├── App.jsx                       # Root component & routing
│   │   ├── index.jsx                     # App entry point
│   │   ├── pages/
│   │   │   ├── Login.jsx
│   │   │   ├── Dashboard.jsx             # Admin system dashboard
│   │   │   ├── UserDashboard.jsx         # Operator daily stats
│   │   │   ├── CameraMonitor.jsx         # Camera status view
│   │   │   ├── Processing.jsx            # Pipeline control & logs
│   │   │   ├── Results.jsx               # Classification history
│   │   │   ├── Settings.jsx              # System configuration
│   │   │   └── AddFruit.jsx              # Upload & classify fruit
│   │   ├── components/
│   │   │   ├── Sidebar.jsx               # Navigation menu
│   │   │   └── ProtectedRoute.jsx        # Role-based access guard
│   │   ├── context/
│   │   │   └── AuthContext.jsx           # Auth state management
│   │   └── utils/                        # Per-feature API clients
│   ├── package.json
│   ├── vite.config.js
│   └── Tests/                            # Frontend test suite
│
├── env_config.py                         # Environment configuration helpers
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python** 3.8 or higher
- **Node.js** 16 or higher with npm
- **MongoDB** (local or remote instance)
- **CUDA-compatible GPU** (optional but recommended for training)

---

### Backend Setup

```bash
cd Backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r utils/requirements.txt

# Create a .env file (see Configuration section below)
cp .env.example .env
# Edit .env with your values

# Run the server
python app.py
```

**Optional startup flags:**

| Flag | Description |
|---|---|
| `--skip-tests` | Skip the pre-startup test suite |
| `--critical-test` | Run only critical tests |
| `--port <port>` | Override default port (5000) |
| `--host <host>` | Override default host (127.0.0.1) |
| `--no-debug` | Disable Flask debug mode |

---

### Frontend Setup

```bash
cd fruit-grading-ui

# Install dependencies
npm install

# Start development server (http://localhost:3000)
npm start

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## Configuration

Create a `.env` file in the `Backend/` directory with the following variables:

```env
# MongoDB
MONGO_CONNECTION_STRING=mongodb://localhost:27017/
DB_NAME=fruit_grading

# Dataset paths
ORIGINAL_DATASET_PATH=/path/to/original/fruit/images
PROCESSED_DATASET_PATH=/path/to/processed/images
STORED_DATASET_PATH=/path/to/stored/images

# Model
MODEL_DIR=saved_models

# Camera system
NUM_OF_CAMERAS=4
CAMERA_FPS=30

# Training
BATCH_SIZE=128
```

The dataset directory should contain fruit images organized by grade:

```
original_dataset/
├── premium/
├── standard/
└── market/
```

---

## API Reference

All backend routes are prefixed with the Flask server base URL (default: `http://localhost:5000`).

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| POST | `/login` | Authenticate user, returns role & session token |

### User Dashboard (`/api/user`)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/user/dashboard-stats` | Daily fruit count by grade |
| GET | `/api/user/recent-results` | Last 5 classification results |
| GET | `/api/user/model-info` | Model accuracy and training metadata |

### Admin Dashboard (`/api/admin`)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/admin/system-status` | Database, model, pipeline, camera health |
| GET | `/api/admin/processing-stats` | Total processed, accuracy, last update |
| GET | `/api/admin/dataset-info` | Training/testing counts, image dimensions |
| GET | `/api/admin/model-performance` | Architecture, train/test accuracy, class count |
| GET | `/api/admin/per-class-performance` | Precision, recall, F1 per grade |
| GET | `/api/admin/training-history` | Loss and accuracy curves |
| GET | `/api/admin/confusion-matrix` | Full confusion matrix data |
| GET | `/api/admin/full-dashboard-data` | All admin data in one response |

### ML Pipeline (`/api/pipeline`)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/pipeline/start` | Trigger the 6-step training pipeline |
| GET | `/api/pipeline/status` | Monitor pipeline step progress |
| GET | `/api/pipeline/logs` | Stream real-time training logs |

### Classification Results (`/api/results`)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/results/list` | Paginated, filtered classification history |
| GET | `/api/results/export` | Export results as CSV |

### Fruit Classification (`/api/fruit`)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/fruit/upload-process` | Upload multi-angle images and receive grade |

### Camera Monitor (`/api/cameras`)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/cameras/status` | Status of all cameras |
| GET | `/api/cameras/<camera_id>` | Details for a specific camera |

### Settings (`/api/settings`)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/settings` | Retrieve current system configuration |
| PUT | `/api/settings` | Update system configuration |
| POST | `/api/settings/test-database` | Validate MongoDB connection |

---

## ML Pipeline

The training pipeline is a 6-step sequential process triggered via the `/api/pipeline/start` endpoint:

| Step | Name | Description |
|---|---|---|
| 1 | Run Tests | Validate system integrity before training |
| 2 | Setup Database | Load and organize fruit images in MongoDB |
| 3 | Preprocess Data | Apply Gaussian blur + CLAHE histogram equalization, resize to 224×224 |
| 4 | Extract Features | ShuffleNetV2 feature extraction from all 4 camera angles |
| 5 | Train Classifier | Train 2-layer FC network with ReLU, dropout, and L2 regularization |
| 6 | Generate Confusion Matrix | Evaluate performance and persist metrics |

### Model Architecture

```
Input Images (4 angles × 224×224×3)
        ↓
ShuffleNetV2 (pre-trained ImageNet)  — Feature Extraction
        ↓
PCA Dimensionality Reduction
        ↓
Fully Connected Layer 1 (ReLU + Dropout)
        ↓
Fully Connected Layer 2
        ↓
Softmax → Grade: Premium / Standard / Market
```

### Image Preprocessing

- Gaussian blur (3×3 kernel) for noise reduction
- CLAHE histogram equalization for contrast enhancement
- Resize to **224×224** pixels
- Data augmentation (rotations, flips) during training

---

## Authentication & Roles

Authentication is session-based using `localStorage`. Sessions expire after **5 hours**.

| Role | Username | Password | Access |
|---|---|---|---|
| Admin | `admin` | `admin123` | All pages including pipeline control, settings, camera monitor |
| User/Operator | `user` | `user123` | User dashboard, results, add fruit |

> **Note:** Default credentials are for development. Change them before deploying to production.

**Admin-only pages:**
- System Dashboard
- Camera Monitor
- Processing (pipeline control)
- Settings

**Shared pages (admin + user):**
- User Dashboard
- Results
- Add Fruit

---

## Running Tests

### Backend

```bash
cd Backend

# Run full test suite
pytest Tests/

# Run with coverage report
pytest Tests/ --cov=. --cov-report=html
```

### Frontend

```bash
cd fruit-grading-ui

# Run tests
npm test
```

The backend also runs a 6-phase automated test suite on startup (unless `--skip-tests` is passed). Use `--critical-test` to run only the essential checks.
