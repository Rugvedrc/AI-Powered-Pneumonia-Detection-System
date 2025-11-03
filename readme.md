# 🫁 AI-Powered Pneumonia Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.33%25-brightgreen)](https://github.com/Rugvedrc/AI-Powered-Pneumonia-Detection-System)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-99.42%25-success)](https://github.com/Rugvedrc/AI-Powered-Pneumonia-Detection-System)

> **Advanced uncertainty-aware attention mechanism achieving 96.33% accuracy in pneumonia detection from chest X-rays**

**📄 Paper Status:** Under review at **IEEE CICT 2025**  
**Title:** *Uncertainty-Aware Attention Mechanism for Pneumonia Detection Using Chest X-Rays*  
**Authors:** Archana W. Bhade, Smita M. Chavan, Rugved Rajesh Chandekar, Dnyanesh Milind Deshmukh, Aditya Keshav Magar

---

## 🚀 Key Highlights

- **🏅 96.33% Accuracy** with 99.42% ROC-AUC on chest X-ray classification
- **🧠 Attention-Augmented DenseNet121** with spatial and channel attention mechanisms
- **📊 Uncertainty Quantification** via Monte Carlo Dropout for confident predictions
- **🔍 Explainable AI** using Grad-CAM visualizations showing model focus areas
- **💡 Ready-to-Use Model** available for immediate deployment and testing

## 📋 Table of Contents

- [Demo](#-demo)
- [Performance Metrics](#-performance-metrics)
- [Installation](#️-installation)
- [Quick Start](#-quick-start)
- [Model Architecture](#-model-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Research Paper](#-research-paper)
- [License](#-license)

## 🎬 Demo

### Live Demo
Try the interactive demo: **[🌐 Pneumonia Detection App](https://huggingface.co/spaces/rugvedrc/ai-powered-pneumonia-detection-system)**

### Key Features
- Upload chest X-rays for instant analysis
- Real-time Grad-CAM visualization highlighting attention regions
- Confidence scoring with uncertainty estimates
- Batch processing support

## 📊 Performance Metrics

| Metric | Value |
|--------|--------|
| **Accuracy** | 96.33% |
| **Sensitivity (Recall)** | 96.26% |
| **Specificity** | 96.53% |
| **F1-Score** | 97.45% |
| **ROC-AUC** | 99.42% |

### Confusion Matrix Results
```
                 Predicted
               Normal  Pneumonia
Actual Normal    308      11
    Pneumonia     32     823
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- GPU support recommended (CUDA-compatible)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rugvedrc/AI-Powered-Pneumonia-Detection-System.git
   cd AI-Powered-Pneumonia-Detection-System
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Web Application
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`

### Python API Usage
```python
from pneumonia_detector import PneumoniaDetector
from PIL import Image

# Initialize detector
detector = PneumoniaDetector()

# Load and predict
image = Image.open("chest_xray.jpg")
result = detector.predict(image)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Uncertainty: {result['uncertainty']:.3f}")
```

### Batch Processing
```python
results = detector.batch_predict([
    "image1.jpg", 
    "image2.jpg", 
    "image3.jpg"
])

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['prediction']} ({result['confidence']:.1%})")
```

## 🧠 Model Architecture

### Core Components

```
Input (224×224×3)
      ↓
DenseNet121 Backbone
      ↓
Channel Attention
      ↓
Spatial Attention
      ↓
Monte Carlo Dropout
      ↓
Binary Classification
      ↓
Grad-CAM Visualization
```

### Key Features

1. **Attention Mechanisms**: Dual attention (channel + spatial) focuses on clinically relevant lung regions
2. **Uncertainty Estimation**: Monte Carlo Dropout (50 samples) quantifies prediction confidence
3. **Explainability**: Grad-CAM heatmaps show exactly where the model detects pneumonia
4. **Optimized Threshold**: Data-driven decision boundary at 0.6446 for optimal performance

## 🎨 Features

### Explainable Predictions
- **Grad-CAM Heatmaps**: Visual explanations showing attention on lung opacities and consolidations
- **Uncertainty Scores**: Confidence metrics to identify borderline cases requiring expert review
- **Clinical Transparency**: Model decisions aligned with radiological features

### Production-Ready Deployment
- **Streamlit Web Interface**: User-friendly application for clinical workflows
- **Model Caching**: Optimized inference for faster predictions
- **Batch Processing**: Efficient handling of multiple X-rays
- **GPU Acceleration**: Support for CUDA-enabled devices

### Medical Safety
- **High Sensitivity (96.26%)**: Minimizes missed pneumonia cases
- **High Specificity (96.53%)**: Reduces false alarms for healthy patients
- **Uncertainty Awareness**: Flags ambiguous cases for human review

## 📁 Project Structure

```
AI-Powered-Pneumonia-Detection-System/
├── app.py                          # Streamlit web application
├── models/
│   ├── attention_model.keras       # Trained model weights
│   └── attention_model_threshold.pkl
├── src/
│   ├── model.py                    # Model architecture
│   ├── utils.py                    # Utility functions
│   ├── gradcam.py                  # Grad-CAM implementation
│   └── detector.py                 # Main detector class
├── samples/                        # Sample X-ray images
├── requirements.txt                # Dependencies
├── README.md
└── LICENSE
```

## 📄 Research Paper

**Title:** Uncertainty-Aware Attention Mechanism for Pneumonia Detection Using Chest X-Rays

**Authors:**
- Archana W. Bhade (Government College of Engineering, Aurangabad)
- Smita M. Chavan (Government College of Engineering, Aurangabad)
- Rugved Rajesh Chandekar (Government College of Engineering, Aurangabad)
- Dnyanesh Milind Deshmukh (Government College of Engineering, Aurangabad)
- Aditya Keshav Magar (Government College of Engineering, Aurangabad)

**Status:** Under review at **IEEE CICT 2025**

**Abstract:** This work proposes an uncertainty-aware deep learning framework integrating spatial and channel-based attention mechanisms with Monte Carlo Dropout for pneumonia detection from chest X-rays. The attention-augmented DenseNet121 model achieves 96.33% accuracy, 96.26% sensitivity, 96.53% specificity, 97.45% F1-score, and 99.42% ROC-AUC, demonstrating superior performance and explainability for clinical decision support.

### Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{bhade2025uncertainty,
  title={Uncertainty-Aware Attention Mechanism for Pneumonia Detection Using Chest X-Rays},
  author={Bhade, Archana W. and Chavan, Smita M. and Chandekar, Rugved Rajesh and Deshmukh, Dnyanesh Milind and Magar, Aditya Keshav},
  booktitle={IEEE International Conference on Computational Intelligence and Computing Technologies (CICT)},
  year={2025},
  note={Under Review}
}
```

## 🔬 Training Pipeline

**Note:** The training pipeline and dataset preparation code are not publicly available as this is ongoing research.
If you need access to the training pipeline for research collaboration or educational purposes, please contact me directly at [rugvedchandekar@gmail.com](mailto:rugvedchandekar@gmail.com).

## ⚖️ Medical Disclaimer

**⚠️ IMPORTANT:** This software is intended for **educational and research purposes only**. It is **NOT approved for clinical use** and should **NOT** be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Government College of Engineering, Aurangabad (Chhatrapati Sambhajinagar)
- Chest X-Ray Images (Pneumonia) dataset from Kaggle
- TensorFlow and Keras development teams
- Streamlit framework

## 📞 Contact

**Rugved Rajesh Chandekar**
- 📧 Email: rugvedchandekar@gmail.com
- 💻 GitHub: [@Rugvedrc](https://github.com/Rugvedrc)
- 🤗 HuggingFace: [rugvedrc](https://huggingface.co/rugvedrc)
- 💬 Issues: [GitHub Issues](https://github.com/Rugvedrc/AI-Powered-Pneumonia-Detection-System/issues)

---

<div align="center">

**⭐ If you find this project helpful, please star the repository! ⭐**

[![GitHub stars](https://img.shields.io/github/stars/Rugvedrc/AI-Powered-Pneumonia-Detection-System.svg?style=social&label=Star)](https://github.com/Rugvedrc/AI-Powered-Pneumonia-Detection-System)
[![GitHub forks](https://img.shields.io/github/forks/Rugvedrc/AI-Powered-Pneumonia-Detection-System.svg?style=social&label=Fork)](https://github.com/Rugvedrc/AI-Powered-Pneumonia-Detection-System/fork)

Made with ❤️ for advancing medical AI research

</div>
