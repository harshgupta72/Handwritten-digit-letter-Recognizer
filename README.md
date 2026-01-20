# 🔢 Handwritten Digit Recognizer

A Convolutional Neural Network (CNN) based system for recognizing handwritten digits from images. This project implements a deep learning model trained on the MNIST dataset, achieving high accuracy in digit classification (0-9).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Optional Enhancements](#optional-enhancements)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project demonstrates the power of Convolutional Neural Networks in computer vision tasks. The model is trained to automatically identify handwritten digits by learning spatial features such as edges, curves, and patterns from grayscale images.

### Key Capabilities
- ✅ Accurate digit classification (0-9)
- ✅ Real-time prediction on custom images
- ✅ Interactive web application
- ✅ Comprehensive evaluation metrics
- ✅ Visual predictions and error analysis

## ✨ Features

- **Data Preprocessing**: Automatic normalization and one-hot encoding
- **CNN Architecture**: Multi-layer convolutional network with pooling and dropout
- **Model Training**: With early stopping, learning rate reduction, and model checkpointing
- **Evaluation Metrics**: Accuracy, loss, confusion matrix, and classification report
- **Visualizations**: Training curves, sample predictions, and error analysis
- **Web Interface**: Streamlit-based interactive application for live predictions
- **Custom Predictions**: Support for uploaded images and drawn digits

## 🛠️ Technologies Used

- **Python** 3.8+
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Streamlit** - Web application framework
- **OpenCV** - Image processing
- **Pillow** - Image manipulation
- **Scikit-learn** - Evaluation metrics

## 📁 Project Structure

```
handwritten-digit-recognizer/
│
├── train_model.py          # Main training script
├── evaluate_model.py       # Model evaluation and visualization
├── predict.py              # Prediction on custom images
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
│
├── models/                 # Trained models directory
│   ├── final_model.h5     # Final trained model
│   └── digit_recognizer_model.h5  # Best model (checkpoint)
│
└── outputs/                # Generated visualizations
    ├── training_history.png
    ├── confusion_matrix.png
    ├── sample_predictions.png
    └── error_analysis.png
```

## 🚀 Installation

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd "C:\Users\Aman Raj\Harsh project"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### 1. Train the Model

Train the CNN on the MNIST dataset:

```bash
python train_model.py
```

This will:
- Download the MNIST dataset automatically
- Preprocess the data (normalization, reshaping)
- Build and train the CNN model
- Save the trained model to `models/final_model.h5`
- Generate training history plots

**Training Parameters:**
- Epochs: 15
- Batch Size: 128
- Optimizer: Adam
- Loss Function: Categorical Cross-entropy

### 2. Evaluate the Model

Evaluate the trained model and generate visualizations:

```bash
python evaluate_model.py
```

This generates:
- Test accuracy and loss metrics
- Classification report
- Confusion matrix
- Sample predictions visualization
- Error analysis

### 3. Predict on Custom Images

Predict digits from your own images:

```bash
python predict.py --image path/to/your/image.png
```

Or test on random MNIST samples:

```bash
python predict.py --mnist --num_samples 10
```

### 4. Run the Web Application

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

The app allows you to:
- Upload digit images
- Draw digits on a canvas
- View predictions with confidence scores
- See prediction probabilities for all digits

## 🏗️ Model Architecture

The CNN architecture consists of:

1. **Convolutional Block 1**
   - Conv2D layer: 32 filters, 3x3 kernel, ReLU activation
   - MaxPooling2D: 2x2 pool size

2. **Convolutional Block 2**
   - Conv2D layer: 64 filters, 3x3 kernel, ReLU activation
   - MaxPooling2D: 2x2 pool size

3. **Convolutional Block 3**
   - Conv2D layer: 64 filters, 3x3 kernel, ReLU activation

4. **Classification Head**
   - Flatten layer
   - Dense layer: 64 units, ReLU activation
   - Dropout: 0.5 (50% dropout rate)
   - Dense layer: 10 units (output), Softmax activation

**Total Parameters:** ~93,000 trainable parameters

### Model Summary
```
Model: Sequential
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320
 max_pooling2d (MaxPooling2D) (None, 13, 13, 32)       0
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496
 max_pooling2d_1 (MaxPooling2D) (None, 5, 5, 64)       0
 conv2d_2 (Conv2D)           (None, 3, 3, 64)          36928
 flatten (Flatten)           (None, 576)               0
 dense (Dense)               (None, 64)                36928
 dropout (Dropout)           (None, 64)                0
 dense_1 (Dense)             (None, 10)                650
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
```

## 📊 Results

### Performance Metrics

The model achieves:
- **Test Accuracy**: ~99%+ (varies with training)
- **Test Loss**: < 0.05

### Training Visualization

The training process includes:
- Real-time accuracy and loss monitoring
- Validation set evaluation
- Early stopping to prevent overfitting
- Learning rate reduction for fine-tuning

## 🔧 Optional Enhancements

This project includes several optional features:

### ✅ Implemented

1. **Web Application** - Streamlit-based GUI for live predictions
2. **Data Visualization** - Comprehensive plots and metrics
3. **Model Checkpointing** - Save best models during training
4. **Error Analysis** - Identify and visualize misclassifications

### 🔄 Can Be Added

1. **Data Augmentation** - Rotate, shift, zoom images to improve generalization
2. **Hyperparameter Tuning** - Grid search or Bayesian optimization
3. **Model Comparison** - Compare CNN with traditional ML models (SVM, Random Forest)
4. **Advanced Architectures** - Experiment with ResNet, VGG, or custom architectures
5. **Model Deployment** - Deploy using Flask, Docker, or cloud platforms

## 🔮 Future Improvements

- [ ] Add data augmentation pipeline
- [ ] Implement hyperparameter tuning with Keras Tuner
- [ ] Add support for multi-digit recognition
- [ ] Create REST API for model serving
- [ ] Deploy to cloud platform (AWS, GCP, Azure)
- [ ] Add real-time camera input support
- [ ] Implement ensemble methods for better accuracy

## 📝 Notes

- The MNIST dataset is automatically downloaded on first run
- Models are saved in the `models/` directory
- All visualizations are saved in the project root
- The Streamlit app requires the trained model to be in `models/final_model.h5`

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Created as part of a Machine Learning/Deep Learning project.

---

**Happy Coding! 🚀**

