"""
Handwritten Digit Recognizer - Prediction Script
Predict digits from custom images
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import argparse

def load_model(model_path='models/final_model.h5'):
    """Load the trained model"""
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_image(image_path, target_size=(28, 28)):
    """Preprocess image for prediction"""
    # Read image
    if isinstance(image_path, str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        # If it's already a numpy array
        img = image_path
    
    # Resize to 28x28
    img = cv2.resize(img, target_size)
    
    # Invert if background is white (MNIST has white background, black digits)
    # If image has white background, invert it
    if np.mean(img) > 127:
        img = 255 - img
    
    # Normalize to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Reshape to match model input (1, 28, 28, 1)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    
    return img


def predict_digit(model, image_path, show_image=True):
    """Predict digit from image"""
    # Preprocess image
    img = preprocess_image(image_path)
    
    # Get prediction
    predictions = model.predict(img, verbose=0)
    predicted_digit = np.argmax(predictions)
    confidence = predictions[0][predicted_digit] * 100
    
    # Get top 3 predictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    top3_probs = predictions[0][top3_indices] * 100
    
    print("\n" + "="*50)
    print("Prediction Results")
    print("="*50)
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nTop 3 Predictions:")
    for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs), 1):
        print(f"  {i}. Digit {idx}: {prob:.2f}%")
    
    if show_image:
        plt.figure(figsize=(8, 4))
        
        # Original image
        plt.subplot(1, 2, 1)
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) if isinstance(image_path, str) else image_path
        plt.imshow(original_img, cmap='gray')
        plt.title('Original Image', fontweight='bold')
        plt.axis('off')
        
        # Preprocessed image
        plt.subplot(1, 2, 2)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'Predicted: {predicted_digit} ({confidence:.1f}%)', 
                 fontweight='bold', fontsize=14, color='green')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return predicted_digit, confidence, predictions


def predict_from_mnist_sample(model, num_samples=10):
    """Predict on random MNIST test samples"""
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        img = x_test[idx]
        true_label = y_test[idx]
        
        # Preprocess and predict
        preprocessed = preprocess_image(img)
        predictions = model.predict(preprocessed, verbose=0)
        pred_label = np.argmax(predictions)
        confidence = predictions[0][pred_label] * 100
        
        # Display
        ax.imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)'
        ax.set_title(title, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('MNIST Test Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Predict handwritten digits')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--mnist', action='store_true', help='Test on MNIST samples')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of MNIST samples')
    parser.add_argument('--model', type=str, default='models/final_model.h5', help='Model path')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    if model is None:
        print("Please train the model first using train_model.py")
        return
    
    if args.mnist:
        # Test on MNIST samples
        predict_from_mnist_sample(model, args.num_samples)
    elif args.image:
        # Predict on custom image
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found")
            return
        predict_digit(model, args.image)
    else:
        print("Please provide either --image <path> or --mnist flag")
        print("\nExample usage:")
        print("  python predict.py --image path/to/image.png")
        print("  python predict.py --mnist --num_samples 10")


if __name__ == "__main__":
    main()

