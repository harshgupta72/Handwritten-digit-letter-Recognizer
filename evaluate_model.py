"""
Handwritten Digit Recognizer - Evaluation Script
Evaluate the trained model and visualize predictions
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def load_model(model_path='models/final_model.h5'):
    """Load the trained model"""
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_test_data():
    """Load and preprocess test data"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)
    
    # One-hot encode labels
    y_test_categorical = keras.utils.to_categorical(y_test, 10)
    
    return x_test, y_test, y_test_categorical


def evaluate_model(model, x_test, y_test_categorical):
    """Evaluate model performance"""
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    # Get predictions
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test_categorical, axis=1)
    
    # Overall accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Classification report
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))
    
    return y_pred, y_true


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.show()


def visualize_predictions(model, x_test, y_test, num_samples=20, save_path='sample_predictions.png'):
    """Visualize sample predictions"""
    # Get random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    samples = x_test[indices]
    true_labels = y_test[indices]
    
    # Get predictions
    predictions = model.predict(samples, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    pred_probs = np.max(predictions, axis=1)
    
    # Create subplot
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < num_samples:
            ax.imshow(samples[i].squeeze(), cmap='gray')
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            confidence = pred_probs[i] * 100
            
            # Color based on correctness
            color = 'green' if true_label == pred_label else 'red'
            title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
            ax.set_title(title, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sample predictions saved to {save_path}")
    plt.show()


def analyze_errors(model, x_test, y_test_categorical, y_test, top_n=10):
    """Analyze prediction errors"""
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test_categorical, axis=1)
    
    # Find misclassified samples
    errors = np.where(y_pred != y_true)[0]
    
    if len(errors) == 0:
        print("No errors found! Perfect predictions.")
        return
    
    print(f"\nTotal errors: {len(errors)} out of {len(y_test)} ({len(errors)/len(y_test)*100:.2f}%)")
    print(f"\nAnalyzing top {min(top_n, len(errors))} errors:")
    
    # Get top errors by confidence
    error_confidences = y_pred_proba[errors, y_pred[errors]]
    top_error_indices = errors[np.argsort(error_confidences)[-top_n:]][::-1]
    
    fig, axes = plt.subplots(2, min(5, top_n//2), figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(top_error_indices[:min(top_n, len(axes))]):
        ax = axes[i]
        ax.imshow(x_test[idx].squeeze(), cmap='gray')
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        confidence = error_confidences[np.where(errors == idx)[0][0]] * 100
        
        title = f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.1f}%'
        ax.set_title(title, color='red', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Top Misclassified Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    print("Error analysis saved to error_analysis.png")
    plt.show()


def main():
    """Main evaluation function"""
    # Load model
    model = load_model('models/final_model.h5')
    if model is None:
        print("Please train the model first using train_model.py")
        return
    
    # Load test data
    x_test, y_test, y_test_categorical = load_test_data()
    
    # Evaluate
    y_pred, y_true = evaluate_model(model, x_test, y_test_categorical)
    
    # Visualizations
    plot_confusion_matrix(y_true, y_pred)
    visualize_predictions(model, x_test, y_test, num_samples=20)
    analyze_errors(model, x_test, y_test_categorical, y_test, top_n=10)


if __name__ == "__main__":
    main()

