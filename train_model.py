"""
Handwritten Digit Recognizer - Training Script
Train a CNN model on the MNIST dataset
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Training set shape: {x_train.shape}")
    print(f"Test set shape: {x_test.shape}")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data to add channel dimension (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Convert labels to categorical one-hot encoding
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Build CNN architecture"""
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten layer
        layers.Flatten(),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


class TestAccuracyCallback(keras.callbacks.Callback):
    """Custom callback to evaluate test accuracy after each epoch"""
    def __init__(self, x_test, y_test):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.test_accuracies = []
        self.test_losses = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Evaluate on test set and print results"""
        test_loss, test_accuracy = self.model.evaluate(
            self.x_test, self.y_test, 
            verbose=0
        )
        self.test_accuracies.append(test_accuracy)
        self.test_losses.append(test_loss)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} - Test Set Evaluation:")
        print(f"  Test Loss:     {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"{'='*60}\n")


def train_model(model, x_train, y_train, x_val, y_val, x_test=None, y_test=None, epochs=10, batch_size=128):
    """Train the CNN model"""
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\n" + "="*50)
    print("Model Architecture:")
    print("="*50)
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            'models/digit_recognizer_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=0.0001
        )
    ]
    
    # Add test accuracy callback if test data is provided
    if x_test is not None and y_test is not None:
        test_callback = TestAccuracyCallback(x_test, y_test)
        callbacks.append(test_callback)
    
    # Train the model
    print("\n" + "="*50)
    print("Starting Training...")
    print("="*50)
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Add test accuracies to history if test callback was used
    if x_test is not None and y_test is not None:
        history.history['test_accuracy'] = test_callback.test_accuracies
        history.history['test_loss'] = test_callback.test_losses
    
    return history


def plot_training_history(history, save_path='training_history.png'):
    """Plot training accuracy and loss"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    
    # Add test accuracy if available
    if 'test_accuracy' in history.history:
        axes[0].plot(history.history['test_accuracy'], label='Test Accuracy', marker='^', linestyle='--')
    
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s')
    
    # Add test loss if available
    if 'test_loss' in history.history:
        axes[1].plot(history.history['test_loss'], label='Test Loss', marker='^', linestyle='--')
    
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history saved to {save_path}")
    plt.show()


def main():
    """Main training function"""
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Split training data into train and validation sets
    # Use 20% of training data for validation
    split_index = int(0.8 * len(x_train))
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_val = x_train[split_index:]
    y_val = y_train[split_index:]
    
    print(f"\nData Split:")
    print(f"  Training set:   {len(x_train_new)} samples")
    print(f"  Validation set: {len(x_val)} samples")
    print(f"  Test set:       {len(x_test)} samples")
    
    # Build model
    print("\n" + "="*50)
    print("Building CNN Model...")
    print("="*50)
    model = build_cnn_model()
    
    # Train model with test set evaluation during training
    history = train_model(
        model, 
        x_train_new, y_train_new,
        x_val, y_val,  # Validation set
        x_test, y_test,  # Test set for evaluation during training
        epochs=15,
        batch_size=128
    )
    
    # Save final model
    model.save('models/final_model.h5')
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Test Set Evaluation...")
    print("="*50)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Test Loss:     {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Print summary of test accuracies during training
    if 'test_accuracy' in history.history:
        print("\n" + "="*50)
        print("Test Accuracy Summary During Training:")
        print("="*50)
        for epoch, acc in enumerate(history.history['test_accuracy'], 1):
            print(f"  Epoch {epoch}: {acc:.4f} ({acc*100:.2f}%)")
        max_test_acc = max(history.history['test_accuracy'])
        max_epoch = history.history['test_accuracy'].index(max_test_acc) + 1
        print(f"\n  Best Test Accuracy: {max_test_acc:.4f} ({max_test_acc*100:.2f}%) at Epoch {max_epoch}")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Model saved to: models/final_model.h5")
    print(f"Best model saved to: models/digit_recognizer_model.h5")


if __name__ == "__main__":
    main()

