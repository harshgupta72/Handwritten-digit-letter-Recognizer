
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import emnist

# ==========================================
# Monkeypatching for EMNIST
# ==========================================
project_cache_dir = os.path.join(os.getcwd(), 'emnist_cache')
emnist.CACHE_FILE_PATH = os.path.join(project_cache_dir, 'emnist.zip')

def patched_parse_idx(data):
    """Parse binary data in IDX format, returning it as a numpy array of the correct shape."""
    data = bytes(data)
    if data[0] != 0 or data[1] != 0:
        raise ValueError("Data is not in IDX format.")
    data_type_code = data[2]
    data_type = emnist.IDX_DATA_TYPE_CODES.get(data_type_code)
    if data_type is None:
        raise ValueError("Unrecognized data type code %s. Is the data in IDX format?" % hex(data_type_code))
    dims = data[3]
    if not dims:
        raise ValueError("Header indicates zero-dimensional data. Is the data in IDX format?")
    shape = []
    for dim in range(dims):
        offset = 4 * (dim + 1)
        dim_size = int(np.frombuffer(data[offset:offset + 4], dtype='>u4')[0])
        shape.append(dim_size)
    shape = tuple(shape)
    offset = 4 * (dims + 1)
    data = np.frombuffer(data[offset:], dtype=np.dtype(data_type).newbyteorder('>'))
    return data.reshape(shape)

emnist.parse_idx = patched_parse_idx
# ==========================================

def load_and_preprocess_data():
    """Load and preprocess EMNIST Letters dataset"""
    print("Loading EMNIST Letters dataset...")
    from emnist import extract_training_samples, extract_test_samples
    
    x_train, y_train = extract_training_samples('letters')
    x_test, y_test = extract_test_samples('letters')
    
    print(f"Training set shape: {x_train.shape}")
    print(f"Test set shape: {x_test.shape}")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data to add channel dimension (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # EMNIST letters are 1-26. Map to 0-25.
    y_train = y_train - 1
    y_test = y_test - 1
    
    # Convert labels to categorical one-hot encoding
    num_classes = 26
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def build_cnn_model(input_shape=(28, 28, 1), num_classes=26):
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
        layers.Dense(128, activation='relu'), # Increased size for more classes
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Validation split
    split_index = int(0.8 * len(x_train))
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_val = x_train[split_index:]
    y_val = y_train[split_index:]
    
    # Build model
    model = build_cnn_model(num_classes=26)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('models/text_recognizer_model.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    print("Starting Training...")
    history = model.fit(
        x_train_new, y_train_new,
        batch_size=128,
        epochs=10, # 10 epochs should be enough
        validation_data=(x_val, y_val),
        callbacks=callbacks
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save final model
    model.save('models/text_recognizer_model.h5')
    print("Model saved to models/text_recognizer_model.h5")

if __name__ == "__main__":
    train_model()
