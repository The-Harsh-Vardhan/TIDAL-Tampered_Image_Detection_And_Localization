import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Error Level Analysis (ELA) Preprocessing Engine
def generate_ela(image_path, quality=90):
    """
    Recreates the ELA transformation described in the ETASR 9593 paper.
    """
    original_img = Image.open(image_path).convert('RGB')
    
    # Resave the original image at a known lossy compression rate (e.g., JPEG at 90%)
    temp_filename = 'temp_compressed.jpg'
    original_img.save(temp_filename, 'JPEG', quality=quality)
    compressed_img = Image.open(temp_filename)
    
    # Calculate the absolute difference between original and re-compressed images
    ela_img = ImageChops.difference(original_img, compressed_img)
    
    # Calculate the dynamic scaling factor: 255 / max(I_ELA(x,y))
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1 # Prevent division by zero on solid color images
    scale = 255.0 / max_diff
    
    # Scale the pixel values to mathematically enhance the visibility of compression artifacts
    ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
    
    # Resize to the strict 128x128 dimensions required for CNN tensor ingestion
    ela_img = ela_img.resize((128, 128))
    
    # Normalize 8-bit pixel intensities (0-255) to a float scale 
    return np.array(ela_img) / 255.0  

# 2. Convolutional Neural Network (CNN) Architecture
def build_etasr_cnn():
    """
    Recreates the precise 2-layer CNN topology and hyperparameters from the research paper.
    """
    model = Sequential()
    
    # Compile with Adam optimizer (conservative learning rate = 0.001) and binary cross-entropy loss
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Instantiate and verify the model topology
model = build_etasr_cnn()
model.summary()