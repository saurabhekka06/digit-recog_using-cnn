from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import json
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Improved CNN model class (matching trained model)
class ImprovedCNN(nn.Module):
    """Improved CNN architecture with batch normalization and dropout"""
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Conv block 1: 28x28 -> 14x14
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 2: 14x14 -> 7x7
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 3: 7x7 -> 3x3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout1(x)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Global variable to store the model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained PyTorch model"""
    global model
    try:
        # Create the improved model
        model = ImprovedCNN().to(device)
        
        # Try to load the trained weights (try best model first)
        try:
            model.load_state_dict(torch.load('mnist_model_best.pth', map_location=device))
            print("✅ Improved PyTorch model loaded from mnist_model_best.pth!")
            model.eval()
            return True
        except FileNotFoundError:
            try:
                model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
                print("✅ Improved PyTorch model loaded from mnist_model.pth!")
                model.eval()
                return True
            except FileNotFoundError:
                print("⚠️ Model file not found. Using untrained model.")
                return True
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}. Using untrained model.")
            return True
            
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def preprocess_image(image_data, debug=False):
    """Preprocess image for MNIST model"""
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if debug:
            print(f"Original image size: {image.size}")
            print(f"Original image mode: {image.mode}")
        
        # Convert to grayscale
        image = image.convert('L')
        
        if debug:
            print(f"Converted to grayscale")
        
        # Convert to numpy array (keep original size for better bounding box detection)
        image_array = np.array(image)
        
        if debug:
            print(f"Original array shape: {image_array.shape}")
            print(f"Original array min/max: {image_array.min()}/{image_array.max()}")
        
        # Auto-detect if we need to invert: check if background is darker than foreground
        # MNIST expects white digit on black background
        mean_value = np.mean(image_array)
        
        if debug:
            print(f"Mean value: {mean_value:.2f}")
        
        # If mean is high (light background), invert
        if mean_value > 127:
            image_array = 255 - image_array
            if debug:
                print("Image inverted (light background detected)")
        else:
            if debug:
                print("Image NOT inverted (dark background detected)")
        
        if debug:
            print(f"After inversion min/max: {image_array.min()}/{image_array.max()}")
        
        # Apply contrast enhancement using percentile normalization
        p2, p98 = np.percentile(image_array, (2, 98))
        if p98 > p2:
            image_array = np.clip((image_array - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
            if debug:
                print(f"Contrast enhanced: p2={p2:.1f}, p98={p98:.1f}")
        
        # Apply binary thresholding using Otsu's method for cleaner digit extraction
        # Calculate optimal threshold
        hist, bin_edges = np.histogram(image_array, bins=256, range=(0, 256))
        hist = hist.astype(float)
        
        total = image_array.size
        current_max = 0
        threshold_otsu = 0
        sum_total = np.sum(np.arange(256) * hist)
        weight_bg = 0
        sum_bg = 0
        
        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var_between > current_max:
                current_max = var_between
                threshold_otsu = t
        
        if debug:
            print(f"Otsu threshold: {threshold_otsu}")
        
        # Apply threshold
        image_array = (image_array > threshold_otsu).astype(np.uint8) * 255
        
        if debug:
            print(f"After thresholding: min={image_array.min()}, max={image_array.max()}")
        
        # Apply morphological operations to clean up the digit
        from scipy.ndimage import binary_closing, binary_opening, binary_dilation
        binary = image_array > 127
        binary = binary_closing(binary, structure=np.ones((3, 3)))
        binary = binary_opening(binary, structure=np.ones((2, 2)))
        binary = binary_dilation(binary, structure=np.ones((2, 2)))
        image_array = (binary * 255).astype(np.uint8)
        
        if debug:
            print("Applied morphological operations")
        
        # Find bounding box
        rows = np.any(image_array > 127, axis=1)
        cols = np.any(image_array > 127, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Crop to bounding box with padding
            cropped = image_array[rmin:rmax+1, cmin:cmax+1]
            
            if debug:
                print(f"Cropped size: {cropped.shape}")
            
            # Add padding (20% of size)
            pad_size = max(cropped.shape) // 5
            padded = np.pad(cropped, pad_size, mode='constant', constant_values=0)
            
            # Calculate new size maintaining aspect ratio (fit in 20x20)
            h, w = padded.shape
            if h > w:
                new_h = 20
                new_w = max(1, int(w * 20 / h))
            else:
                new_w = 20
                new_h = max(1, int(h * 20 / w))
            
            # Create 28x28 image with centered digit
            from PIL import Image as PILImage
            padded_img = PILImage.fromarray(padded.astype('uint8'))
            resized = padded_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            
            # Center in 28x28
            image_array = np.zeros((28, 28), dtype=np.float32)
            y_offset = (28 - new_h) // 2
            x_offset = (28 - new_w) // 2
            image_array[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = np.array(resized, dtype=np.float32)
        else:
            # If no content detected, just resize directly
            image_array = np.array(image.resize((28, 28), Image.Resampling.LANCZOS), dtype=np.float32)
            image_array = 255 - image_array
        
        # Normalize to 0-1 range
        image_array = image_array / 255.0
        
        # Apply MNIST normalization (mean=0.1307, std=0.3081)
        image_array = (image_array - 0.1307) / 0.3081
        
        if debug:
            print(f"Normalized array min/max: {image_array.min()}/{image_array.max()}")
        
        # Reshape to match model input (1, 1, 28, 28)
        image_array = image_array.reshape(1, 1, 28, 28)
        
        # Convert to PyTorch tensor
        tensor = torch.FloatTensor(image_array).to(device)
        
        if debug:
            print(f"Tensor shape: {tensor.shape}")
            print(f"Tensor min/max: {tensor.min().item()}/{tensor.max().item()}")
        
        return tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    """Predict digit from canvas drawing"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        print("\n🔍 Processing new prediction...")
        
        # Preprocess the image with debug info
        processed_image = preprocess_image(image_data, debug=True)
        if processed_image is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # Make prediction
        with torch.no_grad():
            prediction = model(processed_image)
            probabilities = torch.softmax(prediction, dim=1)
            digit = int(torch.argmax(prediction, dim=1).item())
            confidence = float(torch.max(probabilities).item())
        
        print(f"🎯 Prediction: {digit}")
        print(f"🎯 Confidence: {confidence:.4f}")
        print(f"🎯 All probabilities: {probabilities[0].cpu().numpy()}")
        
        return jsonify({
            'digit': digit,
            'confidence': confidence,
            'all_predictions': probabilities[0].cpu().numpy().tolist()
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📱 Loading PyTorch model...")
    
    if load_model():
        print("✅ Model loaded successfully!")
        print("🌐 Starting server on http://localhost:5000")
        print("🔍 Debug mode enabled - check console for detailed info")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model.")
