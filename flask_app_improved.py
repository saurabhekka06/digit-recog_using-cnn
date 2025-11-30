from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Improved CNN model class (must match training script)
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

# Global variables
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the improved PyTorch model"""
    global model
    try:
        model = ImprovedCNN().to(device)
        # Try to load the best model first, then fall back to regular model
        try:
            model.load_state_dict(torch.load('mnist_model_best.pth', map_location=device))
            print("✅ Improved PyTorch model loaded from mnist_model_best.pth!")
        except FileNotFoundError:
            model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
            print("✅ Improved PyTorch model loaded from mnist_model.pth!")
        model.eval()
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def preprocess_image(image_data):
    """
    Advanced preprocessing for better accuracy:
    - Decode base64 image
    - Convert to grayscale
    - Auto-detect if inversion is needed
    - Apply adaptive thresholding
    - Find and crop to bounding box
    - Center and resize to 28x28
    - Normalize
    """
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Convert to numpy array (keep original size for better bounding box detection)
        image_array = np.array(image)
        
        # Auto-detect if we need to invert: check if background is darker than foreground
        # MNIST expects white digit on black background
        mean_value = np.mean(image_array)
        
        # If mean is high (light background), invert
        if mean_value > 127:
            image_array = 255 - image_array
        
        # Apply contrast enhancement using percentile normalization
        p2, p98 = np.percentile(image_array, (2, 98))
        if p98 > p2:
            image_array = np.clip((image_array - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
        
        # Apply binary thresholding using Otsu's method for cleaner digit extraction
        # This works better for photos with uneven lighting
        from scipy import ndimage
        # Calculate optimal threshold (Otsu's method approximation)
        hist, bin_edges = np.histogram(image_array, bins=256, range=(0, 256))
        hist = hist.astype(float)
        
        # Find threshold that maximizes between-class variance
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
        
        # Apply threshold
        image_array = (image_array > threshold_otsu).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the digit
        # Closing to fill small holes
        from scipy.ndimage import binary_closing, binary_opening, binary_dilation
        binary = image_array > 127
        binary = binary_closing(binary, structure=np.ones((3, 3)))
        # Opening to remove small noise
        binary = binary_opening(binary, structure=np.ones((2, 2)))
        # Slight dilation to thicken strokes (MNIST has thicker strokes)
        binary = binary_dilation(binary, structure=np.ones((2, 2)))
        image_array = (binary * 255).astype(np.uint8)
        
        # Find bounding box to center the digit
        rows = np.any(image_array > 127, axis=1)
        cols = np.any(image_array > 127, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Crop to bounding box with padding
            cropped = image_array[rmin:rmax+1, cmin:cmax+1]
            
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
            
            # Resize maintaining aspect ratio
            padded_img = Image.fromarray(padded.astype('uint8'))
            resized = padded_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
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
        
        # Reshape to match model input (1, 1, 28, 28)
        image_array = image_array.reshape(1, 1, 28, 28)
        
        # Convert to PyTorch tensor
        tensor = torch.FloatTensor(image_array).to(device)
        
        return tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    """Predict digit from canvas drawing (single digit)."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # Make prediction
        with torch.no_grad():
            output = model(processed_image)
            probabilities = torch.softmax(output, dim=1)
            digit = int(torch.argmax(output, dim=1).item())
            confidence = float(torch.max(probabilities).item())
            
            # Get top 3 predictions for better insight
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)
            top3_predictions = [
                {'digit': int(idx), 'confidence': float(prob)}
                for idx, prob in zip(top3_indices, top3_probs)
            ]
        
        return jsonify({
            'digit': digit,
            'confidence': confidence,
            'all_predictions': probabilities[0].cpu().numpy().tolist(),
            'top3': top3_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_two_digits', methods=['POST'])
def predict_two_digits():
    """Predict two digits from a single canvas drawing by splitting the image.

    This is a heuristic approach: the 28x28 processed MNIST image is split into
    left and right halves, each resized back to 28x28 and fed separately to the model.
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Get full processed tensor (1, 1, 28, 28)
        full_tensor = preprocess_image(image_data)
        if full_tensor is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400

        # Convert to numpy image (28, 28)
        img = full_tensor.detach().cpu().numpy()[0, 0]

        # Because the tensor is already normalized for MNIST, we roughly invert that
        # normalization back to [0, 1] range before splitting and resizing. This keeps
        # the transformation simple while remaining close to the original digits.
        img_01 = img * 0.3081 + 0.1307

        # Clip to [0, 1]
        img_01 = np.clip(img_01, 0.0, 1.0)

        # Try to find a vertical split (column) where there is a clear gap between digits.
        # We compute the column-wise sum and look for the lowest valley between content
        # on the left and content on the right.
        col_sums = img_01.sum(axis=0)
        cols = np.arange(col_sums.shape[0])

        # Columns that contain any ink
        ink_cols = np.where(col_sums > 0.02)[0]  # threshold is heuristic
        if len(ink_cols) >= 2:
            left_min = ink_cols[0]
            right_max = ink_cols[-1]

            # Search for the minimum column-sum in the middle region as split
            mid_start = left_min + 2
            mid_end = right_max - 2
            if mid_end > mid_start:
                mid_region = col_sums[mid_start:mid_end]
                split_offset = np.argmin(mid_region)
                split_col = mid_start + split_offset
            else:
                split_col = 14
        else:
            # Fallback: simple middle split
            split_col = 14

        # Ensure split is not too extreme
        split_col = max(6, min(22, split_col))

        left_half = img_01[:, :split_col]
        right_half = img_01[:, split_col:]

        # Helper to resize a half back to 28x28 and normalize like MNIST expects
        def half_to_tensor(half_img):
            # Convert to 0-255 uint8 for PIL
            half_uint8 = (half_img * 255.0).astype('uint8')
            pil_img = Image.fromarray(half_uint8)
            # Guard against extremely thin halves: pad to at least 3 pixels wide
            w = max(half_img.shape[1], 3)
            pil_img = pil_img.resize((w, 28), Image.Resampling.LANCZOS)
            pil_resized = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
            arr = np.array(pil_resized).astype(np.float32) / 255.0
            # Apply MNIST normalization again
            arr = (arr - 0.1307) / 0.3081
            arr = arr.reshape(1, 1, 28, 28)
            return torch.FloatTensor(arr).to(device)

        left_tensor = half_to_tensor(left_half)
        right_tensor = half_to_tensor(right_half)

        with torch.no_grad():
            # Left digit
            out_left = model(left_tensor)
            probs_left = torch.softmax(out_left, dim=1)
            digit_left = int(torch.argmax(out_left, dim=1).item())
            conf_left = float(torch.max(probs_left).item())

            # Right digit
            out_right = model(right_tensor)
            probs_right = torch.softmax(out_right, dim=1)
            digit_right = int(torch.argmax(out_right, dim=1).item())
            conf_right = float(torch.max(probs_right).item())

        combined_str = f"{digit_left}{digit_right}"
        combined_conf = float(min(conf_left, conf_right))

        return jsonify({
            'digits': [digit_left, digit_right],
            'combined': combined_str,
            'confidences': [conf_left, conf_right],
            'combined_confidence': combined_conf
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_equation', methods=['POST'])
def predict_equation():
    """Predict equation from canvas drawing (mock implementation)"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Mock equation for now
        mock_equations = [
            '12 + 7', '15 - 8', '6 × 4', '20 ÷ 5', '3 + 4 × 2',
            '10 - 3 + 2', '8 ÷ 2 + 1', '5 × 3 - 7', '9 + 6 ÷ 2'
        ]
        
        equation = mock_equations[np.random.randint(0, len(mock_equations))]
        
        try:
            clean_equation = equation.replace('×', '*').replace('÷', '/')
            result = eval(clean_equation)
        except:
            result = 'Error'
        
        return jsonify({
            'equation': equation,
            'result': result,
            'steps': [
                'Raw Input: Handwritten equation detected',
                f'Segmentation: Individual characters isolated',
                f'Recognition: "{equation}"',
                'Parsing: Mathematical expression parsed',
                f'Calculation: {equation} = {result}'
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'model_type': 'ImprovedCNN'
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server with improved model...")
    print("📱 Loading improved PyTorch model...")
    
    if load_model():
        print("✅ Model loaded successfully!")
        print("🌐 Starting server on http://localhost:5000")
        print("🎯 Using improved CNN architecture for better accuracy")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model.")
        print("💡 Train the model first using: python train_improved_model.py")
