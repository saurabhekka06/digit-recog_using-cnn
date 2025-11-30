# Debug script to test your model directly
# Run this to see what's happening with your model

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Your CNN model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(self.relu(self.conv2(x)))  # 14 -> 7
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_model():
    print("🔍 Testing your model...")
    
    # Load your model
    try:
        model = CNN()
        model.load_state_dict(torch.load('mnist_model.pth', map_location='cpu'))
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with a simple case
    print("\n📊 Testing with random input...")
    test_input = torch.randn(1, 1, 28, 28)
    
    with torch.no_grad():
        output = model(test_input)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.max(probabilities).item()
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"All probabilities: {probabilities[0].numpy()}")
    
    # Test with a more realistic input (all zeros)
    print("\n📊 Testing with zeros input...")
    zeros_input = torch.zeros(1, 1, 28, 28)
    
    with torch.no_grad():
        output = model(zeros_input)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.max(probabilities).item()
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    
    # Test with a simple pattern (center pixel)
    print("\n📊 Testing with center dot...")
    center_input = torch.zeros(1, 1, 28, 28)
    center_input[0, 0, 14, 14] = 1.0  # Single pixel in center
    
    with torch.no_grad():
        output = model(center_input)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.max(probabilities).item()
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    
    print("\n🎯 If all predictions are the same or very low confidence, there might be an issue with the model training or loading.")

if __name__ == "__main__":
    test_model()
