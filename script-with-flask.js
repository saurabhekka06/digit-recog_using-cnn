// Global variables
let currentPage = 'canvas';
let isDrawing = false;
let cameraStream = null;
let drawingContext = null;
let equationContext = null;
let apiBaseUrl = 'http://localhost:5000'; // Flask API URL

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

async function initializeApp() {
    // Check if Flask API is running
    await checkApiHealth();
    
    // Set up navigation
    setupNavigation();
    
    // Initialize drawing canvas
    initializeDrawingCanvas();
    
    // Initialize camera functionality
    initializeCamera();
    
}

// Check if Flask API is running
async function checkApiHealth() {
    try {
        const response = await fetch(`${apiBaseUrl}/health`);
        const data = await response.json();
        
        if (data.model_loaded) {
            showModelStatus('Model loaded successfully!', 'success');
        } else {
            showModelStatus('Model not loaded. Please check Flask server.', 'error');
        }
    } catch (error) {
        console.error('API health check failed:', error);
        showModelStatus('Flask API not running. Please start the server.', 'error');
    }
}

function showModelStatus(message, type) {
    // Create status element
    const statusDiv = document.createElement('div');
    statusDiv.className = `model-status ${type}`;
    statusDiv.textContent = message;
    statusDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 10px 20px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        z-index: 1000;
        background: ${type === 'success' ? '#28a745' : '#dc3545'};
    `;
    
    document.body.appendChild(statusDiv);
    
    // Remove after 5 seconds
    setTimeout(() => {
        if (statusDiv.parentNode) {
            statusDiv.parentNode.removeChild(statusDiv);
        }
    }, 5000);
}

// Navigation functionality
function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const pages = document.querySelectorAll('.page');
    
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetPage = button.getAttribute('data-page');
            switchPage(targetPage);
            
            // Update active button
            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
        });
    });
}

function switchPage(pageName) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show target page
    document.getElementById(pageName + '-page').classList.add('active');
    currentPage = pageName;
    
    // Stop camera if switching away from camera page
    if (pageName !== 'camera' && cameraStream) {
        stopCamera();
    }
}

// Drawing Canvas functionality
function initializeDrawingCanvas() {
    const canvas = document.getElementById('drawingCanvas');
    drawingContext = canvas.getContext('2d');
    
    // Set up drawing styles
    drawingContext.fillStyle = '#FFFFFF'; // White background
    drawingContext.fillRect(0, 0, 400, 400); // Fill the entire canvas with white
    drawingContext.strokeStyle = '#000000'; // Black drawing
    drawingContext.lineWidth = 12;
    drawingContext.lineCap = 'round';
    drawingContext.lineJoin = 'round';
    
    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    // Button events
    document.getElementById('clearCanvas').addEventListener('click', clearCanvas);
    document.getElementById('predictDigit').addEventListener('click', predictDigit);
    const predictTwoBtn = document.getElementById('predictTwoDigits');
    if (predictTwoBtn) {
        predictTwoBtn.addEventListener('click', predictTwoDigits);
    }
}

async function predictTwoDigits() {
    const canvas = document.getElementById('drawingCanvas');

    // Check if canvas has any drawing
    const imageData = drawingContext.getImageData(0, 0, 400, 400);
    const hasContent = imageData.data.some((pixel, index) => {
        return index % 4 === 3 && pixel > 0;
    });

    if (!hasContent) {
        alert('Please draw two digits first!');
        return;
    }

    try {
        const imageDataUrl = canvas.toDataURL('image/png');

        const response = await fetch(`${apiBaseUrl}/predict_two_digits`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageDataUrl
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        if (result.error) {
            throw new Error(result.error);
        }

        // Show combined prediction like "12"
        document.getElementById('predictedDigit').textContent = result.combined;
        document.getElementById('confidenceText').textContent = Math.round(result.combined_confidence * 100) + '%';
        document.getElementById('confidenceFill').style.width = (result.combined_confidence * 100) + '%';

        console.log('Two-digit prediction result:', result);
    } catch (error) {
        console.error('Two-digit prediction error:', error);
        alert('Error making two-digit prediction. Please check if Flask server is running.');
    }
}

function startDrawing(e) {
    isDrawing = true;
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    drawingContext.beginPath();
    drawingContext.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    drawingContext.lineTo(x, y);
    drawingContext.stroke();
}

function stopDrawing() {
    isDrawing = false;
    drawingContext.beginPath();
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                    e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    e.target.dispatchEvent(mouseEvent);
}

function clearCanvas() {
    // Clear and redraw white background
    drawingContext.clearRect(0, 0, 400, 400);
    drawingContext.fillStyle = '#FFFFFF';
    drawingContext.fillRect(0, 0, 400, 400);
    drawingContext.strokeStyle = '#000000'; // Reset stroke color
    
    document.getElementById('predictedDigit').textContent = '-';
    document.getElementById('confidenceText').textContent = '0%';
    document.getElementById('confidenceFill').style.width = '0%';
}

async function predictDigit() {
    const canvas = document.getElementById('drawingCanvas');
    
    // Check if canvas has any drawing by looking at the actual canvas content
    const imageData = drawingContext.getImageData(0, 0, 400, 400);
    const hasContent = imageData.data.some((pixel, index) => {
        // Check every 4th value (alpha channel) or any non-zero pixel
        return index % 4 === 3 && pixel > 0;
    });
    
    if (!hasContent) {
        alert('Please draw a digit first!');
        return;
    }
    
    try {
        // Convert canvas to base64 image
        const imageDataUrl = canvas.toDataURL('image/png');
        
        console.log('Canvas size:', canvas.width, 'x', canvas.height);
        console.log('Image data URL length:', imageDataUrl.length);
        
        // Send to Flask API
        const response = await fetch(`${apiBaseUrl}/predict_digit`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageDataUrl
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Update UI with prediction
        document.getElementById('predictedDigit').textContent = result.digit;
        document.getElementById('confidenceText').textContent = Math.round(result.confidence * 100) + '%';
        document.getElementById('confidenceFill').style.width = (result.confidence * 100) + '%';
        
        console.log('Prediction result:', result);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction. Please check if Flask server is running.');
    }
}

// Camera functionality
function initializeCamera() {
    document.getElementById('startCamera').addEventListener('click', startCamera);
    document.getElementById('capturePhoto').addEventListener('click', capturePhoto);
    document.getElementById('stopCamera').addEventListener('click', stopCamera);
    document.getElementById('imageUpload').addEventListener('change', handleImageUpload);
}

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        
        const video = document.getElementById('cameraVideo');
        video.srcObject = cameraStream;
        
        document.getElementById('startCamera').disabled = true;
        document.getElementById('capturePhoto').disabled = false;
        document.getElementById('stopCamera').disabled = false;
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Unable to access camera. Please check permissions.');
    }
}

function capturePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    // Convert to image
    const imageData = canvas.toDataURL('image/png');
    const img = document.getElementById('capturedImg');
    img.src = imageData;
    img.style.display = 'block';
    
    // Predict the digit
    predictImageDigit(imageData);
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        
        document.getElementById('cameraVideo').srcObject = null;
        document.getElementById('startCamera').disabled = false;
        document.getElementById('capturePhoto').disabled = true;
        document.getElementById('stopCamera').disabled = false;
    }
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.getElementById('capturedImg');
            img.src = e.target.result;
            img.style.display = 'block';
            predictImageDigit(e.target.result);
        };
        reader.readAsDataURL(file);
    }
}

async function predictImageDigit(imageData) {
    try {
        // Send to Flask API
        const response = await fetch(`${apiBaseUrl}/predict_digit`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Update UI with prediction
        document.getElementById('imagePredictedDigit').textContent = result.digit;
        document.getElementById('imageConfidenceText').textContent = Math.round(result.confidence * 100) + '%';
        document.getElementById('imageConfidenceFill').style.width = (result.confidence * 100) + '%';
        
        console.log('Image prediction result:', result);
        
    } catch (error) {
        console.error('Image prediction error:', error);
        alert('Error processing image. Please check if Flask server is running.');
    }
}


// Utility functions
function resizeCanvasToDisplaySize(canvas) {
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;
    
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
    }
}

// Handle window resize
window.addEventListener('resize', () => {
    resizeCanvasToDisplaySize(document.getElementById('drawingCanvas'));
});
