from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms, models

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model_path = 'kidney_stone_resnet18.pth'

# Initialize the ResNet-18 model architecture
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # Adjust output layer for 4 classes
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Classes for kidney stone detection
classes = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Transformations for the uploaded images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file selected'})

    try:
        # Process the image
        image = Image.open(file).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = probabilities.argmax(1).item()
            confidence = probabilities[0][predicted_class].item()

        prediction = classes[predicted_class]
        return jsonify({'prediction': prediction, 'confidence': round(confidence * 100, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
