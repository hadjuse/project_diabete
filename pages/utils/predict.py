import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
import io
class DiabetiCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.features= nn.Sequential(
      nn.Conv2d(3,8, kernel_size=3, stride=1),
      nn.BatchNorm2d(8,eps=1e-5),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

      nn.Conv2d(8,16, kernel_size=3, stride=1),
      nn.BatchNorm2d(16,eps=1e-5),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )

    self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 2),
        nn.Dropout(p=0.3),
        nn.Linear(2, 2),
        nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x

def load_model(model: DiabetiCNN, device, path:Path)->DiabetiCNN:
    # Map to CPU if loading a model trained on GPU but running on CPU
    map_location = torch.device('cpu') if device == 'cpu' or not torch.cuda.is_available() else device
    model.load_state_dict(torch.load(path, weights_only=True, map_location=map_location))
    model.to(device)
    return model

def preprocess_image(image_file, target_size=(224, 224)):
    """
    Preprocess uploaded image for model prediction.
    
    Args:
        image_file: Streamlit UploadedFile or PIL Image
        target_size: Target image size (width, height)
    
    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 3, H, W)
    """
    # Load image
    if hasattr(image_file, 'read'):
        # It's a file-like object from Streamlit
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Reset file pointer if needed later
        if hasattr(image_file, 'seek'):
            image_file.seek(0)
    else:
        # It's already a PIL Image
        image = image_file.convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor
    
def prediction(model, image, device='cpu'):
    """
    Make prediction on an uploaded image.
    
    Args:
        model: Trained DiabetiCNN model
        image: Uploaded image file or PIL Image
        device: Device to run inference on
    
    Returns:
        dict: Prediction results with probabilities
    """
    model.eval()
    
    # Preprocess image
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get probabilities and predicted class
    probabilities = output[0].cpu().numpy()
    predicted_class = probabilities.argmax()
    
    # Map to labels (assuming 0=Good, 1=Bad)
    labels = ['Good (Non-Diabetic)', 'Bad (Diabetic)']
    
    result = {
        'prediction': labels[predicted_class],
        'confidence': float(probabilities[predicted_class]),
        'probabilities': {
            'Good': float(probabilities[0]),
            'Bad': float(probabilities[1])
        }
    }
    
    return result
    