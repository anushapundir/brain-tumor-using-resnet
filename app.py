import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from typing import Tuple, Dict
import os


# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device (GPU or CPU).
    
    Returns:
        torch.device: CUDA if available, otherwise CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model(checkpoint_path: str, class_names: list) -> nn.Module:
    """
    Load the trained ResNet18 model from checkpoint.
    Cached with @st.cache_resource so it only loads once per session.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        class_names: List of class names for the classification task
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If checkpoint doesn't contain required keys
    """
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Get the device
    device = get_device()
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Verify checkpoint structure
        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint doesn't contain 'model_state_dict'")
        
        # Build ResNet18 with pretrained ImageNet weights
        # Using weights parameter (newer torchvision API)
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            # Fallback for older torchvision versions
            model = models.resnet18(pretrained=True)
        
        # Get the number of input features for the final layer
        num_features = model.fc.in_features
        
        # Replace the final fully connected layer
        # Original ResNet18 has 1000 outputs (ImageNet classes)
        # We replace it with our number of tumor classes
        model.fc = nn.Linear(num_features, len(class_names))
        
        # Load the trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to the appropriate device
        model = model.to(device)
        
        # Set model to evaluation mode (disables dropout, batch norm, etc.)
        model.eval()
        
        return model
        
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess the input image to match training preprocessing.
    
    Applies the same transformations used during training:
    - Resize to 256x256
    - Center crop to 224x224
    - Convert to tensor
    - Normalize with ImageNet mean and std
    
    Args:
        image: PIL Image object
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Define the same preprocessing pipeline as training
    preprocess = transforms.Compose([
        transforms.Resize(256),                          # Resize shortest side to 256
        transforms.CenterCrop(224),                      # Crop center 224x224
        transforms.ToTensor(),                           # Convert to tensor [0, 1]
        transforms.Normalize(                            # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406],                 # ImageNet mean
            std=[0.229, 0.224, 0.225]                   # ImageNet std
        )
    ])
    
    # Apply transformations
    return preprocess(image)


def predict(
    image: Image.Image, 
    model: nn.Module, 
    class_names: list
) -> Tuple[str, float, Dict[str, float]]:
    """
    Make a prediction on the input image.
    
    Args:
        image: PIL Image object to classify
        model: Trained PyTorch model
        class_names: List of class names
        
    Returns:
        Tuple containing:
            - predicted_class (str): Name of predicted class
            - confidence (float): Confidence score (0-1) for predicted class
            - class_probs (dict): Dictionary mapping class names to probabilities
    """
    # Get device
    device = get_device()
    
    # Preprocess the image
    img_tensor = preprocess_image(image)
    
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0)
    
    # Move to the same device as model
    img_tensor = img_tensor.to(device)
    
    # Disable gradient computation for inference (saves memory and speeds up)
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(img_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the predicted class (highest probability)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Convert to Python native types
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        
        # Get predicted class name
        predicted_class = class_names[predicted_idx]
        
        # Create dictionary of all class probabilities
        class_probs = {
            class_names[i]: probabilities[0][i].item() 
            for i in range(len(class_names))
        }
        
    return predicted_class, confidence, class_probs


def main():
    """
    Main function that defines the Streamlit UI and handles user interactions.
    """
    # Title and description
    st.title("🧠 Brain Tumor Detection using Transfer Learning")
    st.markdown("**PyTorch ResNet18 + Streamlit**")
    
    # Disclaimer
    st.warning(
        "⚠️ **Important:** This is a student project for educational purposes only. "
        "It should NOT be used for actual medical diagnosis or treatment decisions."
    )
    
    st.markdown("---")
    
    # Define file paths
    checkpoint_path = "brain_tumor_resnet18.pth"
    class_names_path = "class_names.json"
    
    # Load class names
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        st.success(f"✅ Loaded {len(class_names)} tumor classes: {', '.join(class_names)}")
        
    except FileNotFoundError:
        st.error(f"❌ Error: Could not find '{class_names_path}'. Please ensure it's in the same directory as app.py")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"❌ Error: '{class_names_path}' is not a valid JSON file.")
        st.stop()
    
    # Load the model
    try:
        with st.spinner("Loading model... This may take a moment on first run."):
            model = load_model(checkpoint_path, class_names)
        
        device = get_device()
        st.success(f"✅ Model loaded successfully on {device}")
        
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.info("Please ensure 'brain_tumor_resnet18.pth' is in the same directory as app.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    st.markdown("---")
    
    # File uploader
    st.subheader("📤 Upload an MRI Image")
    uploaded_file = st.file_uploader(
        "Choose an MRI scan image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI scan in JPG or PNG format"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display the image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Display the uploaded image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded MRI Image", use_container_width=True)
            
            st.markdown("---")
            
            # Prediction button
            if st.button("🔍 Predict Tumor Type", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    predicted_class, confidence, class_probs = predict(image, model, class_names)
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                # Main prediction
                st.markdown(f"### Predicted Class: **{predicted_class}**")
                st.markdown(f"### Confidence: **{confidence * 100:.2f}%**")
                
                # Confidence meter
                st.progress(confidence)
                
                st.markdown("---")
                
                # All class probabilities
                st.subheader("📈 Probability Distribution")
                
                # Sort probabilities in descending order
                sorted_probs = sorted(
                    class_probs.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Display as a table
                for class_name, prob in sorted_probs:
                    # Highlight the predicted class
                    if class_name == predicted_class:
                        st.markdown(f"**{class_name}**: {prob * 100:.2f}% ✓")
                    else:
                        st.markdown(f"{class_name}: {prob * 100:.2f}%")
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please make sure you uploaded a valid image file.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit 🎈 | Powered by PyTorch 🔥"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
