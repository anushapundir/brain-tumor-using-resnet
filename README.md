# 🧠 Brain Tumor Detection App

A Streamlit web application for classifying brain tumor MRI images using transfer learning with PyTorch ResNet18. This is an educational project demonstrating deep learning model deployment.

**⚠️ Disclaimer:** This is a student project for educational purposes only and should NOT be used for actual medical diagnosis.

## Features

- Upload MRI brain scan images (JPG, PNG)
- Real-time tumor classification using ResNet18
- Display prediction confidence and probability distribution
- Support for 4 tumor types: Glioma, Meningioma, No-tumor, Pituitary
- Simple, clean UI built with Streamlit

## Project Structure

```
brain_tumor_app/
├── app.py                        # Main Streamlit application
├── brain_tumor_resnet18.pth      # Trained model checkpoint
├── class_names.json              # List of tumor class names
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Running Locally

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** at `http://localhost:8501`

## Deploying to Streamlit Cloud

### Steps

1. **Push your code to GitHub**
   - Create a new repository on GitHub
   - Push all files (app.py, requirements.txt, .pth file, class_names.json)

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository, branch, and `app.py` as the main file
   - Click "Deploy"

3. **Wait for deployment**
   - Streamlit Cloud will install dependencies and launch your app
   - You'll get a public URL to share

### Important Notes

- Make sure `brain_tumor_resnet18.pth` is included in your repository (even if it's large)
- GitHub has a 100MB file size limit; if your model is larger, consider using Git LFS
- For faster deployment on Streamlit Cloud, you can use CPU-only PyTorch (see requirements.txt comments)

## Model Information

- **Architecture:** ResNet18 (transfer learning from ImageNet)
- **Input size:** 224x224 RGB images
- **Preprocessing:** Resize to 256x256, center crop to 224x224, ImageNet normalization
- **Output:** 4 classes (Glioma, Meningioma, No-tumor, Pituitary)

## Credits

Model trained using PyTorch and transfer learning techniques as part of an educational project.

## License

This project is for educational purposes only.
