# 🌿 Plant Disease Prediction with CNN

A Deep Learning project that uses Convolutional Neural Networks to identify plant diseases from leaf images. The model can classify 38 different plant disease categories with high accuracy.

## ✨ Features

- 🔬 **38 Disease Classes**: Identifies diseases across 14 different plant species
- 🎯 **High Accuracy**: CNN-based classification model
- 🌐 **Web Interface**: User-friendly Streamlit web application
- 📊 **Visualization**: Training history and prediction results
- 🚀 **Easy Deployment**: Dockerized application ready for deployment

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training the Model](#training-the-model)
- [Running the Web App](#running-the-web-app)
- [Project Structure](#project-structure)
- [Disease Classes](#disease-classes)
- [Model Architecture](#model-architecture)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Plant-Disease-Prediction-with-CNN
```

### 2. Python Environment Setup

**Already configured!** ✅ A Python 3.11.6 virtual environment is set up.

**Activate the environment:**
```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install Required Packages

**Already installed!** ✅ All required packages are installed:
- TensorFlow 2.15
- NumPy 1.26.3
- Streamlit 1.30.0
- Matplotlib, Pillow, Kaggle

## 📥 Dataset Setup

You need the PlantVillage dataset to train the model. Choose one of these methods:

### Method 1: Kaggle API (Automated)

1. **Get Kaggle API Token:**
   - Visit https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - Save the downloaded `kaggle.json`

2. **Place kaggle.json:**
   ```
   Windows: C:\Users\YourUsername\.kaggle\kaggle.json
   ```
   
   Or copy it to the project root directory.

3. **Run the download script:**
   ```powershell
   .venv\Scripts\python.exe download_dataset.py
   ```

### Method 2: Manual Download

1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Click "Download" (requires Kaggle login)
3. Extract the ZIP file to the project directory
4. Rename folder to `plantvillage_dataset`
5. Verify structure: `plantvillage_dataset/color/`

**Expected structure:**
```
plantvillage_dataset/
└── color/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── Apple___Cedar_apple_rust/
    ├── ... (35 more disease classes)
    └── Tomato___healthy/
```

## 🚀 Training the Model

### Quick Start Training

```powershell
# Make sure you're in the project directory
cd D:\Plant-Disease-Prediction-with-CNN

# Run the training script
.venv\Scripts\python.exe train_model_local.py
```

### Training Configuration

Default parameters (can be modified in `train_model_local.py`):
- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Epochs**: 5
- **Train/Val Split**: 80/20
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy

### What Happens During Training

1. ✅ Loads and preprocesses the dataset
2. ✅ Creates train/validation split (80/20)
3. ✅ Builds CNN model architecture
4. ✅ Trains for 5 epochs (~30-60 mins)
5. ✅ Evaluates on validation set
6. ✅ Saves model to `app/trained_model/plant_disease_prediction_model.h5`
7. ✅ Saves class indices to `app/class_indices.json`
8. ✅ Generates training history plot

### Expected Output

```
✓ Training samples: 43456
✓ Validation samples: 10865
✓ Number of classes: 38
...
Epoch 5/5
1357/1357 [==============================] - 123s 91ms/step
✓ Validation Accuracy: 93.45%
🎉 TRAINING COMPLETED SUCCESSFULLY!
```

### Files Generated After Training

- `app/trained_model/plant_disease_prediction_model.h5` - Trained model (~180MB)
- `app/class_indices.json` - Class name mappings
- `training_history.png` - Accuracy/Loss plots
- `plant_disease_prediction_model.h5` - Backup model (root directory)

## 🌐 Running the Web App

### 1. Ensure Model is Trained

Verify the model file exists:
```powershell
dir app\trained_model\plant_disease_prediction_model.h5
```

### 2. Launch Streamlit App

```powershell
.venv\Scripts\python.exe -m streamlit run app\main.py
```

### 3. Open in Browser

The app will automatically open at: `http://localhost:8501`

### Using the Web Interface

1. 📤 **Upload Image**: Click "Browse files" and select a plant leaf image
2. 🖼️ **Preview**: View the uploaded image
3. 🔍 **Classify**: Click "Classify" button
4. 📊 **Results**: See the predicted disease classification

### Test with Sample Images

Use the provided test images in `test_images/`:
- `test_apple_black_rot.JPG`
- `test_blueberry_healthy.jpg`
- `test_potato_early_blight.jpg`

## 📁 Project Structure

```
Plant-Disease-Prediction-with-CNN/
│
├── app/                                    # Web application
│   ├── trained_model/
│   │   └── plant_disease_prediction_model.h5    # Trained model (generated)
│   ├── class_indices.json                  # Class mappings
│   ├── main.py                             # Streamlit app
│   ├── requirements.txt                    # App dependencies
│   └── Dockerfile                          # Docker configuration
│
├── model_training_notebook/
│   └── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb
│
├── test_images/                            # Sample test images
│   ├── test_apple_black_rot.JPG
│   ├── test_blueberry_healthy.jpg
│   └── test_potato_early_blight.jpg
│
├── train_model_local.py                    # Main training script
├── download_dataset.py                     # Dataset downloader
├── Plant_disease_PredictionCoLab.py        # Original Colab script
├── TRAINING_GUIDE.md                       # Detailed training guide
└── README.md                               # This file
```

## 🎯 Disease Classes (38 Total)

The model identifies the following plant diseases:

### Apple (4 classes)
- Apple scab
- Black rot
- Cedar apple rust
- Healthy

### Blueberry (1 class)
- Healthy

### Cherry (2 classes)
- Powdery mildew
- Healthy

### Corn/Maize (4 classes)
- Cercospora leaf spot (Gray leaf spot)
- Common rust
- Northern Leaf Blight
- Healthy

### Grape (4 classes)
- Black rot
- Esca (Black Measles)
- Leaf blight (Isariopsis Leaf Spot)
- Healthy

### Orange (1 class)
- Huanglongbing (Citrus greening)

### Peach (2 classes)
- Bacterial spot
- Healthy

### Pepper/Bell Pepper (2 classes)
- Bacterial spot
- Healthy

### Potato (3 classes)
- Early blight
- Late blight
- Healthy

### Raspberry (1 class)
- Healthy

### Soybean (1 class)
- Healthy

### Squash (1 class)
- Powdery mildew

### Strawberry (2 classes)
- Leaf scorch
- Healthy

### Tomato (10 classes)
- Bacterial spot
- Early blight
- Late blight
- Leaf Mold
- Septoria leaf spot
- Spider mites (Two-spotted spider mite)
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato mosaic virus
- Healthy

## 🏗️ Model Architecture

```
Sequential CNN Model:

Input: (224, 224, 3) RGB images

Layer 1: Conv2D(32 filters, 3x3) + ReLU
         MaxPooling2D(2x2)

Layer 2: Conv2D(64 filters, 3x3) + ReLU
         MaxPooling2D(2x2)

Layer 3: Flatten()

Layer 4: Dense(256) + ReLU

Output:  Dense(38) + Softmax

Total Parameters: ~47.8 Million
```

### Training Metrics

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Expected Validation Accuracy**: 90-95%

## 🔧 Customization

### Modify Training Parameters

Edit `train_model_local.py`:

```python
# Line 50-51: Image and batch size
img_size = 224        # Change image dimensions
batch_size = 32       # Change batch size (16 or 8 for less memory)

# Line 109: Number of epochs
epochs = 10           # Train for more epochs
```

### Add Data Augmentation

Modify the `ImageDataGenerator` (line 56):

```python
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,          # Add rotation
    width_shift_range=0.2,      # Add width shift
    height_shift_range=0.2,     # Add height shift
    horizontal_flip=True,       # Add horizontal flip
    zoom_range=0.2              # Add zoom
)
```

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size
```python
batch_size = 16  # or 8
```

### Issue: Training Too Slow

**Solutions**:
- Check if GPU is being used: Model will show "GPU Available: True"
- Reduce epochs for quick testing
- Use a smaller subset of data initially

### Issue: Kaggle API Not Working

**Solution**: Use manual download (Method 2)

### Issue: Model File Not Found

**Solution**: Ensure training completed successfully and check:
```powershell
dir app\trained_model\plant_disease_prediction_model.h5
```

### Issue: Low Accuracy

**Solutions**:
- Train for more epochs (10-20)
- Add data augmentation
- Use transfer learning (ResNet, VGG, etc.)

## 📊 Performance Tips

1. **Use GPU**: TensorFlow automatically detects and uses GPU if available
2. **Larger Batch Size**: Increase if you have more RAM/VRAM (64, 128)
3. **More Epochs**: Train longer for better accuracy
4. **Data Augmentation**: Helps prevent overfitting
5. **Transfer Learning**: Use pre-trained models for better performance

## 🚢 Deployment

### Docker Deployment

A Dockerfile is included for containerized deployment:

```bash
docker build -t plant-disease-app .
docker run -p 8501:8501 plant-disease-app
```

## 📝 Notes

- First training run takes longer due to TensorFlow compilation
- Model file is ~180MB
- Dataset is ~800MB compressed, ~2GB extracted
- Training time: 30-60 minutes on modern CPU, 5-10 minutes on GPU
- Web app loads instantly after model is trained

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- PlantVillage dataset from Kaggle
- TensorFlow and Keras communities
- Streamlit for the web framework

---

## 🚀 Quick Command Reference

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Download dataset (if kaggle.json is configured)
.venv\Scripts\python.exe download_dataset.py

# Train the model
.venv\Scripts\python.exe train_model_local.py

# Run web application
.venv\Scripts\python.exe -m streamlit run app\main.py

# Check package versions
.venv\Scripts\python.exe -m pip list
```

---

**Ready to start?** Follow the [Dataset Setup](#dataset-setup) section and then [train your model](#training-the-model)! 🌱
