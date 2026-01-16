# Plant Disease Prediction - Setup and Training Guide

## 📋 Project Overview
This project uses a Convolutional Neural Network (CNN) to classify plant diseases from leaf images. It can identify 38 different plant disease classes.

## 🔧 Prerequisites

### 1. Python Environment
✅ Already configured: Python 3.11.6 virtual environment

### 2. Required Packages
✅ Already installed:
- tensorflow
- numpy
- matplotlib
- pillow
- kaggle
- streamlit (for web app)

## 📥 Dataset Download Options

### Option 1: Using Kaggle API (Recommended)

1. **Get Kaggle API Credentials:**
   - Go to https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json` file

2. **Place the kaggle.json file:**
   - Windows: `C:\Users\YourUsername\.kaggle\kaggle.json`
   - Or place it in the project root directory

3. **Run the download script:**
   ```powershell
   .venv\Scripts\python.exe download_dataset.py
   ```

### Option 2: Manual Download

1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download the dataset (about 800MB)
3. Extract it to the project directory
4. Rename the folder to `plantvillage_dataset`
5. Ensure the structure is: `plantvillage_dataset/color/`

### Option 3: Use Alternative Dataset

You can use any plant disease dataset with the following structure:
```
your_dataset/
├── Class1_Name/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Class2_Name/
│   └── ...
└── ...
```

Update the `base_dir` variable in `train_model_local.py` to point to your dataset.

## 🚀 Training the Model

### Step 1: Ensure Dataset is Ready
Check that the dataset exists:
```powershell
dir plantvillage_dataset\color
```

You should see 38 folders (one for each disease class).

### Step 2: Run Training Script
```powershell
.venv\Scripts\python.exe train_model_local.py
```

**Training Parameters:**
- Image Size: 224x224
- Batch Size: 32
- Epochs: 5
- Train/Validation Split: 80/20

**Expected Output:**
- Training will take 30-60 minutes depending on your hardware
- Model will be saved to `app/trained_model/plant_disease_prediction_model.h5`
- Class indices saved to `app/class_indices.json`
- Training history plot saved as `training_history.png`

### Step 3: Verify Model Files
After training, check that these files exist:
```powershell
dir app\trained_model\plant_disease_prediction_model.h5
dir app\class_indices.json
```

## 🌐 Running the Web Application

### Install Streamlit (if not already installed)
```powershell
.venv\Scripts\python.exe -m pip install streamlit
```

### Launch the Web App
```powershell
.venv\Scripts\python.exe -m streamlit run app\main.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Web App
1. Click "Browse files" to upload a plant leaf image
2. Click "Classify" button
3. View the predicted disease classification

## 🧪 Testing with Sample Images

The project includes test images in the `test_images/` folder:
- `test_apple_black_rot.JPG`
- `test_blueberry_healthy.jpg`
- `test_potato_early_blight.jpg`

Use these to test the web app after training.

## 📊 Model Architecture

```
Layer (type)                Output Shape              Params
Conv2D                      (None, 222, 222, 32)      896
MaxPooling2D                (None, 111, 111, 32)      0
Conv2D                      (None, 109, 109, 64)      18,496
MaxPooling2D                (None, 54, 54, 64)        0
Flatten                     (None, 186,624)           0
Dense                       (None, 256)               47,776,000
Dense                       (None, 38)                9,766
```

Total params: 47,805,158

## 🎯 38 Disease Classes

The model can identify:
- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Powdery mildew, Healthy
- **Corn**: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- **Grape**: Black rot, Esca, Leaf blight, Healthy
- **Orange**: Huanglongbing (Citrus greening)
- **Peach**: Bacterial spot, Healthy
- **Pepper**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery mildew
- **Strawberry**: Leaf scorch, Healthy
- **Tomato**: 10 different diseases and healthy

## 🔍 Troubleshooting

### Issue: Kaggle API not working
**Solution**: Download dataset manually (Option 2 above)

### Issue: Out of memory during training
**Solution**: Reduce batch_size in `train_model_local.py` (line 51):
```python
batch_size = 16  # or even 8
```

### Issue: Training is too slow
**Solutions**:
- Reduce number of epochs (line 109)
- Reduce image size (line 50)
- Use GPU if available (TensorFlow will automatically use it)

### Issue: Model accuracy is low
**Solutions**:
- Train for more epochs
- Add data augmentation
- Use a pre-trained model (transfer learning)

### Issue: Web app can't find model
**Solution**: Ensure model file exists at:
```
app/trained_model/plant_disease_prediction_model.h5
```

## 📁 Project Structure

```
Plant-Disease-Prediction-with-CNN/
├── app/
│   ├── trained_model/
│   │   └── plant_disease_prediction_model.h5  (generated after training)
│   ├── class_indices.json
│   ├── main.py                                (Streamlit web app)
│   ├── requirements.txt
│   └── Dockerfile
├── test_images/                               (sample test images)
├── model_training_notebook/
│   └── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb
├── train_model_local.py                       (training script - NEW)
├── download_dataset.py                        (dataset downloader - NEW)
└── README.md
```

## 🎉 Quick Start (Summary)

1. **Download dataset** (using Kaggle API or manually)
2. **Train model**: `.venv\Scripts\python.exe train_model_local.py`
3. **Run web app**: `.venv\Scripts\python.exe -m streamlit run app\main.py`
4. **Test** with images from `test_images/` folder

## 📝 Notes

- First training will take longer as TensorFlow compiles operations
- GPU acceleration will significantly speed up training
- The model file (~180MB) will be created after training
- You can modify hyperparameters in `train_model_local.py` for experimentation

## 🆘 Need Help?

If you encounter any issues:
1. Check that all required files exist
2. Verify Python environment is activated
3. Ensure dataset path is correct
4. Check error messages carefully
5. Make sure you have enough disk space (~3GB for dataset + model)
