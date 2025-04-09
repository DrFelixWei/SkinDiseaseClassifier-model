## Project Description
This project focuses on training and evaluating a skin disease classification model using a transfer learning approach with the VGG19 architecture. 
The model was trained using images from the Dermnet Dataset and the Oily, Dry and Normal Skin Types Dataset from Kaggle.

The model classifies skin images into six possible classes: 
1. Normal* 
2. Acne/Rosacea
3. Eczema
4. Atopic Dermatitis
5. Psoriasis
6. Tinea
7. Melanoma
* The current model is fairly weak at classifying normal skin due to a less robust dataset of images.

## Contributors
- Felix Wei

## Hosting
1. https://skindiseaseclassifier-model.onrender.com
2. https://myskinhealth-ml-wtbt8.ondigitalocean.app

## Local Testing
1. python -m venv venv
2. venv\Scripts\activate
3. pip install -r requirements.txt
4. uvicorn server:app --reload
5. (then execute test.py)

*data is not uploaded in this repo


## 🧠 Model Overview

- **Feature Extractor**: [VGG19](https://keras.io/api/applications/vgg/#vgg19-function), pre-trained on ImageNet, used with `include_top=False`.
- **Classifier**: A simple fully connected neural network:
  - Dense(200, relu)
  - Dense(170, relu)
  - Dense(5, softmax)

---

## 🚀 Training Process (`training.py`)

### Key Steps:

1. **Data Loading**:
   - Images are read from folders.
   - Labels are assigned based on the folder name.

2. **Preprocessing**:
   - Images are resized to `180x180`.
   - Converted to RGB.
   - Labels are one-hot encoded.

3. **Feature Extraction**:
   - VGG19 is used to extract deep features (convolutional output).
   - Features are flattened before being passed to the classifier.

4. **Model Training**:
   - 3 k-folds (iterations) of random `train_test_split` to simulate cross-validation.
   - Each iteration:
     - Trains the model using extracted features.
     - Saves the best model based on validation accuracy (`model_fold_{n}.h5`).
     - Uses callbacks:
       - `ModelCheckpoint`
       - `ReduceLROnPlateau`
     - Plots accuracy/loss curves (`training_history_fold_{n}.png`).

5. **Final Model**:
   - The last trained model is saved as `model.h5`.

6. **Evaluation**:
   - Accuracy is reported for each fold.
   - Mean and standard deviation of accuracy across folds are printed.

---

## 📊 Evaluation Process (`evaluation.py`)


## API Service
The api service is hosted by running server.py

---
