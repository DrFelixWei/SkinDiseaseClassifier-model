## Project Description
This project focuses on training and evaluating a skin disease classification model using a transfer learning approach with the VGG19 architecture. 
The model classifies skin images into five categories based on data from the DermNet dataset.

## Contributors
- Felix Wei

## Hosting


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
   - 3 iterations of random `train_test_split` to simulate cross-validation.
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

This script duplicates the logic of `training.py`. You can use it independently to:

- Validate data loading and preprocessing.
- Retrain the model for additional verification or testing.
- Reproduce the results using the same methodology.

---
