import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Define disease categories
categories = [
    'Normal',
    'Acne and Rosacea Photos',
    'Eczema Photos',
    'Atopic Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Melanoma Skin Cancer Nevi and Moles',
]


# Function to load data
def data_dictionary():
    path_train = "./data/dermnet/train/"
    train_dictionary = {"image_path": [], "target": []}

    # First load general categories
    for k, category in enumerate(categories):
        path_disease_train = path_train + category
        if os.path.exists(path_disease_train):
            image_list_train = os.listdir(path_disease_train)
            for image in image_list_train:
                img_path_train = path_disease_train + "/" + image
                train_dictionary["image_path"].append(img_path_train)
                train_dictionary['target'].append(k)

    train_df = pd.DataFrame(train_dictionary)
    return train_df


# Load data
train_df = data_dictionary()
print("Data loaded successfully:")
train_df.info()

# Load and preprocess images
images = []
labels = []

for i, label in zip(train_df['image_path'], train_df['target']):
    try:
        img = cv2.imread(i)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (180, 180))
        images.append(img)
        labels.append(label)
    except Exception as e:
        print(f"Error loading image {i}: {e}")

# Convert to numpy arrays
data = np.array(images)
labels = np.array(labels)
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")


# Display sample images
def display_samples(images_list, n_samples=6):
    sample_indices = np.random.choice(len(images_list), n_samples, replace=False)
    example_list = [images_list[i] for i in sample_indices]

    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    for ax, img_array in zip(axes.ravel(), example_list):
        ax.imshow(img_array)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


display_samples(images)

# Convert labels to categorical
num_classes = len(categories)
labels_categorical = to_categorical(labels, num_classes)

# Load pre-trained VGG19 model for feature extraction
vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))

# Freeze VGG19 layers
for layer in vgg_model.layers:
    layer.trainable = False

# features = vgg_model.predict(data, batch_size=32, verbose=1)
# np.save("vgg19_features.npy", features)
# np.save("vgg19_labels.npy", labels_categorical)
# features = np.load("vgg19_features.npy")
# labels_categorical = np.load("vgg19_labels.npy")
# features = features.reshape(features.shape[0], -1)

# Setup K-fold cross validation
kf = KFold(n_splits=3)
fold_no = 1
acc_per_fold = []


# Enable early stopping
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=4,            # stop if no improvement after 4 epochs
    restore_best_weights=True  # load best weights at the end
)

# Initialize the model
def create_model():
    model = Sequential([
        Dense(200, activation='relu'),
        Dense(170, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Cross-validation training loop
for i in range(3):  # Run 3 times with different random splits (mirroring original code)
    # Split data with different random state each time
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels_categorical,
        test_size=0.2,
        random_state=np.random.randint(1, 1000, 1)[0]
    )

    print(f"Fold {fold_no}, Training samples: {x_train.shape[0]}, Testing samples: {x_test.shape[0]}")

    # Extract features using VGG19
    print("Extracting features using VGG19...")
    features_train = vgg_model.predict(x_train)
    features_test = vgg_model.predict(x_test)

    # Reshape features
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    x_train_features = features_train.reshape(num_train, -1)
    x_test_features = features_test.reshape(num_test, -1)

    # Create fresh model for each fold
    model = create_model()

    # Create callbacks
    checkpoint = ModelCheckpoint(f'model_fold_{fold_no}.h5', save_best_only=True, monitor='val_accuracy')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1)

    # Train the model
    print(f"Training fold {fold_no}...")
    history = model.fit(
        x_train_features,
        y_train,
        epochs=25,
        batch_size=32,
        validation_data=(x_test_features, y_test),
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1
    )

    # Evaluate on test set
    scores = model.evaluate(x_test_features, y_test, verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)

    # Plot training history for this fold
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Fold {fold_no} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Fold {fold_no} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')

    plt.tight_layout()
    plt.savefig(f'training_history_fold_{fold_no}.png')
    plt.show()

    fold_no += 1

# Save the last model as the final model
model.save('model.h5')
print("Final model saved as model.h5")

# Show average performance across folds
print('\nAverage accuracy across all folds: {:.2f}%'.format(np.mean(acc_per_fold)))
print('Standard deviation: {:.2f}%'.format(np.std(acc_per_fold)))

print("Training completed!")