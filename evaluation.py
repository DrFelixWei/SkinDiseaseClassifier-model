import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# Define disease categories - same as in training.py
categories = [
    'Normal',
    'Acne and Rosacea Photos',
    'Eczema Photos',
    'Atopic Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Melanoma Skin Cancer Nevi and Moles',
]

# Simplified class names for display
class_names = [
    "Normal",
    "Acne",
    "Eczema",
    "Atopic Dermatitis",
    "Tinea",
    "Melanoma",
]

# Load the VGG19 model for feature extraction
vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))

# Load the trained model
model = tf.keras.models.load_model('model.h5')
print("Model loaded successfully!")


def predict_skin_disease(image_path):
    """
    Predicts skin disease from an image

    Args:
        image_path (str): Path to the image file

    Returns:
        tuple: (predicted class name, confidence percentage)
    """
    # Load and preprocess image - exactly matching the original code's approach
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not load image", 0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (180, 180))
    # Don't normalize here as the original doesn't

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Extract features using VGG19
    img_features = vgg_model.predict(img)
    img_features = img_features.reshape(1, -1)

    # Make prediction
    pred = model.predict(img_features)[0]
    predicted_class_index = np.argmax(pred)
    predicted_class_name = class_names[predicted_class_index]
    confidence = pred[predicted_class_index] * 100

    return predicted_class_name, confidence


def evaluate_test_set(test_dir):
    """
    Evaluates the model on a test set

    Args:
        test_dir (str): Directory containing test images organized in class folders
    """
    true_labels = []
    predicted_labels = []
    confidences = []
    image_paths = []

    for i, category in enumerate(categories):
        category_dir = os.path.join(test_dir, category)
        if not os.path.exists(category_dir):
            print(f"Warning: {category_dir} does not exist")
            continue

        image_files = os.listdir(category_dir)
        for image_file in image_files[:50]:  # Limit to 50 images per class for speed
            image_path = os.path.join(category_dir, image_file)
            try:
                prediction, confidence = predict_skin_disease(image_path)
                pred_index = class_names.index(prediction)

                true_labels.append(i)
                predicted_labels.append(pred_index)
                confidences.append(confidence)
                image_paths.append(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=class_names)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.title("Skin Disease Classification - Confusion Matrix")
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(report)

    # Create a dataframe of results for further analysis
    results_df = pd.DataFrame({
        'image_path': image_paths,
        'true_label': [class_names[i] for i in true_labels],
        'predicted_label': [class_names[i] for i in predicted_labels],
        'confidence': confidences,
        'correct': [t == p for t, p in zip(true_labels, predicted_labels)]
    })

    # Save results to CSV
    results_df.to_csv('evaluation_results.csv', index=False)

    # Print overall accuracy
    accuracy = (results_df['correct'].sum() / len(results_df)) * 100
    print(f"\nOverall accuracy: {accuracy:.2f}%")

    # Show some misclassifications
    misclassifications = results_df[~results_df['correct']].sort_values('confidence', ascending=False)
    print(f"\nTop misclassifications (highest confidence incorrect predictions):")
    print(misclassifications.head())

    return results_df


def visualize_predictions(results_df, num_samples=10):
    """
    Visualizes predictions from the evaluation results

    Args:
        results_df (pd.DataFrame): DataFrame with evaluation results
        num_samples (int): Number of images to display
    """
    # Select samples - mix of correct and incorrect predictions
    correct_samples = results_df[results_df['correct']].sample(
        min(num_samples // 2, len(results_df[results_df['correct']])))
    incorrect_samples = results_df[~results_df['correct']].sample(
        min(num_samples - len(correct_samples), len(results_df[~results_df['correct']])))

    samples = pd.concat([correct_samples, incorrect_samples]).sample(frac=1)

    # Display images with predictions
    rows = int(np.ceil(len(samples) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5))
    axes = axes.flatten()

    for i, (_, row) in enumerate(samples.iterrows()):
        img = cv2.imread(row['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[i].imshow(img)
        color = 'green' if row['correct'] else 'red'
        axes[i].set_title(f"True: {row['true_label']}\nPred: {row['predicted_label']}\nConf: {row['confidence']:.1f}%",
                          color=color)
        axes[i].axis('off')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()


if __name__ == "__main__":
    # Demo on a single test image
    test_image = "./data/dermnet/test/Acne and Rosacea Photos/PerioralDermEye.jpg"
    if os.path.exists(test_image):
        prediction, confidence = predict_skin_disease(test_image)
        print(f"Prediction for test image: {prediction} (Confidence: {confidence:.2f}%)")

        # Display the image and prediction
        img = cv2.imread(test_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Prediction: {prediction} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()
    else:
        print(f"Test image not found: {test_image}")

    # Run evaluation on test set
    print("\nRunning evaluation on test set...")
    results = evaluate_test_set("./data/dermnet/test")

    # Visualize some predictions
    visualize_predictions(results)