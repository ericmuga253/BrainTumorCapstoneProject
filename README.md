

# Brain MRI Images for Brain Tumor Detection
![Screenshot 2024-11-08 125002](https://github.com/user-attachments/assets/851ab050-7519-46a7-b476-2d86cdf0bbb6)


This project utilizes a convolutional neural network (CNN) model, specifically a modified VGG16 architecture, to classify brain MRI images as either containing a tumor or not. The project aims to leverage deep learning to enable faster, more accurate brain tumor detection, supporting early intervention and reducing diagnostic errors. 

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training and Evaluation](#training-and-evaluation)
7. [Error Analysis](#error-analysis)
8. [Results](#results)
9. [License](#license)

---

## Project Overview

Brain tumors are life-threatening and account for a significant number of cancer-related deaths globally. Early detection can significantly improve treatment outcomes. Traditional diagnostic methods are time-consuming and prone to errors. This project demonstrates the use of CNNs, specifically a modified VGG16 model, to accurately classify brain MRI images for tumor detection.

---

## Objective

Develop a deep learning model to:
- Accurately classify brain MRI images as either tumor-positive or tumor-negative.
- Enable faster and more reliable diagnostics.
- Support early intervention, improve patient outcomes, and reduce healthcare burdens.

---

## Dataset

The dataset, **Brain MRI Images for Brain Tumor Detection**, is sourced from Kaggle:
- [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

### Dataset Details:
- **Classes**:
  - `Yes`: Tumor present (~1,500 images).
  - `No`: Tumor absent (~1,400 images).
- **Total Images**: 2,903.
- **Preprocessing**:
  - Images resized to **224x224 pixels** for VGG16 compatibility.
  - Converted to **RGB format**.
- **Balanced Dataset**: Equal representation of both classes for reliable evaluation.

---

## Data Preprocessing

- **Loading Data**: Images are organized into `yes` (tumor) and `no` (no tumor) categories.
- **Resizing**: All images are resized to **224x224 pixels**.
- **Normalization**: Pixel values are scaled to the range [0, 1] for faster convergence.
- **Label Encoding**: Labels (`yes` → 1, `no` → 0) are converted to numerical format.
- **Train-Test Split**: Data is split into **67% training** and **33% testing** for evaluation.

---

## Model Architecture

### VGG16 Base Model:
- Pre-trained on ImageNet.
- Top layers excluded for feature extraction.

### Custom Layers:
- **Global Average Pooling**: Reduces dimensionality.
- **Dense Layers**: With **ReLU activation** for learning complex patterns.
- **Dropout**: Regularization to prevent overfitting.
- **Softmax Output Layer**: Binary classification (tumor/no tumor).

### Regularization Techniques:
- **L2 Regularization**: Penalizes large weights.
- **Lower Learning Rate**: Ensures smoother convergence.

---

## Training and Evaluation

- The model is trained for **5 epochs** with a validation split.
- Early results showed a divergence between training and validation accuracy due to overfitting. Adjustments such as increased L2 regularization strength and reduced learning rate were applied.

---

## Error Analysis
![Screenshot 2024-11-12 142809](https://github.com/user-attachments/assets/520ba04a-0c26-438f-a6a3-378c4c5b3e24)


### Insights:
- KDE plots of residuals revealed:
  - Broader confidence distribution (0.4–0.6) for incorrect predictions, indicating uncertainty/confusion.
  - Incorrect predictions should ideally have confidence closer to 0 to reflect model uncertainty.

### Adjustments:
- Increased L2 regularization strength.
- Reduced learning rate to improve generalization.

---

## Results

### Initial Results:
![Screenshot 2024-11-27 113836](https://github.com/user-attachments/assets/b336d01e-7f80-4a10-962e-a2d7099e69a7)

- **Training Accuracy**: >90%, showing effective learning.
- **Validation Accuracy**: Initially dipped but steadily increased to ~85%, indicating good generalization.
- Early overfitting was mitigated through regularization and preprocessing adjustments.

### Adjusted Results:
![Screenshot 2024-11-27 114252](https://github.com/user-attachments/assets/9feeca97-4798-4873-8639-eccef7d36bb3)

- **Training Accuracy**: Consistently above 90%.
- **Validation Accuracy**: Reached ~85%, with a small gap between training and validation metrics.

---

### Final Interpretations:
- The model demonstrates effective learning with strong generalization.
- Error analysis and regularization adjustments reduced overfitting and improved validation performance.
- The model is ready for further optimization or deployment in diagnostic pipelines.

---

## License

This project adheres to the licensing terms of the Kaggle dataset and any pre-trained models used (e.g., VGG16 from Keras).

---


