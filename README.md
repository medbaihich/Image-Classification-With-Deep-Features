# Image Classification: Raw Pixels vs. Deep Features

## 📌 Project Overview

This project explores two different approaches to building an image classifier using **Turi Create**. The goal is to classify images into four distinct categories: **Bird, Cat, Dog, and Automobile**.

The project demonstrates the power of **Transfer Learning** by comparing:

1. **Raw Pixel Classification:** A baseline model using raw RGB values.
2. **Deep Feature Classification:** A robust model using high-level semantic features extracted from a pre-trained Convolutional Neural Network (CNN).

## 📂 Dataset

The project utilizes the CIFAR-10 subset provided via Turi Create SFrames (`image_train_data/` and `image_test_data/`).

* **Total Training Images:** ~2,000
* **Target Labels:** `bird`, `cat`, `dog`, `automobile`
* **Data Features:**
* `image_array`: Raw pixel values (flattened vector).
* `deep_features`: A numeric vector representing semantic features extracted from the image.



## 🚀 Methodology

### Approach 1: The "Raw Pixel" Model

This model trains a Logistic Classifier directly on the `image_array`.

* **Concept:** The model tries to learn patterns based solely on specific color values at specific coordinate locations.
* **Limitation:** It is highly sensitive to rotation, lighting, and position (e.g., a blue car facing left looks completely different in raw pixels than a blue car facing right).

### Approach 2: The "Deep Features" Model

This model trains a Logistic Classifier on `deep_features`.

* **Concept:** Uses a pre-trained neural network to "see" the image first. The network converts the image into a vector of "deep features" (detecting edges, textures, and shapes) before classification.
* **Advantage:** Provides robustness against simple variations in the image, capturing the *semantic* content rather than just pixel grids.

## 📊 Performance Results

The models were evaluated on the test dataset. The results demonstrate a significant performance gap:

| Model Type | Feature Used | Accuracy | Conclusion |
| --- | --- | --- | --- |
| **Raw Pixels** | `image_array` | **47.0%** | Poor generalization. Struggles to distinguish classes. |
| **Deep Features** | `deep_features` | **79.2%** | **Superior performance.** Successfully captures object identity. |

## 🛠 Installation & Usage

### Prerequisites

* Python 3.8+
* Turi Create (`pip install turicreate`)

### Running the Code

1. **Load the Data:**
```python
import turicreate as tc
image_train = tc.SFrame('image_train_data/')
image_test = tc.SFrame('image_test_data/')

```


2. **Train Raw Pixel Model:**
```python
raw_pixel_model = tc.logistic_classifier.create(
    image_train,
    target='label',
    features=['image_array']
)

```


3. **Train Deep Features Model:**
```python
deep_features_model = tc.logistic_classifier.create(
    image_train,
    target='label',
    features=['deep_features']
)

```


4. **Evaluate:**
```python
raw_pixel_model.evaluate(image_test)
deep_features_model.evaluate(image_test)

```

**Author:** BAIHICH Mohamed
