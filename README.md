# Image Classifier with Transfer Learning

This project trains a simple image classifier using TensorFlow and Keras transfer learning. It uses a pretrained MobileNetV2 model as a feature extractor and fine-tunes a small classification head on a custom image dataset.

## Overview

The workflow is:

1. Resize all images in the dataset to 224x224 pixels.
2. Load the images from subfolders in the dataset directory.
3. Use a pretrained MobileNetV2 model (trained on ImageNet) to extract features.
4. Add a new classification head on top of the pretrained network.
5. Train the classifier for a few epochs.
6. Save the trained model and use it to predict new images.

---

## Model Used

The project uses MobileNetV2 as the base convolutional neural network.

### Why MobileNetV2?

- It is a lightweight CNN architecture that performs well on image classification tasks.
- It is pretrained on ImageNet, which provides strong feature extraction capabilities.
- It is efficient and works well for transfer learning on smaller custom datasets.

### Architecture Details

The model is built as follows:

- Base model: MobileNetV2
- Pretrained weights: ImageNet
- Input shape: 224x224x3
- `include_top=False` to remove the original ImageNet classification head
- Added layers:
  - `GlobalAveragePooling2D()`
  - `Dense(128, activation="relu")`
  - `Dense(num_classes, activation="softmax")`

This approach is a common transfer learning strategy: the pretrained convolutional layers are kept frozen, and only the newly added classification layers are trained on the custom dataset.

---

## Dataset Structure

The training code expects the dataset to be organized as folders where each folder name is a class name.

```text
dataset/
  cats/
  dogs/
```

Each folder contains images belonging to that class.

The code uses TensorFlow's `image_dataset_from_directory()` to automatically discover all classes from the subfolders.

---

## Training Process

### 1. Image Resizing

Before training, images are resized to 224x224 pixels using Pillow.

The scripts used are:

- `resize_dataset.py` - resizes all images in the dataset folder
- `resizeimage.py` - resizes a single image to test input size

### 2. Loading the Dataset

The training script loads data from the `dataset/` directory using:

```python
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    image_size=(224, 224),
    batch_size=32
)
```

This automatically creates a dataset and infers the class names from the folder names.

### 3. Building the Model

The model is created using MobileNetV2 as a pretrained base:

```python
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False
```

### 4. Adding the Classification Head

A small custom head is added on top of the pretrained base:

```python
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
output = Dense(len(class_names), activation="softmax")(x)
```

### 5. Compiling and Training

The model is compiled with:

- Optimizer: Adam
- Loss: `sparse_categorical_crossentropy`
- Metric: Accuracy

Then it is trained for 8 epochs using:

```python
history = model.fit(train_ds, epochs=8)
```

### 6. Saving the Model

After training, the model is saved as:

```text
model/saved_model.keras
```

---

## How Classification Works

The prediction workflow is implemented in `predict.py`.

### Steps

1. Load the saved trained model from `model/saved_model.keras`.
2. Load an input image.
3. Resize it to 224x224 pixels.
4. Convert the image to a NumPy array.
5. Expand the array to create a batch dimension.
6. Normalize pixel values to the range `[0, 1]`.
7. Pass the image through the model.
8. Take the class with the highest probability as the predicted class.

### Layer Summary

- Input layer: 224x224x3 image
- MobileNetV2 pretrained convolutional layers
- GlobalAveragePooling2D layer
- Dense hidden layer with ReLU activation
- Output layer with softmax for class probabilities

### Example Prediction Flow

```python
img = image.load_img(TEST_IMAGE, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predictions = model.predict(img_array)
class_index = np.argmax(predictions)
confidence = predictions[0][class_index]
```

The output is:

- `Predicted Class`: index of the predicted class
- `Confidence`: probability score for that prediction

> In the current script, the prediction prints the class index. If you want human-readable labels, you can map the index to the class names using `class_names` from the training step.

---

## Project Files

- `train.py` – trains the classifier and saves the model
- `predict.py` – loads the trained model and predicts a class for a single image
- `resize_dataset.py` – resizes all images in the dataset folder
- `resizeimage.py` – resizes one image for testing
- `model/saved_model.keras` – the saved trained model

---

## Requirements

Install the following dependencies:

```bash
pip install tensorflow pillow numpy
```

If you are using a virtual environment, activate it first and then install the packages.

---

## How to Run

### 1. Resize the Dataset

```bash
python resize_dataset.py
```

### 2. Train the Model

```bash
python train.py
```

### 3. Predict a New Image

Edit `predict.py` and set `TEST_IMAGE` to the path of your image, then run:

```bash
python predict.py
```

---

## Example Output

```text
Predicted Class: 0
Confidence: 0.97
```

## Training Logs and Outputs

A sample training log file is included in this repository as [trainingLogs.txt](trainingLogs.txt).

You can also save future logs by running:

```bash
python train.py > training_logs.txt 2>&1
```

To save a prediction result for a sample image, you can write it to a text file:

```python
with open("prediction_output.txt", "w") as f:
    f.write(f"Predicted Class: {class_index}\n")
    f.write(f"Confidence: {confidence:.2f}\n")
```

---

## Push the Project to GitHub

Follow these steps to upload this project to your GitHub profile.

### 1. Initialize Git in the project folder

```bash
git init
```

### 2. Add the files to the repository

```bash
git add .
```

### 3. Commit the files

```bash
git commit -m "Initial commit"
```

### 4. Create a new repository on GitHub

Go to GitHub and create a new repository.

- Repository name: for example, `image-classifier`
- Keep it public or private as desired
- Do not initialize it with a README if you already have one

### 5. Connect your local repo to GitHub

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/image-classifier.git
```

### 6. Push the code to GitHub

```bash
git push -u origin main
```

### 7. Update later

After changes, use:

```bash
git add .
git commit -m "Update model and README"
git push
```

> If you already created a GitHub repo and want to link it, replace the remote URL with your repository URL.

---

## Notes

- The model is a transfer learning model, so it is much faster to train than building a CNN from scratch.
- The current project uses a small number of training epochs; increasing `epochs` can improve accuracy.
- For better results, use a larger and more diverse dataset.
- For a production-ready version, you may want to add:
  - validation split
  - data augmentation
  - class name mapping in predictions
  - confusion matrix and accuracy plots

---

## Summary

This project demonstrates how to:

- use MobileNetV2 for image classification,
- train a model with transfer learning,
- save the trained model,
- and run inference on a single image.

It is a simple and effective starting point for building a custom image classifier on GitHub.
