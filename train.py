# Placeholder training script
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Path to your dataset folder
DATASET_PATH = "dataset/"

# Load dataset
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    image_size=(224, 224),  
    batch_size=32
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Load pretrained MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pretrained layers

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
output = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(train_ds, epochs=8)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/saved_model.keras")

print("Model saved successfully!")
