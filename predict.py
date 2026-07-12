# Placeholder prediction script
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model/saved_model.keras"
TEST_IMAGE = "test_224.jpg"   # Change this to your test image

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load and preprocess image
img = image.load_img(TEST_IMAGE, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
predictions = model.predict(img_array)
class_index = np.argmax(predictions)
confidence = predictions[0][class_index]

print(f"Predicted Class: {class_index}")
print(f"Confidence: {confidence:.2f}")
