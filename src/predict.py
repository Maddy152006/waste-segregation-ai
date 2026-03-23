import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = load_model("../model/final_model.keras")

# Class labels
class_names = ['metal', 'paper', 'plastic', 'trash']

def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]

# Test example
if __name__ == "__main__":
    result = predict("../test.jpg")  # add a test image
    print("Prediction:", result)