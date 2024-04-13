import tensorflow as tf
from tensorflow.keras.preprocessing import image  # Importing from tf.keras
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

# Load the saved model
loaded_model = tf.keras.models.load_model('Model/is_tree_model.keras')

TREE_TYPES = ['Dood', 'Goed', 'Matig', 'Redelijk', 'Slecht', 'Zeer Slecht']
FOLDER_PATH_INPUT = 'RetrieveGSV/images/'

for tree_path in TREE_TYPES:
    input_tree_type = os.path.join(FOLDER_PATH_INPUT, tree_path)

    # Iterate over each file in the directory
    for filename in os.listdir(input_tree_type):
        print(input_tree_type)
        image_input_path = os.path.join(input_tree_type, filename)
        
        if os.path.isfile(image_input_path):
            # Load the image using TensorFlow's Keras preprocessing
            img = image.load_img(image_input_path, target_size=(200, 200))
            X = image.img_to_array(img)
            X = np.expand_dims(X, axis=0)

            # Predict the class of the image
            prediction = loaded_model.predict(X)

            if prediction == 0:
                # If classified as not_tree, delete the image
                os.remove(image_input_path)

