import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def predict_image_class(model, img_path, class_names):
    """
    Predicts the class of an image using a trained model.

    Parameters:
    - model: Trained Keras model
    - img_path: Path to the image file
    - class_names: List of class labels

    Returns:
    - Predicted class name
    - Display the image with its prediction
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to match model's input size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image (since we rescaled the training data)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability

    # Print predicted class
    predicted_class_name = class_names[predicted_class[0]]
    print(f"Predicted class: {predicted_class_name}")

    # Rescale the image for display (multiply by 255 to get the pixel values in [0, 255])
    img_display = img_array[0] * 255  # Rescale pixel values
    img_display = img_display.astype("uint8")  # Convert to uint8 for display

    # Display the image and the prediction
    plt.imshow(img_display)
    plt.title(f"Predicted: {predicted_class_name}")
    plt.axis('off')  # Hide axes
    plt.show()

    return predicted_class_name
