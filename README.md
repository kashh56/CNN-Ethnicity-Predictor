CNN Ethnicity Predictor
=======================

**Project by: Akash Anandani**

🔍 About the Project
--------------------

This project demonstrates the use of deep learning for ethnicity prediction. I fine-tuned the powerful `InceptionV3` model to predict the ethnicity of a person based on their image.

Users can either upload an image or use their webcam to predict the ethnicity of the person in the image. The application also shows detailed information about the predicted ethnicity.

🧠 What I Did
-------------

To achieve this, I used **InceptionV3**, a pre-trained CNN model developed by Google. Instead of retraining the entire model, I focused on fine-tuning only the last layer and added a custom `ANN` (Artificial Neural Network) layer on top for better performance on the ethnicity classification task.

This allowed me to take advantage of InceptionV3's powerful image recognition features, while tailoring it specifically for my custom dataset.

💻 Tech Stack
-------------

*   **Deep Learning Frameworks:** TensorFlow, Keras
*   **Pre-trained Model:** InceptionV3 (Fine-Tuned)
*   **Frontend Framework:** Streamlit
*   **Deployment:** Streamlit Cloud

📝 How to Run the Project
-------------------------

Follow the steps below to get the project up and running on your local machine:

1.  Clone this repository:
    
        git clone https://github.com/kashh56/CNN-Ethnicity-Predictor.git
    
2.  Navigate into the project directory:
    
        cd CNN-Ethnicity-Predictor
    
3.  Install the necessary dependencies:
    
        pip install -r requirements.txt
    
4.  Run the app using Streamlit:
    
        streamlit run app.py
    

🌍 Live Demo
------------

You can check out the live version of the app here: [Live App](https://cnn-ethnicity-predictor-akash.streamlit.app/)

💬 Contributions
----------------

Feel free to open an issue or submit a pull request if you find any bugs or have suggestions for improvements!

🚀 Future Enhancements
----------------------

*   Expand the dataset for better accuracy
*   Improve the frontend design with more interactive features
*   Deploy the model on a larger scale with more optimized infrastructure

📄 License
----------

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
