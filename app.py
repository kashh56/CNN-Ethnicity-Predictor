import streamlit as st
from PIL import Image
import io
from utils import icon
import time
import random
from tensorflow.keras.models import load_model  
from predict import predict_image_class  
import requests
import os



# Page Configuration
st.set_page_config(page_title="Ethnicity Detection Classifier",
                   page_icon=":bridge_at_night:",
                   layout="wide")

MODEL_URL = "https://github.com/kashh56/CNN-Ethnicity-Predictor/releases/download/v1.0.0/fine_tuned_InceptionV3_model.h5"
MODEL_PATH = "model_file.h5"

# Fun Facts List
fun_facts = [
    "🔥 Deep learning was inspired by the human brain!",
    "📊 Overfitting is like memorizing answers instead of understanding concepts.",
    "🎯 The first CNN, LeNet-5, was created in 1998!",
    "🤖 The term 'Artificial Intelligence' was first coined in 1956!",
    "⚡ GPUs revolutionized deep learning by making training much faster!",
    "🧠 Neural networks were first proposed in 1943 but became popular only in the 21st century!",
    "🔍 The Transformer architecture, used in ChatGPT, was introduced in 2017!",
    "🌍 AI can now generate realistic human faces that don’t even exist!",
    "🚀 Google's AlphaGo defeated a human world champion in the game of Go in 2016!"
]

def download_model():
    """Download model with a progress bar and fun facts."""
    if os.path.exists(MODEL_PATH):
        return  # Skip download if file already exists

    progress = st.progress(0)  # Progress bar
    fact_display = st.empty()   # Placeholder for fun facts
    percentage_display = st.empty()  # Placeholder for percentage

    response = requests.get(MODEL_URL, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        last_fact_time = time.time()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # Update progress bar with percentage
                    percentage = min(100, int((downloaded_size / total_size) * 100))
                    progress.progress(percentage)
                    percentage_display.text(f"📦 Downloading model... {percentage}%")

                    # Show a fun fact every 5 seconds
                    if time.time() - last_fact_time >= 5:
                        fact_display.markdown(f"**💡 Fun Fact:** {random.choice(fun_facts)}")
                        last_fact_time = time.time()

        st.success("✅ Model downloaded successfully!")
    else:
        st.error(f"❌ Failed to download the model. HTTP Status Code: {response.status_code}")

@st.cache_resource
def get_model():
    """Download and load the model only once per session."""
    download_model()

    # Load model
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load the model: {str(e)}")
        return None

# Load model only once (cached)
model = get_model()

if model:
    st.success("🎯 Model loaded successfully!")
else:
    st.error("❌ Model could not be loaded. Try restarting the app.")


# icon.show_icon(":foggy:")
st.markdown("# :rainbow[Ethnicity Detection using Images]")




# Sidebar for Information with images
with st.sidebar:

    st.image("images\logo.webp", width=350) 

    st.header(":rainbow[About the App]")
    st.markdown("""
    This web application uses a **fine-tuned InceptionV3 model** to predict the ethnicity of a given image. The model has been trained on seven ethnicity categories:  
    - Black  
    - East Asian  
    - Indian  
    - Latino_Hispanic  
    - Middle Eastern  
    - Southeast Asian  
    - White  

    The model is fine-tuned on top of a pre-trained architecture (InceptionV3), which helps in extracting important features from the images. Fine-tuning is the process of adjusting the model's weights for your specific task, allowing it to specialize in ethnicity classification. 
    """)
    # Displaying sample images for each ethnicity in the sidebar
    st.subheader(":rainbow[Sample Images of Different Ethnicities]")

    ethnicity_images = {
        'Black': 'images/black_sample.jpg',
        'East Asian': 'images/east_asian_sample.jpg',
        'Indian': 'images/indian_sample.jpg',
        'Latino_Hispanic': 'images/latino_sample.jpg',
        'Middle Eastern': 'images/middle_eastern_sample.jpg',
        'Southeast Asian': 'images/southeast_asian_sample.jpg',
        'White': 'images/white_sample.jpg'
    }
    
    for ethnicity, img_path in ethnicity_images.items():
        st.image(img_path, caption=ethnicity, width=150)


# Information about ethnicities after prediction
ethnicity_info = {
    'Black': {
        'origin': "🌍 Black people primarily originate from Sub-Saharan Africa, home to ancient kingdoms like Mali, Ghana, and Songhai. African heritage is deeply rooted in storytelling, music, and community.",
        'food': "🍛 A fusion of bold flavors! Dishes include Jollof rice (West Africa), injera (Ethiopia), fufu with soup, plantains, and suya—a spicy grilled meat delicacy.",
        'clothing': "👗 Vibrant patterns and textiles! Traditional clothing includes Dashikis, Kente cloth (Ghana), Ankara prints, and beaded jewelry reflecting deep cultural symbolism.",
        'culture': "🥁 A legacy of rhythm and soul! African culture is rich in drumming, storytelling, and dance forms like Afrobeat, Amapiano, and traditional tribal dances."
    },
    'East Asian': {
        'origin': "🏯 East Asians come from China, Japan, Korea, and Mongolia, with ancient civilizations that contributed to philosophy, martial arts, and tea culture.",
        'food': "🍣 A balance of flavors! Sushi, dim sum, ramen, kimchi, and Peking duck are iconic. Tea ceremonies and chopstick etiquette are also integral.",
        'clothing': "👘 Timeless elegance! Traditional outfits include the Chinese Cheongsam (Qipao), Japanese Kimono, Korean Hanbok, and Mongolian Deel.",
        'culture': "🎎 Harmony and discipline! Confucian values emphasize respect for elders, calligraphy, traditional painting, and martial arts like Kung Fu and Taekwondo."
    },
    'Indian': {
        'origin': "🕌 The Indian subcontinent boasts one of the oldest civilizations, the Indus Valley Civilization, and is home to multiple religions like Hinduism, Buddhism, Sikhism, and Jainism.",
        'food': "🍛 Spices and flavors galore! Indian cuisine includes curry, biryani, dosa, paneer dishes, samosas, and sweets like Gulab Jamun. Chai (tea) is a daily staple.",
        'clothing': "👗 Grace and tradition! Sarees, Lehengas, and Salwar Kameez for women, while men wear Kurta-Pajama and Sherwanis, often adorned with intricate embroidery.",
        'culture': "🪷 A land of festivals! India is known for Diwali, Holi, Navratri, and Eid celebrations, alongside Bollywood, classical dance forms like Bharatanatyam, and yoga."
    },
    'Latino_Hispanic': {
        'origin': "🎭 Latino culture is shaped by indigenous civilizations (Aztecs, Incas, Mayans) and European influence, especially from Spain and Portugal.",
        'food': "🌮 Bold and flavorful! Popular dishes include tacos, empanadas, ceviche, feijoada (Brazil), and churros. Corn, beans, and plantains are staples.",
        'clothing': "💃 Passionate and colorful! Flamenco dresses (Spain), Ponchos (Andes), Guayaberas (Caribbean), and embroidered dresses reflect vibrant heritage.",
        'culture': "🎶 Fiesta spirit! Latin America is known for its love of music and dance—Salsa, Tango, Reggaeton, and Mariachi music bring communities together."
    },
    'Middle Eastern': {
        'origin': "🏜️ The Middle East is the birthplace of ancient civilizations like Mesopotamia and is home to the three major Abrahamic religions—Islam, Christianity, and Judaism.",
        'food': "🍢 Aromatic delights! Hummus, kebabs, shawarma, baklava, and mansaf are famous dishes. Spices like sumac, saffron, and za'atar are widely used.",
        'clothing': "🧕 Modesty with elegance! Traditional garments include the Abaya, Thobe, Keffiyeh, and Jalabiya, often made of flowing fabrics and intricate embroidery.",
        'culture': "🎤 Poetry and storytelling! Middle Eastern culture values hospitality, traditional calligraphy, oud music, and the deep influence of Arabian Nights tales."
    },
    'Southeast Asian': {
        'origin': "🏝️ Southeast Asia is a diverse region influenced by Hindu, Buddhist, and Islamic cultures, spanning Thailand, Vietnam, Indonesia, the Philippines, and Malaysia.",
        'food': "🥢 A fusion of sweet, spicy, and sour! Dishes include Pad Thai, Pho, Satay, Laksa, and Nasi Goreng. Coconut milk and lemongrass are commonly used.",
        'clothing': "🧵 Unique cultural identity! Traditional wear includes the Ao Dai (Vietnam), Batik (Indonesia & Malaysia), Barong Tagalog (Philippines), and Sarong.",
        'culture': "🌺 Deep spiritual roots! Festivals like Songkran (Thailand), Nyepi (Bali), and Lunar New Year are celebrated, alongside shadow puppetry and temple dances."
    },
    'White': {
        'origin': "🏰 White people predominantly come from Europe, a region that shaped global history through the Renaissance, Enlightenment, and Industrial Revolution.",
        'food': "🥖 Diverse European cuisine! Dishes range from Italian pasta and French pastries to British fish and chips, German sausages, and Mediterranean olives.",
        'clothing': "🧥 Western fashion dominance! Styles range from traditional kilts (Scotland) and Lederhosen (Germany) to modern business suits and high fashion.",
        'culture': "🎭 A legacy of innovation! Western culture emphasizes philosophy, democracy, classical music, and the arts, with literature from Shakespeare to modern cinema."
    }
}

# Load the model
# model = load_model('fine_tuned_InceptionV3_model.h5')

# Define the class names
class_names = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']

# Provide both upload image and webcam capture options
st.header("Upload an Image or Capture via Webcam for Prediction")

# # Option to upload an image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# # Option to capture an image using the webcam
# webcam_image = st.camera_input("Take a picture using your webcam")

col1, col2 = st.columns([3, 1])  # Adjust the ratio of column widths (3:1 for a smaller webcam view)

with col1:
    # The image upload area
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with col2:
    # The webcam input area
    webcam_image = st.camera_input("Take a picture using your webcam")




if uploaded_file is not None:
    # Handle image upload
    img = Image.open(uploaded_file)

    
    img = img.resize((600, int(img.height * 600 / img.width)))
    st.image(img, caption="Uploaded Image", width=600)


    # st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Save the uploaded image temporarily
    img_path = "temp_image.jpg"
    img.save(img_path)
    
    # Predict the class
    predicted_class = predict_image_class(model, img_path, class_names)

elif webcam_image is not None:

    try:
    # Handle webcam image capture
        img = Image.open(io.BytesIO(webcam_image.getvalue()))

        img = img.resize((600, int(img.height * 600 / img.width)))
    
        # Display the resized image
        st.image(img, caption="Captured Image", width=600)


        # st.image(img, caption="Captured Image", use_container_width=True)
        
        # Save the captured image temporarily
        img_path = "temp_webcam_image.jpg"
        img.save(img_path)
        
        # Predict the class
        predicted_class = predict_image_class(model, img_path, class_names)
    
    except Exception as e:
        st.error(f"Error processing webcam image: {e}")    

else:
    st.warning("Please upload an image or use the webcam to take a picture.")


if 'predicted_class' in locals():
    # Use a colorful header with emoji for the predicted class
    st.markdown(f"<h2 style='color: #FF6347;'>🔮 Predicted Ethnicity: <span style='color: #4682B4;'>🌍 {predicted_class}</span></h2>", unsafe_allow_html=True)

    # Use colorful subheader with emoji for ethnicity information
    st.markdown(f"<h3 style='color: #32CD32;'>🧐 Information about <span style='color: #8A2BE2;'>✨ {predicted_class}</span></h3>", unsafe_allow_html=True)

    # Display each section with added emojis, colors, and highlighting
    st.markdown(f"<p style='font-size: 18px; color: #D2691E;'><strong>🌎 Origin:</strong> {ethnicity_info[predicted_class]['origin']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 18px; color: #D2691E;'><strong>🍽️ Food:</strong> {ethnicity_info[predicted_class]['food']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 18px; color: #D2691E;'><strong>👗 Clothing:</strong> {ethnicity_info[predicted_class]['clothing']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 18px; color: #D2691E;'><strong>🎨 Culture:</strong> {ethnicity_info[predicted_class]['culture']}</p>", unsafe_allow_html=True)

    # Add some fun styling to the sample images section with emoji
    st.markdown(f"<h4 style='color: #FF4500;'>📸 Sample Images of {predicted_class}</h4>", unsafe_allow_html=True)



    # Display sample images related to the predicted ethnicity
    st.subheader("Sample Images")
    st.image(ethnicity_images[predicted_class], caption=f"Sample {predicted_class}", width=200)
