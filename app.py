import streamlit as st
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import streamlit as st
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import random
import time


# Your existing model and breed mapping code
def get_breed_mapping():
    breeds = [
        'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel',
        'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle',
        'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound',
        'redbone', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound',
        'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner',
        'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier',
        'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier',
        'Norwich_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier',
        'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont',
        'Boston_bull', 'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer',
        'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier',
        'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever', 'curly-coated_retriever',
        'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short-haired_pointer',
        'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_spaniel',
        'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel',
        'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael',
        'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog',
        'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd',
        'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog',
        'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog',
        'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky', 'affenpinscher',
        'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees', 'Samoyed',
        'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan',
        'toy_poodle', 'miniature_poodle', 'standard_poodle', 'Mexican_hairless', 'dingo',
        'dhole', 'African_hunting_dog'
    ]
    return breeds

# Define the CNN class (your existing model architecture)
class CNN(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CNN, self).__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.base_model(x)
        return self.classifier(x)

def load_model(model_path, device):
    # Load the pre-trained ResNet50
    resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Create model with same architecture
    model = CNN(resnet, num_classes=120)
    model = model.to(device)
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    return model

def predict_image(model, image, device):
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Transform image
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
    return top5_idx, top5_prob

# Streamlit app
def main():
    st.set_page_config(page_title="Dog Breed Classifier", layout="wide")
    
    # Title
    st.title("🐕 Dog Breed Classifier")
    st.write("Upload a dog image and I'll tell you the breed!")
    
    # Model loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'C:/Users/anujp/Desktop/dog_breed_classification/best_model_20241226_000726.pth'  # Update with your model path
    
    try:
        model = load_model(model_path, device)
        breeds = get_breed_mapping()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            try:
                with st.spinner('Analyzing image...'):
                    # Get predictions
                    top5_idx, top5_prob = predict_image(model, image, device)
                    
                    # Display results
                    st.subheader("Predictions:")
                    
                    # Create a progress bar for each prediction
                    for idx, prob in zip(top5_idx, top5_prob):
                        breed = breeds[idx]
                        confidence = prob.item() * 100
                        st.write(f"**{breed}**")
                        st.progress(confidence / 100)
                        st.write(f"Confidence: {confidence:.2f}%")
                        st.write("---")
                        
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def get_dog_fact():
    facts = [
        "Dogs' noses are unique, just like human fingerprints!",
        "A dog's sense of smell is about 40 times greater than humans.",
        "Dogs can understand over 150 words and can count up to five.",
        "A dog's nose print is unique, much like a human's fingerprint.",
        "Dogs have about 1,700 taste buds. Humans have approximately 9,000.",
        "Dogs have three eyelids, including one for lubrication and protection.",
        "A dog's normal temperature is between 101 and 102.5 degrees Fahrenheit.",
        "Dogs are as smart as a two-year-old human toddler.",
    ]
    return random.choice(facts)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Smart Dog Breed Classifier",
        page_icon="🐕",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTitle {
            color: #2c3e50;
            font-size: 3rem !important;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .fun-fact {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("C:/Users/anujp/Desktop/123.png", width=100)  # Replace with your image path

        st.title("About")
        st.info("""
        This AI-powered dog breed classifier can identify 120 different dog breeds! 
        
        Simply upload a photo of a dog, and the model will provide the top 5 most likely breeds.
        
        Built with:
        - PyTorch
        - ResNet50
        - Streamlit
        """)
        
        st.markdown("---")
        st.subheader("🎯 Did you know?")
        st.markdown(f"*{get_dog_fact()}*")

    # Main content
    st.title("🐕 Smart Dog Breed Classifier")
    st.markdown("<p style='text-align: center; color: #666;'>Upload a dog photo and let AI identify the breed!</p>", 
                unsafe_allow_html=True)
    
    # Model loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'C:/Users/anujp/Desktop/dog_breed_classification/best_model_20241226_000726.pth'
    
    # Load model with error handling
    try:
        with st.spinner("🔄 Loading AI model..."):
            model = load_model(model_path, device)
            breeds = get_breed_mapping()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return

    # File upload section
    uploaded_file = st.file_uploader("Choose a dog image...", 
                                   type=["jpg", "jpeg", "png"],
                                   help="Supported formats: JPG, JPEG, PNG")

    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Your Dog', use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            try:
                with st.spinner('🔍 Analyzing image...'):
                    # Add artificial delay for effect
                    time.sleep(1)
                    top5_idx, top5_prob = predict_image(model, image, device)
                    
                    st.subheader("🎯 Breed Predictions")
                    
                    # Display predictions with enhanced styling
                    for i, (idx, prob) in enumerate(zip(top5_idx, top5_prob), 1):
                        breed = breeds[idx]
                        confidence = prob.item() * 100
                        
                        # Color coding based on confidence
                        if confidence > 80:
                            color = "#4CAF50"  # Green
                        elif confidence > 60:
                            color = "#FFA726"  # Orange
                        else:
                            color = "#EF5350"  # Red
                            
                        st.markdown(f"**{i}. {breed}**")
                        st.progress(confidence / 100)
                        st.markdown(f"<p style='color: {color};'>Confidence: {confidence:.2f}%</p>", 
                                  unsafe_allow_html=True)
                        st.markdown("---")
                    
                    # Add a fun fact about the predicted breed
                    st.markdown("<div class='fun-fact'>", unsafe_allow_html=True)
                    st.markdown(f"**🎨 Fun Fact:** {get_dog_fact()}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional information section
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h4>Want to learn more about dogs?</h4>
            <p>Check out these resources:</p>
            <ul style='list-style-type: none;'>
                <li>🐕 <a href='https://www.akc.org/'>American Kennel Club</a></li>
                <li>📚 <a href='https://www.dogs.info/'>Dog Breeds Encyclopedia</a></li>
                <li>🏥 <a href='https://www.avma.org/'>American Veterinary Medical Association</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
