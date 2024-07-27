# app.py
import streamlit as st
from qa_setup import llm_ans
import base64

# Function to get base64 of image
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set page configuration
st.set_page_config(
    page_title="LOTR Questionnaire Chatbot",
    page_icon="🧙‍♂️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Get the background image as base64
img_file = "middle-earth-wallpaper.jpg"
img_base64 = get_base64(img_file)

# Add custom CSS for styling
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
    }}
    .main {{
        background: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }}
    .stButton>button {{
        background-color: #f4d03f;
        color: black;
        border-radius: 10px;
        padding: 10px;
    }}
    .stTextInput>div>div>input {{
        background-color: white;
        color: black;
    }}
    .stTitle h1 {{
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.markdown("<h1 style='color: white;'>🧙‍♂️ LOTR Questionnaire Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white;'>Ask me anything about Middle-earth!</h2>", unsafe_allow_html=True)

# Input text box for questions
question = st.text_input('Enter your travel-related question:')

# Add a button to submit the question
if st.button('Ask'):
    if question:
        answer = llm_ans(question)
        st.markdown(f"**Answer:** {answer}")
    else:
        st.markdown("<p style='color: red;'>Please enter a question.</p>", unsafe_allow_html=True)
