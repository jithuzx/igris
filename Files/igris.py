import streamlit as st  # Ensure this is the first import
st.set_page_config(page_title="Igris AI", page_icon="🤖", layout="wide")  # Must be the first Streamlit command

import ollama
import pytesseract
import pdfplumber
import easyocr
import numpy as np
from PIL import Image
import asyncio
import traceback
from googlesearch import search
from concurrent.futures import ThreadPoolExecutor

# Set up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Cache EasyOCR for better performance
@st.cache_resource
def get_handwriting_reader():
    return easyocr.Reader(["en"])
handwriting_reader = get_handwriting_reader()

# Sidebar - Model Selection
st.sidebar.title("🧠 AI Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a Model:",
    ["deepseek-r1:7b", "mistral:7b", "llama3:8b", "gemma:7b", "phi3"],
    index=0
)

st.sidebar.image("icon.ico", caption="Igris AI", width=280)

# Web Search Toggle
use_web_search = st.sidebar.checkbox("Enable Web Search Mode")

# Main Chat UI
st.title("⚡ Igris AI - Offline & Online Assistant")
st.markdown("Type below and click **Generate Response** to chat with Igris.")

user_input = st.text_area("Your message:", height=150)

# Async Web Search Function
def web_search(query, num_results=3):
    """Perform a Google search asynchronously using ThreadPoolExecutor."""
    with ThreadPoolExecutor() as executor:
        return list(executor.map(lambda x: x, search(query, num_results=num_results)))

# Custom Styled Response Box
def display_response_box(response_text):
    """Display AI response inside a styled box for better readability."""
    st.markdown(
        f"""
        <div style="
            background-color: #1E1E1E; 
            border-radius: 10px; 
            padding: 15px; 
            color: white; 
            border: 2px solid #00FFFF;
            box-shadow: 2px 2px 10px rgba(0, 255, 255, 0.3);
            white-space: pre-wrap;
        ">
            {response_text}
        </div>
        """,
        unsafe_allow_html=True
    )

# Generate AI Response
def generate_response():
    """Handles AI response generation."""
    if not user_input.strip():
        st.warning("⚠️ Please enter a message.")
        return

    with st.spinner("Thinking... 🤖"):
        try:
            prompt = user_input
            if use_web_search:
                search_results = web_search(user_input)
                prompt = f"User Query: {user_input}\n\nWeb Results:\n" + "\n".join(search_results[:3])

            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            st.subheader("🧠 Igris AI Response")
            display_response_box(response["message"]["content"])

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.text(traceback.format_exc())

if st.button("Generate Response", key="generate_response"):
    generate_response()

# OCR Functions
def extract_text_from_image(image):
    """Extract printed text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image).strip()

def extract_text_from_handwriting(image):
    """Extract handwritten text using EasyOCR."""
    try:
        image_np = np.array(image)
        result = handwriting_reader.readtext(image_np)
        return " ".join([text[1] for text in result])
    except Exception as e:
        return f"Error in handwriting OCR: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF using pdfplumber."""
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages).strip() or "No text found in PDF."

# File Upload & OCR Processing
st.subheader("📂 Upload Document for OCR")
uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

ocr_extracted_text = ""

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension in ["png", "jpg", "jpeg"]:
        image = Image.open(uploaded_file)
        is_handwritten = st.radio("Is this handwritten text?", ["No", "Yes"], index=0)

        ocr_extracted_text = extract_text_from_handwriting(image) if is_handwritten == "Yes" else extract_text_from_image(image)
        image.thumbnail((300, 300))
        st.image(image, caption="Uploaded Image", use_container_width=False)

    elif file_extension == "pdf":
        ocr_extracted_text = extract_text_from_pdf(uploaded_file)

    else:
        ocr_extracted_text = "Unsupported file type!"

    st.subheader("📜 Extracted Text:")
    st.text_area("OCR Output", ocr_extracted_text, height=200)

# Summarization Mode
if len(ocr_extracted_text) > 500 and st.button("Summarize OCR Text"):
    with st.spinner("Summarizing... 🤖"):
        try:
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": f"Summarize this text:\n{ocr_extracted_text}"}])
            display_response_box(response["message"]["content"])
        except Exception as e:
            st.error(f"Error summarizing text: {str(e)}")

# Search OCR Text in AI
if ocr_extracted_text and st.button("Search OCR in AI"):
    with st.spinner("Processing OCR text in AI... 🤖"):
        try:
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": ocr_extracted_text}])
            display_response_box(response["message"]["content"])
        except Exception as e:
            st.error(f"Error processing OCR text: {str(e)}")

# Document Q&A
if ocr_extracted_text:
    st.subheader("❓ Ask Questions About Extracted Text")
    question = st.text_input("Enter your question:")
    if st.button("Ask AI"):
        with st.spinner("Thinking... 🤖"):
            try:
                response = ollama.chat(model=model_name, messages=[
                    {"role": "user", "content": f"Based on this document, answer the following question:\n\nDocument:\n{ocr_extracted_text}\n\nQuestion:\n{question}"}
                ])
                display_response_box(response["message"]["content"])
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

# Sticky Footer
st.markdown(
    """
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #111; color: white; text-align: center; padding: 15px; font-size: 14px;">
        🔥 Made by <strong>Jithu.S</strong> | Contact: <a href="mailto:sjithu1203@gmail.com" style="color: #00FFFF;">sjithu1203@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True
)
