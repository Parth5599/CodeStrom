import ssl
# Mac ke SSL error ko bypass karne ke liye
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw
from deep_translator import GoogleTranslator
from langdetect import detect
import json
import base64
import fitz  # NAYA: PDF read karne ke liye (PyMuPDF)

# --- Language Codes ko Full Name mein badalna ---
try:
    langs_dict = GoogleTranslator().get_supported_languages(as_dict=True)
    CODE_TO_NAME = {v: k.capitalize() for k, v in langs_dict.items()}
    CODE_TO_NAME['zh-cn'] = "Chinese (Simplified)"
    CODE_TO_NAME['zh-tw'] = "Chinese (Traditional)"
    CODE_TO_NAME['tl'] = "Tagalog (Filipino)"
except:
    CODE_TO_NAME = {}

# --- Page Setup ---
st.set_page_config(page_title="AI Multi-Lang OCR", layout="wide")
st.title("🌐 AI-Based OCR Solution for Multi-Foreign Languages")

# --- Smart Feature: Sidebar Settings ---
st.sidebar.title("⚙️ AI Settings")
st.sidebar.info("Unique scripts require manual selection for high accuracy. European languages can be auto-detected.")

language_options = [
    "Auto-Detect (Latin/European)", 
    "Hindi", 
    "Arabic", 
    "Chinese (Simplified)", 
    "Japanese",
    "Korean",
    "Russian",
    "French",
    "Spanish",
    "German"
]
doc_language = st.sidebar.selectbox("Select Document Language:", language_options)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Image Processing")
apply_cleaning = st.sidebar.checkbox("🧹 Enhance Image (Remove Noise)", value=False)

# --- Load AI Models ---
@st.cache_resource
def load_ocr_model(lang_selection):
    if lang_selection == "Auto-Detect (Latin/European)": return easyocr.Reader(['en', 'fr', 'es', 'de'])
    elif lang_selection == "Hindi": return easyocr.Reader(['en', 'hi'])
    elif lang_selection == "Arabic": return easyocr.Reader(['en', 'ar'])
    elif lang_selection == "Chinese (Simplified)": return easyocr.Reader(['en', 'ch_sim'])
    elif lang_selection == "Japanese": return easyocr.Reader(['en', 'ja'])
    elif lang_selection == "Korean": return easyocr.Reader(['en', 'ko'])
    elif lang_selection == "Russian": return easyocr.Reader(['en', 'ru'])
    elif lang_selection == "French": return easyocr.Reader(['en', 'fr'])
    elif lang_selection == "Spanish": return easyocr.Reader(['en', 'es'])
    elif lang_selection == "German": return easyocr.Reader(['en', 'de'])
    else: return easyocr.Reader(['en'])

reader = load_ocr_model(doc_language)

# --- Main App ---
# NAYA: ab 'pdf' bhi allow kar diya hai
uploaded_file = st.file_uploader("Upload an Image or PDF Document", type=["jpg", "png", "jpeg", "pdf"])

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()
    
    # --- NAYA LOGIC: PDF vs Image Handle karna ---
    if uploaded_file.name.lower().endswith('.pdf'):
        st.info("📄 PDF Detected: Extracting Page 1 for AI Analysis...")
        # PDF open karo
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        # Pehla page load karo (Hackathon ke liye Page 1 fast aur best hai)
        page = doc.load_page(0) 
        # High Quality (300 DPI) image banao PDF se
        pix = page.get_pixmap(dpi=300) 
        
        # Pixmap ko OpenCV format me badlo
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: # Agar PDF transparent hai (RGBA)
            opencv_image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else: # Standard PDF (RGB)
            opencv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
    else:
        # Normal Image Logic
        file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

    # OpenCV (BGR) ko Standard Display (RGB) me badalna
    img_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    with st.spinner(f"Processing document with {doc_language} AI Model..."):
        
        if apply_cleaning:
            gray_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            ocr_ready_image = cv2.medianBlur(gray_img, 3)
        else:
            ocr_ready_image = opencv_image
            
        results = reader.readtext(ocr_ready_image)
        
        processed_data = []
        annotated_image = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(annotated_image)
        
        for (bbox, text, prob) in results:
            try:
                lang_code = detect(text).lower()
                full_lang_name = CODE_TO_NAME.get(lang_code, lang_code.upper())
                translation = GoogleTranslator(source='auto', target='en').translate(text)
            except:
                translation = "Error/No text"
                full_lang_name = "Unknown"
                lang_code = "unknown"

            conf_score = round(prob * 100, 2)
            
            processed_data.append({
                "original_text": text,
                "translated_text": translation,
                "detected_language": full_lang_name,
                "confidence": f"{conf_score}%",
                "coordinates": [list(map(int, p)) for p in bbox]
            })
            
            points = [tuple(map(int, p)) for p in bbox]
            draw.polygon(points, outline="red", width=3)
            draw.text((points[0][0], points[0][1] - 15), f"[{lang_code.upper()}]", fill="red")

        # --- Show Results ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Document")
            st.image(img_rgb, use_container_width=True)
        with col2:
            st.subheader("AI Layout Detection")
            st.image(annotated_image, use_container_width=True)

        st.divider()
        
        if processed_data:
            avg_conf = sum([float(x['confidence'].strip('%')) for x in processed_data]) / len(processed_data)
            
            if avg_conf >= 80:
                st.success(f"✅ High Accuracy: {round(avg_conf, 2)}%")
            else:
                st.warning(f"⚠️ Low Accuracy: {round(avg_conf, 2)}%. Try enabling 'Enhance Image' from the sidebar!")

            st.download_button(
                label="📥 Download JSON Result",
                data=json.dumps(processed_data, indent=4),
                file_name="ocr_output.json",
                mime="application/json"
            )
            st.dataframe(processed_data)
else:
    st.info("Please upload an image or PDF to start.")