import streamlit as st
import pandas as pd
import re
import os
import cv2
import json
import numpy as np
import pdf2image
import easyocr
from pdfminer.high_level import extract_text
from transformers import pipeline
import torch
import random
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import io
import concurrent.futures # For potential future parallel processing of PDFs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import subprocess # For ColBERT training

# --- Configuration ---
PDF_DPI = 200
EASYOCR_LANGS = ["en"]
SBERT_EPOCHS = 1
COLBERT_MAX_STEPS = 1000 # Reduce for faster runs, but affects quality
COLBERT_BSZ = 32

# Model Save Paths (relative to the script's execution directory)
SBERT_MODEL_SAVE_PATH = 'sbert_finetuned'
COLBERT_TRAIN_OUTPUT_DIR = './ColBERT/experiments/jobmatch_colbert' # ColBERT's training output directory

# --- Global Setup for Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ["WANDB_DISABLED"] = "true" # Disable Weights & Biases logging

# Define Resume Section Titles for classification.
SECTION_TITLES = ["Experience", "Education", "Projects", "Skills", "Certifications"]

# --- Utility Functions ---

def clean_text(text):
    """
    Cleans text by removing special characters, extra spaces, and converting to lowercase.
    Handles non-string inputs gracefully.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def pdf_bytes_to_images(pdf_bytes):
    """
    Converts PDF bytes (in memory) to a list of PIL Image objects.
    Uses pdf2image.convert_from_bytes for in-memory conversion.
    """
    try:
        return pdf2image.convert_from_bytes(io.BytesIO(pdf_bytes), dpi=PDF_DPI)
    except Exception as e:
        print(f"Error converting PDF bytes to images: {e}")
        return []

def extract_text_pdfminer_from_bytes(pdf_bytes):
    """
    Extracts text from PDF bytes using pdfminer.high_level.extract_text.
    Ideal for 'born-digital' PDFs where text is selectable.
    """
    try:
        text = extract_text(io.BytesIO(pdf_bytes)).strip()
        return text if text else None
    except Exception: # Suppress specific pdfminer errors for cleaner output
        return None

def preprocess_image_for_ocr(pil_image):
    """
    Converts a PIL Image object to an OpenCV (numpy array) format and
    applies basic image preprocessing (grayscale, blur, thresholding) for OCR.
    """
    open_cv_image = np.array(pil_image.convert('RGB')) # Convert to RGB first for consistent color channels
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Reduce noise
    # Apply Otsu's thresholding for automatic threshold calculation
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def detect_titles(pil_image):
    """
    Placeholder for a layout detection model (e.g., Detectron2LayoutModel).
    This part was commented out in the original code and remains so,
    as it would require additional model loading and configuration.
    It currently returns an empty list, meaning OCR will process larger blocks.
    """
    return []

def extract_text_ocr(cv_image_segment, _reader_instance, bbox=None):
    """
    Performs OCR on a preprocessed OpenCV image segment using EasyOCR.
    Can operate on a specific bounding box within the image.
    """
    try:
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = cv_image_segment.shape[:2] # Get height and width from the image
            # Basic validation for bounding box coordinates
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                cropped = cv_image_segment
            else:
                cropped = cv_image_segment[y1:y2, x1:x2]
        else:
            cropped = cv_image_segment

        text = _reader_instance.readtext(cropped, detail=0)
        return " ".join(text)
    except Exception as e:
        print(f"Error in OCR extraction: {e}")
        return ""

def classify_section(text, _classifier_instance, section_titles):
    """
    Classifies a given text into one of the predefined resume section titles
    using a zero-shot classification model.
    """
    try:
        if not text:
            return "Unknown" # Handle empty text gracefully
        result = _classifier_instance(text, section_titles)
        return result["labels"][0] if result["labels"] else "Unknown"
    except Exception as e:
        print(f"Error classifying section: {e}")
        return "Unknown"

def extract_sections_from_image(preprocessed_cv_image, title_blocks, _reader_instance, _classifier_instance, section_titles):
    """
    Extracts and classifies sections from a preprocessed OpenCV image based on detected titles.
    If no titles are detected, it attempts to OCR the entire page.
    """
    sections = {}
    if not title_blocks:
        # If no titles, try to OCR the whole page as a general section
        full_page_text = extract_text_ocr(preprocessed_cv_image, _reader_instance)
        if full_page_text:
            sections["Full Page Text"] = full_page_text
        return sections

    for i, (title, bbox) in enumerate(title_blocks):
        try:
            # Determine the end y-coordinate for the current section
            if i < len(title_blocks) - 1:
                next_y = title_blocks[i + 1][1][1]
            else:
                next_y = preprocessed_cv_image.shape[0] # End of image for the last section

            # Extract text for the section using OCR
            section_text = extract_text_ocr(preprocessed_cv_image, _reader_instance, (bbox[0], bbox[1], bbox[2], next_y))
            # Classify the section type based on its title
            section_type = classify_section(title, _classifier_instance, section_titles)
            sections[section_type] = section_text
        except Exception as e:
            print(f"Error extracting section '{title}': {e}")
            continue
    return sections

def process_single_resume_in_memory(resume_bytes, file_name, _reader_instance, _classifier_instance, section_titles):
    """
    Processes a single resume from its bytes representation entirely in memory.
    Attempts text extraction, then falls back to image conversion and OCR if needed.
    Returns a dictionary containing the extracted data.
    """
    resume_data = {}
    text_data = extract_text_pdfminer_from_bytes(resume_bytes)
    if text_data:
        resume_data["Text Extracted"] = text_data
    else:
        images = pdf_bytes_to_images(resume_bytes)
        if images:
            for page_num, pil_img in enumerate(images):
                preprocessed_cv_img = preprocess_image_for_ocr(pil_img)
                title_blocks = detect_titles(pil_img)
                sections = extract_sections_from_image(preprocessed_cv_img, title_blocks, _reader_instance, _classifier_instance, section_titles)
                resume_data[f"Page {page_num+1}"] = sections
    return file_name, resume_data

# --- Cached Model Loaders (for efficiency in Streamlit) ---

@st.cache_resource
def get_easyocr_reader():
    """Caches and returns the EasyOCR reader instance."""
    print("Initializing EasyOCR reader...")
    return easyocr.Reader(EASYOCR_LANGS, gpu=torch.cuda.is_available())

@st.cache_resource
def get_hf_classifier():
    """Caches and returns the Hugging Face zero-shot classification pipeline."""
    print("Initializing Hugging Face zero-shot classifier...")
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

@st.cache_resource
def get_sbert_model(model_path, labeled_df_for_training):
    """
    Caches and returns the fine-tuned SBERT model.
    Trains it if not found or if retraining is forced.
    """
    sbert_model = None
    if os.path.exists(model_path):
        try:
            sbert_model = SentenceTransformer(model_path)
            st.success(f"Loaded fine-tuned SBERT model from {model_path}.")
            return sbert_model
        except Exception as e:
            st.warning(f"Error loading SBERT model from {model_path}: {e}. Retraining SBERT.")
            sbert_model = None # Force retraining

    if sbert_model is None:
        st.info("Fine-tuning SBERT model (this may take a moment)...")
        if labeled_df_for_training.empty:
            st.error("Cannot fine-tune SBERT: Labeled dataset is empty.")
            return None

        train_examples = [InputExample(texts=[row['Category'], row['Resume']], label=1.0) for _, row in labeled_df_for_training.iterrows()]
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model=sbert_model)
        sbert_model.fit(train_objectives=[(train_loader, train_loss)], epochs=SBERT_EPOCHS, warmup_steps=10)
        sbert_model.save(model_path)
        st.success("SBERT fine-tuning complete and model saved.")
        return sbert_model


# --- Core Pipeline Functions (Encapsulating previous phases) ---

@st.cache_data(show_spinner=False) # Cache data loading results
def load_and_preprocess_labeled_data(uploaded_file_bytes):
    """Loads and preprocesses the labeled dataset from uploaded file bytes."""
    st.info("Loading and preprocessing labeled dataset...")
    labeled_df = pd.DataFrame()
    if uploaded_file_bytes is None:
        st.error("No labeled dataset file uploaded.")
        return labeled_df

    try:
        labeled_df = pd.read_csv(io.BytesIO(uploaded_file_bytes))
        labeled_df.dropna(inplace=True)
        labeled_df['Resume'] = labeled_df['Resume'].apply(clean_text)
        st.success("Labeled dataset preprocessed.")
    except Exception as e:
        st.error(f"Error loading or preprocessing labeled data: {e}")
    return labeled_df

@st.cache_data(show_spinner=False) # Cache job description loading
def load_job_description(job_description_text_input):
    """Loads job description from text input."""
    st.info("Loading job description...")
    if not job_description_text_input:
        st.warning("Job description input is empty.")
        return ""
    st.success("Job description loaded.")
    return job_description_text_input

@st.cache_data(show_spinner=False) # Cache processed resumes
def process_all_resumes(uploaded_resume_files, _reader_instance, _classifier_instance, section_titles):
    """Processes all uploaded PDF resumes."""
    st.info("Processing PDF resumes...")
    extracted_resumes_in_memory = {}
    if not uploaded_resume_files:
        st.error("No resume files uploaded.")
        return pd.DataFrame()

    for uploaded_file in uploaded_resume_files:
        file_name = uploaded_file.name
        pdf_bytes = uploaded_file.getvalue()
        processed_file_name, resume_data_for_file = process_single_resume_in_memory(
            pdf_bytes, file_name, _reader_instance, _classifier_instance, section_titles
        )
        extracted_resumes_in_memory[processed_file_name] = resume_data_for_file

    all_extracted_resumes_list = []
    for file_name, data in extracted_resumes_in_memory.items():
        if "Text Extracted" in data:
            all_extracted_resumes_list.append({"File": file_name, "Resume": data["Text Extracted"]})
        else:
            combined_text_for_resume = []
            for page_key in sorted(data.keys()):
                page_sections = data.get(page_key, {})
                if isinstance(page_sections, dict):
                    combined_text_for_resume.extend([str(v) for v in page_sections.values() if v])
                elif isinstance(page_sections, str) and page_sections:
                    combined_text_for_resume.append(page_sections)
            if combined_text_for_resume:
                all_extracted_resumes_list.append({"File": file_name, "Resume": " ".join(combined_text_for_resume)})

    preprocessed_extracted_df = pd.DataFrame(all_extracted_resumes_list)
    if not preprocessed_extracted_df.empty:
        preprocessed_extracted_df["Resume"] = preprocessed_extracted_df["Resume"].apply(clean_text)
        st.success("PDF resume processing and preprocessing complete.")
    else:
        st.warning("No resume data extracted for further processing.")
    return preprocessed_extracted_df

@st.cache_data(show_spinner=False) # Cache classification results
def train_classify_and_predict_categories(labeled_df_for_training, extracted_resumes_df_for_classification, job_description_text):
    """Trains a classifier, classifies extracted resumes and the job description."""
    st.info("Training classification model and predicting categories...")
    predicted_job_category_for_jd = "Unknown"

    if labeled_df_for_training.empty:
        st.error("Cannot train classifier: Labeled dataset is empty.")
        return extracted_resumes_df_for_classification, predicted_job_category_for_jd

    X = labeled_df_for_training['Resume']
    y = labeled_df_for_training['Category']

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Train Logistic Regression
    classifier_model = LogisticRegression(max_iter=1000, random_state=SEED)
    classifier_model.fit(X_tfidf, y)
    st.success("Classifier training complete.")

    # Classify Extracted Resumes
    if not extracted_resumes_df_for_classification.empty:
        extracted_resumes_tfidf = tfidf_vectorizer.transform(extracted_resumes_df_for_classification['Resume'])
        predicted_categories = classifier_model.predict(extracted_resumes_tfidf)
        extracted_resumes_df_for_classification['Predicted_Category'] = predicted_categories
        st.success("Extracted resumes classified.")
    else:
        st.warning("No extracted resumes to classify.")

    # Classify Job Description
    if job_description_text:
        cleaned_job_description = clean_text(job_description_text)
        job_description_tfidf = tfidf_vectorizer.transform([cleaned_job_description])
        predicted_job_category_for_jd = classifier_model.predict(job_description_tfidf)[0]
        st.success(f"Job description classified into category: '{predicted_job_category_for_jd}'.")
    else:
        st.warning("Job description is empty. Cannot classify its category.")

    return extracted_resumes_df_for_classification, predicted_job_category_for_jd

@st.cache_data(show_spinner=False) # Cache ranking results
def perform_targeted_ranking(labeled_df_for_sbert_training, preprocessed_extracted_df_with_categories, job_description_text, predicted_job_category_for_jd, _sbert_model_instance, num_requirements):
    """Performs targeted job matching and ranking using SBERT."""
    st.info("Performing targeted job matching and ranking...")
    final_ranked_resumes_data = []

    if preprocessed_extracted_df_with_categories.empty or not job_description_text or not predicted_job_category_for_jd:
        st.warning("Insufficient data for ranking. Please ensure resumes are extracted, classified, and job description is provided.")
        return []

    # Filter resumes based on predicted job category
    filtered_candidate_resumes_df = preprocessed_extracted_df_with_categories[
        preprocessed_extracted_df_with_categories['Predicted_Category'] == predicted_job_category_for_jd
    ].copy()

    if filtered_candidate_resumes_df.empty:
        st.warning(f"No resumes found in the extracted set that match the category '{predicted_job_category_for_jd}'. Cannot perform ranking.")
        return []

    resumes_for_ranking = filtered_candidate_resumes_df['Resume'].tolist()
    st.info(f"Found {len(resumes_for_ranking)} resumes matching the category '{predicted_job_category_for_jd}' for ranking.")

    # SBERT Reranking
    query_embed = _sbert_model_instance.encode(job_description_text, convert_to_tensor=True)
    cand_embeds = _sbert_model_instance.encode(resumes_for_ranking, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embed, cand_embeds)[0]
    top_indices = scores.argsort(descending=True)

    # Prepare results for display
    for i, idx_tensor in enumerate(top_indices):
        if i >= num_requirements: # Limit to 'num_requirements'
            break
        idx = idx_tensor.item() # Convert tensor index to integer
        if i < len(resumes_for_ranking):
            resume_content = resumes_for_ranking[idx]
            original_file_name = filtered_candidate_resumes_df.iloc[idx]['File']
            final_ranked_resumes_data.append({
                "Rank": i + 1,
                "File": original_file_name,
                "Score": scores[idx].item(),
                "Resume_Excerpt": resume_content[:500] + "..." if len(resume_content) > 500 else resume_content
            })
    st.success("Targeted job matching and ranking complete.")
    return final_ranked_resumes_data

# --- Streamlit UI Layout ---

st.set_page_config(layout="wide", page_title="Job Matching System")

st.title("📄 Job Matching System")
st.markdown("""
Welcome to the Job Matching System! Upload your resumes, provide a job description,
and get the most relevant candidates ranked by category.
""")

# Input Widgets
st.header("Inputs")

# Replaced st.text_input with st.file_uploader for labeled dataset
uploaded_labeled_data_file = st.file_uploader(
    "Upload Labeled Dataset CSV",
    type=["csv"],
    help="Upload your CSV file containing labeled resume data (e.g., UpdatedResumeDataSet.csv)"
)

# Replaced st.text_input with st.file_uploader for resumes directory
uploaded_resume_files = st.file_uploader(
    "Upload Resumes (PDFs)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload multiple PDF resume files from your system"
)

num_requirements = st.number_input(
    "Number of Top Resumes to Display",
    min_value=1,
    value=5,
    step=1
)
job_description_input = st.text_area(
    "Job Description",
    height=200,
    placeholder="Paste the job description here..."
)

process_button = st.button("Process Resumes and Job Description")

# --- Main Processing Logic ---
if process_button:
    st.header("Processing Results")
    status_placeholder = st.empty()

    # Phase 1: Load and Preprocess Labeled Data
    with st.spinner("Loading and preprocessing labeled dataset..."):
        labeled_df = load_and_preprocess_labeled_data(uploaded_labeled_data_file.getvalue() if uploaded_labeled_data_file else None)
        if labeled_df.empty:
            st.error("Labeled dataset could not be loaded or is empty. Please upload a valid CSV file.")
            st.stop()

    # Phase 2: Text Extraction and Preprocessing from Resumes
    with st.spinner("Extracting and preprocessing resumes..."):
        reader_instance = get_easyocr_reader()
        classifier_instance = get_hf_classifier()
        # Pass unhashable objects with a leading underscore to tell Streamlit not to hash them
        preprocessed_extracted_df = process_all_resumes(
            uploaded_resume_files, _reader_instance=reader_instance, _classifier_instance=classifier_instance, section_titles=SECTION_TITLES
        )
        if preprocessed_extracted_df.empty:
            st.error("No resumes were extracted or processed successfully. Please upload PDF resume files.")
            st.stop()

    # Phase 3: Job Classification
    with st.spinner("Classifying resumes and job description by category..."):
        preprocessed_extracted_df, predicted_job_category = train_classify_and_predict_categories(
            labeled_df, preprocessed_extracted_df, job_description_input
        )
        if not predicted_job_category or predicted_job_category == "Unknown":
            st.warning("Could not reliably classify the job description. This might affect targeted ranking.")

    st.subheader("Job Description Category")
    st.write(f"The job description is classified into the category: **{predicted_job_category}**")

    st.subheader("Extracted Resumes with Predicted Categories")
    if not preprocessed_extracted_df.empty:
        st.dataframe(preprocessed_extracted_df[['File', 'Predicted_Category', 'Resume']].head(10)) # Show first 10
        st.download_button(
            label="Download All Extracted Resumes with Categories (CSV)",
            data=preprocessed_extracted_df.to_csv(index=False).encode('utf-8'),
            file_name="extracted_resumes_with_categories.csv",
            mime="text/csv",
        )
    else:
        st.info("No resumes to display or download.")

    # Phase 4: Targeted Job Matching and Ranking
    with st.spinner("Performing targeted job matching and ranking..."):
        sbert_model_instance = get_sbert_model(SBERT_MODEL_SAVE_PATH, labeled_df)
        if sbert_model_instance is None:
            st.error("SBERT model could not be loaded or trained. Skipping ranking.")
            st.stop()

        # Pass sbert_model_instance with a leading underscore
        ranked_results = perform_targeted_ranking(
            labeled_df, preprocessed_extracted_df, job_description_input,
            predicted_job_category, _sbert_model_instance=sbert_model_instance, num_requirements=num_requirements
        )

    st.subheader(f"Top {num_requirements} Ranked Resumes for '{predicted_job_category}'")
    if ranked_results:
        for item in ranked_results:
            st.markdown(f"**Rank {item['Rank']}**: {item['File']} (Score: {item['Score']:.4f})")
            with st.expander("View Resume Excerpt"):
                st.write(item['Resume_Excerpt'])
            st.markdown("---")
    else:
        st.info("No resumes ranked. This might be because no resumes matched the predicted job category.")

    st.success("Processing complete!")

