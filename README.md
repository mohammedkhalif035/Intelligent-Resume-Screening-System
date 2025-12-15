# Intelligent-Resume-Screening-System

📌 Overview

The Intelligent Resume Screening System is an AI-powered application that automates resume analysis, classification, and ranking. It helps recruiters efficiently screen large volumes of resumes by extracting text from PDF files (including scanned resumes), classifying them into job categories, and ranking candidates based on their relevance to a given job description using semantic similarity.

The system reduces manual effort, improves accuracy, and supports fair and efficient hiring decisions.

🎯 Objectives

Automate resume screening

Extract text from digital and scanned resumes

Classify resumes into predefined job roles

Match resumes with job descriptions

Rank candidates based on relevance

Improve recruitment efficiency

🚀 Key Features

Upload multiple PDF resumes

Text extraction using PDFMiner and EasyOCR

Resume and job classification using TF-IDF + Logistic Regression

Semantic resume ranking using Sentence-BERT (SBERT)

Interactive Streamlit web interface

Downloadable CSV results

🛠 Technology Stack

Language: Python

Frontend: Streamlit

ML & NLP: TF-IDF, Logistic Regression, SBERT

OCR: EasyOCR, OpenCV

Deep Learning: PyTorch

PDF Processing: PDFMiner, pdf2image

Data Handling: Pandas, NumPy


⚙️ Installation & Run
pip install -r requirements.txt
streamlit run app.py


Open: http://localhost:8501

🔄 Workflow

Upload labeled dataset

Upload PDF resumes

Enter job description

Extract and preprocess text

Classify resumes and job role

Rank resumes using SBERT

📈 Output

Ranked resumes with similarity scores

Resume preview excerpts

Downloadable CSV file

