🧠 Intelligent Resume Screening System
📌 Project Overview

In today’s competitive recruitment environment, organizations receive hundreds of resumes for a single job opening. Manual screening is inefficient, time-consuming, and prone to bias.
The Intelligent Resume Screening System is an AI-powered application designed to automate resume analysis, classification, and ranking using advanced Natural Language Processing (NLP), Machine Learning, and Deep Learning techniques.

The system extracts textual data from resumes (including scanned PDFs), classifies them into relevant job categories, and ranks candidates based on their semantic similarity to a given job description.

🎯 Project Objectives

Automate the resume screening process

Extract accurate information from both digital and scanned resumes

Classify resumes into predefined job roles

Match resumes intelligently with job descriptions

Rank candidates based on relevance score

Reduce recruiter workload and improve hiring efficiency

🚀 Key Features
📑 Resume Upload

Supports multiple PDF resumes

Handles both text-based and scanned resumes

🧠 Smart Text Extraction

PDFMiner for digital resumes

EasyOCR + OpenCV for scanned resumes

Automatic fallback to OCR when needed

🏷 Resume Classification

TF-IDF vectorization

Logistic Regression classifier

Predicts job category for resumes and job description

🔍 Intelligent Resume Ranking

Uses Sentence-BERT (SBERT)

Semantic similarity using cosine similarity

Displays top N relevant candidates

🖥 Interactive Web Interface

Built with Streamlit

Real-time results

Downloadable CSV outputs

🛠 Technology Stack
Programming Language

Python 3

Libraries & Tools
Category	Technologies
Frontend	Streamlit
NLP	TF-IDF, Sentence-BERT
ML	Logistic Regression
OCR	EasyOCR, OpenCV
DL	PyTorch
Transformers	Hugging Face
PDF Handling	PDFMiner, pdf2image
Data Processing	Pandas, NumPy
📂 Project Structure
Intelligent-Resume-Screening-System/
│
├── app.py
│   └── Main Streamlit application
│
├── requirements.txt
│   └── Project dependencies
│
├── README.md
│   └── Documentation
│
├── sbert_finetuned/
│   └── Fine-tuned SBERT model
│
└── sample_data/
    ├── resumes/
    │   └── Sample PDF resumes
    └── labeled_dataset.csv
        └── Training dataset

⚙️ Installation & Setup
Step 1: Clone Repository
git clone https://github.com/your-username/Intelligent-Resume-Screening-System.git
cd Intelligent-Resume-Screening-System

Step 2: Create Virtual Environment
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate

Step 3: Install Requirements
pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py


Access at:

http://localhost:8501

📊 Dataset Description
Labeled Resume Dataset (CSV)

Used to train the resume classification model.

Format:

Category,Resume
Data Science,Experience in Python, ML, and analytics...
Web Development,Frontend developer with React skills...

🔄 System Workflow

Upload labeled dataset

Upload multiple resumes (PDF)

Enter job description

Extract resume text

Preprocess and clean data

Train classification model

Predict job & resume categories

Filter relevant resumes

Rank resumes using SBERT

Display top candidates

🧪 Algorithms Used
TF-IDF

Converts text into numerical vectors

Highlights relevant keywords

Logistic Regression

Predicts job categories

Efficient and interpretable

Sentence-BERT (SBERT)

Captures contextual meaning

Improves matching accuracy

📈 Output

Ranked resumes with similarity score

Resume preview excerpts

Categorized resume CSV download

✅ Advantages

Reduces manual screening effort

Handles unstructured resume formats

Supports scanned PDFs

Scalable and efficient

⚠️ Limitations

Requires labeled training dataset

OCR accuracy depends on document quality

Training may be slow without GPU
