from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz  # PyMuPDF
from docx import Document

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    file_path = os.path.join('/mnt/data/', file.filename)
    file.save(file_path)
    
    # Extract text based on file type
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file.filename.endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

    # Perform keyword extraction
    keywords = extract_keywords(text)
    return jsonify({'keywords': keywords})

def extract_text_from_pdf(pdf_path):
    text = ''
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ' '.join([para.text for para in doc.paragraphs])
    return text

def extract_keywords(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    indices = X[0].nonzero()[1]
    keywords = [vectorizer.get_feature_names_out()[i] for i in indices]
    return keywords

if __name__ == '__main__':
    app.run(port=5000, debug=True)
