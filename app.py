from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from transformers import pipeline
import fitz  # PyMuPDF
from docx import Document

app = Flask(__name__)
CORS(app)

# Load summarization pipeline
summarizer = pipeline("summarization")

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

    # Perform summarization
    summary = summarize_text(text)
    return jsonify({'summary': summary})

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

def summarize_text(text):
    # Summarize the text using the pipeline
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
