import os
import re
import textwrap
import numpy as np
import nltk
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    T5TokenizerFast,
    T5ForConditionalGeneration
)
import torch

nltk.download('punkt')

class LegalAssistant:
    def __init__(self):
        self.qa_model_name = "deepset/roberta-base-squad2"
        self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)

        self.t5_model_name = "google/flan-t5-large"
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(self.t5_model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_model_name)

        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

        self.document_chunks = []
        self.document_embeddings = None
        self.document_metadata = {}
        self.section_map = {}

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text

    def process_pdf(self, pdf_path: str, chunk_size: int = 500):
        reader = PdfReader(pdf_path)
        full_text = ""
        section_texts = {"Default": ""}
        current_section = "Default"

        for page in reader.pages:
            text = page.extract_text() or ""
            lines = text.split('\n')
            for line in lines:
                if len(line.strip()) < 100 and line.strip().isupper():
                    current_section = line.strip()
                    if current_section not in section_texts:
                        section_texts[current_section] = ""
                section_texts[current_section] += line + " "
                full_text += text

        self.document_metadata = {
            'total_pages': len(reader.pages),
            'sections': list(section_texts.keys()),
            'file_name': os.path.basename(pdf_path)
        }

        processed_chunks = []
        for section, content in section_texts.items():
            cleaned_content = self.preprocess_text(content)
            section_chunks = textwrap.wrap(cleaned_content, chunk_size, break_long_words=False)
            for chunk in section_chunks:
                chunk_id = len(processed_chunks)
                self.section_map[chunk_id] = section
                processed_chunks.append(chunk)

        self.document_chunks.extend(processed_chunks)
        if self.document_embeddings is None:
            self.document_embeddings = self.sentence_transformer.encode(self.document_chunks)
        else:
            new_embeddings = self.sentence_transformer.encode(processed_chunks)
            self.document_embeddings = np.vstack((self.document_embeddings, new_embeddings))

    def find_relevant_context(self, question: str, top_k: int = 3) -> dict:
        question_embedding = self.sentence_transformer.encode([question])[0]
        similarities = np.dot(self.document_embeddings, question_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return {
            'chunks': [self.document_chunks[i] for i in top_indices],
            'scores': [float(similarities[i]) for i in top_indices]
        }

    def generate_descriptive_answer(self, question: str, context: dict) -> dict:
        if not context['chunks']:
            return {
                'solution': "No relevant context found.",
                'confidence': 0.0,
                'source_pdf': self.document_metadata['file_name']
            }

        weighted_context = " ".join(context['chunks'])

        t5_prompt = f"Answer this legal question in detail: {question}\nContext: {weighted_context}\nProvide a comprehensive and clear answer."
        t5_inputs = self.t5_tokenizer(t5_prompt, return_tensors="pt", max_length=1024, truncation=True)
        t5_outputs = self.t5_model.generate(
            t5_inputs.input_ids, max_length=500, num_beams=10, temperature=0.5, no_repeat_ngram_size=2
        )
        generated_answer = self.t5_tokenizer.decode(t5_outputs[0], skip_special_tokens=True)

        return {
            'solution': generated_answer,
            'confidence': max(context['scores']),
            'source_pdf': self.document_metadata['file_name'],
            'sources': [
                {
                    'section': self.section_map[i],
                    'relevance': context['scores'][i],
                    'file_name': self.document_metadata['file_name']
                } for i in range(len(context['chunks']))
            ]
        }

    def answer_question(self, question: str) -> dict:
        if not self.document_chunks:
            return {"error": "Please load a legal document first."}
        context = self.find_relevant_context(question)
        return self.generate_descriptive_answer(question, context)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

assistant = LegalAssistant()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        files = request.files.getlist('file')
        if not files:
            return jsonify({"error": "No selected files"}), 400
        file_names = []
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            assistant.process_pdf(file_path)
            file_names.append(filename)
        return jsonify({"message": "Files uploaded and processed successfully", "file_names": file_names}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred while processing the upload: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({"error": "No question provided"}), 400
        response = assistant.answer_question(question)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred while processing the question: {str(e)}"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except SystemExit as e:
        print(f"SystemExit: {e}")
    except OSError as e:
        print(f"OSError: {e}")
