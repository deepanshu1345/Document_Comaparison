import os
import requests
from flask import Flask, render_template, request, redirect, url_for
from PyPDF2 import PdfReader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path to the pre-set standard document
STANDARD_DOCUMENT_PATH = 'static/MAHA RERA MODEL AGREEMENT FOR SALE-2.pdf'

# Hugging Face Token
HUGGINGFACE_API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
}


def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def compare_documents(user_doc_path):
    """
    Compares the user document with the pre-set standard document and identifies:
    - Missing points (lines in standard but not in user).
    - Additional points (lines in user but not in standard).
    """
    # Extract text from both documents
    standard_text = extract_text_from_pdf(STANDARD_DOCUMENT_PATH).splitlines()
    user_text = extract_text_from_pdf(user_doc_path).splitlines()

    # Find missing and additional points
    missing_points = [line for line in standard_text if line not in user_text]
    additional_points = [line for line in user_text if line not in standard_text]

    return missing_points, additional_points


def get_gpt_neo_suggestions(diff_text):
    """
    Get suggestions from GPT-Neo based on the missing or additional points using the Hugging Face API.
    """
    prompt = f"Here are the differences between a standard document and a user-submitted document:\n\n{diff_text}\n\nPlease provide suggestions for improving the user document based on these differences. Be specific and actionable."

    payload = {
        "inputs": prompt
    }

    # Send the request to the Hugging Face API
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        # Assuming the model's output is in the 'generated_text' field
        return response_data[0]['generated_text'].strip()
    else:
        return "Error: Could not generate suggestions from the model."


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user_file' not in request.files:
        return redirect(url_for('index'))

    user_file = request.files['user_file']
    if not user_file.filename.endswith('.pdf'):
        return "Please upload a valid PDF file.", 400

    # Save the uploaded file
    user_file_path = os.path.join(app.config['UPLOAD_FOLDER'], user_file.filename)
    user_file.save(user_file_path)

    # Compare documents
    missing_points, additional_points = compare_documents(user_file_path)

    # Format the differences for GPT-Neo
    diff_text = "\n".join(
        [f"Missing: {line}" for line in missing_points] + [f"Additional: {line}" for line in additional_points])

    # Get suggestions from GPT-Neo based on the differences
    gpt_neo_suggestions = get_gpt_neo_suggestions(diff_text)

    return render_template(
        'result.html',
        missing_points=missing_points,
        additional_points=additional_points,
        gpt_neo_suggestions=gpt_neo_suggestions
    )


if __name__ == '__main__':
    app.run(debug=True, port=2323)
