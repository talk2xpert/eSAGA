import os
import secrets
import time

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS

import rnd.llm_vector_similarity as vector_simi
from rnd import  llm_rag_pdf as llm_rag
import rnd.LLMUTILS as LLMU
from rnd import  llm_simple_prompt as simple
app = Flask(__name__)
CORS(app)

# Directory to store pickle files
EMBEDDINGS_DIR = 'embeddings'
if not os.path.exists(EMBEDDINGS_DIR):
    os.makedirs(EMBEDDINGS_DIR)
    # Directory to temporarily store uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set a secret key for the session
app.secret_key = secrets.token_hex(16)

@app.route('/')
def index():
    return render_template('index_rag.html')
@app.route('/for_bot')
def index1():
    return render_template('index_rag1.html')

@app.route('/bot', methods=['POST'])
def simple_prompt():

    data = request.json
    query = data.get('query')
    prompt_type = data.get('type')
    print("******************In Chat Bot*****************")

    # Simulate response based on prompt type
    if prompt_type == 'question':
        response = f"You asked: '{query}'. Here is a response to your question." + LLMU.simple_prompt_qa(query)
    elif prompt_type == 'translation':
        language = data.get('language')
        response = f"You asked to translate: '{query}'. Here is the translation." ++ LLMU.simple_prompt_translater(query,language)
    else:
        response = "Sorry, I didn't understand your request."

    return jsonify({'response': response})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    file = request.files['file']
    print(file)

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})

    # Store the filename in session
    session['uploaded_filename'] = file.filename

    # Simulate file processing
    time.sleep(5)  # Simulate a delay for file

    # Save the uploaded file temporarily
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)


    return jsonify({'status': 'success', 'message': 'File uploaded successfully'})

@app.route('/query', methods=['POST'])
def query_file():
    MODEL_NAME = "gemma:2b"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'],session['uploaded_filename'] )
    data = request.json
    query = data.get('query')
    print(query)
    use_llm = data.get('use_llm')
    #use_llm=False
    print(use_llm)
    results=[]
    if(use_llm==False):
        results = vector_simi.get_Vector_DB_Results(file_path,query)
        result_strings = [x.dict()['page_content'] for x in results]
        return jsonify(status='success', response=result_strings)
    else:
        results=llm_rag.get_llm_rag_query_results(file_path,query)
        return jsonify(status='success', response=results)


if __name__ == '__main__':
    app.run(debug=True)
    #http://127.0.0.1:8081/for_bot
