from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Conversation

app = Flask(__name__)

# Initialize Hugging Face Transformers pipelines and models
classifier = pipeline("sentiment-analysis")
generator = pipeline("text-generation")
ner = pipeline("ner")
summarizer = pipeline("summarization")
translator = pipeline("translation_en_to_fr")
qa = pipeline("question-answering")
zero_shot_classifier = pipeline("zero-shot-classification")
conversational_agent = pipeline("conversational")
text_generator = pipeline("text-generation")

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/sentiment', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        text = request.form['text']
        sentiment_result = classifier(text)
        return render_template('index1.html', sentiment_result=sentiment_result, task='sentiment')

@app.route('/generation', methods=['POST'])
def generate_text():
    if request.method == 'POST':
        prompt = request.form['prompt']
        generated_text = generator(prompt, max_length=50, num_return_sequences=1)
        return render_template('index1.html', generated_text=generated_text, task='generation')

@app.route('/ner', methods=['POST'])
def recognize_entities():
    if request.method == 'POST':
        text = request.form['text']
        entities = ner(text)
        return render_template('index1.html', entities=entities, task='ner')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    if request.method == 'POST':
        text = request.form['text']
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return render_template('index1.html', summary=summary[0]['summary_text'], task='summarization')

@app.route('/translate', methods=['POST'])
def translate_text():
    if request.method == 'POST':
        text = request.form['text']
        translation = translator(text)
        return render_template('index1.html', translation=translation[0]['translation_text'], task='translation')

@app.route('/qa', methods=['POST'])
def answer_question():
    if request.method == 'POST':

        question = request.form['question']
        context = request.form['context']
        # Get the answer from the QA pipeline
        answer = qa(question=question, context=context)["answer"]

        return render_template('index1.html', answer=answer, task='question-answering')

@app.route('/zero_shot', methods=['POST'])
def classify_text():
    if request.method == 'POST':
        text = request.form['text']
        candidate_labels = request.form.getlist('candidate_labels')
        result = zero_shot_classifier(text, candidate_labels=candidate_labels)
        return render_template('index1.html', classification=result['labels'][0], confidence=result['scores'][0], task='zero-shot-classification')

@app.route('/chat', methods=['POST'])
def chat():
    if request.method == 'POST':
        print("****************************")
        chatbot = pipeline("conversational")
        print("*****************")
        user_input = request.form['user_input']
        # Generate a response from the chatbot based on user input
        # Generate a response from the chatbot based on user input
        response = chatbot(user_input, attention_mask=[[1] * len(user_input.split())])
        print(response)
        print(f"Chatbot: {response['generated_text']}")
        return render_template('index1.html', response=response['generated_text'], task='conversational')

@app.route('/generate_text', methods=['POST'])
def generate_text_lm():
    if request.method == 'POST':
        prompt = request.form['prompt']
        generated_text = text_generator(prompt, max_length=100, do_sample=True)
        return render_template('index1.html', generated_text=generated_text[0]['generated_text'], task='language-modeling')

@app.route('/classification', methods=['POST'])
def classify_sentiment():
    if request.method == 'POST':
        text = request.form['text']
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = tokenizer.decode(outputs.logits.argmax())
        return render_template('index1.html', predicted_sentiment=predicted_class, input_text=text, task='classification')

if __name__ == '__main__':
    app.run(debug=True)
