import torch
from transformers import DistilBertTokenizer, DistilBertModel, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import faiss
import numpy as np

# Load the Wikipedia dataset
dataset = load_dataset('wikipedia', '20220301.simple', split='train[:1%]', trust_remote_code=True)


# Preprocess the dataset
def preprocess_function(examples):
    return {'text': examples['text'][:512]}


# Load tokenizer and model for DistilBert
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Tokenize and encode the dataset
encoded_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding=True), batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Create document embeddings
embeddings = []
with torch.no_grad():
    for batch in encoded_dataset:
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        embeddings.append(outputs.last_hidden_state[:, 0, :].detach().cpu().numpy())

embeddings = np.vstack(embeddings)

# Set up FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Load GPT-2 model for generation
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')


# Function for RAG
def rag_query(query, k=5):
    # Encode the query
    query_tokens = tokenizer(query, return_tensors='pt')
    query_embedding = model(**query_tokens).last_hidden_state[:, 0, :].detach().cpu().numpy()

    # Retrieve relevant documents
    _, indices = index.search(query_embedding, k)
    context = "\n".join([dataset[i]['text'] for i in indices[0]])

    # Generate response using GPT-2
    input_text = query + "\n\n" + context
    input_tokens = gpt2_tokenizer.encode(input_text, return_tensors='pt')
    output_tokens = gpt2_model.generate(input_tokens, max_length=512, num_return_sequences=1)

    return gpt2_tokenizer.decode(output_tokens[0], skip_special_tokens=True)


# Example query
query = "What is machine learning?"
response = rag_query(query)
print(response)
