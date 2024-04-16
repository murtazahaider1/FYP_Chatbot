from flask import Flask, request, jsonify
import pandas as pd
import json
from pathlib import Path
import openai
from openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from deepgram import Deepgram
import os

app = Flask(__name__)

CHROMA_PATH = "chroma_db"
chat_history = []
PROMPT_TEMPLATE = """
Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Please make a standalone question using the above information and answer the question based on the context and chat history but don't mention the standalone question when generating the response.
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)

openai_api_key = os.getenv()
client = OpenAI(api_key=openai_api_key)

csv_file_path = 'Data Classification - Sheet1.csv'
data_classification = pd.read_csv(csv_file_path)
json_file_path = 'query_category_counts.json'

deepgram_api_key = 'your_deepgram_api_key'
dg_client = Deepgram(deepgram_api_key)

def load_or_initialize_counts(csv_data, json_path):
    if Path(json_path).exists():
        with open(json_path, 'r') as file:
            return json.load(file)
    else:
        unique_categories = csv_data['Category'].unique()
        counts = {category: 0 for category in unique_categories}
        with open(json_path, 'w') as file:
            json.dump(counts, file, indent=4)
        return counts

def update_category_count(document_name, csv_data, json_path):
    category_series = csv_data[csv_data['Document_Name'] == document_name]['Category']
    if not category_series.empty:
        category = category_series.iloc[0]
        counts = load_or_initialize_counts(csv_data, json_path)
        if category in counts:
            counts[category] += 1
        else:
            counts[category] = 1
        with open(json_path, 'w') as file:
            json.dump(counts, file, indent=4)

@app.route('/input', methods=['POST'])
def handle_input():
    if 'file' in request.files:
        file = request.files['file']
        audio_data = file.read()
        transcript = transcribe_audio(audio_data)
        input_text = transcript
    else:
        input_text = request.form.get('text', '')

    response_text = process_query(input_text)
    return jsonify({'response': response_text})

def transcribe_audio(audio_data):
    # This should contain the logic to transcribe audio to text
    response = dg_client.transcription.prerecorded(
        {'buffer': audio_data, 'mimetype': 'audio/wav'},
        {'punctuate': True}
    ).result()
    transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
    return transcript

def process_query(query_text):
    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    global chat_history

    if query_text.lower() == 'exit':
        return 'Goodbye!'

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        messages_for_api = [{"role": "system", "content": "You are a helpful assistant."}]
        for entry in chat_history[-3:]:
            messages_for_api.append(entry)
        messages_for_api.append({"role": "user", "content": query_text})

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=messages_for_api
        )
        response_text = completion.choices[0].message.content
    else:
        context_text = "\\n\\n---\\n\\n".join([doc.page_content for doc, _score in results])
        chat_history_str = "\\n".join([entry['content'] for entry in chat_history])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, chat_history=chat_history_str, question=query_text)
        
        model = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-4")
        response_text = model.predict(prompt)

        if len(results) > 0 and results[0][1] >= 0.7:
            sources = [doc.metadata.get("source", None) for doc, _score in results]
            update_category_count(sources[0], data_classification, json_file_path)

    # Update chat history, keep last 3 interactions only
    chat_history.append({"role": "user", "content": query_text})
    chat_history.append({"role": "assistant", "content": response_text})
    if len(chat_history) > 6:  # Keeping only the last 3 interactions (6 entries: 3 questions, 3 answers)
        chat_history = chat_history[-6:]

    return response_text

if __name__ == "__main__":
    app.run(debug=True)
