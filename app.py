import sys
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from google import genai
from google.genai import types
from groq import Groq
from threading import Thread
import requests
import time

app = Flask(__name__)
CORS(app)

print("Loading law_data.json...")
with open("law_data.json", "r") as f:
    embedded_data = json.load(f)

print("Building vector database...")
db_client = chromadb.Client()
collection = db_client.get_or_create_collection(
    name="pakistan_law",
    metadata={"hnsw:space": "cosine"}
)

BATCH_SIZE = 100
for i in range(0, len(embedded_data), BATCH_SIZE):
    batch = embedded_data[i:i+BATCH_SIZE]
    collection.add(
        documents=[x["response"] for x in batch],
        embeddings=[x["embedding"] for x in batch],
        metadatas=[{"question": x["question"]} for x in batch],
        ids=[x["id"] for x in batch]
    )

print(f"Database ready with {collection.count()} entries.")

gemini_key = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_key)

groq_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

def keep_alive():
    while True:
        try:
            render_url = os.environ.get("RENDER_EXTERNAL_URL")
            if render_url:
                requests.get(f"{render_url}/health", timeout=5)
        except:
            pass
        time.sleep(840)

pinger_thread = Thread(target=keep_alive, daemon=True)
pinger_thread.start()

def get_query_embedding(text):
    try:
        result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=256
            )
        )
        return [round(v, 6) for v in result.embeddings[0].values]
    except Exception as e:
        print(f"Embedding error: {e}")
        raise

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        translation_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""Translate to English. If already English, repeat exactly.
Output only the translated question, nothing else.
Question: {question}"""
            }],
            timeout=10
        )
        search_query = translation_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        search_query = question
    
    try:
        query_embedding = get_query_embedding(search_query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        context = "\n\n".join(results["documents"][0])
    except Exception as e:
        print(f"Database error: {e}")
        return jsonify({"answer": "معاف کریں، سوال سمجھ نہیں آیا۔ دوبارہ کریں۔"}), 200
    
    try:
        answer_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""You are a legal assistant for Pakistani law.
Answer using ONLY the context below. Reply in the same language as the user.
Urdu → Urdu, Roman Urdu → Roman Urdu, English → English.

Context:
{context}

Question: {question}

Answer:"""
            }],
            timeout=15
        )
        return jsonify({"answer": answer_response.choices[0].message.content}), 200
    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"answer": "معاف کریں، جواب نہیں بن سکے۔ دوبارہ کریں۔"}), 200

@app.route("/health", methods=["GET"])
def health():
    try:
        return jsonify({"status": "ok", "entries": collection.count()}), 200
    except Exception as e:
        return jsonify({"status": "error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
