import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from google import genai
from google.genai import types
from groq import Groq
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
# Gemini for embeddings only
gemini_key = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_key)
# Groq for text generation
groq_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)
def get_query_embedding(text):
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=[text],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=256
        )
    )
    return [round(v, 6) for v in result.embeddings[0].values]
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        # Translate using Groq
        translation_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""Translate the following question to English.
If already in English, repeat it exactly.
Output only the translated question, nothing else.
Question: {question}"""
            }]
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
        print(f"Embedding error: {e}")
        return jsonify({"answer": "Sorry, could not process your question. Please try again."}), 200
    try:
        # Generate answer using Groq
        answer_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""You are a helpful legal assistant for Pakistani law (Family, Criminal, and Property law).
Answer using ONLY the legal context below.
Reply in the SAME language the user used: Urdu script → Urdu, Roman Urdu → Roman Urdu, English → English.
If the context does not answer the question, say so honestly.
Legal context:
{context}
User question: {question}
Answer:"""
            }]
        )
        return jsonify({"answer": answer_response.choices[0].message.content})
    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"answer": "Sorry, could not generate answer. Please try again."}), 200
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "entries": collection.count()})
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

Thats the code I generated to repalce it with app.py code
