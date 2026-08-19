import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from google import genai
from google.genai import types
from anthropic import Anthropic
from threading import Thread
import requests
import time

app = Flask(__name__)
CORS(app)

print("\n[STARTUP] Loading database...")
with open("law_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"[STARTUP] Loaded {len(data)} entries")

db = chromadb.Client()
collection = db.get_or_create_collection(
    name="pakistan_law",
    metadata={"hnsw:space": "cosine"}
)

for i in range(0, len(data), 100):
    batch = data[i:i+100]
    collection.add(
        documents=[x["response"] for x in batch],
        embeddings=[x["embedding"] for x in batch],
        metadatas=[{"question": x["question"]} for x in batch],
        ids=[x["id"] for x in batch]
    )

print(f"[STARTUP] Database ready: {collection.count()} entries\n")

gemini_key = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_key)

claude_key = os.environ.get("Claude_API_KEY")
claude_client = Anthropic(api_key=claude_key)

def keep_alive():
    while True:
        try:
            url = os.environ.get("RENDER_EXTERNAL_URL")
            if url:
                requests.get(f"{url}/health", timeout=5)
        except:
            pass
        time.sleep(840)

Thread(target=keep_alive, daemon=True).start()

def get_embedding(text):
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
        return jsonify({"error": "No question"}), 400
    
    print(f"\n[QUERY] {question}")
    
    try:
        embedding = get_embedding(question)
        results = collection.query(query_embeddings=[embedding], n_results=3)
        context = "\n\n".join(results["documents"][0])
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({
            "answer": "معاف کریں، سوال سمجھ نہیں آیا۔ دوبارہ کریں۔"
        }), 200
    
    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""You are a Pakistani legal assistant. Answer in the SAME language as the user.

ONLY use this context:
{context}

Question: {question}

Answer:"""
            }]
        )
        
        answer = response.content[0].text
        print(f"[SUCCESS] Answer generated")
        return jsonify({"answer": answer}), 200
    
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({
            "answer": "معاف کریں، جواب نہیں بن سکے۔ دوبارہ کریں۔"
        }), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "entries": collection.count()}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
