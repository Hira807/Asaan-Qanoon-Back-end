import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from groq import Groq
from threading import Thread
import requests
import time

app = Flask(__name__)
CORS(app)

# Active, Fast and Stable Groq Model
MODEL_NAME = "llama-3.1-8b-instant"

print("\n" + "="*70)
print("ASAAN QANOON - STARTING UP")
print("="*70)

# 1. Load law database
print("\n[STARTUP] Loading law_data.json...")
try:
    with open("law_data.json", "r", encoding="utf-8") as f:
        embedded_data = json.load(f)
    print(f"[STARTUP] ✅ Loaded {len(embedded_data)} law entries")
except Exception as e:
    print(f"[STARTUP] ❌ Failed to load: {e}")
    exit(1)

# 2. Build ChromaDB Vector DB
print("[STARTUP] Building ChromaDB...")
try:
    db = chromadb.Client()
    
    try:
        db.delete_collection("pakistan_law")
    except Exception:
        pass

    collection = db.get_or_create_collection(
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
    
    print(f"[STARTUP] ✅ Database ready: {collection.count()} entries")
except Exception as e:
    print(f"[STARTUP] ❌ Database error: {e}")
    exit(1)

# 3. Setup Groq Client
print("\n[STARTUP] Initializing Groq client...")
try:
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found")
    groq_client = Groq(api_key=groq_key)
    print(f"[STARTUP] ✅ Groq client ready ({MODEL_NAME})")
except Exception as e:
    print(f"[STARTUP] ❌ Groq error: {e}")
    exit(1)

print("\n" + "="*70)
print("STARTUP COMPLETE - SERVER READY")
print("="*70 + "\n")

# Keep-alive pinger for Render
def keep_alive():
    while True:
        try:
            url = os.environ.get("RENDER_EXTERNAL_URL")
            if url:
                requests.get(f"{url}/health", timeout=5)
        except Exception:
            pass
        time.sleep(840)

Thread(target=keep_alive, daemon=True).start()

@app.route("/ask", methods=["POST"])
def ask():
    req_data = request.get_json()
    question = req_data.get("question", "").strip() if req_data else ""
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    print(f"\n{'='*70}")
    print(f"[QUERY] Question: {question}")
    print(f"{'='*70}")
    
    # Step 1: Search database using pre-computed embeddings
    try:
        print("[STEP 1] Searching database...")
        
        question_embedding = None
        for entry in embedded_data:
            if entry["question"].lower() == question.lower():
                question_embedding = entry["embedding"]
                break
        
        if not question_embedding:
            question_embedding = embedded_data[0]["embedding"]
        
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        
        if not results["documents"] or not results["documents"][0]:
            raise ValueError("No matching laws found")
        
        context = "\n\n".join(results["documents"][0])
        print(f"[STEP 1] ✅ Found {len(results['documents'][0])} relevant entries")
        
    except Exception as e:
        print(f"[ERROR] Database error: {e}")
        return jsonify({
            "answer": "معاف کریں، سوال سمجھ نہیں آیا۔ براہ کرم دوبارہ کوشش کریں۔"
        }), 200
    
    # Step 2: Generate answer with Groq
    try:
        print("[STEP 2] Generating answer with Groq...")
        
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"""You are a helpful Pakistani legal assistant for Family, Criminal, and Property law.

IMPORTANT RULES:
1. Answer in the SAME language as the user
   - Urdu script → Answer in Urdu
   - Roman Urdu → Answer in Roman Urdu
   - English → Answer in English
2. Use ONLY the legal context below
3. Be clear, practical, and concise
4. Keep answer 200-400 words

LEGAL CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""
            }],
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        print(f"[STEP 2] ✅ Answer generated ({len(answer)} chars)")
        print(f"{'='*70}\n")
        
        return jsonify({"answer": answer}), 200
    
    except Exception as e:
        print(f"[ERROR] Groq generation error: {e}")
        return jsonify({
            "answer": "معاف کریں، جواب تیار نہیں ہو سکے۔ براہ کرم دوبارہ کوشش کریں۔"
        }), 200

@app.route("/health", methods=["GET"])
def health():
    try:
        return jsonify({
            "status": "ok",
            "entries": collection.count(),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Asaan Qanoon API on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
