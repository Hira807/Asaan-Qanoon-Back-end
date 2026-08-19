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

print("\n" + "="*60)
print("ASAAN QANOON - STARTING UP")
print("="*60)

# Load law database
print("\n[STARTUP] Loading law_data.json...")
try:
    with open("law_data.json", "r", encoding="utf-8") as f:
        embedded_data = json.load(f)
    print(f"[STARTUP] ✅ Loaded {len(embedded_data)} law entries")
except Exception as e:
    print(f"[STARTUP] ❌ Failed to load law_data.json: {e}")
    sys.exit(1)

# Build vector database
print("\n[STARTUP] Building ChromaDB vector database...")
try:
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
    
    print(f"[STARTUP] ✅ Database ready with {collection.count()} entries")
except Exception as e:
    print(f"[STARTUP] ❌ Database error: {e}")
    sys.exit(1)

# Initialize API clients
print("\n[STARTUP] Initializing API clients...")
try:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not set")
    gemini_client = genai.Client(api_key=gemini_key)
    print("[STARTUP] ✅ Gemini client initialized")
except Exception as e:
    print(f"[STARTUP] ❌ Gemini error: {e}")
    sys.exit(1)

try:
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not set")
    groq_client = Groq(api_key=groq_key)
    print("[STARTUP] ✅ Groq client initialized")
except Exception as e:
    print(f"[STARTUP] ❌ Groq error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("STARTUP COMPLETE - SERVER READY")
print("="*60 + "\n")

# ===== KEEP-ALIVE PINGER =====
def keep_alive():
    """Prevent Render free tier spin-down"""
    while True:
        try:
            render_url = os.environ.get("RENDER_EXTERNAL_URL")
            if render_url:
                requests.get(f"{render_url}/health", timeout=5)
        except:
            pass
        time.sleep(840)  # Every 14 minutes

pinger_thread = Thread(target=keep_alive, daemon=True)
pinger_thread.start()

def get_query_embedding(text):
    """Convert text to embedding vector"""
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
        print(f"[Embedding] ❌ Error: {e}")
        raise

@app.route("/ask", methods=["POST"])
def ask():
    """Main endpoint for legal questions"""
    data = request.get_json()
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    print(f"\n{'='*60}")
    print(f"[QUERY] Question: {question}")
    print(f"{'='*60}")
    
    # Step 1: Translate to English
    try:
        print("[STEP 1] Translating to English...")
        translation_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""Translate this to English.
If already English, repeat exactly.
Output ONLY the translated text, nothing else.

Question: {question}"""
            }],
            timeout=10
        )
        search_query = translation_response.choices[0].message.content.strip()
        print(f"[STEP 1] ✅ Translated: {search_query}")
        
    except Exception as e:
        print(f"[STEP 1] ❌ Translation failed: {e}")
        search_query = question

    # Step 2: Get embedding + search
    try:
        print("[STEP 2] Getting embedding & searching database...")
        query_embedding = get_query_embedding(search_query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        if not results["documents"] or not results["documents"][0]:
            raise ValueError("No results found in database")
        
        context = "\n\n".join(results["documents"][0])
        print(f"[STEP 2] ✅ Found {len(results['documents'][0])} relevant laws")
        
    except Exception as e:
        print(f"[STEP 2] ❌ Database error: {e}")
        return jsonify({
            "answer": "معاف کریں، آپ کے سوال کو سمجھنے میں مسئلہ ہو رہا ہے۔ براہ کرم دوبارہ کوشش کریں۔\n\nSorry, could not process your question. Please try again."
        }), 200

    # Step 3: Generate answer
    try:
        print("[STEP 3] Generating answer with Groq...")
        answer_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""You are a helpful legal assistant for Pakistani law (Family, Criminal, Property).

ANSWER RULES:
1. Use ONLY the legal context below
2. Answer in the SAME language as user: Urdu → Urdu, Roman Urdu → Roman Urdu, English → English
3. Be clear and practical
4. If context doesn't answer, say so honestly
5. Keep answer 200-400 words

LEGAL CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""
            }],
            timeout=20
        )
        
        answer = answer_response.choices[0].message.content.strip()
        print(f"[STEP 3] ✅ Answer generated ({len(answer)} chars)")
        print(f"{'='*60}\n")
        
        return jsonify({"answer": answer}), 200
        
    except Exception as e:
        print(f"[STEP 3] ❌ Generation error: {e}")
        return jsonify({
            "answer": "معاف کریں، جواب تیار نہیں ہو سکے۔ براہ کرم دوبارہ کوشش کریں۔\n\nSorry, could not generate answer. Please try again."
        }), 200

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    try:
        count = collection.count()
        return jsonify({
            "status": "ok",
            "entries": count,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Asaan Qanoon API on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
