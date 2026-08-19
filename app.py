import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from groq import Groq

app = Flask(__name__)
CORS(app)

# Official Groq Model Name
# Updated active model ID for Groq
MODEL_NAME = "llama3-8b-8192"

# 1. Load Law Data
print("Loading law_data.json...")
with open("law_data.json", "r") as f:
    embedded_data = json.load(f)

# 2. Build Chroma Vector DB
print("Building vector database...")
db_client = chromadb.Client()

try:
    db_client.delete_collection("pakistan_law")
except Exception:
    pass

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

# 3. Setup Groq Client
groq_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Step 1: Translate Question using Groq
    try:
        translation_res = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"""Translate the following question to English.
If already in English, repeat it exactly.
Output ONLY the translated text without extra explanation or quotes.

Question: {question}"""
            }]
        )
        search_query = translation_res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        search_query = question

    # Step 2: Query Vector DB
    try:
        matching_item = None
        for item in embedded_data:
            if search_query.lower() in item["question"].lower():
                matching_item = item
                break

        if matching_item:
            query_emb = matching_item["embedding"]
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=3
            )
            context = "\n\n".join(results["documents"][0])
        else:
            context = "\n\n".join([x["response"] for x in embedded_data[:3]])
    except Exception as e:
        print(f"Retrieval error: {e}")
        context = "\n\n".join([x["response"] for x in embedded_data[:3]])

    # Step 3: Generate Final Answer using Groq
    try:
        prompt = f"""You are a helpful legal assistant for Pakistani law (Family, Criminal, and Property law).
Answer using ONLY the legal context below.
Reply in the EXACT SAME language the user used:
- Urdu script -> Urdu script
- Roman Urdu -> Roman Urdu
- English -> English

If the context does not answer the question, say so honestly.

Legal context:
{context}

User question: {question}
Answer:"""

        answer_res = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"answer": answer_res.choices[0].message.content.strip()})
    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"answer": f"Generation Error: {str(e)}"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "entries": collection.count()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
