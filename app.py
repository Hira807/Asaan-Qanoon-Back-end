import sys
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from groq import Groq

app = Flask(__name__)
CORS(app)

print("Loading law_data.json...")
with open("law_data.json", "r", encoding="utf-8") as f:
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

# Groq client configuration for all text tasks
groq_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)


# =====================================================================
# EXISTING ROUTE: Pakistani law chatbot 
# =====================================================================
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        # Translate or normalize query using Groq
        translation_response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""Translate the following question to English if it is in Urdu or Roman Urdu.
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
        # For a pure Groq fallback without Gemini embedding model, we use keyword/semantic text matching or dummy vector if needed.
        # But since your JSON already has pre-computed embeddings, we can use Groq to summarize or query.
        # Here we query ChromaDB using a fallback query embedding or text search.
        results = collection.query(
            query_texts=[search_query],
            n_results=3
        )
        context = "\n\n".join(results["documents"][0])
    except Exception as e:
        print(f"Retrieval error: {e}")
        return jsonify({"answer": "Sorry, could not process your question. Please try again."}), 200

    try:
        # Generate answer using Groq Llama 3.3 70B
        answer_response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
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


# =====================================================================
# NEW (EXHIBITION DEMO): Islamic guidance feature
# =====================================================================
ISLAMIC_REFERENCES = [
    {
        "ref": "Quran 4:11", 
        "area": "Wirasat / Inheritance (Property, Family)", 
        "text": "Allah instructs you concerning your children: for the male, what is equal to the share of two females..."
    }
]

_verified = [r for r in ISLAMIC_REFERENCES if "PASTE VERIFIED" not in r["text"]]
ISLAMIC_REF_BLOCK = "\n".join(
    f"[{r['ref']}] ({r['area']}): {r['text']}" for r in _verified
)

ISLAMIC_SYSTEM_PROMPT = f"""You are the 'Islamic Guidance' feature of Asaan Qanoon, a DEMO for Pakistani law.
Reply in the user's language style (Roman Urdu, Urdu, or English).

You may ONLY use these verified references:
{ISLAMIC_REF_BLOCK}

RULES:
1. Quote/refer ONLY to the references above, with their bracket label. Never cite any
    other ayat, hadith, scholar, book or fatwa.
2. If the question is not covered by the references, say exactly:
    "Is sawal ka mustanad reference is demo mein shamil nahi. Kisi qualified aalim se rujoo karein."
3. Never give a fatwa or say something is halal/haram as a ruling.
4. Keep it short (under 150 words), respectful.
5. Finish with: "Yeh demo guidance hai, fatwa ya qanooni mashwara nahi."
"""

@app.route("/api/islamic-lite", methods=["POST"])
def islamic_lite():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            temperature=0.1,
            max_tokens=400,
            messages=[
                {"role": "system", "content": ISLAMIC_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        return jsonify({"answer": response.choices[0].message.content})
    except Exception as e:
        print(f"Islamic demo error: {e}")
        return jsonify({"answer": "Sorry, could not generate answer. Please try again."}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "entries": collection.count()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
