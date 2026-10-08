import sys
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


# =====================================================================
# EXISTING ROUTE: Pakistani law chatbot (unchanged)
# =====================================================================
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


# =====================================================================
# NEW (EXHIBITION DEMO): Islamic guidance, curated references, no RAG
# Exhibition ke baad hatane ke liye: yeh poora section delete kar do.
# =====================================================================

# Har "text" tanzil.net se KHUD verify karke paste karo (translation wording copy-paste).
# Jab tak "PASTE VERIFIED TRANSLATION HERE" likha hai, woh entry bot ko nahi dikhegi.
ISLAMIC_REFERENCES = [
    {"ref": "Quran 4:11-12",   "area": "Wirasat / Inheritance (Property, Family)", "text": "PASTE VERIFIED TRANSLATION HERE"},
    {"ref": "Quran 4:4",       "area": "Haq mehr (Family)",                         "text": "PASTE VERIFIED TRANSLATION HERE"},
    {"ref": "Quran 2:228",     "area": "Iddat (Family)",                            "text": "PASTE VERIFIED TRANSLATION HERE"},
    {"ref": "Quran 2:229",     "area": "Talaq / Khula (Family)",                    "text": "PASTE VERIFIED TRANSLATION HERE"},
    {"ref": "Quran 4:19",      "area": "Husn-e-sulook with spouse (Family)",        "text": "PASTE VERIFIED TRANSLATION HERE"},
    {"ref": "Quran 4:2, 4:10", "area": "Orphans' property (Property)",              "text": "PASTE VERIFIED TRANSLATION HERE"},
    {"ref": "Quran 4:29",      "area": "Unlawful consumption of wealth (Property)", "text": "PASTE VERIFIED TRANSLATION HERE"},
    {"ref": "Quran 2:282",     "area": "Documenting debts/transactions (Property)", "text": "PASTE VERIFIED TRANSLATION HERE"},
    {"ref": "Quran 2:178",     "area": "Qisas (Criminal)",                          "text": "PASTE VERIFIED TRANSLATION HERE"},
    {"ref": "Quran 5:32",      "area": "Sanctity of human life (Criminal)",         "text": "PASTE VERIFIED TRANSLATION HERE"},
    {"ref": "Quran 17:33",     "area": "Prohibition of unjust killing (Criminal)",  "text": "PASTE VERIFIED TRANSLATION HERE"},
]

_verified = [r for r in ISLAMIC_REFERENCES if "PASTE VERIFIED" not in r["text"]]
ISLAMIC_REF_BLOCK = "\n".join(
    f"[{r['ref']}] ({r['area']}): {r['text']}" for r in _verified
)
print(f"Islamic demo: {len(_verified)} verified references loaded.")

ISLAMIC_SYSTEM_PROMPT = f"""You are the 'Islamic Guidance' feature of Asaan Qanoon, a DEMO for Pakistani law.
Reply in the user's language style (Roman Urdu, Urdu, or English).

You may ONLY use these verified references:
{ISLAMIC_REF_BLOCK}

RULES:
1. Quote/refer ONLY to the references above, with their bracket label. Never cite any
   other ayat, hadith, scholar, book or fatwa, and never recall references from memory.
2. If the question is not covered by the references, say exactly:
   "Is sawal ka mustanad reference is demo mein shamil nahi. Kisi qualified aalim se rujoo karein."
3. Never give a fatwa or say something is halal/haram as a ruling. Explain what the
   reference says and mention that scholars/fiqh details may differ.
4. For sensitive criminal topics (qisas, hudood), explain only; give no advice on actions.
5. Keep it short (under 150 words), respectful, no sectarian or political comments.
6. Finish with: "Yeh demo guidance hai, fatwa ya qanooni mashwara nahi."
"""


@app.route("/api/islamic-lite", methods=["POST"])
def islamic_lite():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not _verified:
        return jsonify({"answer": "Islamic guidance demo abhi set up ho raha hai. Please thori der baad try karein."}), 200

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
