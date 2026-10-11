import os
import json
import logging
import threading

from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("asaan-qanoon")

app = Flask(__name__)

# Production mein ALLOWED_ORIGINS set karo, e.g. "https://your-frontend.com,http://localhost:3000"
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
CORS(app, origins=_origins)

MODEL = "llama-3.3-70b-versatile"
MAX_QUESTION_CHARS = 1000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# Groq client (key missing ho to app crash na kare)
# ---------------------------------------------------------------------
groq_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key) if groq_key else None
if not groq_client:
    log.error("GROQ_API_KEY is not set. /ask and /api/islamic-lite will return 503.")

# ---------------------------------------------------------------------
# Vector DB: background thread mein load hota hai taake server foran
# port bind kare (Render timeout / gunicorn worker timeout se bachao)
# ---------------------------------------------------------------------
collection = None
db_state = {"status": "loading", "error": None}


def build_db():
    global collection
    try:
        log.info("Loading law_data.json...")
        with open(os.path.join(BASE_DIR, "law_data.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        log.info("Building vector database...")
        client = chromadb.Client()
        col = client.get_or_create_collection(
            name="pakistan_law_safe",
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            col.upsert(  # upsert: duplicate ids pe crash nahi hoga
                documents=[x["response"] for x in batch],
                metadatas=[{"question": x["question"]} for x in batch],
                ids=[str(x["id"]) for x in batch],
            )

        collection = col
        db_state["status"] = "ready"
        log.info("Database ready with %s entries.", col.count())
    except Exception as e:
        log.exception("Database build failed")
        db_state["status"] = "error"
        db_state["error"] = str(e)


threading.Thread(target=build_db, daemon=True).start()


def read_question():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return None, (jsonify({"error": "No question provided"}), 400)
    if len(question) > MAX_QUESTION_CHARS:
        return None, (jsonify({"error": f"Question too long (max {MAX_QUESTION_CHARS} characters)"}), 400)
    return question, None


# =====================================================================
# ASK ROUTE: Pakistani law chatbot
# =====================================================================
LAW_SYSTEM_PROMPT = """You are a helpful legal information assistant for Pakistani law (Family, Criminal, and Property law).

Rules:
1. Answer ONLY using the legal context provided. If the context does not cover the question, say so clearly and suggest consulting a qualified lawyer. Do not invent section numbers, case names or penalties.
2. Reply in the SAME language the user used: Urdu script -> Urdu, Roman Urdu -> Roman Urdu, English -> English.
3. Keep the answer clear and practical.
4. End with a short note that this is general information, not legal advice."""


@app.route("/ask", methods=["POST"])
def ask():
    question, err = read_question()
    if err:
        return err

    if not groq_client:
        return jsonify({"error": "Server is not configured (missing API key)."}), 503
    if collection is None:
        msg = "Database is still loading, please try again in a minute."
        if db_state["status"] == "error":
            msg = "Database failed to load."
        return jsonify({"error": msg}), 503

    try:
        results = collection.query(query_texts=[question], n_results=3)
        docs = (results.get("documents") or [[]])[0]
        context = "\n\n".join(docs) if docs else ""
    except Exception as e:
        log.exception("Retrieval error: %s", e)
        context = ""

    if not context:
        context = "NO RELEVANT LEGAL CONTEXT WAS FOUND for this question."

    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            max_tokens=800,
            messages=[
                {"role": "system", "content": LAW_SYSTEM_PROMPT},
                {"role": "user", "content": f"Legal context:\n{context}\n\nUser question: {question}"},
            ],
        )
        return jsonify({"answer": resp.choices[0].message.content})
    except Exception as e:
        log.exception("Generation error: %s", e)
        return jsonify({"error": "Could not generate an answer. Please try again."}), 502


# =====================================================================
# ISLAMIC GUIDANCE ROUTE
# =====================================================================
ISLAMIC_REFERENCES = [
    {
        "ref": "Quran 4:11",
        "area": "Wirasat / Inheritance",
        # TODO: yahan poori verified translation paste karo (abhi "..." se truncated hai)
        "text": "Allah instructs you concerning your children: for the male, what is equal to the share of two females...",
    }
]

ISLAMIC_REF_BLOCK = "\n".join(
    f"[{r['ref']}] ({r['area']}): {r['text']}" for r in ISLAMIC_REFERENCES
)

ISLAMIC_SYSTEM_PROMPT = f"""You are the 'Islamic Guidance' feature of Asaan Qanoon, a DEMO for Pakistani law.
Reply in the user's language style (Roman Urdu, Urdu, or English).

You may ONLY use these verified references:
{ISLAMIC_REF_BLOCK}

RULES:
1. Quote/refer ONLY to the references above.
2. If the question is not covered, say: "Is sawal ka mustanad reference is demo mein shamil nahi. Kisi qualified aalim se rujoo karein."
3. Keep it short and respectful.
4. Finish with: "Yeh demo guidance hai, fatwa ya qanooni mashwara nahi."
"""


@app.route("/api/islamic-lite", methods=["POST"])
def islamic_lite():
    question, err = read_question()
    if err:
        return err
    if not groq_client:
        return jsonify({"error": "Server is not configured (missing API key)."}), 503

    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            temperature=0.1,
            max_tokens=400,
            messages=[
                {"role": "system", "content": ISLAMIC_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        return jsonify({"answer": resp.choices[0].message.content})
    except Exception as e:
        log.exception("Islamic demo error: %s", e)
        return jsonify({"error": "Could not generate an answer. Please try again."}), 502


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if db_state["status"] == "ready" else db_state["status"],
        "entries": collection.count() if collection is not None else 0,
        "groq_configured": groq_client is not None,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
