import os
import re
import json
import math
import logging
from collections import Counter, defaultdict

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("asaan-qanoon")

app = Flask(__name__)

# Production mein Render Environment mein ALLOWED_ORIGINS set kar sakti ho, e.g.
# "https://your-site.com,http://127.0.0.1:5500,http://localhost:5500"
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
CORS(app, origins=_origins)

MODEL = "llama-3.3-70b-versatile"
REWRITE_MODEL = "llama-3.1-8b-instant"
MAX_QUESTION_CHARS = 1000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.environ.get("LAW_DATA_FILE", "law_data.json")

groq_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key) if groq_key else None
if not groq_client:
    log.error("GROQ_API_KEY is not set. /ask and /api/islamic-lite will return 503.")


# ---------------------------------------------------------------------
# Halka BM25 search (chromadb ki jagah): koi model download nahi, RAM bohat kam
# ---------------------------------------------------------------------
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    "the", "is", "are", "a", "an", "of", "to", "in", "on", "for", "and", "or", "how", "what",
    "can", "do", "does", "i", "my", "me", "be", "by", "with", "at", "it", "this", "that",
    "kya", "hai", "ka", "ki", "ke", "mein", "main", "se", "ko", "aur", "kar", "karna", "kaise",
}


def tokenize(text):
    return [t for t in TOKEN_RE.findall((text or "").lower()) if len(t) > 1 and t not in STOPWORDS]


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.dl = [len(d) for d in docs_tokens]
        n = len(docs_tokens)
        self.avgdl = (sum(self.dl) / n) if n else 1.0
        df = Counter()
        self.inv = defaultdict(list)
        for i, d in enumerate(self.tf):
            for t in d:
                df[t] += 1
                self.inv[t].append(i)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query_tokens, k=3):
        scores = defaultdict(float)
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.inv[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


LAWS = []      # list of {"question":..., "response":...}
INDEX = None
LOAD_ERROR = None


def load_laws():
    global INDEX, LOAD_ERROR
    try:
        path = os.path.join(BASE_DIR, DATA_FILE)
        log.info("Loading %s ...", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs_tokens = []
        for x in data:
            q = (x.get("question") or "").strip()
            r = (x.get("response") or x.get("answer") or "").strip()
            if not r:
                continue
            LAWS.append({"question": q, "response": r})
            # question ko double weight dete hain
            docs_tokens.append(tokenize(q) * 2 + tokenize(r))

        INDEX = BM25(docs_tokens)
        log.info("Search index ready with %s entries.", len(LAWS))
    except Exception as e:
        LOAD_ERROR = str(e)
        log.exception("Failed to load law data")


load_laws()


def rewrite_query(question):
    """Roman Urdu / Urdu sawal ko English search keywords mein badalta hai (agar fail ho to '' )."""
    if not groq_client:
        return ""
    try:
        r = groq_client.chat.completions.create(
            model=REWRITE_MODEL,
            temperature=0,
            max_tokens=40,
            messages=[
                {"role": "system", "content": "Convert the user's legal question (Urdu, Roman Urdu or English) into 5-8 English search keywords about Pakistani law. Output only the keywords, nothing else."},
                {"role": "user", "content": question},
            ],
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        log.warning("Query rewrite failed: %s", e)
        return ""


def retrieve(question, k=3):
    if INDEX is None:
        return []
    tokens = tokenize(question) + tokenize(rewrite_query(question))
    hits = INDEX.search(tokens, k=k)
    return [LAWS[i] for i, _ in hits]


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
    if INDEX is None:
        return jsonify({"error": "Law database failed to load."}), 503

    try:
        hits = retrieve(question, k=3)
    except Exception as e:
        log.exception("Retrieval error: %s", e)
        hits = []

    if hits:
        context = "\n\n".join(f"Q: {h['question']}\nA: {h['response'][:1500]}" for h in hits)
    else:
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
        "status": "ok" if INDEX is not None else "error",
        "entries": len(LAWS),
        "groq_configured": groq_client is not None,
        "load_error": LOAD_ERROR,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
