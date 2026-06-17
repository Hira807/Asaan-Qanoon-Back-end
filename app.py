import os
import gradio as gr
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

print("Loading dataset...")
dataset = load_dataset("heyIamUmair/pakistani-law-family-criminal-property")
data = dataset["train"]
queries = list(data["Query"])
responses = list(data["Response"])

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Building vector database...")
db_client = chromadb.Client()
collection = db_client.get_or_create_collection(name="pakistan_law")

embeddings = embedder.encode(queries)
collection.add(
    documents=responses,
    embeddings=embeddings.tolist(),
    metadatas=[{"question": q} for q in queries],
    ids=[str(i) for i in range(len(queries))]
)
print(f"Database ready with {collection.count()} entries.")

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def ask_legal_question(question, n_results=3):
    translation_prompt = f"""Translate the following question to English.
If it's already in English, just repeat it exactly as-is.
Only output the translated question, nothing else, no explanation.

Question: {question}"""
    translation_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=translation_prompt
    )
    search_query = translation_response.text.strip()

    results = collection.query(
        query_embeddings=embedder.encode([search_query]).tolist(),
        n_results=n_results
    )
    retrieved_texts = results["documents"][0]
    context = "\n\n".join(retrieved_texts)

    prompt = f"""You are a helpful legal assistant for Pakistani law (Family, Criminal, and Property law).
Answer the user's question using ONLY the legal context provided below.
Respond in the SAME language and script the user used in their original question.
If the context doesn't clearly answer the question, say so honestly instead of guessing.

Legal context:
{context}

User's original question: {question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

demo = gr.Interface(
    fn=ask_legal_question,
    inputs=gr.Textbox(label="Apna sawal likhein (English, Urdu, ya Roman Urdu mein)"),
    outputs=gr.Textbox(label="Jawab"),
    title="Asaan Qanoon",
    description="Pakistan ke Family, Criminal aur Property Law ke baare mein sawal poochain."
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
