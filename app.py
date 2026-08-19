import os
from groq import Groq

# Set your API key
groq_key = os.environ.get("GROQ_API_KEY") # ya direct "gsk_..." likh kar test karein
client = Groq(api_key=groq_key)

# 1. Available Models List Print Karein
print("--- Available Models on Groq ---")
models = client.models.list()
for m in models.data:
    print(m.id)

print("\n--- Testing Llama 3.3 ---")
# 2. Llama 3.3 Test Request
try:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Hello Llama 3.3, respond in 5 words."}]
    )
    print("Success! Llama 3.3 Response:", response.choices[0].message.content)
except Exception as e:
    print("Llama 3.3 Test Failed:", e)
