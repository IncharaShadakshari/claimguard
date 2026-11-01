import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

print("Models available to you:")
for m in client.models.list().data:
    print(" ", m.id)

resp = client.chat.completions.create(
    model="openai/gpt-oss-20b",   # change this if it is not in the list above
    messages=[{"role": "user", "content": "Reply with the single word: ready"}],
)
print("\nModel replied:", resp.choices[0].message.content)