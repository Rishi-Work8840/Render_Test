from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise SystemExit("Error: Set the GROQ_API_KEY environment variable first.")

client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key=api_key,
  timeout=30.0,
)

print("Connecting to Groq...")

try:
    response = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[
        {"role": "user", "content": "Explain the role of api keys in API authentication."}
      ]
    )
    print("\nSuccess! Here is the response:")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\nAn error occurred: {e}")