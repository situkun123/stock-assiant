from dotenv import load_dotenv
import os
from openai import OpenAI

# Load env file
load_dotenv()

# Read API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env or export it in your shell.")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "user", "content": "Say this is a test"}
  ]
)

print(response.choices[0].message.content)