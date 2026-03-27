import os
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI


load_dotenv()


llm = GoogleGenAI(
    model="models/gemini-3-flash-preview", 
    api_key=os.getenv("GOOGLE_API_KEY")
)


print("Sending test message...")
response = llm.complete("Hello! Are you ready to analyze some urban planning documents?")


print("\n--- AI RESPONSE ---")
print(response)