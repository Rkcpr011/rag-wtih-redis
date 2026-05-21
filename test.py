from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = OpenAI(base_url=endpoint,api_key=api_key)
 

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "explain RAG in simple terms",
        }
    ],
)
 
print(completion.choices[0].message.content.strip())