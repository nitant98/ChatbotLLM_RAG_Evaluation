# model_setup.py
import openai
from config import CFG

# Set up OpenAI API key
openai.api_key = CFG.openai_api_key

def get_llm_response(prompt):
    response = openai.ChatCompletion.create(
        model="GPT-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=CFG.temperature,
        max_tokens=100,
        top_p=CFG.top_p,
    )
    return response.choices[0].message["content"]
