
'''     # WOOOOOOOOOOOOOOOOOOOOOOOOOOOKRRRRRRRRRINNNNNNNNNNNNNNNNN
# model_setup.py
# model_setup.py
# model_setup.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain import HuggingFacePipeline
from config import CFG

def get_model(model="facebook/bart-base"):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)
    return tokenizer, model

# Load model and tokenizer
tokenizer, model = get_model()

# Set up Hugging Face pipeline
pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    temperature=max(CFG.temperature, 0.7),
    top_p=CFG.top_p,
    repetition_penalty=CFG.repetition_penalty,
    max_length=1024,
    max_new_tokens=100
)

# Langchain pipeline
llm = HuggingFacePipeline(pipeline=pipe)
'''
'''      #     GPTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
# model_setup.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from config import CFG

def get_model(model="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    return tokenizer, model

# Load model and tokenizer
tokenizer, model = get_model()

# Set up Hugging Face pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    temperature=max(CFG.temperature, 0.7),
    top_p=CFG.top_p,
    repetition_penalty=CFG.repetition_penalty,
    max_length=1024,
    max_new_tokens=100
)

# Langchain pipeline
llm = HuggingFacePipeline(pipeline=pipe)
'''
# model_setup.py
# model_setup.py
import openai
from config import CFG

# Set up OpenAI API key
openai.api_key = CFG.openai_api_key

def get_llm_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=CFG.temperature,
        max_tokens=100,
        top_p=CFG.top_p,
    )
    return response.choices[0].message["content"]



