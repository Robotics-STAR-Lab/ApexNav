from ollama import chat
from ollama import ChatResponse
from llm.utils.get_sys_prompt import get_similar_answer_prompt

def ollama_respond(model, prompt):
    system_prompts = get_similar_answer_prompt()
    msg = {
        "role": "user",
        "content": prompt,
    }
    history = system_prompts + [msg]
    
    response = chat(model=model, messages=history, stream=False),
    return response[0].message.content
    
