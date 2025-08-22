from llm.client.deepseek_answer import deepseek_respond
from llm.utils.only_answer import only_answer
from llm.client.ollama_answer import ollama_respond

def get_answer(client, prompt=None):
    if client.llm_client == 'deepseek':
        respond = deepseek_respond(prompt=prompt)
    if client.llm_client == 'ollama':
        respond = ollama_respond(model=client.ollama, prompt=prompt)
    else:
        respond = []
        
    similar_answer = only_answer(respond)
    
    return similar_answer, respond