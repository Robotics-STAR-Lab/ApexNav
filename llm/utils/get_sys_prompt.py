from llm.prompt.get_llm_answer import SYSTEM_PROMPT, USER1, ASSISTANT1, USER2, ASSISTANT2

def get_similar_answer_prompt():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER1},
        {"role": "assistant", "content": ASSISTANT1},
        {"role": "user", "content": USER2},
        {"role": "assistant", "content": ASSISTANT2}
    ]
    return messages