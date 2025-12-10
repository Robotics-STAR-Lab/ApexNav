from llm.answer import get_answer

def read_answer(llm_answer_path, llm_response_path, label, llm_client):
    label_existing = False

    with open(llm_answer_path, "a+") as f:
        f.seek(0)
        lines = f.readlines()

        for line in lines:
            if line.startswith(f"{label}:"):
                label_existing = True
                llm_answer = eval(line[len(label) + 1 :].strip())
                print(f"Already have Answer for {label}: {llm_answer}")
                break

        if not label_existing:
            llm_answer, response = get_answer(prompt=label, client=llm_client)
            print(llm_answer)
            f.write(f"\n{label}: {llm_answer}")
            print(f"New Answer for {label}: {llm_answer}")
            # Write the response to the llm_response_path file
            with open(llm_response_path, "a+") as response_file:
                response_file.write(
                    f"\n{label}: {response}"
                )  # Write the label and its corresponding response to the file
                print(f"Response saved to {llm_response_path}: {response}")
                
    if isinstance(llm_answer[-1], str):
        room = llm_answer[-1]
        llm_answer.pop()
    else:
        raise ValueError("Room answer is not correct!!!!")

    if isinstance(llm_answer[-1], float):
        fusion_score = llm_answer[-1]
        llm_answer.pop()
    else:
        raise ValueError("Score answer is not correct!!!!")
    
    return llm_answer, room, fusion_score