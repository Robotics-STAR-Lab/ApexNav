import os


def write_record(
    scene_id, episode_id, table, result_text, label, num_total, time_spend, file_path
):
    """
    Write navigation episode results to record file
    
    This function formats and saves navigation episode results including
    performance metrics, success status, and timing information to a
    structured record file for later analysis.
    
    Args:
        scene_id: Identifier for the navigation scene
        episode_id: Identifier for the specific episode
        table: Formatted table of performance metrics
        result_text: Success/failure result description
        label: Target object label being searched for
        num_total: Total episode number completed
        time_spend: Time spent on this episode (seconds)
        file_path: Path to the record file to write to
    """
    new_info = f"""
    Scene ID: {scene_id}
    Episode ID: {episode_id}
    {table}
    success or not: {result_text}
    target to find is {label}
    No.{num_total} task is finished
    {time_spend:.2f} seconds spend in this task
    """
    new_info = remove_all_indents(new_info)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            existing_content = file.read()
    else:
        existing_content = ""

    updated_content = new_info + "\n" + existing_content
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(updated_content)


def remove_all_indents(text):
    """
    Remove leading whitespace from all lines in text
    
    Args:
        text (str): Input text with potential indentation
        
    Returns:
        str: Text with all leading whitespace removed from each line
    """
    # Split into lines, apply lstrip to each line, then recombine
    lines = text.splitlines()
    stripped_lines = [line.lstrip() for line in lines]
    return "\n".join(stripped_lines)
