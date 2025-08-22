import ast
import re

def only_answer(response):
    """
    Extract and parse answer list from LLM response
    
    This function searches for answers in the format "Answer: [item1, item2, ...]"
    and converts them to Python lists, automatically adding quotes to string elements.
    
    Args:
        response (str): LLM response text containing answer
        
    Returns:
        list: Parsed answer list, or None if parsing fails
    """
    match = re.search(r'Answer:\s*\[([^\]]+)\]', response)
    if match:
        content = match.group(1)
        # Add quotes to strings without quotes
        elements = []
        for item in content.split(','):
            item = item.strip()
            if re.match(r'^-?\d+(\.\d+)?$', item):  # Is a number
                elements.append(item)
            else:
                elements.append(f'"{item}"')
        list_str = '[' + ', '.join(elements) + ']'
        try:
            return ast.literal_eval(list_str)
        except Exception as e:
            print(f"Error parsing fixed answer: {e}")
            return None
    return None
