import re


def extract_html_from_completion(completion: str) -> str | None:
    """Extract HTML code from a fenced ```html code block in the completion.

    Returns the HTML string if found, or None if no valid block is present.
    """
    pattern = r"```html\s*\n(.*?)```"
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
