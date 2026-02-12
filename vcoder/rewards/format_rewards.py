import re


def format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    """Check that the completion contains a fenced ```html block with basic HTML tags.

    Scoring:
    - 0.5 for having a ```html ... ``` fenced block
    - 0.25 for containing <html or <!DOCTYPE
    - 0.25 for containing </html>
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        score = 0.0

        # Check for fenced html block
        if re.search(r"```html\s*\n.*?```", text, re.DOTALL):
            score += 0.5

        # Check for opening html tag
        if re.search(r"(<html|<!doctype\s+html)", text, re.IGNORECASE):
            score += 0.25

        # Check for closing html tag
        if re.search(r"</html>", text, re.IGNORECASE):
            score += 0.25

        rewards.append(score)
    return rewards
