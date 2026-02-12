from __future__ import annotations

from bs4 import BeautifulSoup

from vcoder.utils.html_utils import extract_html_from_completion


# Tags that must be properly closed in valid HTML
IMPORTANT_TAGS = {"html", "head", "body", "div", "p", "span", "table", "ul", "ol", "li", "a", "section", "header", "footer", "nav", "main"}


def html_validity_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    """Score HTML validity: parseability, proper nesting, and tag richness.

    Scoring breakdown (sums to 1.0):
    - 0.4 for parseable HTML (BeautifulSoup doesn't raise errors)
    - 0.3 for having <html>, <head>, and <body> structure
    - 0.3 for tag diversity (proportion of important tags used, capped at 1.0)
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        html_str = extract_html_from_completion(text)
        if html_str is None:
            rewards.append(0.0)
            continue

        score = 0.0

        try:
            soup = BeautifulSoup(html_str, "html5lib")
        except Exception:
            rewards.append(0.0)
            continue

        # Parseable
        score += 0.4

        # Structural completeness
        has_html = soup.find("html") is not None
        has_head = soup.find("head") is not None
        has_body = soup.find("body") is not None
        structural_score = (int(has_html) + int(has_head) + int(has_body)) / 3
        score += 0.3 * structural_score

        # Tag diversity
        all_tags = {tag.name for tag in soup.find_all(True)}
        overlap = all_tags & IMPORTANT_TAGS
        diversity = min(len(overlap) / 6, 1.0)  # 6+ important tags = full score
        score += 0.3 * diversity

        rewards.append(score)
    return rewards
