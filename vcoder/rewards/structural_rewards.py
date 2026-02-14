from __future__ import annotations

from bs4 import BeautifulSoup

from vcoder.utils.html_utils import extract_html_from_completion


def _get_tag_sequence(html: str) -> list[str]:
    """Extract ordered list of tag names from HTML."""
    try:
        soup = BeautifulSoup(html, "html5lib")
        return [tag.name for tag in soup.find_all(True)]
    except Exception:
        return []


def _get_css_classes(html: str) -> set[str]:
    """Extract all CSS class names from HTML."""
    try:
        soup = BeautifulSoup(html, "html5lib")
        classes: set[str] = set()
        for tag in soup.find_all(True):
            tag_classes = tag.get("class", [])
            classes.update(tag_classes)
        return classes
    except Exception:
        return set()


def _sequence_similarity(seq_a: list[str], seq_b: list[str]) -> float:
    """Compute similarity between two tag sequences using longest common subsequence ratio."""
    if not seq_a or not seq_b:
        return 0.0
    m, n = len(seq_a), len(seq_b)
    # Truncate very long sequences for performance
    max_len = 200
    seq_a = seq_a[:max_len]
    seq_b = seq_b[:max_len]
    m, n = len(seq_a), len(seq_b)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    return 2 * lcs_len / (m + n)


def _class_overlap(classes_a: set[str], classes_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets of CSS classes."""
    if not classes_a and not classes_b:
        return 1.0
    if not classes_a or not classes_b:
        return 0.0
    return len(classes_a & classes_b) / len(classes_a | classes_b)


def structural_similarity_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs,
) -> list[float]:
    """Score structural similarity between generated HTML and ground truth.

    Compares:
    - DOM tag sequence similarity (weight 0.6)
    - CSS class name overlap (weight 0.4)

    Args:
        completions: Model completions (each is a list with one dict containing 'content')
        solution: Ground truth HTML strings
    """
    rewards = []
    for completion, gt_html in zip(completions, solution):
        text = completion[0]["content"]
        pred_html = extract_html_from_completion(text)
        if pred_html is None:
            rewards.append(0.0)
            continue

        # Tag sequence similarity
        pred_tags = _get_tag_sequence(pred_html)
        gt_tags = _get_tag_sequence(gt_html)
        tag_sim = _sequence_similarity(pred_tags, gt_tags)

        # CSS class overlap
        pred_classes = _get_css_classes(pred_html)
        gt_classes = _get_css_classes(gt_html)
        class_sim = _class_overlap(pred_classes, gt_classes)

        # score = 0.6 * tag_sim + 0.4 * class_sim
        score = tag_sim
        rewards.append(score)
    return rewards
