from __future__ import annotations

from itertools import islice
from typing import Any

DEFAULT_WEBSIGHT_DATASET = "HuggingFaceM4/WebSight"


def load_websight_dataset(
    processor: Any,
    split: str = "train",
    *,
    dataset_name: str = DEFAULT_WEBSIGHT_DATASET,
    streaming: bool = False,
    max_samples: int | None = None,
    **load_kwargs: Any,
):
    """Load the WebSight dataset and preprocess each sample.

    Args:
        processor: Callable used to convert each raw dataset row into model-ready features.
        split: Any Hugging Face split expression (for example ``"train[:1%]"``) to support
            partial dataset downloads for large datasets.
        dataset_name: Hugging Face dataset id to load.
        streaming: If ``True``, stream examples without downloading full shards.
        max_samples: Optional cap on the number of samples after split selection.
        **load_kwargs: Extra keyword arguments forwarded to ``datasets.load_dataset``.

    Returns:
        A processed ``datasets.Dataset``.
    """
    from datasets import Dataset, load_dataset

    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        **load_kwargs,
    )

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be a positive integer when provided")

    if streaming:
        if max_samples is None:
            raise ValueError(
                "max_samples is required when streaming=True so the result can be "
                "materialized as a non-iterable dataset"
            )
        dataset = Dataset.from_list(list(islice(dataset, max_samples)))
    elif max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return dataset.map(processor)
