"""Output processing utilities for model generation.

Provides functions for cleaning and post-processing model outputs.
"""


def clean_model_output(text: str) -> str:
    """Clean raw model output by removing echoed instructions.

    Removes prompt templates, instruction markers, and collapses
    consecutive duplicate lines.

    Args:
        text: Raw text from the model.

    Returns:
        Cleaned text with artifacts removed.
    """
    # Trim off any prompt instructions the model might copy back.
    markers = [
        "Follow this structure:",
        "INSTRUCTIONS FOR YOUR ANSWER:",
        "Write your answer in three sections",
        "Now write your answer.",
        "Now give the final answer.",
        "Assistant:",
    ]
    for m in markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]

    # Collapse consecutive duplicate lines and extra blank lines.
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned: list[str] = []
    last_nonempty: str | None = None

    for ln in lines:
        if not ln.strip():
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        if last_nonempty is None or ln.strip() != last_nonempty.strip():
            cleaned.append(ln)
            last_nonempty = ln

    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned).strip()


def truncate_at_stop_sequences(
    text: str,
    stop_sequences: list[str] | None = None,
    include_stop_sequence: bool = False,
) -> str:
    """Truncate text at the first occurrence of any stop sequence.

    Args:
        text: Generated text to process.
        stop_sequences: List of strings to stop at. If None, uses defaults.

    Returns:
        Text truncated at the first stop sequence, or original if none found.
    """
    if stop_sequences is None:
        stop_sequences = [
            "\n\nUser:",
            "\n\nHuman:",
            "<|endoftext|>",
            "</s>",
        ]

    min_idx = len(text)
    matched_seq = None
    for seq in stop_sequences:
        idx = text.find(seq)
        if idx != -1 and idx < min_idx:
            min_idx = idx
            matched_seq = seq

    if matched_seq is None:
        return text.strip()

    if include_stop_sequence:
        return text[: min_idx + len(matched_seq)].strip()
    return text[:min_idx].strip()
