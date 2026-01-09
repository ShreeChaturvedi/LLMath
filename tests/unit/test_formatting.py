"""Tests for training formatting utilities."""

import pytest

pytest.importorskip("datasets", reason="datasets required for formatting tests")

from llmath.training.formatting import format_for_deepseek  # noqa: E402


class DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_format_for_deepseek_messages():
    tokenizer = DummyTokenizer()
    examples = {
        "messages": [
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        ]
    }
    formatted = format_for_deepseek(examples, tokenizer)
    assert formatted == ["user:Q|assistant:A"]
