"""Tests for the ReAct output parser."""

from llmath.agent.parser import ReActOutputParser


def test_parse_extracts_latest_tags():
    parser = ReActOutputParser()
    text = (
        "<think>first</think><tool>simplify: x + x</tool>"
        "<think>second</think><tool>solve: x**2 - 1</tool>"
        "<answer>done</answer>"
    )
    result = parser.parse(text)

    assert result.thought == "second"
    assert result.tool == "solve: x**2 - 1"
    assert result.answer == "done"


def test_parse_handles_missing_tags():
    parser = ReActOutputParser()
    text = "no tags here"
    result = parser.parse(text)

    assert result.thought is None
    assert result.tool is None
    assert result.answer is None
