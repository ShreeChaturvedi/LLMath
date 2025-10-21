"""Tests for ReAct state handling."""

from llmath.agent.state import ReActState, ReActStep


def test_build_context_includes_steps():
    state = ReActState(question="What is 2 + 2?")
    step = ReActStep(
        thought="Compute quickly.",
        tool="simplify: 2 + 2",
        observation="4",
    )
    state.add_step(step)

    context = state.build_context()
    assert "Question:" in context
    assert "<think>Compute quickly.</think>" in context
    assert "<tool>simplify: 2 + 2</tool>" in context
    assert "<observe>4</observe>" in context


def test_format_skips_empty_fields():
    step = ReActStep(thought="Only thought.")
    formatted = step.format()
    assert formatted == "<think>Only thought.</think>"
