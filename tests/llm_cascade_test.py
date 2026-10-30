"""Tests for prompt parsing, the budget and the cascade router."""

import asyncio

import pandas as pd
import pytest

from opensrbench import cascade
from opensrbench.llm import client
from opensrbench.llm import prompts


def test_parse_tag_last_tag_wins_and_abstention():
    assert prompts.parse_tag("<decision>EXCLUDE</decision>") == (0, True)
    assert prompts.parse_tag(
        "<decision>EXCLUDE</decision> ... <decision> include </decision>"
    ) == (1, True)
    assert prompts.parse_tag("<decision>UNCERTAIN</decision>") == (1, False)
    assert prompts.parse_tag("I think exclude") == (1, False)
    assert prompts.parse_tag(None) == (1, False)


def test_parse_cao_tail_markers():
    assert prompts.parse_cao("blah " * 200 + "XXX") == (0, True)
    assert prompts.parse_cao("YYY" + " pad" * 300) == (1, False)  # Not in tail.
    assert prompts.parse_cao("... YYY") == (1, True)
    assert prompts.parse_cao("YYY XXX") == (1, False)


def test_render_and_placeholder_guard(tmp_path):
    out = prompts.render("{objective}|{title}|{abstract}", "o", "c", "t", "")
    assert out == "o|t|(no abstract)"
    path = tmp_path / "p.txt"
    path.write_text("PASTE_THE_VERBATIM prompt")
    with pytest.raises(prompts.PromptNotConfiguredError):
        prompts.load_template(str(path))


def test_budget_reservation_blocks_overspend():
    async def go():
        budget = client.Budget(1.0)
        await budget.reserve(0.6)
        with pytest.raises(client.BudgetExceededError):
            await budget.reserve(0.5)
        await budget.settle(0.6, 0.1)
        await budget.reserve(0.5)
        return budget

    budget = asyncio.run(go())
    assert budget.spent == pytest.approx(0.1)


def test_cascade_routes_gray_zone_to_verifier():
    scored = pd.DataFrame(
        {
            "set": "b",
            "review_id": "r",
            "record_id": ["low", "mid1", "mid2", "high"],
            "score": [0.01, 0.4, 0.5, 0.9],
        }
    )
    verifier = pd.DataFrame(
        {
            "review_id": "r",
            "record_id": ["mid1", "low"],
            "decision_keep": [0, 1],
            "is_valid_answer": [True, True],
            "cost_usd": [0.5, 0.5],
        }
    )
    out = cascade.cascade_decisions(scored, 0.1, 0.8, verifier).set_index(
        "record_id"
    )
    assert out.loc["low", "decision_keep"] == 0  # Auto-exclude ignores LLM.
    assert out.loc["mid1", "decision_keep"] == 0
    assert out.loc["mid2", "decision_keep"] == 1  # No verdict -> keep.
    assert not out.loc["mid2", "is_valid_answer"]
    assert out.loc["high", "decision_keep"] == 1
    assert out["cost_usd"].sum() == 0.5
