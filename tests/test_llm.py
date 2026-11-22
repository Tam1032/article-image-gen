import json

import pytest

from app.services.llm import LLMError, build_user_message, parse_plans


def _answer(*prompts: str) -> str:
    return json.dumps({"images": [{"prompt": p, "alt_text": f"alt {p}"} for p in prompts]})


def test_parse_exact_count():
    plans = parse_plans(_answer("a", "b"), 2)
    assert [p.prompt for p in plans] == ["a", "b"]


def test_parse_pads_when_llm_returns_too_few():
    plans = parse_plans(_answer("a"), 3)
    assert len(plans) == 3


def test_parse_truncates_when_llm_returns_too_many():
    assert len(parse_plans(_answer("a", "b", "c"), 2)) == 2


@pytest.mark.parametrize("raw", ["not json", "{}", '{"images": []}', '{"images": [{"x": 1}]}'])
def test_parse_rejects_bad_answers(raw):
    with pytest.raises(LLMError):
        parse_plans(raw, 1)


def test_user_message_truncates_long_content():
    msg = build_user_message("Title", "x" * 50_000, 2)
    assert "Title" in msg and "2" in msg
    assert len(msg) < 7000
