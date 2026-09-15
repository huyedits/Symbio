"""The boundary that leaves the machine.

Everything else in this project redacts at WRITE boundaries — the corpus, the
session store, mistake notes — on the reasoning that a secret on disk becomes a
secret in the weights. Nothing guarded the boundary that leaves the machine
entirely, which is the only one that cannot be undone: a key handed to a search
engine is disclosed the instant it is sent, and deleting it afterwards changes
nothing.
"""
import pytest

from symbio.app.web import outbound_query, web_search


@pytest.mark.parametrize("query", [
    "how do I use sk-abc123def456ghi789jkl012mno345pq with openai",
    "why does AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCY fail",
    'curl -H "Authorization: Bearer ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"',
    "https://api.example.com/v1?token=supersecret123456 returns 403",
    "postgres://admin:hunter2@db.internal:5432/prod refuses connections",
])
def test_a_credential_refuses_the_search(query):
    """Refuses rather than redacts: "search for [redacted]" is not the search
    anyone wanted, and sending it would teach the model its query went
    through."""
    sent, refusal = outbound_query(query)

    assert refusal is not None
    assert "credential" in refusal


def test_the_refusal_says_what_to_do_instead():
    _sent, refusal = outbound_query("my key sk-abc123def456ghi789jkl012mno is rejected")

    assert "without the secret" in refusal


def test_an_address_is_redacted_rather_than_refused():
    """The query around it usually still means something without it."""
    sent, refusal = outbound_query("postfix relay config for henry.tranvan@gmail.com")

    assert refusal is None
    assert "henry.tranvan" not in sent
    assert "postfix relay config" in sent


@pytest.mark.parametrize("query", [
    "what is the capital of France",
    "mlx_lm lora early stopping best practice",
    "BSD sed in place edit syntax",
])
def test_an_innocent_query_is_untouched(query):
    """A filter that mangles ordinary searches is one that gets switched off."""
    sent, refusal = outbound_query(query)

    assert refusal is None
    assert sent == query


def test_web_search_itself_refuses_before_any_request(monkeypatch):
    """Checked before a backend is reached, not after: the point is that
    nothing leaves the machine."""
    called = []
    import symbio.app.web as web
    monkeypatch.setattr(web, "_search_duckduckgo",
                        lambda *a, **k: called.append("ddg") or [])

    ok, message = web_search("token=abcdef1234567890abcdef please help",
                             {"web": {"search_results": 3, "http_timeout": 5},
                              "agent": {"max_output_len": 2000}})

    assert ok is False
    assert "not sent" in message
    assert called == []
