"""Tests for `rie_ingest.robots.RobotsPolicy` — pure parse, no I/O."""

from __future__ import annotations

from rie_ingest.robots import RobotsPolicy


def test_empty_robots_allows_everything() -> None:
    policy = RobotsPolicy.from_text("")
    assert policy.allowed("https://laws.example.gov/acts/a.pdf")
    assert policy.allowed("/anything")


def test_disallow_blocks_matching_prefix() -> None:
    policy = RobotsPolicy.from_text("User-agent: *\nDisallow: /private/\n")
    assert not policy.allowed("https://laws.example.gov/private/secret.pdf")
    assert policy.allowed("https://laws.example.gov/public/open.pdf")


def test_wildcard_user_agent_applies_to_named_bot() -> None:
    policy = RobotsPolicy.from_text("User-agent: *\nDisallow: /blocked\n")
    assert not policy.allowed("https://x.gov/blocked/page.html", user_agent="rie-bot")


def test_specific_user_agent_group_overrides_wildcard() -> None:
    txt = "User-agent: *\nDisallow: /\n\nUser-agent: rie-bot\nDisallow: /admin/\n"
    policy = RobotsPolicy.from_text(txt)
    # rie-bot has its own, narrower group → only /admin/ is blocked.
    assert policy.allowed("https://x.gov/acts/a.pdf", user_agent="rie-bot")
    assert not policy.allowed("https://x.gov/admin/x", user_agent="rie-bot")
    # An unlisted agent falls back to the catch-all `*` group → all blocked.
    assert not policy.allowed("https://x.gov/acts/a.pdf", user_agent="other-bot")


def test_disallow_empty_value_allows_all() -> None:
    policy = RobotsPolicy.from_text("User-agent: *\nDisallow:\n")
    assert policy.allowed("https://x.gov/anything")


def test_path_only_url_matches() -> None:
    policy = RobotsPolicy.from_text("User-agent: *\nDisallow: /no/\n")
    assert not policy.allowed("/no/here")
    assert policy.allowed("/yes/here")
