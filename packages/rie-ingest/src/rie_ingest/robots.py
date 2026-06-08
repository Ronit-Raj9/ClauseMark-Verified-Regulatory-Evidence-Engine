"""Pure `robots.txt` policy — parse text, answer allow/disallow per URL.

Wraps the stdlib `urllib.robotparser` behind a tiny pure value object so the
crawler can construct a policy from already-fetched (or offline-seeded) text
without any I/O. Network fetching of `robots.txt` lives in the crawler, never
here — this module is trivially testable with no transport.

Semantics (RFC 9309):
  * An empty / absent `robots.txt` means "no rules" → everything allowed.
  * The most specific matching `User-agent` group wins; `*` is the fallback.
  * `Disallow:` with an empty value un-blocks (allows) the whole path space.
"""

from __future__ import annotations

import urllib.parse
import urllib.robotparser
from dataclasses import dataclass

DEFAULT_USER_AGENT = "rie-bot"


@dataclass(frozen=True)
class RobotsPolicy:
    """Immutable allow/disallow oracle parsed from a `robots.txt` body."""

    _parser: urllib.robotparser.RobotFileParser

    @classmethod
    def from_text(cls, txt: str) -> RobotsPolicy:
        """Build a policy from raw `robots.txt` text. Empty text → allow-all."""
        parser = urllib.robotparser.RobotFileParser()
        # `parse` expects an iterable of lines. An empty body parses to "no
        # rules", which `can_fetch` then treats as allow-all.
        parser.parse(txt.splitlines())
        return cls(_parser=parser)

    def allowed(self, url: str, user_agent: str = DEFAULT_USER_AGENT) -> bool:
        """Is `user_agent` permitted to fetch `url` under this policy?

        A path-only `url` (no scheme/host) is matched as-is; an absolute URL
        is reduced to its path+query before matching, mirroring how crawlers
        compare against `robots.txt` rules.
        """
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme or parsed.netloc:
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"
            match_target = path
        else:
            match_target = url
        return self._parser.can_fetch(user_agent, match_target)
