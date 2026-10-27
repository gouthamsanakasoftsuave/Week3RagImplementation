"""Detect the Week 11 production failure: the wrong termination clause.

The logged answer (trace trc_7f3a91c2e0) told the user that Clause 12.9 lets
either party terminate on sixty days' notice. Clause 12.9 only says where a
notice is delivered. The termination right is Clause 7.1: 30 days' prior
written notice.
"""

from __future__ import annotations

import re

_SIXTY = re.compile(r"\b60\b|sixty", re.IGNORECASE)
_TERMINATION = re.compile(r"terminat", re.IGNORECASE)


def is_wrong_termination_citation(answer: str) -> bool:
    """True only for the failure we found: 12.9 cited as a 60-day termination right."""
    text = answer or ""
    if not _TERMINATION.search(text):
        return False
    if "12.9" not in text:
        return False
    return _SIXTY.search(text) is not None
