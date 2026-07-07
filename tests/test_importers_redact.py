"""Tests for the secret-redaction module - WOW-9 Phase 2.

The load-bearing privacy gate. The core safety property under test is
"zero missed matches": every secret in the externally-sourced corpus
(tests/redaction_corpus.py - gitleaks-derived reference fixtures + synthetic
cases) must be absent from the redacted output. The precision guard is the
mirror: ordinary prose that merely mentions sensitive words must be left
untouched, because over-redaction destroys the decision/gotcha context that
gives the import its value.

Zero-LLM, zero-network: pure regex/string transforms, no model calls.

Provider-shaped secrets used inline below are fragment-joined via `_fj` for the
same reason as the corpus: so this source file never contains a contiguous
scannable token (GitHub push-protection / gitleaks), while the runtime value
handed to `redact()` stays complete.
"""

from __future__ import annotations

import pytest

from kairn.importers.redact import (
    RedactionFinding,
    RedactionResult,
    confirm_import,
    redact,
    render_preview,
)
from tests.redaction_corpus import NEGATIVE_CASES, POSITIVE_CASES


def _fj(*parts: str) -> str:
    return "".join(parts)


_AWS_KEY = _fj("AKIA", "IOSFODNN7EXAMPLE")
_STRIPE_KEY = _fj("sk_live_", "4eC39HqLyjWDarjtT1zdp7dc")
_ANTHROPIC_KEY = _fj("sk-ant-", "api03-abcDEF1234567890_ghijklmnopqrstuvwxyz-ABCDEFGH")
_URL_PW = _fj("S3cr3t", "P4ssw0rd")
_PEM_HDR = _fj("-----BEGIN RSA ", "PRIVATE KEY-----")
_PEM_END = _fj("-----END RSA ", "PRIVATE KEY-----")
_PEM_BODY = _fj("MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1", "Pt8Q")


# --- the gate: zero missed matches over the external corpus ---


@pytest.mark.parametrize("case", POSITIVE_CASES, ids=[c.name for c in POSITIVE_CASES])
def test_secret_is_removed_from_redacted_output(case):
    """Zero missed matches: the exact secret substring must not survive."""
    result = redact(case.text)
    assert case.secret not in result.text, f"{case.name}: secret survived redaction:\n{result.text}"
    assert result.findings, f"{case.name}: expected at least one finding"
    categories = {f.category for f in result.findings}
    assert case.category in categories, (
        f"{case.name}: expected category {case.category!r} in {categories}"
    )


# --- the precision guard: prose must not be over-redacted ---


@pytest.mark.parametrize("name,text", NEGATIVE_CASES, ids=[n for n, _ in NEGATIVE_CASES])
def test_prose_is_not_redacted(name, text):
    result = redact(text)
    assert result.findings == (), f"{name}: false-positive findings: {result.findings}"
    assert result.text == text, f"{name}: prose was mutated: {result.text!r}"


# --- result shape ---


def test_redact_returns_result_with_text_and_findings():
    result = redact("nothing sensitive here")
    assert isinstance(result, RedactionResult)
    assert result.text == "nothing sensitive here"
    assert result.findings == ()
    assert result.redacted is False


def test_redacted_property_true_when_something_matched():
    result = redact(f"key {_AWS_KEY} leaked")
    assert result.redacted is True


def test_finding_exposes_rule_category_span_and_placeholder():
    result = redact(f"key {_AWS_KEY} leaked")
    (finding,) = result.findings
    assert isinstance(finding, RedactionFinding)
    assert finding.rule
    assert finding.category == "vendor-key"
    assert finding.start < finding.end
    assert finding.placeholder in result.text
    assert finding.placeholder.startswith("[REDACTED:")


def test_placeholder_replaces_the_secret_in_place():
    result = redact(f"before {_AWS_KEY} after")
    assert _AWS_KEY not in result.text
    assert result.text.startswith("before ")
    assert result.text.endswith(" after")
    assert "[REDACTED:" in result.text


# --- discriminating / env-dependent bug-class guards ---


def test_case_literal_prefix_is_not_ignorecased():
    """AWS access-key IDs are uppercase-literal - a lowercased lookalike is
    not a key and must not be redacted (guards against a stray re.IGNORECASE
    on case-significant vendor prefixes)."""
    result = redact(f"the string {_AWS_KEY.lower()} is just lowercase text")
    assert result.findings == ()


def test_two_secrets_in_one_text_are_both_redacted():
    text = f"aws {_AWS_KEY} and stripe {_STRIPE_KEY} together"
    result = redact(text)
    assert _AWS_KEY not in result.text
    assert _STRIPE_KEY not in result.text
    assert len(result.findings) == 2


def test_overlapping_auth_header_and_vendor_key_redacts_cleanly():
    """`Authorization: Bearer <anthropic-key>` matches both the auth-header
    rule and the vendor-key rule on overlapping spans - the result must be a
    single clean placeholder, never a nested/garbled double-redaction."""
    text = f"Authorization: Bearer {_ANTHROPIC_KEY}"
    result = redact(text)
    assert _ANTHROPIC_KEY not in result.text
    assert "[REDACTED:[REDACTED:" not in result.text
    assert result.text.count("[REDACTED:") == 1


def test_unicode_context_is_preserved_around_redaction():
    """Locale/encoding bug-class guard: non-ASCII prose around a secret must
    survive intact (the git importer hit an analogous LANG=C locale bug)."""
    text = f"Schlüssel für Käse: {_AWS_KEY} 🧀 gespeichert"
    result = redact(text)
    assert _AWS_KEY not in result.text
    assert "Schlüssel für Käse" in result.text
    assert "🧀 gespeichert" in result.text


def test_redaction_is_idempotent():
    """Re-redacting already-redacted text is a no-op: placeholders must not
    themselves re-trigger any rule, and a second pass changes nothing."""
    text = f"key {_AWS_KEY} and password=s3cr3tvalue99 here"
    once = redact(text).text
    twice = redact(once)
    assert twice.text == once
    assert twice.findings == ()


def test_private_key_block_is_redacted_whole():
    text = f"{_PEM_HDR}\n{_PEM_BODY}\n{_PEM_END}"
    result = redact(text)
    assert _PEM_BODY not in result.text
    assert any(f.category == "private-key" for f in result.findings)


# --- render_preview: the dry-run "what would be stored" surface ---


def test_preview_never_leaks_the_raw_secret():
    """The preview is shown to a human BEFORE any write - it must itself be
    post-redaction. A preview that echoed the raw secret would defeat the
    entire purpose of the dry-run gate."""
    preview = render_preview(f"token {_AWS_KEY} leaked")
    assert _AWS_KEY not in preview
    assert "[REDACTED:" in preview


def test_preview_shows_the_redacted_body_verbatim():
    text = "password=s3cr3tvalue99 in the env file"
    preview = render_preview(text)
    assert "password=[REDACTED:secret-assignment] in the env file" in preview


def test_preview_summarizes_redaction_count_and_categories():
    text = f"aws {_AWS_KEY} and Authorization: Bearer abcdef1234567890ABCDEF"
    preview = render_preview(text)
    assert "2" in preview  # two redactions
    assert "vendor-key" in preview
    assert "auth-header" in preview


def test_preview_includes_source_and_type_when_given():
    preview = render_preview(
        "nothing sensitive", source="claude-code:/p/x.jsonl#L3", exp_type="decision"
    )
    assert "claude-code:/p/x.jsonl#L3" in preview
    assert "decision" in preview


def test_preview_of_clean_text_reports_zero_redactions_and_full_text():
    preview = render_preview("just an ordinary sentence about tests")
    assert "just an ordinary sentence about tests" in preview
    assert "0" in preview


# --- confirm_import: no silent first-run writes ---


def _boom(_msg: str) -> str:
    raise AssertionError("prompt_fn must not be called")


def test_confirm_dry_run_never_writes_and_never_prompts():
    assert confirm_import(dry_run=True, prompt_fn=_boom) is False


def test_confirm_assume_yes_proceeds_without_prompting():
    assert confirm_import(dry_run=False, assume_yes=True, prompt_fn=_boom) is True


def test_confirm_real_run_requires_explicit_yes():
    assert confirm_import(dry_run=False, prompt_fn=lambda _: "y") is True
    assert confirm_import(dry_run=False, prompt_fn=lambda _: "yes") is True


def test_confirm_real_run_defaults_to_no():
    """The whole point of the gate: silence / anything-but-yes means DO NOT
    write. A blank enter, an 'n', or garbage all abort."""
    assert confirm_import(dry_run=False, prompt_fn=lambda _: "") is False
    assert confirm_import(dry_run=False, prompt_fn=lambda _: "n") is False
    assert confirm_import(dry_run=False, prompt_fn=lambda _: "maybe later") is False


def test_confirm_is_case_insensitive_and_strips_whitespace():
    assert confirm_import(dry_run=False, prompt_fn=lambda _: "  Y  ") is True
    assert confirm_import(dry_run=False, prompt_fn=lambda _: "YES") is True


def test_confirm_shows_preview_before_prompting():
    shown: list[str] = []
    prompted: list[str] = []

    def _prompt(msg: str) -> str:
        prompted.append(msg)
        return "n"

    confirm_import(
        dry_run=False,
        prompt_fn=_prompt,
        preview="--- would store ---\npassword=[REDACTED:secret-assignment]",
        out=shown.append,
    )
    joined = "\n".join(shown)
    assert "password=[REDACTED:secret-assignment]" in joined
    assert prompted, "expected the user to be prompted"


# --- RC-gate regression guards ---


def test_jwt_pattern_has_no_catastrophic_backtracking():
    """ReDoS guard (RC-gate HIGH): a long run of `eyJ` with no `.` separators
    must not trigger quadratic backtracking - the gate processes ALL extracted
    text, so a degenerate input cannot be allowed to hang it."""
    import time

    pathological = "eyJ" * 20000  # 60KB, zero dots -> no complete JWT
    start = time.perf_counter()
    result = redact(pathological)
    elapsed = time.perf_counter() - start
    assert result.findings == ()
    assert elapsed < 1.0, f"possible ReDoS: {elapsed:.2f}s for 60KB input"


def test_preview_redacts_secrets_in_the_source_field():
    """render_preview (RC-gate MEDIUM): its docstring promises it never echoes
    a raw secret. That must hold for the source/metadata line too, not just the
    redacted body."""
    preview = render_preview("ok", source=f"postgres://admin:{_URL_PW}@db.example.com/prod")
    assert _URL_PW not in preview


def test_truncated_private_key_does_not_eat_following_prose():
    """The truncated-PEM fallback must stop at the first non-base64 line and
    not swallow the leading word of the following prose (guards the RC-gate
    MEDIUM fix - masking the body must not over-redact real context)."""
    text = f"{_PEM_HDR}\n{_PEM_BODY}\nGotcha: this prose must survive intact."
    result = redact(text)
    assert _PEM_BODY not in result.text
    assert "Gotcha: this prose must survive intact." in result.text
