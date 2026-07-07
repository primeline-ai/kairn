"""Secret redaction - WOW-9 Phase 2 (the load-bearing privacy gate).

Deterministic, zero-LLM, zero-network: pure regex/string transforms only. No
model calls, no I/O. This module is deliberately parser-independent (it knows
nothing about Claude Code transcripts) so it can be validated in isolation
against a fixed test corpus before any real transcript ever touches it.

Redaction policy (documented here + in the README):

- Default action is MASK, not drop: a matched secret span is replaced in place
  by a stable ``[REDACTED:<rule>]`` placeholder, and the surrounding text is
  preserved untouched. Masking (not dropping the whole line/block) is the right
  default because the surrounding decision/gotcha prose is exactly the value an
  import is meant to capture; dropping entire blocks is a documented Phase 3
  escalation for the case where in-line masking proves insufficient, not the
  Phase 2 default.
- PEM ``PRIVATE KEY`` blocks are masked whole (the block itself is the secret).
- For ``key = value`` assignments and ``Authorization`` headers, only the
  credential (the value / token) is masked; the key and scheme word are kept so
  the redacted line stays readable (``password=[REDACTED:secret-assignment]``).

Two properties are what "correct" means here, and both are tested:

1. Zero missed matches - every secret in the (externally-sourced) corpus is
   absent from the output. This is the primary safety bar.
2. Precision - ordinary prose that merely mentions a sensitive word
   (``password``, ``secret``, ``Authorization``, ``bearer``, ``AKIA``) is left
   byte-for-byte unchanged. Over-redaction destroys the imported context.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_PLACEHOLDER = "[REDACTED:{rule}]"


@dataclass(frozen=True)
class RedactionFinding:
    """One masked span, in ORIGINAL-text coordinates.

    ``rule`` is the specific pattern that matched (e.g. ``aws-access-key-id``);
    ``category`` is the coarse bucket (``vendor-key`` / ``auth-header`` /
    ``assignment`` / ``private-key``) that callers/tests assert against.
    """

    rule: str
    category: str
    start: int
    end: int
    placeholder: str


@dataclass(frozen=True)
class RedactionResult:
    text: str
    findings: tuple[RedactionFinding, ...]

    @property
    def redacted(self) -> bool:
        return bool(self.findings)


@dataclass(frozen=True)
class _Rule:
    rule: str
    category: str
    pattern: re.Pattern[str]
    group: int | str = 0  # which capture group's span is the secret


# A leading boundary that forbids a vendor prefix from matching mid-identifier
# (so "ask-questions-..." never looks like an "sk-..." key). Kept as a string so
# it can be prepended uniformly to every case-literal vendor pattern.
_B = r"(?<![A-Za-z0-9])"

# Value charset for `key = value` assignments and header credentials. It stops
# at whitespace and quotes (a value never spans a newline), and - critically -
# excludes square brackets so a `[REDACTED:...]` placeholder is NOT itself a
# valid value, which makes a second redaction pass a no-op (idempotency).
_VALUE = r"[^\s\"'\[\]]"
_CRED = r"[A-Za-z0-9._~+/=-]"

# Sensitive assignment keys. A key may be a *substring* of a longer identifier
# (DB_PASSWORD, aws_secret_access_key, X-Api-Key): finditer scans every offset,
# so the keyword is found even mid-identifier, and the trailing `[a-z0-9_.-]*`
# absorbs the rest up to the separator. There is deliberately NO leading
# `[a-z0-9_.-]*` - anchoring on the literal keyword (not a greedy prefix) is
# what keeps this rule linear: a leading `.*`-style prefix made it backtrack
# quadratically on large base64-heavy inputs (7MB transcript: 5.2s -> ~0.1s).
_ASSIGN_KEYS = (
    r"secret[_-]?access[_-]?key|client[_-]?secret|access[_-]?token|"
    r"refresh[_-]?token|auth[_-]?token|private[_-]?key|api[_-]?key|apikey|"
    r"access[_-]?key|password|passwd|secret|token|pwd"
)

# Rules are declared most-specific first; declaration order is the tie-break
# priority when two matches cover the exact same span (see redact()).
_RULES: tuple[_Rule, ...] = (
    _Rule(
        "private-key",
        "private-key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    _Rule(
        # Fallback for a PEM block truncated before its -----END----- line
        # (pagination / copy-paste). Masks header + only the base64 body lines
        # that follow: each captured line must be pure base64 AND end at a
        # newline/EOF (the `(?=\r?\n|$)` line anchor), so the match stops at the
        # first line of ordinary prose instead of eating its leading word. The
        # literal PEM header makes false positives impossible; a real full block
        # is caught by the longer-span rule above.
        "private-key-header",
        "private-key",
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----(?:\r?\n[A-Za-z0-9+/=]+(?=\r?\n|$))+"),
    ),
    _Rule(
        "auth-header",
        "auth-header",
        re.compile(
            r"(?i)(?:proxy-)?authorization\s*[:=]\s*"
            r"(?:(?:bearer|basic|digest|token)\s+)?"
            rf"(?P<cred>{_CRED}{{8,}})",
        ),
        group="cred",
    ),
    _Rule(
        "bearer-token",
        "auth-header",
        re.compile(rf"(?i)(?<![A-Za-z])bearer\s+(?P<cred>{_CRED}{{16,}})"),
        group="cred",
    ),
    _Rule(
        # URI-embedded basic-auth credential: scheme://user:PASSWORD@host.
        # Only the password (between `:` and `@`) is masked. Requires the full
        # user:pass@ shape, so a plain URL or a host:port (no `@`) never matches.
        # The scheme is BOUNDED {0,30}: an unbounded `*` before `://` under
        # `(?i)` backtracks quadratically on letter-heavy input (measured 8.9s
        # on a 60KB blob); real URI schemes are short, so the bound is free.
        "url-credential",
        "url-credential",
        re.compile(r"(?i)[a-z][a-z0-9+.\-]{0,30}://[^\s:/@]+:(?P<cred>[^\s:/@]+)@"),
        group="cred",
    ),
    _Rule(
        "aws-access-key-id",
        "vendor-key",
        re.compile(_B + r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}"),
    ),
    _Rule(
        "anthropic-key",
        "vendor-key",
        re.compile(_B + r"sk-ant-[A-Za-z0-9_-]{20,}"),
    ),
    _Rule(
        # Project-scoped keys carry `_`/`-` in the body; the distinctive
        # `proj-` prefix makes it safe to allow them without broadening the
        # generic `sk-` rule (which would false-positive on `sk-learn-...`).
        "openai-project-key",
        "vendor-key",
        re.compile(_B + r"sk-proj-[A-Za-z0-9_-]{20,}"),
    ),
    _Rule(
        "openai-key",
        "vendor-key",
        re.compile(_B + r"sk-[A-Za-z0-9]{20,}"),
    ),
    _Rule(
        "github-token",
        "vendor-key",
        re.compile(_B + r"gh[pousr]_[A-Za-z0-9]{36,}"),
    ),
    _Rule(
        "github-pat",
        "vendor-key",
        re.compile(_B + r"github_pat_[A-Za-z0-9_]{22,}"),
    ),
    _Rule(
        # xox[baprse]- covers bot/app/user/refresh/session/rotation tokens;
        # xapp- covers app-level tokens.
        "slack-token",
        "vendor-key",
        re.compile(_B + r"(?:xox[baprse]|xapp)-[A-Za-z0-9-]{10,}"),
    ),
    _Rule(
        "google-api-key",
        "vendor-key",
        re.compile(_B + r"AIza[0-9A-Za-z_-]{35}"),
    ),
    _Rule(
        "google-oauth-token",
        "vendor-key",
        re.compile(_B + r"ya29\.[A-Za-z0-9_-]{20,}"),
    ),
    _Rule(
        "stripe-key",
        "vendor-key",
        re.compile(_B + r"(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{10,}"),
    ),
    _Rule(
        # Greedy {10,} is DELIBERATE and measured linear (720KB of dot-less
        # `eyJ` -> 7.6ms): CPython fast-fails a greedy class-run followed by an
        # absent literal. Do NOT "harden" this to possessive `{10,}+` - that was
        # measured ~10000x SLOWER (8.9s on the same input) because it defeats
        # that fast-fail and re-scans to EOF at every `eyJ` offset. The
        # test_jwt_pattern_has_no_catastrophic_backtracking guard locks this in.
        "jwt",
        "vendor-key",
        re.compile(_B + r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ),
    _Rule(
        "secret-assignment",
        "assignment",
        re.compile(
            rf"(?i)(?P<key>(?:{_ASSIGN_KEYS})[a-z0-9_.\-]*)\s*[:=]\s*"
            rf"(?P<q>[\"']?)(?P<val>{_VALUE}{{6,}})(?P=q)"
        ),
        group="val",
    ),
)


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    rule: str
    category: str
    priority: int


def _collect(text: str) -> list[_Span]:
    spans: list[_Span] = []
    for priority, rule in enumerate(_RULES):
        for match in rule.pattern.finditer(text):
            start, end = match.span(rule.group)
            if start < 0 or end <= start:  # group did not participate
                continue
            spans.append(_Span(start, end, rule.rule, rule.category, priority))
    return spans


def _resolve(spans: list[_Span]) -> list[_Span]:
    """Greedily keep non-overlapping spans, widest first (then most-specific).

    Widest-first guarantees the longest secret always wins - a shorter,
    overlapping fragment can never truncate a larger sensitive span and leave
    part of it exposed. Ties on width fall back to rule priority (declaration
    order), so a specific vendor-key beats a generic assignment on an identical
    span (e.g. `aws_access_key_id = AKIA...`)."""
    ordered = sorted(spans, key=lambda s: (-(s.end - s.start), s.priority, s.start))
    accepted: list[_Span] = []
    for span in ordered:
        if any(not (span.end <= a.start or span.start >= a.end) for a in accepted):
            continue
        accepted.append(span)
    accepted.sort(key=lambda s: s.start)
    return accepted


def redact(text: str) -> RedactionResult:
    """Mask every secret in ``text``; return the masked text plus findings.

    Deterministic and idempotent: ``redact(redact(t).text).findings == ()``.
    """
    accepted = _resolve(_collect(text))
    if not accepted:
        return RedactionResult(text, ())

    out: list[str] = []
    findings: list[RedactionFinding] = []
    pos = 0
    for span in accepted:
        out.append(text[pos : span.start])
        placeholder = _PLACEHOLDER.format(rule=span.rule)
        out.append(placeholder)
        findings.append(
            RedactionFinding(span.rule, span.category, span.start, span.end, placeholder)
        )
        pos = span.end
    out.append(text[pos:])
    return RedactionResult("".join(out), tuple(findings))


def render_preview(
    text: str,
    *,
    source: str | None = None,
    exp_type: str | None = None,
) -> str:
    """Render the dry-run preview of what would be stored for one candidate.

    The returned string is ALWAYS post-redaction - it is the human-review
    surface shown before the confirmation gate, so it must never echo a raw
    secret. Callers (the Phase 3 CLI) render one of these per candidate under
    ``--dry-run`` and, on a real run, again for the sample shown at the
    confirmation prompt.
    """
    result = redact(text)
    lines = ["--- would store (post-redaction) ---"]
    if source is not None:
        # A source_ref is a file path / URI and could itself embed a credential;
        # redact it too so the preview upholds its never-echo-a-raw-secret contract.
        lines.append(f"source: {redact(source).text}")
    if exp_type is not None:
        # exp_type is a controlled enum from the importer, never user content.
        lines.append(f"type: {exp_type}")
    if result.findings:
        counts: dict[str, int] = {}
        for finding in result.findings:
            counts[finding.category] = counts.get(finding.category, 0) + 1
        summary = ", ".join(f"{cat} ({n})" for cat, n in sorted(counts.items()))
        lines.append(f"redactions: {len(result.findings)} [{summary}]")
    else:
        lines.append("redactions: 0")
    lines.append(result.text)
    return "\n".join(lines)


def confirm_import(
    *,
    dry_run: bool,
    assume_yes: bool = False,
    preview: str | None = None,
    prompt_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> bool:
    """Gate a real (writing) import behind an explicit human confirmation.

    Returns True iff a write should proceed. The contract is deliberately
    fail-safe - anything other than an explicit yes means DO NOT write:

    - ``dry_run``    -> False, no prompt (a dry run writes nothing by design).
    - ``assume_yes`` -> True, no prompt (non-interactive / automation opt-in).
    - otherwise      -> show ``preview`` (if any), prompt, and return True only
      on an explicit ``y``/``yes`` (case-insensitive, whitespace-stripped).

    ``prompt_fn`` and ``out`` are injectable so the gate is testable without a
    real terminal. This is what enforces "no silent first-run writes".
    """
    if dry_run:
        return False
    if assume_yes:
        return True
    if preview is not None:
        out(preview)
    answer = prompt_fn("Proceed with import and write these to the store? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}
