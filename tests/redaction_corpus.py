"""Secret-redaction test corpus - WOW-9 Phase 2.

The "zero missed matches" gate (plan Success Criteria) is only meaningful if
the corpus is NOT purely self-authored: writing the redaction rules and their
own test cases together makes a zero-miss bar trivially satisfiable by
construction. So this corpus deliberately incorporates a sample of gitleaks'
public default-rule reference fixtures *as data* (canonical, non-functional
example secrets that gitleaks' own rules are documented/tested against - e.g.
AWS's own `AKIA...EXAMPLE` docs placeholder, the jwt.io sample token)
alongside synthetic cases. gitleaks is NOT a dependency - these are reference
pattern shapes, hand-copied, none are live credentials.

IMPORTANT - fragment-joined literals: because these fixtures are deliberately
secret-SHAPED, a naive literal like `"ghp_<36 chars>"` in the source would trip
GitHub push-protection / gitleaks on THIS repo (validated: it did). So each
positive case is built from a `{s}` template plus fragment-joined parts via
`_case(...)`: the source file never contains a contiguous scannable token, but
the RUNTIME value handed to `redact()` is the full secret, so nothing about the
test changes. (Reusable pattern, Kairn a9712791.)

Case shape:
  RedactionCase(name, text, secret, category)
    - text     : a realistic line/block that embeds the secret
    - secret   : the exact substring that MUST be absent from redacted output
                 (this is the real safety property - "zero missed matches")
    - category : a category that MUST appear among the redaction findings

NEGATIVE_CASES are ordinary prose that happens to contain sensitive-adjacent
words (password, secret, token, Authorization, AKIA, bearer) WITHOUT being a
real secret - they must produce ZERO findings (precision guard: over-redaction
destroys the surrounding decision/gotcha context that gives the import value).
"""

from __future__ import annotations

from collections import namedtuple

RedactionCase = namedtuple("RedactionCase", ["name", "text", "secret", "category"])


def _fj(*parts: str) -> str:
    """Join a secret's fragments at runtime (see module docstring)."""
    return "".join(parts)


def _case(name: str, template: str, parts: tuple[str, ...], category: str) -> RedactionCase:
    """Build a case: reconstruct the secret from `parts`, splice it into
    `template` at the `{s}` marker. The source file holds only the split parts
    and the template - never the contiguous secret."""
    secret = _fj(*parts)
    return RedactionCase(name, template.replace("{s}", secret), secret, category)


# PEM header/body/footer are split too - a contiguous
# `-----BEGIN ... PRIVATE KEY-----` + base64 would itself be flagged.
_PEM_HDR = _fj("-----BEGIN RSA ", "PRIVATE KEY-----")
_PEM_END = _fj("-----END RSA ", "PRIVATE KEY-----")
_PEM_BODY = _fj("MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1", "Pt8Q")
_PEM_BODY2 = "uKUpRKfFLfRYC9AIKjbJTWit+Cqvjfmk6E="  # short, prefix-less: not scannable

# --- gitleaks-derived reference fixtures (canonical public example secrets) ---
# Sourced conceptually from gitleaks' default-rule reference fixtures + the
# respective vendors' own documentation placeholders. Non-functional by design.
GITLEAKS_DERIVED: list[RedactionCase] = [
    _case(
        "aws-access-key-id-docs-example",
        "aws_access_key_id = {s}",
        ("AKIA", "IOSFODNN7EXAMPLE"),
        "vendor-key",
    ),
    _case(
        "aws-secret-access-key-docs-example",
        "aws_secret_access_key = {s}",
        ("wJalrXUtnFEMI/K7", "MDENG/bPxRfiCYEXAMPLEKEY"),
        "assignment",
    ),
    _case(
        "aws-temp-access-key-asia",
        "credentials: {s} were rotated",
        ("ASIA", "ABCDEFGHIJKLMNOP"),
        "vendor-key",
    ),
    _case(
        "github-pat-classic",
        "leaked in the logs: {s} oops",
        ("ghp_", "wWPw5k4aXcaT4fNP0UcnZwJUVFk6LO0pINUx"),
        "vendor-key",
    ),
    _case(
        "github-pat-fine-grained",
        "token {s} found",
        ("github_pat_", "11ABCDEFG0abcdefghijkl_1234567890abcdefghijklmnopqrstuvwxyzABCDEFGH"),
        "vendor-key",
    ),
    _case(
        "slack-bot-token",
        "slack webhook uses {s} here",
        ("xoxb-", "1234567890-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"),
        "vendor-key",
    ),
    _case(
        "google-api-key",
        "maps key {s} embedded",
        ("AIza", "SyA1234567890abcdefghijklmnopqrstuv"),
        "vendor-key",
    ),
    _case(
        "stripe-secret-key",
        "billing configured with {s} last week",
        ("sk_live_", "4eC39HqLyjWDarjtT1zdp7dc"),
        "vendor-key",
    ),
    _case(
        "openai-api-key",
        "the OpenAI call used {s} and failed",
        ("sk-", "abcdefghijklmnopqrstuvwxyz1234567890ABCDefgh"),
        "vendor-key",
    ),
    _case(
        "anthropic-api-key",
        "set ANTHROPIC key to {s} now",
        ("sk-ant-", "api03-abcDEF1234567890_ghijklmnopqrstuvwxyz-ABCDEFGH"),
        "vendor-key",
    ),
    _case(
        "jwt-jwtio-sample",
        "bearer session {s} stored",
        (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.",
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        ),
        "vendor-key",
    ),
    RedactionCase(
        "rsa-private-key-block",
        f"{_PEM_HDR}\n{_PEM_BODY}\n{_PEM_BODY2}\n{_PEM_END}",
        _PEM_BODY,
        "private-key",
    ),
]

# --- synthetic fixtures (hand-authored assignment / header shapes) ---
SYNTHETIC: list[RedactionCase] = [
    _case(
        "password-double-quoted",
        'db config: password="{s}" in the connection string',
        ("hunter2", "trombone"),
        "assignment",
    ),
    _case(
        "db-password-env",
        "export DB_PASSWORD={s} before running",
        ("s3cr3t", "P4ssw0rd"),
        "assignment",
    ),
    _case(
        "client-secret-single-quoted",
        "oauth client_secret: '{s}' rotated",
        ("abcdef1234567890", "zzzz"),
        "assignment",
    ),
    _case(
        "api-key-assignment",
        "api_key = {s} configured",
        ("9f8e7d6c5b4a", "32100011"),
        "assignment",
    ),
    _case(
        "x-api-key-header",
        "curl -H 'X-Api-Key: {s}' https://api.example.com",
        ("1234567890", "abcdef1234"),
        "assignment",
    ),
    _case(
        "authorization-bearer-header",
        "Authorization: Bearer {s}",
        ("abcdef1234567890", "ABCDEF9876"),
        "auth-header",
    ),
    _case(
        "authorization-basic-header",
        "sent Authorization: Basic {s} to the proxy",
        ("dXNlcjpzdXBlcnNl", "Y3JldHBhc3N3b3Jk"),
        "auth-header",
    ),
    _case(
        "bare-bearer-token",
        "retry with Bearer {s} and it worked",
        ("eyJ0eXAabcdef", "1234567890ghijklmnop"),
        "auth-header",
    ),
    _case(
        # Opaque (non-vendor) value on purpose, so this stays a clean test of
        # the assignment rule; the vendor `ya29.` path is covered separately by
        # bare-google-oauth-access-token below.
        "access-token-assignment",
        "the access_token={s} expired",
        ("8f3a9c2e", "7b1d4056aa991122"),
        "assignment",
    ),
    # Self-review additions (real false-negative shapes found during Phase 2 EPT):
    _case(
        # sk-proj keys carry `_`/`-` in the body and a short first segment, which
        # an alnum-only body pattern would miss entirely.
        "openai-project-key",
        "the call used {s} and 429'd",
        ("sk-proj-", "Ab12_cd34-Ef56gh78ij90klMNOPqrst_uvwxYZ01"),
        "vendor-key",
    ),
    _case(
        "slack-app-level-token",
        "app token {s} set",
        ("xapp-", "1-A012BC34DEF-0123456789012-abcdef1234567890abcdef1234"),
        "vendor-key",
    ),
    _case(
        "bare-google-oauth-access-token",
        "the token {s} leaked in a log",
        ("ya29.", "a0AbCdEf1234567890_ghijklmnopqrstuvwx-YZ"),
        "vendor-key",
    ),
    # RC-gate (independent code-reviewer) additions - real leak shapes it found:
    _case(
        # URI-embedded basic-auth credential; the password after `user:` and
        # before `@` is the secret. No assignment keyword is present.
        "uri-embedded-credential",
        "DATABASE_URL=postgres://admin:{s}@db.example.com:5432/prod",
        ("S3cr3t", "P4ssw0rd"),
        "url-credential",
    ),
    _case(
        # Slack rotation/refresh token family `xoxe-` (class had no `e`).
        "slack-refresh-token-xoxe",
        "rotated {s} today",
        ("xoxe-", "1-A01234567-1234567890123-abcdefghijklmnopqrstuvwxyz123456"),
        "vendor-key",
    ),
    RedactionCase(
        # A PEM block truncated by pagination/copy-paste (no -----END----- line);
        # the body must still be masked.
        "truncated-private-key-no-end",
        f"{_PEM_HDR}\n{_PEM_BODY}",
        _PEM_BODY,
        "private-key",
    ),
]

POSITIVE_CASES: list[RedactionCase] = GITLEAKS_DERIVED + SYNTHETIC

# --- precision guard: prose that must NOT be redacted (zero findings) ---
NEGATIVE_CASES: list[tuple[str, str]] = [
    ("password-mention", "I reset my password yesterday and it worked fine."),
    ("secret-idiom", "The secret to good code is writing the test first."),
    ("token-based-auth", "We switched to token-based authentication last sprint."),
    ("authorization-header-docs", "See the Authorization header docs for details."),
    ("bearer-of-letter", "The bearer of this letter is a trustworthy person."),
    ("api-word", "Update the api documentation before the release."),
    ("access-level", "Set the access level to admin for the new role."),
    ("akia-prefix-explained", "AKIA is the standard prefix for AWS access key IDs."),
    ("conventional-commit", "feat: add token bucket rate limiter for the gateway"),
    ("password-requirements", "Password requirements: at least eight characters long."),
    ("ask-questions-hyphenated", "This is an ask-questions-about-everything kind of team."),
    ("secret-colon-short", "It is a secret: no."),
    ("json-fields-prose", 'The config exposes a "name" field and a "type" field.'),
    ("honest-take-colon", "Here is my honest take: tests matter more than coverage."),
    # A plain URL with a port (no `user:pass@`) must not trip the url-credential rule.
    ("url-with-port-no-creds", "connect to https://api.example.com:5432/v1/data please"),
    ("git-ssh-url", "clone git@github.com:primeline-ai/kairn.git into your workspace"),
]
