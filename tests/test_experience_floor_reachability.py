"""The floor's suggested value has to be REACHABLE, and the message says so.

`_validate_experience_min_match` tells a user to try 0.65. That sentence is a
claim about a quantity nobody had measured: `bm25_match * term_coverage` is a
saturating transform of a score whose magnitude grows with corpus size, so
"0.65" means different things on a one-row store and on a real one.

Measured here rather than asserted, because the error message now quotes these
numbers and a number in a message with no test behind it is exactly the
"exemption asserted without a test" shape this branch already fixed once.

The bounds are deliberately loose and one-sided. What must hold is the SHAPE -
unreachable when empty, reachable at any real size, and still discriminating -
not the fourth decimal of a bm25 constant.
"""

from __future__ import annotations

import pytest

from kairn.cli import _validate_experience_min_match as cli_validate
from kairn.core.experience import ExperienceEngine
from kairn.core.fts import bm25_match, fts_keywords, term_coverage, to_fts_query
from kairn.events.bus import EventBus
from kairn.server import _validate_experience_min_match as server_validate

SUGGESTED = 0.65

TARGET = "postgres connection pool exhaustion under pgbouncer transaction mode"
FILLER = [
    "kubernetes ingress annotation rewrite target regression",
    "rust borrow checker lifetime elision on nested closures",
    "swift concurrency actor reentrancy deadlock in main actor",
    "terraform state lock dynamodb table missing after import",
    "elasticsearch shard allocation awareness rack id mismatch",
    "grpc deadline propagation across a sidecar proxy hop",
    "webpack tree shaking sideEffects false breaks css import",
    "ffmpeg hwaccel videotoolbox colour range mismatch on export",
]


async def _strength(store, question: str) -> float:
    """The number the abstention gate actually reads, for the target row."""
    fts = to_fts_query(question)
    terms = fts_keywords(question)
    rows = await store.query_experiences(text=fts, limit=100000, offset=0)
    for data in rows:
        if data.get("content") == TARGET:
            return bm25_match(data.get("rank")) * term_coverage(
                terms, data.get("content"), data.get("context")
            )
    raise AssertionError(f"the target row was not returned for {question!r}")


async def _store_of(store, n_rows: int):
    """The target row plus `n_rows - 1` unrelated ones."""
    eng = ExperienceEngine(store, EventBus())
    await eng.save(type="solution", content=TARGET, confidence="high")
    for i in range(n_rows - 1):
        await eng.save(
            type="solution",
            content=f"{FILLER[i % len(FILLER)]} variant {i}",
            confidence="high",
        )
    return store


@pytest.mark.asyncio
async def test_the_suggested_floor_is_unreachable_on_a_one_row_store(store):
    """bm25 IDF is ~0 when every document is the only document.

    A row queried with its OWN EXACT CONTENT - the strongest question anyone
    can ask of it - scores essentially zero. So the suggested floor rejects a
    perfect match here, and the message must not present 0.65 as a
    scale-free percentage.
    """
    await _store_of(store, 1)
    strength = await _strength(store, TARGET)
    assert strength < 0.01, strength
    assert strength < SUGGESTED


@pytest.mark.asyncio
async def test_the_suggested_floor_is_reachable_once_the_store_has_ten_rows(store):
    """Positive control: the floor is not unreachable in general, only when empty."""
    await _store_of(store, 10)
    strength = await _strength(store, TARGET)
    assert strength >= SUGGESTED, strength


@pytest.mark.asyncio
async def test_strength_grows_with_the_store_and_stays_below_one(store):
    """The saturating transform never reaches 1.0, which is why >1.0 rejects all."""
    await _store_of(store, 400)
    strength = await _strength(store, TARGET)
    assert SUGGESTED < strength < 1.0, strength


@pytest.mark.asyncio
async def test_the_floor_still_discriminates_on_a_real_sized_store(store):
    """Negative control: reachable must not mean "everything clears it".

    The message claims 0.65 rejects a one-word question and passes a two-word
    one. If that ever stops being true the floor is decorative and the advice
    is wrong, so it is pinned here rather than left in prose.
    """
    await _store_of(store, 400)
    one_word = await _strength(store, "postgres")
    two_words = await _strength(store, "postgres pool")
    assert one_word < SUGGESTED <= two_words, (one_word, two_words)


@pytest.mark.parametrize("validate", [server_validate, cli_validate])
def test_the_message_carries_the_measurement_not_a_percentage(validate):
    """Both copies say what the number means, and neither calls it a percentage."""
    with pytest.raises(ValueError) as excinfo:
        validate(65)
    message = str(excinfo.value)
    assert "NOT a percentage" in message, message
    assert "term coverage" in message, message
    assert str(SUGGESTED) in message, message
    # Negative control: the retired wording equated the floor with a percent.
    assert "for 65%" not in message, message
