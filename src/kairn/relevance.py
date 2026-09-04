"""What the `relevance` number in a result actually means.

TOP-LEVEL, NOT UNDER `core/`. `models/` imports this, and `models/` importing
`core/` would invert the package layering (`core` imports `models`, never the
reverse). No cycle fires today only because `core/__init__.py` is a bare
docstring - add one re-export there and `import kairn.models.experience` becomes
a partially-initialised-module ImportError. A constants module belongs above
both layers.

THE PROBLEM THIS NAMES. Every retrieval surface returns rows carrying a
`relevance` float, and the number is computed FOUR different ways depending on
where the row came from and which config is on:

    node, keyword recall      _bm25_to_relevance(rank)  how well the TEXT matched
    node, semantic recall     cosine similarity         embedding distance
    node, crossref / browse   the literal 1.0           nothing - a constant
    experience, all paths     Experience.relevance(at)  how OLD it is

A caller reading `relevance: 0.98` on an experience reasonably concludes it was
an excellent match for their query. It means the row is a few hours old. Sorting
them together compares a decay against a constant.

Nothing here changes how any number is computed. This module only lets a result
SAY which of the four it is.

USE THE CONSTANTS, never the bare strings. Three call sites hand-typing
"recency" is how the fourth one ends up typing "recent".
"""

# Age, via Experience.relevance(at) = score * exp(-decay_rate * days).
# NOT pure time: `score` is currently always 1.0, but `decay_rate` is derived
# from the experience TYPE's half-life and its CONFIDENCE, so two rows created
# in the same second can report different numbers. It is age-driven and says
# NOTHING about whether the row matched the query - a completely unrelated
# experience saved this morning scores near 1.0.
RELEVANCE_KIND_RECENCY = "recency"

# Lexical match strength, via `_bm25_to_relevance(rank)` in
# `kairn.core.intelligence`. Corpus-size dependent, so it is comparable within
# one result set and not across two.
RELEVANCE_KIND_MATCH = "match"

# Lexical match strength with a BOUNDED recency nudge, from the experience
# recall path. It is `bm25 * term_coverage` multiplied by a factor in
# [0.9, 1.1] derived from the row's time-decay, so recency can order two
# comparable matches and can never promote a weak fresh hit over a strong old
# one. It gets its OWN label rather than reusing MATCH because this module
# exists to stop one word covering two quantities: a caller comparing an
# experience against a node needs to know that one of the two carries an age
# term. Rows that never went through a text recall fall back to RECENCY.
RELEVANCE_KIND_MATCH_RECENCY = "match_recency"

# Embedding cosine similarity, from the semantic recall path. A different scale
# from MATCH and corpus-independent - which is why it does not share that label
# even though both answer "how well did this match".
RELEVANCE_KIND_SIMILARITY = "similarity"

# Not a score at all - a placeholder a surface fills in with a constant because
# it has no ranking to report. Two paths produce it: crossref, which never ranks
# its nodes, and a text-less browse query, where there is no rank to convert.
# Present so a caller sorting on `relevance` can see that some rows carry no
# information rather than reading the constant as a perfect score.
RELEVANCE_KIND_UNSCORED = "unscored"

RELEVANCE_KINDS = frozenset(
    {
        RELEVANCE_KIND_RECENCY,
        RELEVANCE_KIND_MATCH,
        RELEVANCE_KIND_MATCH_RECENCY,
        RELEVANCE_KIND_SIMILARITY,
        RELEVANCE_KIND_UNSCORED,
    }
)
