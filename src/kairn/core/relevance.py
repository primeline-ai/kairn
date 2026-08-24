"""What the `relevance` number in a result actually means.

THE PROBLEM THIS NAMES. Every retrieval surface returns rows carrying a
`relevance` float, and the number is computed three different ways depending on
where the row came from:

    node, keyword recall   bm25_to_relevance(rank)   how well the TEXT matched
    node, crossref         the literal 1.0           nothing - it is a constant
    experience, all paths  Experience.relevance(at)  how RECENT it is

A caller reading `relevance: 0.98` on an experience reasonably concludes it was
an excellent match for their query. It means the row is a few hours old. The
same caller reading `relevance: 0.62` on a node is looking at a match score.
Sorting them together, which `crossref` does, compares a recency to a constant.

Nothing here changes how any number is computed. This module only lets a result
SAY which of the three it is, so a caller can stop guessing - the mismatch is
what the plan calls the defect, and mislabelling is the cheapest half of it to
close.

USE THE CONSTANTS, never the bare strings. Three call sites hand-typing
"recency" is how the fourth one ends up typing "recent".
"""

# How recent the row is. Pure time-decay from Experience.relevance(at). Says
# NOTHING about whether the row matched the query - a completely unrelated
# experience created this morning scores near 1.0.
RELEVANCE_KIND_RECENCY = "recency"

# How well the row's text matched the query, via bm25_to_relevance(rank).
# Corpus-size dependent, so it is comparable within one result set and not
# across two.
RELEVANCE_KIND_MATCH = "match"

# Not a score at all - a placeholder the surface fills in with a constant
# because it has no ranking to report. Present so that a caller sorting on
# `relevance` can see that some rows carry no information, rather than reading
# the constant as a perfect score.
RELEVANCE_KIND_UNSCORED = "unscored"

RELEVANCE_KINDS = frozenset(
    {RELEVANCE_KIND_RECENCY, RELEVANCE_KIND_MATCH, RELEVANCE_KIND_UNSCORED}
)
