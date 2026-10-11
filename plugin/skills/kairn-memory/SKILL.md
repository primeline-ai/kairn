---
name: kairn-memory
description: Use when starting a task in a known project, before making a decision, and after fixing, learning, or deciding something worth remembering.
---

Recall before acting: call `kn_recall(topic="<project and topic>")` before
starting work or choosing an approach. Use relevant past knowledge; an empty
result means nothing was found for that topic.

Save after learning: call `kn_learn(content="<what was learned>",
type="solution", confidence="high")` after a useful fix or decision.
Choose type from `decision`, `pattern`, `solution`, `workaround`, `gotcha`,
or `preference`. Use confidence `high` for permanent knowledge, `medium` for
likely conclusions, and `low` for exploratory findings. Include enough
context to make the entry useful later.

Use `kn_context(keywords="<topic>")` for related graph context,
`kn_memories(text="<query>")` to search saved experiences, and `kn_status()`
to inspect the store.

Never save secrets, credentials, or personal data. Everything stays in
`~/.kairn` on this machine.
