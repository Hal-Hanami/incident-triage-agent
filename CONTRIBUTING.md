# Contributing

## Commit messages

This repository uses [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <summary in the imperative, lower case, no trailing period>

<body: why this change is needed, and anything a reader would otherwise
have to reconstruct from the diff>
```

**Types**

| Type | Use for |
|---|---|
| `feat` | new capability |
| `fix` | corrected behaviour |
| `docs` | documentation only |
| `test` | tests only |
| `refactor` | restructuring with no behavioural change |
| `perf` | measured performance or cost improvement |
| `build` | packaging, dependencies, CI |
| `chore` | housekeeping that fits nothing above |

**Scopes** follow the pipeline: `schema`, `classify`, `retrieve`, `draft`,
`decide`, `agent`, `eval`, `observe`, `cli`, `demo`, `fixtures`.

```
feat(decide): escalate when the runbook itself directs a human handoff
fix(observe): price cache reads at 0.1x so the cross-check stops undercounting
docs(evaluation): report the false-abstention count the design target missed
test(agent): deny every mutating built-in by name, not by tool input
```

**The body explains why.** The diff already shows what changed; a reader a year
from now needs the reason. Constraints discovered, alternatives rejected, and
measurements that motivated the change all belong here.

**Keep it about the system.** Commit messages describe the software, not the
process that produced it and not the author's circumstances. If a sentence would
not make sense to someone who has never met the author, it belongs in a personal
note rather than in the history of a public repository.

## Code comments

Comments explain **why**, not what. The code already states what it does, and a
comment restating it will drift out of date and start lying.

Worth a comment: a constraint that is not visible locally, a non-obvious ordering
requirement, a rejected alternative, a value that came from a measurement.
Not worth a comment: anything a reader can get from the line itself.

Module and function docstrings carry the reasoning; inline `#` comments are for
the specific line that would otherwise make a reader stop and squint.

## Numbers

Every number in the documentation must name the date it was measured, the set it
was measured over, and a command that reproduces it — that is what
[`docs/EVALUATION.md`](docs/EVALUATION.md) is for. A number that cannot carry
those three things does not belong in a headline; it belongs in a labelled note,
or nowhere.

A design target that was not met is reported as not met.

## Tests

Every LLM and network boundary sits behind a Protocol (`Classifier`, `Retriever`,
`Drafter`, `Judge`), so the decision table, the guardrail policy, the pricing
math, and the demo round trip are all exercised offline with fakes. A test that
needs an API key or a network connection is a sign that policy and adapter have
become entangled.

```sh
uv run --with pytest python -m pytest -q     # or: pip install -e '.[dev]' && pytest -q
```

Tests must pass without `ANTHROPIC_API_KEY` or `VOYAGE_API_KEY` set, and without
the sibling tech-docs-rag checkout present.

The guardrail tests are the ones to be careful with. They assert that mutating
tools are denied **by name**, never by inspecting tool input — a guard that reads
command strings can be talked around, and a test that checks the input path would
quietly bless that design.
