# D-XXX — <Title>

| | |
|---|---|
| **ID** | D-XXX |
| **Date** | YYYY-MM-DD |
| **Status** | Proposed / Ruled / Superseded-by-D-XXX / Retired |

## Context

What situation forced a decision. 2–5 sentences; link the deliberation
brief in `docs/briefs/` if one exists.

## Options considered

| Option | Summary | Why (not) |
|---|---|---|
| A | … | … |
| B | … | … |

## Evidence

What was actually measured/observed, with links — never "we felt".
Numbers belong here, sources beside them.

## Ruling + rationale

The decision, stated so a future reader can apply it without this
conversation. Why this option won, in the decider's terms.

## Consequences

What this commits us to, what it forecloses, what debt it accepts.

## Revisit triggers

Specific, observable conditions that re-open this record — "when X is
seen in production" / "when harness Y contradicts Z". Never "someday".

## Retest recipe

The exact command(s)/config that re-run this record's evidence, e.g.:

```
python3 test_regime.py
python3 scripts/backtest_regime.py --skip-grid
```

A record whose evidence cannot be re-run documents that honestly and
says what WOULD run.

## Links

- Jira: PER-XXX comment <id>
- Brief: docs/briefs/<file>.md
- Commits: <sha> …
- Docs: docs/<file>.md §…
