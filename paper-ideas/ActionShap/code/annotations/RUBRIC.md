# Modifiability Elicitation Rubric

Version 1.0

This rubric operationalizes Definition 1 of the manuscript. It is reproduced
verbatim in Appendix A3 of the paper, so treat any change to it as a change to
the paper.

## The three levels

Assign each factor exactly one of:

| Value | Meaning |
|---|---|
| **1.0** | Directly controllable by the decision-maker. They can set or adjust this factor as a deliberate act, within a normal operating cycle, without first changing something else. |
| **0.5** | Controllable indirectly, or at substantial cost or delay. The decision-maker can influence it, but only through another factor, over a long horizon, or at a cost that would make routine adjustment impractical. |
| **0.0** | Observable but immutable. The decision-maker can measure it and reason about it, but cannot change it at all. |

Only these three values are admissible. The loader rejects anything else,
because a value outside the scale means the rubric was not followed rather than
that a finer judgment was made.

## Who the decision-maker is

Modifiability is a property of the deployment context, not of the model or the
data. Before annotating a dataset, write down who the decision-maker is and
what powers they have, and annotate consistently against that person. State
this at the top of the annotation file.

- **Wine:** a production winemaker adjusting a batch within a normal
  vinification cycle.
- **Air quality:** a municipal air-quality regulator setting emissions policy
  over a season.

A factor's level can differ between contexts. That is expected and is why the
decision-maker must be named.

## Procedure

1. Read the decision-maker description at the top of the annotation file.
2. Work through the factor list **independently**. Do not confer, and do not
   look at any attribution output, intervention result, or model behaviour.
3. For each factor ask, in order:
   - Can the decision-maker change this at all? If no, assign **0.0**.
   - Can they change it directly, as a deliberate act within one operating
     cycle? If yes, assign **1.0**.
   - Otherwise assign **0.5**.
4. Record your values under your own annotator key.
5. Only once all three annotators are done: compute agreement, discuss
   disagreements, and record the resolution. Do not revise your own values
   before this step.

## Deciding the hard cases

**Derived quantities.** A factor that is a consequence of other factors rather
than an input is **0.0** even when it is highly predictive. Wine density is the
worked example: it follows from sugar and alcohol content, so a winemaker
changes it only by changing those, and cannot set it directly. Recording it as
modifiable would credit an intervention nobody can perform.

**Preventable but not reversible.** A factor that can be avoided in advance but
not corrected afterwards is **0.5**, not 1.0. The intervention under study acts
on an existing batch or an existing state, not on a counterfactual past.

**Regulated or contested.** Whether the decision-maker is *permitted* to change
a factor is out of scope. Annotate physical and operational capability only,
and note any legal constraint in the comments rather than folding it into the
value.

## Freezing

Annotation must be complete and committed **before any attribution or
intervention result is inspected**. This is the substantive safeguard in the
protocol: independence between annotators only shows the rubric is clear,
whereas freezing is what prevents modifiability from being tuned toward a
favourable finding.

Compute the payload hash and freeze with:

```bash
python scripts/freeze_annotation.py annotations/wine.yaml
git add annotations/wine.yaml && git commit -m "Freeze wine modifiability annotation"
```

The loader recomputes the hash on every run and refuses to proceed if the
payload has changed. If you must revise an annotation after freezing, re-freeze
and disclose the revision in the paper. Do not edit silently — the
pre-registration claim in §4.2 depends on this.
