# MoE regime routing: validation results and explanation

This validates `moe_router.py` the same way `losses/faoc_loss_validation.md`
validated `faoc_loss.py`: no real data, no training — just hand-picked
inputs, checking that the mechanism is wired correctly before anyone builds
on top of it. If you haven't read the FAOC-Loss write-up yet, the short
version of the philosophy is: **this proves the code matches its design,
not that the design has learned anything useful.** Nothing here has been
trained.

## What "MoE regime routing" means, plainly

Instead of one network handling every situation the same way, this splits
the work across four small "expert" sub-networks — one per disruption
regime this project already scopes itself to (`README.md`, `PLAN.md`):
**normal** (no disruption), **GDP**, **Ground Stop**, and **MIT**. A "gate"
looks at the input and decides how much to trust each expert for that
specific case — like four specialists in a room and a triage nurse deciding
how much weight to give each opinion, rather than sending every patient to
the same one doctor.

Because the network was just created with random starting weights and has
never seen real data, **the gate does not yet know what a GDP-like input
looks like versus a Ground-Stop-like one.** All this validation can check is
that the plumbing works: right shapes, valid probabilities, and that it
actually plugs into `FAOCLoss` without breaking.

## How to reproduce this

```bash
cd nas_project
python -m models.planning.validate_moe_router
```

The script fixes a random seed, so the numbers below will come out
identical every time it's run.

## Results

### 1. Shapes and output range

```
pred_probs shape: (5,) (expect (5,))
gate_weights shape: (5, 4) (expect (5, 4))
pred_probs in (0, 1): True
```

Fed 5 fake examples with 6 made-up features each. Got back 5 disruption
probabilities (one per example) and a 5×4 grid of gate weights (one weight
per example per regime). Both match what the design promises, and every
probability landed strictly between 0 and 1 — a valid probability, not a
raw unbounded number.

### 2. Do the gate weights actually behave like a probability distribution?

```
sample 0: sum=1.000000
sample 1: sum=1.000000
sample 2: sum=1.000000
sample 3: sum=1.000000
sample 4: sum=1.000000
```

For every example, its four regime weights add up to exactly 1.0. That's
what a softmax is supposed to guarantee — this confirms the gate is a
genuine "how do I split my trust across 4 options" distribution, not just
four unrelated numbers.

```
sample        normal           gdp   ground_stop           mit
     0        0.2701        0.1818        0.3984        0.1497
     1        0.0195        0.0550        0.5070        0.4184
     2        0.2087        0.1027        0.5160        0.1726
     3        0.1416        0.0517        0.7164        0.0903
     4        0.1757        0.2097        0.3880        0.2267
```

**Important:** don't read anything into *which* regime got the highest
weight here. These are random features going into a randomly-initialized
gate — "ground_stop" winning most rows is a coincidence of the random seed,
not a sign the model thinks these examples look like ground stops. That
column would only become meaningful after training on real data.

### 3. Do different inputs get routed differently?

```
gate for all-zero input:   [0.2277, 0.2539, 0.3519, 0.1665]
gate for all-5.0 input:    [0.0169, 0.4904, 0.0354, 0.4574]
identical routing for both inputs: False (expect False)
```

Two deliberately extreme, very different fake inputs get two clearly
different gate distributions. This confirms the gate is actually a function
of the input — it's not accidentally ignoring the features and returning
some fixed weighting no matter what comes in. (Again: the *specific*
direction of that difference means nothing yet, since nothing has been
trained — only the fact that it *does* differ matters here.)

### 4. Does this actually work together with FAOC-Loss?

```
pred_probs: [0.4745, 0.4586, 0.4706, 0.4611, 0.5043]
labels:     [1.0, 0.0, 1.0, 0.0, 1.0]
airports:   ['ORD', 'ADK', 'LGB', 'ATL', 'ZZZ']
FAOCLoss on router output: 8,971.52
(no error raised -> the two shared utilities compose end to end)
```

This is the check that matters most for the project, not just for this
file: the router's output was fed directly into `FAOCLoss` from Step 1,
with no conversion or glue code needed, and it produced a single loss
number without erroring. That confirms the two shared utilities — the loss
function and the routing architecture — are compatible by design, which
matters because Step 7 (the actual Planning-tier model stub) is supposed to
wire both of these into one model at once.

## What this validation does *not* cover

- **No real features.** The 6 input numbers per example are random noise,
  not real airport-hour features from `common/data_loader.py`. This checks
  wiring, not whether the gate can learn anything meaningful from real
  data — that's Phase 2/3 work.
- **No training.** The gate weights and expert outputs shown above are pure
  artifacts of random initialization. They will look completely different,
  and hopefully *meaningful*, only after a model is actually trained.
- **Soft routing only.** Every expert runs on every example here (a
  weighted blend, not a hard choice). That's a deliberate, documented
  simplification for this first stub — see `moe_router.py`'s docstring for
  why sparse top-k routing was left out for now.
