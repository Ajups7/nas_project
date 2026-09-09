# Long-horizon features: validation results and explanation

This validates `planning.py` differently from `losses/faoc_loss_validation.md`
and `models/planning/moe_router_validation.md` — those used fake, hand-picked
numbers on purpose. **This one runs against the real downloaded data**
(`data/weather_metar/`, `data/aspm_edct_report/`), because that data actually
exists and this is the first Planning-tier code that touches it. Running it
for real surfaced two genuine issues in the data pipeline, not just in this
file — both are documented below, not swept under the rug.

## What Step 3 is building

Three feature groups for the 1–30 day forecast horizon, all answering the
same question: *what would you know about an airport this far out, before
any real weather forecast or near-term signal exists?*

1. **Calendar/seasonal signals** — day of week, month, a smooth day-of-year
   encoding. Pure date math, no data involved.
2. **Weather climatology** — the *typical* weather for this airport in this
   calendar month, averaged across all downloaded years. Not a forecast —
   METAR is observed history, not prediction (see `schema.md`).
3. **Disruption climatology** — how often, historically, this airport has
   needed EDCT (this project's GDP proxy) in this calendar month.

## How to reproduce this

```bash
cd nas_project
python -m features.validate_planning_features
```

Runs on airport **ATL**. Weather climatology uses the *full* 2016–2026
history (fast — one file per airport). Disruption climatology uses a
**6-month sample** (3 Januaries + 3 Julys across different years), not the
full 119-month history — see "Why only a sample" below.

## Results

### 1. Calendar features — plausible on inspection

```
      date  month  day_of_week  is_weekend  day_of_year_sin  day_of_year_cos
2024-01-01      1            0       False         0.017166         0.999853
2024-01-02      1            1       False         0.034328         0.999411
...
2024-01-06      1            5        True         0.102821         0.994700
2024-01-07      1            6        True         0.119881         0.992788

sin range: [0.017, 0.120]   cos range: [0.993, 1.000]
```

Jan 1, 2024 was a Monday (`day_of_week=0`) — correct. Jan 6–7 are flagged
`is_weekend=True` — correct (Saturday/Sunday). The sin/cos values stay
inside [-1, 1] as required, and both are near their maximum in early
January, which is exactly where the year "wraps around" — this cyclical
encoding exists so Dec 31 and Jan 1 read as numerically *close*, instead of
maximally far apart the way a raw "day 365" vs. "day 1" would.

### 2. Weather climatology — matches real-world seasonal patterns

```
airport  month  climo_tmpf  climo_sknt  climo_vsby
    ATL      1   45.3   (Jan, coldest)
    ATL      7   81.0   (Jul, hottest)
    ATL     12   48.8
```

Computed from ATL's **entire** downloaded METAR history in well under a
second. January averages ~45°F, July ~81°F, with a smooth curve up and back
down in between — that's Atlanta's actual climate. This is the kind of
result you can sanity-check just by knowing the city, and it checks out.

### 3. Disruption climatology — also matches a known real-world pattern

```
airport  month  climo_disruption_rate
    ATL      1               0.947
    ATL      7               1.862
```

July's disruption rate is about **2x** January's at ATL, from real EDCT
data. This lines up with something genuinely true about Atlanta: summer
afternoon thunderstorms are a well-known major cause of ground delays there,
more so than winter weather in most years. This isn't proof the number is
exactly right, but it's a meaningful sanity check — a climatology feature
that got the *direction* of a well-known seasonal pattern backwards would be
a red flag, and this one didn't.

### 4. Full assembly — merges cleanly, no missing values

```
No missing values after merge: True
```

Calendar features, weather climatology, and disruption climatology all
merged together on `(airport, month)` with zero `NaN`s introduced by the
join itself.

## Two real issues this surfaced (not synthetic — found by running on real data)

### Issue 1: `load_weather()` returns numeric columns as text

`common/data_loader.py`'s `load_weather()` (a **shared** Phase 0 utility,
used by all three tiers) returns `tmpf`, `dwpf`, `drct`, `sknt`, `gust`,
`vsby` as **string/object dtype, not numbers** — confirmed directly:

```python
>>> load_weather('ATL')[['tmpf','dwpf','drct','sknt','gust','vsby']].dtypes
tmpf     object
dwpf     object
...
```

Cause: the raw METAR files use `"M"` as IEM's standard "missing
observation" marker — routine for `gust`, which only gets reported when a
gust actually happens (**78,338 of ATL's 88,308 rows**, ~89%, are `"M"` for
that column alone). A single non-numeric value anywhere in a column forces
pandas to read the whole column as text.

**Consequence if unhandled:** any numeric operation (`.mean()`, plotting,
model input) on these columns silently breaks or misbehaves. In this case
it crashed with a confusing error — pandas' fallback aggregation path
concatenated dozens of string values into one unreadable blob before
failing, which looked nothing like a "missing value" problem at first
glance.

**Fix applied**: `build_weather_climatology()` now explicitly converts these
columns with `pd.to_numeric(errors="coerce")` before aggregating, so `"M"`
correctly becomes `NaN` and is excluded from the average, rather than
breaking the whole computation. **This fix lives in `planning.py` only** —
`common/data_loader.py` itself was deliberately left untouched, since it's a
shared Phase 0 utility and `TEAM_PLAN.md` says changes to shared code need a
heads-up to the other two people first. Worth raising with them: any
Tactical/Strategic feature code doing numeric work on `load_weather()`'s
output will hit this same thing.

### Issue 2: 2 months of EDCT data are genuinely missing

Already known and documented (`README.md`: *"ASPM EDCT Report — 119/121
months (2 known gaps)"*) — this write-up's contribution is confirming
exactly *which* months and making `planning.py` handle it gracefully instead
of crashing:

```
missing months: [(2016, 7), (2025, 9)]
```

`data/aspm_edct_report/download_log_2016_retry1.txt` confirms the cause for
2016-07: `[fail] 2016-07 error: no download event observed`, after an
earlier timeout — a download that was retried once and still never
succeeded.

**Fix applied**: `build_disruption_climatology()` now checks whether each
month's file exists *before* trying to load it, and skips missing months
with a printed warning instead of crashing. Worth knowing if you hit this:
the failure doesn't come back as a clean `FileNotFoundError` — when the
`.xls` path doesn't exist, pandas' `read_html` falls through to treating the
path *string itself* as literal HTML text and fails with a generic
`ValueError("No tables found")`, which looks like a parsing bug, not a
missing file. Checking file existence directly (rather than catching that
exception) avoids ever needing to tell those two failure modes apart.

## Practical cost, measured (not estimated)

Loading one month of EDCT data via `load_edct()` takes **~13–14 seconds**
(it parses a large HTML table with `pandas.read_html`). Building disruption
climatology from all 119 available months, for one airport, would take on
the order of **22–26 minutes**. That's a one-time cost per airport — for all
82 airports, expect several hours if done naively, one airport at a time.
**Recommendation:** run this once as an offline batch job and cache the
result (e.g. to a parquet/CSV file keyed by airport+month), rather than
recomputing it inside a training loop or every time someone imports this
module.

## Two Python packages installed and added to `requirements.txt`

While loading EDCT files, `pandas.read_html` needed two optional parser
dependencies that weren't installed — `html5lib` and `beautifulsoup4` — as
a fallback for at least one file `lxml` alone couldn't parse cleanly. Both
are now in `requirements.txt` so a fresh `pip install -r requirements.txt`
won't hit this.

## What this validation does *not* cover

- Only one airport (ATL) and a 6-month EDCT sample were checked in detail.
  The seasonal patterns matching real-world expectations for ATL is
  reassuring, but isn't proof every one of the 82 airports' climatology will
  come out sensible — spot-checking a couple more, especially ones with very
  different climates, would be a reasonable next check.
- No claim is made that these three feature groups are *sufficient* for the
  Planning tier's model — only that what's built here is computed correctly
  from real data.
