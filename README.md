# Which Transformer Blocks Should You Keep?

**An Exhaustive Map of Layer-Subset Selection in a Pretrained ViT**

Steffen Jung and Margret Keuper
NeurIPS 2026 Workshop on On-Device Intelligence:
Foundation Models under Real-World Constraints

This repository contains the complete measured landscape of all 4,095
nonempty block subsets of DINOv3 ViT-S/16 after three epochs of end-to-end
finetuning on ImageNet-1k, together with figure and search-replay code.


## Files

```text
search.db                Complete subset measurements and training times.
make\_figures\_1to3.py      Landscape, truncation, and block-selection plots.
make\_figure\_4.py          Four-panel interaction and search figure.
replay\_search.py         Shared replay implementation and numerical results.
README.md                These instructions.
```

## Installation and reproduction

Tested with Python 3.12.7, NumPy 2.3.3, and Matplotlib 3.10.6.
SQLite support is included in Python. Install the plotting dependencies:

```bash
python -m pip install numpy==2.3.3 matplotlib==3.10.6
```

From the repository directory, run:

```bash
python make\_figures\_1to3.py
python make\_figure\_4.py
python replay\_search.py
```

The scripts read search.db and save generated figures and results in the
same directory as the scripts. The database is read-only.

make\_figures\_1to3.py produces:

* fig1\_frontier.pdf: accuracy landscape and truncation ranking.
* fig2\_structure\_and\_marginal\_value.pdf: combined block-selection and
marginal-value panels.
* fig2\_kept\_dropped\_grid.pdf and fig3\_marginal\_value.pdf: standalone panels.
* interaction.pdf: a two-panel deletion/scoring diagnostic.

make\_figure\_4.py produces fig4\_interaction\_and\_search.pdf, containing
single-block deletions, two-block interactions, additive scoring, and search.
Some figures also have PNG previews.

The four-panel figure defaults to budgets B=10,20,...,200, with mean curves
and 10th-90th percentile bands. Axis labels are sparse for readability.
--no-baselines generates a mean-only alternative without bands.
--with-baselines also adds the truncation and additive-predictor references.

## Database schema

search.db has one table, search\_runs, with exactly 4,095 rows:

```text
schedule\_key          Twelve comma-separated 0/1 indicators for blocks 1-12.
val\_top1              Best validation top-1 accuracy over three epochs (%).
val\_top5              Top-5 accuracy at the selected top-1 checkpoint (%).
num\_trainable\_params  Trainable parameter count, including the classifier.
num\_layers            Number of retained blocks (sum of the indicators).
training\_time         Elapsed training-start-to-completion time in minutes.
```

Example: 0,0,1,1,0,1,0,0,0,0,0,1 executes blocks \[3,4,6,12] in that order.
All schedules are nonempty and binary. Every depth k
contains exactly binomial(12,k) subsets. Trainings used one NVIDIA RTX8000 GPU per candidate. The recorded intervals
total 4,034.11 GPU-hours and include initialization, training, validation,
checkpoint writing, and runtime delays.

## Search protocol

make\_figure\_4.py imports the implementation in replay\_search.py. Both use:

* Random-restart first-improvement local search and uniform random sampling
with replacement.
* 300 search trials per method and depth, at k=3,...,9.
* Budgets B=10,20,...,200, using prefixes of the same 200-evaluation run.
* One charged evaluation per lookup, including restarts and repeats.

Local search swaps one retained block with one omitted block. It checks
neighbors in shuffled order, accepts the first improvement greater than
1e-12, and restarts when no neighbor improves the current subset.

For trial t=0,...,299, depth k, and method index m (0 for local search,
1 for random search), the seed is:

```text
42 + t + 300\*((k-3) + 7\*m)
```

The first seed range is 42-341; each method/depth pair has its own range.
At each budget, selected accuracy minus best observed accuracy is averaged
equally over the seven depths for each trial. Curves show the mean over
300 depth-averaged trials. Bands show their 10th and 90th percentiles using
linear interpolation.

Both commands generate identical fig4\_search\_bands\_statistics.json files,
including seeds, raw gaps, summary statistics, and a database checksum.
The replay command also prints per-depth results and optimum-hit rates.
Preserve database row order for exact seeded reproduction: populations
are loaded with ORDER BY rowid.

The mean gaps below the best observed subset at B=50/100 are 0.18/0.02
percentage points for local search and 0.79/0.50 for random search.

## Citation

If you use these measurements or analysis code, please cite:

```bibtex
@inproceedings{jung2026which,
  title     = {Which Transformer Blocks Should You Keep? An Exhaustive Map of Layer-Subset Selection in a Pretrained {ViT}},
  author    = {Jung, Steffen and Keuper, Margret},
  booktitle = {NeurIPS 2026 Workshop on On-Device Intelligence: Foundation Models under Real-World Constraints},
  year      = {2026}
}
```

