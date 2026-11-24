# Long-Term Time Series Forecasting with PatchTST

**Team:** Ian Fisk, Ryan Baldwin · CS 545, Colorado State University · **Due:** December 5, 2026

We explain and test **PatchTST** (Nie et al., 2023), a transformer for long-term time series forecasting (LTSF). Our deliverable is one teaching notebook, [`finalreport.ipynb`](finalreport.ipynb), written in the style of the course notebooks for a reader who knows transformers up to BERT and small LLMs.

The story the notebook tells:

1. **Time series basics:** what the data looks like, trend and seasons, and why we predict the whole future at once instead of one step at a time.
2. **Where earlier models fall short:** Zeng et al. (2023) showed that a single linear layer beat the transformer forecasters of the time.
3. **How PatchTST answers that:** *patching* (groups of time steps become tokens) and *channel independence* (every series shares one backbone).
4. **Experiments:** PatchTST compared against seasonal, linear and other baselines, plus experiments on the architecture.

## Repository layout

| Path | What it is |
|---|---|
| `finalreport.ipynb` | The final report notebook |
| `src/ltsf_data.py` | Shared data loader: downloads and loads every benchmark dataset with the papers' splits |
| `src/ian/` | Ian's experiment notebooks (PatchTST, baselines, CSU weather) |
| `src/ryan/` | Ryan's notebooks: explanations, figures and animations, reproduction experiments |
| `data/` | Datasets. Downloads are gitignored; only the CSU weather files are committed |

## Getting the data

```python
import sys; sys.path.append("../")   # from a notebook in src/<name>/
from ltsf_data import load, list_datasets

list_datasets()           # what's available and what's already downloaded
ts = load("traffic")      # downloads to data/ on first use, then loads from disk
ts.values                 # (n_steps, M) float32: one row per time step, one column per channel
ts.split_ratios           # the train/val/test split the papers use for this dataset
```

The benchmark CSVs (ETTh1/h2/m1/m2, Weather, Traffic, Electricity, Exchange Rate, ILI) come from the [Hugging Face mirror](https://huggingface.co/datasets/AutonLab/Timeseries-PILE/tree/main/forecasting/autoformer) of the files both papers used. All ten datasets take **268 MB on disk** (Traffic and Electricity are 232 MB of that). Files download only when first loaded. `ts.values` plugs straight into `split_time_series_data` in Ian's notebooks.

## Who does what

| | Ian | Ryan |
|---|---|---|
| **Focus** | Implementation and architecture experiments | Explaining the papers, diagrams and animations, simple reproduction experiments |
| **Notebook sections** | PatchTST implementation walkthrough, architecture experiments, results | Time series basics, where earlier models fall short, patching and channel-independence explanations |
| **Shared** | Results table, references, final read-through, run-all check | ← same |

## Remaining work

**Ryan**
- [x] Shared data loader for all benchmark datasets (`src/ltsf_data.py`)
- [ ] Time series basics section: the data's shape, trend and seasons, sliding windows, whole-horizon vs. step-by-step forecasting
- [ ] Figures and animations: sliding window, patching, channel independence, architecture diagram with tensor shapes, instance normalization (RevIN)
- [ ] Reproduction experiments: look-back window sweep (PatchTST Fig. 2, Zeng Fig. 4) and input-shuffle test (Zeng Table 5)

**Ian**
- [x] PatchTST from scratch, plus seasonal-naive, linear and linear-on-patches baselines
- [ ] Architecture experiments (candidates from his notes): patching and channel-independence ablation (PatchTST Table 7), held-out channels, distribution shift over time

**Together**
- [ ] Decide whether to keep the vanilla one-token-per-step transformer baseline from the proposal
- [ ] Rough draft of `finalreport.ipynb` (target: Nov 28)
- [ ] Review each other's sections
- [ ] References, dataset citations, restart and run all, submit (Dec 5)

## Contributions

*The assignment asks the notebook to end with each member's contributions, how AI was used, and the resources used. We keep the running record here and copy it over at the end.*

**Ian Fisk**
- Implemented PatchTST from scratch: RevIN, patching, patch embedding, BatchNorm transformer encoder, channel independence, flatten + linear head.
- Wrote the training and evaluation utilities, an experiment runner, and `RandomizedChannelDataset`, which samples channels so 862-channel Traffic fits in GPU memory.
- Built the baselines: seasonal naive, shared and per-channel linear (Zeng et al.), and a linear head on patch embeddings.
- Ran PatchTST and the linear baseline on Traffic at several look-back/horizon settings. At horizon T=720 the test MSE came out close to the published values (PatchTST/64: 0.432; DLinear: 0.466).
- Wrote a scraper for CSU's hourly weather data (`src/ian/data/csu-weather.js`) and ran experiments on it: learning-rate scheduling and a patch-size sweep.

**Ryan Baldwin**
- Set up the repository, the project timeline and the report notebook.
- Wrote the shared data loader (`src/ltsf_data.py`): it downloads all benchmarks, applies the papers' splits, and puts the CSU data on a complete hourly grid.
- Planned the notebook's teaching sequence, figures and reproduction experiments.

**AI use**
- Ryan: Claude Code (Claude Opus) helped write the data loader, plan the figures and notebook structure, and rewrite this README.
- Ian: _to fill in_

**Resources and existing code**
- RevIN implementation copied (MIT license) from [ts-kim/RevIN](https://github.com/ts-kim/RevIN), the version the official PatchTST code uses.
- Course notebooks (CS 545, module 04) for the transformer building blocks and the animation style.

## References

1. Nie, Nguyen, Sinthong, Kalagnanam. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
2. Zeng, Chen, Zhang, Xu. *Are Transformers Effective for Time Series Forecasting?* AAAI 2023. [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)
3. Wu, Xu, Wang, Long. *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.* NeurIPS 2021. (Source of the benchmark dataset files.)

**Datasets:** [ETT](https://github.com/zhouhaoyi/ETDataset) · [Weather](https://www.bgc-jena.mpg.de/wetter/) · [Traffic](https://pems.dot.ca.gov/) · [Electricity](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) · [CSU Weather](https://climate.colostate.edu/autowx/)
