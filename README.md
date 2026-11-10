# Project Timeline: Long-Term Time Series Forecasting with PatchTST

**Team:** Ian Fisk, Ryan Baldwin

**Duration:** November 1 – December 5, 2026 (5 weeks)

Each task has a **Type** that tells you how many checkmarks it needs:

| Type | Meaning | Done when |
|:---:|---|---|
| 👥 **Each** | Individual work | **Both** columns are ✅ Done |
| 📓 **Once** | Shared or main-notebook work | **Either** column is ✅ Done |

⬜ Todo · 🔄 In progress · ✅ Done · ➖ N/A (the other person has the 📓 task)

To claim a 📓 task, set yours to 🔄 In progress and the other person's to ➖ N/A.

### Dataset split

Standard time-series datasets can be found here: https://github.com/laiguokun/multivariate-time-series-data

| | Ian | Ryan |
|---|---|---|
| Datasets | _TBD_ | _TBD_ |

> Suggested split, which balances one small and one large dataset each: **ETTh1 + Traffic** / **Weather + Electricity**. CSU weather data is optional and can go to whoever has more time.

---

## Week 1: Nov 1 – 7 · Setup, Reading & Baselines (start)

| Task | Type | Ian | Ryan |
|---|:---:|:---:|:---:|
| Create the shared GitHub repo and add the collaborator | 📓 Once | ⬜ | ✅ |
| Set up folder structure (`notebooks/`, `data/`, `src/`, `results/`) and `.gitignore` for data files | 📓 Once | ⬜ | ⬜ |
| Set up the environment (`requirements.txt` / conda env) and confirm GPU access on department machines | 📓 Once | ⬜ | ⬜ |
| Create the main `final_project.ipynb` with a section skeleton | 📓 Once | ⬜ | ⬜ |
| Create a personal experiment notebook | 👥 Each | ⬜ | ⬜ |
| Agree on the dataset split (fill in the table above) | 📓 Once | ⬜ | ⬜ |
| Download and load your datasets | 👥 Each | ⬜ | ⬜ |
| Write a shared data-loading and train/val/test split utility so splits match the papers | 📓 Once | ⬜ | ⬜ |
| Agree on evaluation metrics (MSE, MAE) and prediction horizons (e.g., 96 / 192 / 336 / 720) | 📓 Once | ⬜ | ⬜ |
| Read: Nie et al., *A Time Series is Worth 64 Words* (PatchTST) | 👥 Each | ⬜ | ⬜ |
| Read: Zeng et al., *Are Transformers Effective for Time Series Forecasting?* | 👥 Each | ⬜ | ⬜ |
| Read: Rizvi et al., *GLinear* | 👥 Each | ⬜ | ⬜ |
| Draft a mathematical definition of LTSF (look-back window *L*, horizon *T*, channels *M*) in the main notebook | 📓 Once | ⬜ | ⬜ |
| Start the seasonal-average baseline (average of `nDays % period` observations) | 👥 Each | ⬜ | ⬜ |
| Start the linear baseline (Zeng et al.) | 👥 Each | ⬜ | ⬜ |
| **Sync meeting:** share progress and questions from the reading | 📓 Once | ⬜ | ⬜ |

---

## Week 2: Nov 8 – 14 · Finish Baselines, Start PatchTST

| Task | Type | Ian | Ryan |
|---|:---:|:---:|:---:|
| Finish the seasonal-average baseline and record results | 👥 Each | ⬜ | ⬜ |
| Finish the linear baseline and record results | 👥 Each | ⬜ | ⬜ |
| Implement the vanilla encoder-decoder transformer (one token per time step) | 👥 Each | ⬜ | ⬜ |
| Run the vanilla transformer on your datasets and record results | 👥 Each | ⬜ | ⬜ |
| Create a shared results table format (CSV or dict) so results merge cleanly | 📓 Once | ⬜ | ⬜ |
| Implement the PatchTST patching mechanism (patch length, stride, padding) | 👥 Each | ⬜ | ⬜ |
| Implement channel independence (shared weights across univariate channels) | 👥 Each | ⬜ | ⬜ |
| Start training PatchTST on your datasets | 👥 Each | ⬜ | ⬜ |
| Decide which experiments best show off PatchTST, using the papers and our own interests | 📓 Once | ⬜ | ⬜ |
| Divide the chosen experiments between us | 📓 Once | ⬜ | ⬜ |
| Add the baseline descriptions to the main notebook | 📓 Once | ⬜ | ⬜ |
| **Sync meeting:** compare baseline numbers and PatchTST issues | 📓 Once | ⬜ | ⬜ |

---

## Week 3: Nov 15 – 21 · Evaluate PatchTST & Start Documenting

| Task | Type | Ian | Ryan |
|---|:---:|:---:|:---:|
| Finish PatchTST training on your datasets | 👥 Each | ⬜ | ⬜ |
| Run your assigned experiments (ablations, e.g., patching vs. no patching, channel-independent vs. mixed, look-back length) | 👥 Each | ⬜ | ⬜ |
| Compare PatchTST to each baseline and note *why* the results differ | 👥 Each | ⬜ | ⬜ |
| Create plots of your results (forecast vs. ground truth, error by horizon) | 👥 Each | ⬜ | ⬜ |
| Merge both sets of results into one comparison table in the main notebook | 📓 Once | ⬜ | ⬜ |
| Outline the "learning journey" structure of the main notebook (section order and what each teaches) | 📓 Once | ⬜ | ⬜ |
| Write up what you learned about the architecture in the main notebook | 👥 Each | ⬜ | ⬜ |
| Draft the "Motivation behind the design choices" section | 📓 Once | ⬜ | ⬜ |
| **Sync meeting:** plan the rough draft and divide notebook sections | 📓 Once | ⬜ | ⬜ |

---

## Week 4: Nov 22 – 28 · Rough Draft

| Task | Type | Ian | Ryan |
|---|:---:|:---:|:---:|
| Write your assigned half of the notebook sections | 👥 Each | ⬜ | ⬜ |
| Refine the introduction, LTSF definition, and use cases | 📓 Once | ⬜ | ⬜ |
| Write the "Where previous models fail" section (vanilla transformer and linear baseline) | 📓 Once | ⬜ | ⬜ |
| Write the transitions that lead into the experiments showing how PatchTST solves those issues | 📓 Once | ⬜ | ⬜ |
| Write the PatchTST architecture walkthrough (patching and channel independence) | 📓 Once | ⬜ | ⬜ |
| Make sure the notebook runs top to bottom (with cached results or small subsets if training is slow) | 📓 Once | ⬜ | ⬜ |
| Review the other person's sections and leave comments | 👥 Each | ⬜ | ⬜ |
| **Rough draft complete** | 📓 Once | ⬜ | ⬜ |

---

## Week 5: Nov 29 – Dec 5 · Final Development & Polish

| Task | Type | Ian | Ryan |
|---|:---:|:---:|:---:|
| Address review comments on your sections | 👥 Each | ⬜ | ⬜ |
| Fit-and-finish pass: consistent formatting, figure labels, equation notation | 📓 Once | ⬜ | ⬜ |
| Add or improve visual aids (e.g., patching diagram, attention maps) | 👥 Each | ⬜ | ⬜ |
| *(Stretch)* Animated vignettes like the module04 transformer notebooks | 📓 Once | ⬜ | ⬜ |
| Add the references section and cite the datasets | 📓 Once | ⬜ | ⬜ |
| Clean up the repo: README, remove dead code, pin requirements | 📓 Once | ⬜ | ⬜ |
| Final read-through of the whole notebook | 👥 Each | ⬜ | ⬜ |
| Restart the kernel, run all cells, and confirm there are no errors | 📓 Once | ⬜ | ⬜ |
| **Submit final notebook** | 📓 Once | ⬜ | ⬜ |

---

## References

1. Nie et al. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* 2023. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
2. Rizvi, Kanwal, Naeem. *Bridging simplicity and sophistication using GLinear.* Digital Signal Processing, 2025.
3. Zeng et al. *Are Transformers Effective for Time Series Forecasting?* AAAI 2023.

**Datasets:** [ETTh1](https://github.com/zhouhaoyi/ETDataset) · [Weather](https://www.bgc-jena.mpg.de/wetter/) · [Traffic](https://pems.dot.ca.gov/) · [Electricity](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) · [CSU Weather](https://climate.colostate.edu/autowx/)
