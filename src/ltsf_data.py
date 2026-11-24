"""
Download and load the standard long-term time series forecasting (LTSF) benchmarks.

Both papers we build on (Zeng et al. 2023, "Are Transformers Effective for Time Series
Forecasting?"; Nie et al. 2023, PatchTST) evaluate on the benchmark CSVs released with
Autoformer (Wu et al. 2021). Those files are mirrored on Hugging Face in the
AutonLab/Timeseries-PILE dataset (MIT license), which is where we download them from.

Usage
-----
    import sys; sys.path.append("../")        # from a notebook in src/<name>/
    from ltsf_data import load, list_datasets

    ts = load("ETTh1")                        # downloads to data/ on first use
    ts.values.shape                           # (n_steps, M) float32 = (time steps, channels)
    ts.timestamps[:3], ts.channels            # when each row was recorded, what each column is
    ts.df                                     # the same thing as a pandas DataFrame

    # Feeds straight into split_time_series_data from Ian's notebooks:
    split_time_series_data(ts.values, lookback_size=336, prediction_size=96,
                           split_ratios=ts.split_ratios)

Splits
------
`ts.split_ratios` is the train/val/test split the papers use. The ETT datasets use
12/4/4 months (30-day months) and drop everything after month 20, so `load` truncates
them to 20 months and the ratio (0.6, 0.2, 0.2) reproduces the papers' row borders
exactly. All other datasets use 70/10/20.

Data size
---------
TOTAL: 268 MB on disk in data/ for all 10 datasets, and 105 MB in memory if every
dataset is loaded at once (float32 arrays). Traffic and Electricity are 232 MB of the
disk total. Each file is downloaded only the first time `load` asks for it, so loading
just the small datasets costs ~36 MB. The downloads are gitignored (data/*.csv); only
the CSU weather files are committed.

    dataset        shape (steps x M)   on disk    in memory
    ETTh1            14,400 x   7        2.6 MB      0.4 MB   (file holds 17,420 rows)
    ETTh2            14,400 x   7        2.4 MB      0.4 MB   (file holds 17,420 rows)
    ETTm1            57,600 x   7       10.4 MB      1.6 MB   (file holds 69,680 rows)
    ETTm2            57,600 x   7        9.7 MB      1.6 MB   (file holds 69,680 rows)
    weather          52,696 x  21        7.2 MB      4.4 MB
    traffic          17,544 x 862      136.5 MB     60.5 MB
    electricity      26,304 x 321       95.6 MB     33.8 MB
    exchange_rate     7,588 x   8        0.6 MB      0.2 MB
    illness             966 x   7        0.1 MB      0.0 MB
    csu_weather      52,608 x  10        3.4 MB      2.1 MB   (in repo, not downloaded)
    TOTAL                              268.4 MB    105.1 MB

Only NumPy and pandas are needed here, so this module does not import torch.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Repository root is the parent of src/.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

_HF_BASE = ("https://huggingface.co/datasets/AutonLab/Timeseries-PILE/"
            "resolve/main/forecasting/autoformer/")


@dataclass(frozen=True)
class _Spec:
    filename: str
    url: str | None          # None means the file ships with the repo
    size: int | None         # expected download size in bytes, for a sanity check
    freq: str                # sampling interval, as a pandas offset alias
    periods: dict            # natural seasonal periods, in time steps
    split_ratios: tuple
    keep_rows: int | None = None   # truncate to this many rows (ETT, to match the papers)
    description: str = ""


# Number of time steps in an ETT month under the papers' 30-day-month convention.
_ETT_MONTH_H = 30 * 24
_ETT_MONTH_M = 30 * 24 * 4

_SPECS: dict[str, _Spec] = {
    "ETTh1": _Spec("ETTh1.csv", _HF_BASE + "ETTh1.csv", 2589657, "h",
                   {"day": 24, "week": 168}, (0.6, 0.2, 0.2), 20 * _ETT_MONTH_H,
                   "Electricity transformer: 6 load features + oil temperature (OT), hourly."),
    "ETTh2": _Spec("ETTh2.csv", _HF_BASE + "ETTh2.csv", 2417960, "h",
                   {"day": 24, "week": 168}, (0.6, 0.2, 0.2), 20 * _ETT_MONTH_H,
                   "Second transformer station, same 7 features, hourly."),
    "ETTm1": _Spec("ETTm1.csv", _HF_BASE + "ETTm1.csv", 10360719, "15min",
                   {"day": 96, "week": 672}, (0.6, 0.2, 0.2), 20 * _ETT_MONTH_M,
                   "ETTh1's station at 15-minute resolution."),
    "ETTm2": _Spec("ETTm2.csv", _HF_BASE + "ETTm2.csv", 9677236, "15min",
                   {"day": 96, "week": 672}, (0.6, 0.2, 0.2), 20 * _ETT_MONTH_M,
                   "ETTh2's station at 15-minute resolution."),
    "weather": _Spec("weather.csv", _HF_BASE + "weather.csv", 7182728, "10min",
                     {"day": 144, "week": 1008}, (0.7, 0.1, 0.2), None,
                     "21 meteorological measurements, Jena (Germany), 2020, every 10 minutes."),
    "traffic": _Spec("traffic.csv", _HF_BASE + "traffic.csv", 136478119, "h",
                     {"day": 24, "week": 168}, (0.7, 0.1, 0.2), None,
                     "Road occupancy rate (0-1) from 862 San Francisco freeway sensors, hourly. "
                     "Dates in the file are placeholders: spacing is right, calendar labels are not."),
    "electricity": _Spec("electricity.csv", _HF_BASE + "electricity.csv", 95581762, "h",
                         {"day": 24, "week": 168}, (0.7, 0.1, 0.2), None,
                         "Hourly kWh consumption of 321 clients (2012-2014). "
                         "Dates in the file are placeholders: spacing is right, calendar labels are not."),
    "exchange_rate": _Spec("exchange_rate.csv", _HF_BASE + "exchange_rate.csv", 630212, "D",
                           {}, (0.7, 0.1, 0.2), None,
                           "Daily exchange rates of 8 countries. No real seasonality."),
    "illness": _Spec("national_illness.csv", _HF_BASE + "national_illness.csv", 66653, "W",
                     {"year": 52}, (0.7, 0.1, 0.2), None,
                     "Weekly US influenza-like-illness counts (ILI), 7 features."),
    "csu_weather": _Spec("csu_weather_2020_2025_dates.csv", None, None, "h",
                         {"day": 24, "year": 8760}, (0.6, 0.2, 0.2), None,
                         "CSU Fort Collins weather station, hourly, 2020-2025 (in repo)."),
}


@dataclass
class TimeSeries:
    """A multivariate time series: M channels observed at the same T time steps."""
    name: str
    values: np.ndarray            # (n_steps, M) float32
    timestamps: pd.DatetimeIndex  # (n_steps,)
    channels: list[str]           # (M,)
    freq: str
    periods: dict = field(default_factory=dict)
    split_ratios: tuple = (0.7, 0.1, 0.2)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, index=self.timestamps, columns=self.channels)

    def channel(self, name_or_index) -> np.ndarray:
        """One univariate series, shape (n_steps,)."""
        i = name_or_index if isinstance(name_or_index, int) else self.channels.index(name_or_index)
        return self.values[:, i]

    def __repr__(self):
        n_steps, M = self.values.shape
        return (f"TimeSeries({self.name!r}, {n_steps} steps x M={M} channels, every {self.freq}, "
                f"{self.timestamps[0]} -> {self.timestamps[-1]})")


def list_datasets() -> pd.DataFrame:
    """Table of available datasets and whether each is already on disk."""
    rows = []
    for name, s in _SPECS.items():
        rows.append({"name": name, "freq": s.freq, "split": s.split_ratios,
                     "download_MB": round(s.size / 1e6, 1) if s.size else 0,
                     "on_disk": (DATA_DIR / s.filename).exists(),
                     "description": s.description})
    return pd.DataFrame(rows).set_index("name")


def download(name: str, force: bool = False) -> Path:
    """Fetch a dataset's CSV into data/ (skipped if already present). Returns its path."""
    spec = _spec(name)
    path = DATA_DIR / spec.filename
    if path.exists() and not force:
        return path
    if spec.url is None:
        raise FileNotFoundError(f"{path} should ship with the repo but is missing.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {name} ({spec.size / 1e6:.0f} MB) to {path} ...", file=sys.stderr)
    with urllib.request.urlopen(spec.url) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)

    got = tmp.stat().st_size
    if spec.size is not None and got != spec.size:
        tmp.unlink()
        raise IOError(f"{name}: expected {spec.size} bytes, got {got}. Try again.")
    tmp.rename(path)
    return path


def load(name: str, download_if_missing: bool = True) -> TimeSeries:
    """Load a dataset as a TimeSeries, downloading it first if needed."""
    spec = _spec(name)
    path = DATA_DIR / spec.filename
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(path)
        download(name)

    if name == "csu_weather":
        df = _read_csu_weather(path)
    else:
        # weather.csv has a few Latin-1 characters in its units (e.g. W/m²).
        df = pd.read_csv(path, encoding="latin-1", parse_dates=["date"], index_col="date")

    if spec.keep_rows is not None:
        df = df.iloc[:spec.keep_rows]

    return TimeSeries(
        name=name,
        values=df.to_numpy(dtype=np.float32),
        timestamps=pd.DatetimeIndex(df.index),
        channels=[str(c) for c in df.columns],
        freq=spec.freq,
        periods=dict(spec.periods),
        split_ratios=spec.split_ratios,
    )


def _spec(name: str) -> _Spec:
    if name not in _SPECS:
        raise KeyError(f"Unknown dataset {name!r}. Choose from: {', '.join(_SPECS)}")
    return _SPECS[name]


def _read_csu_weather(path: Path) -> pd.DataFrame:
    """
    The CSU file has two header rows (names, then units), a 'Dir' column name used twice
    (wind and gust direction), and '***' for readings the station missed.

    The station also skipped ~270 whole hours (10 outages of up to 2 days, plus both
    Feb 29ths). We put the data on a complete hourly grid so that "24 steps back" is
    always "one day back", which the seasonal structure depends on. Missing readings
    are filled by linear interpolation in time, which is safer than 0 for channels like
    pressure (~25 inHg) where a 0 would be a huge fake spike.
    """
    df = pd.read_csv(path, skiprows=[1], na_values=["***"])
    df.columns = ["Date", "Time", "Temp", "RH", "DewPt", "Wind", "WindDir",
                  "Gust", "GustDir", "Press", "Solar", "Prec"]
    df.index = pd.to_datetime(df.pop("Date") + " " + df.pop("Time"))
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="h"))
    df.index.name = "date"
    return df.interpolate(method="time", limit_direction="both")
