# Dataset Notes

Load any dataset with the shared loader, which downloads it here on first use:

```python
from ltsf_data import load      # src/ltsf_data.py
ts = load("traffic")
```

The benchmark files come from the [Hugging Face mirror](https://huggingface.co/datasets/AutonLab/Timeseries-PILE/tree/main/forecasting/autoformer) of the CSVs released with Autoformer (Wu et al. 2021). Both of our papers used these files. Downloads are gitignored; only the CSU weather files are committed. Sizes are listed in the docstring of `src/ltsf_data.py` (268 MB on disk for everything).

| Dataset | Steps × channels | Interval | What it measures |
|---|---|---|---|
| ETTh1, ETTh2 | 17,420 × 7 | 1 hour | Electricity transformer load features + oil temperature (OT) |
| ETTm1, ETTm2 | 69,680 × 7 | 15 min | Same stations as ETTh1/h2, finer resolution |
| Weather | 52,696 × 21 | 10 min | Meteorological measurements, Jena (Germany), 2020 |
| Traffic | 17,544 × 862 | 1 hour | Freeway occupancy rate, San Francisco Bay Area |
| Electricity | 26,304 × 321 | 1 hour | kWh used by each of 321 clients |
| Exchange rate | 7,588 × 8 | 1 day | Exchange rates of 8 countries |
| ILI | 966 × 7 | 1 week | US influenza-like-illness counts |
| CSU weather | 52,608 × 10 | 1 hour | Fort Collins weather station, 2020–2025 |

The loader truncates ETT to the first 20 months (14,400 hourly / 57,600 15-minute rows), as the papers do.

## Timestamps in Traffic and Electricity are placeholders

The Hugging Face files hold the same values as the older [laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data) files we used at first. We checked: Traffic matches to float precision and Electricity matches exactly. So results from either source are comparable.

But the `date` column in these two files does not give the real recording dates. Electricity is labeled 2016-07-01 to 2019-07-02, but the data is from 2012–2014. Traffic is labeled 2016-07-01 to 2018-07-02, but the original source says 2015–2016. **The hourly spacing is right, so seasonal periods in steps (24 = a day, 168 = a week) are fine. Weekday, month and year labels are not.** Use CSU, Weather or ETT for any plot that depends on the calendar.

## Traffic

Each line records the hourly _occupancy rate_ for 862 sensors.

- A value of 0.00 means the sensor was completely clear of traffic for that entire hour window (empty road).
- A value of 0.15 means a vehicle was physically occupying the space directly above the sensor for 15% of that hour interval.
- A value approaching 1.00 indicates a severe traffic jam or bumper-to-bumper standstill where vehicles are parked over the sensor

## Electricity

The dataset is a processed form of the raw dataset from https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014.
Only years 2012-2014 are kept to avoid 0 values in 2011, and consumption values are kWh. So, each row is the use for one hour and each column is one client (n = 321).
It is much smaller than the original dataset.

I'm not sure what they did to handle the special time change hours as documented in the original dataset.

## CSU weather

Hourly readings from CSU's Fort Collins station, fetched with `src/ian/data/csu-weather.js`.

- `csu_weather_2020_2025_dates.csv` has the timestamps and two header rows (names, units). `csu_weather_2020_2025.csv` is the same data without them.
- Channels: Temp (°F), RH (%), DewPt (°F), Wind (mph), WindDir (deg), Gust (mph), GustDir (deg), Press (inHg), Solar (W/m²), Prec (in).
- The station missed ~270 whole hours: 10 outages of up to 2 days, and both Feb 29ths. Individual readings it missed are written as `***`. The loader puts the data on a complete hourly grid and fills both kinds of gap by linear interpolation in time, so 24 steps is always one day.
- **Gust has bad readings** (maximum 1,052 mph). Leave it out, or clip it, before training.
