# Dataset Notes

https://github.com/laiguokun/multivariate-time-series-data

## Traffic

Each line records the 10 minute _occupancy rate_ for 862 sensors.

- A value of 0.00 means the sensor was completely clear of traffic for that entire 10-minute window (empty road).
- A value of 0.15 means a vehicle was physically occupying the space directly above the sensor for 15% of that 10-minute interval.
- A value approaching 1.00 indicates a severe traffic jam or bumper-to-bumper standstill where vehicles are parked over the sensor

## Electricity

The dataset from the above GitHub link is a processed form of the raw dataset from https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014.
Only years 2012-2014 are kept to avoid 0 values in 2011, and consumption values are kWh. So, each row is the use for one hour and each column is one client (n = 321).
It is much smaller than the original dataset.

I'm not sure what they did to handle the special time change hours as documented in the original dataset.
