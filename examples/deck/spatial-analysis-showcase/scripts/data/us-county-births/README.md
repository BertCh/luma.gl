# us-county-births

Births per county 2021-2023 and the women aged 15-44 at risk, aligned row-for-row to `us-counties`
(sorted by FIPS). The general fertility rate (births per 1,000 women aged 15-44 per year) is a count of
events over a population at risk; a few hundred small counties have only a handful of births, so their
raw rates swing widely. Built for `statistics/rate-smoothing` (empirical-Bayes smoothing), replacing the
traffic-death counts of `us-county-mortality` (wholesome-data decision of 2026-10-07).

Source: US Census Bureau, Population Estimates Program, Vintage 2023
(`co-est2023-alldata.csv`: BIRTHS2021-2023 by mother's residence; `cc-est2023-agesex-all.csv`:
AGE1544_FEM on July 1 of 2021, 2022, 2023). https://www.census.gov/programs-surveys/popest.html
Licence: US Government work, public domain. Vintage 2023 already uses the 9 Connecticut planning
regions, matching `us-counties`; Alaska and Hawaii are not in `us-counties`.

Run (geo venv): `python -I build.py` then `python -I validate.py`. Raw downloads (~8.7 MB) are cached
outside the app tree (`US_COUNTY_BIRTHS_RAW`). Output ~130 KB: float32 columns `births`, `womenYears`,
`population`, `rawRate`, `births2021..2023`, `women2021..2023`.

Caveats: PEP births come from NCHS records, the latest year partly estimated; women 15-44 is itself an
estimate, and students inflate it in college towns (low rates there are a denominator effect).
