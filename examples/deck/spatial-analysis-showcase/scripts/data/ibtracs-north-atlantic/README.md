# ibtracs-north-atlantic

Source: NOAA NCEI IBTrACS v04r01, North Atlantic list CSV (public domain; cite Knapp et al. 2010, BAMS 91, 363-376, doi:10.1175/2009BAMS2755.1; dataset doi:10.25921/82ty-9e16).
No key needed:

```sh
mkdir -p downloads && curl -o downloads/ibtracs.NA.list.v04r01.csv \
  https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv
python3 -I scripts/data/ibtracs-north-atlantic/build.py downloads/ibtracs.NA.list.v04r01.csv public/data/ibtracs-north-atlantic
```

Window: seasons 1980-2025 (satellite era; the 2026 season is in progress), 6-hourly fixes of the main track,
storms with at least 4 fixes, fixes west of 105 W cut. 739 storms, about 20.8k fixes, 0.45 MB.
Per vertex: wind (kt, USA_WIND else WMO_WIND), pressure (hPa), Saffir-Simpson category from wind, nature, distance to land.
Per storm: season, SID, name, peak wind and category.

The poopdeck.gl `hurricanes` archive (2020-2023) is a subset of the same IBTrACS source, so it is not exported separately.
