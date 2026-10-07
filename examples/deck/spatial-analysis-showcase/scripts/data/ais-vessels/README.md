# ais-vessels (+ ais-zones)

Source: NOAA / USCG Nationwide AIS 2024 daily GeoParquet, `https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/ais-2024-06-12.parquet` (CC0 1.0). Zones: NOAA ENCDirect `NavigationChartData` Anchorage_Areas and MarineTransportation layer 1.

Steps: `build.py` downloads the 306 MB day file into the raw cache, clips to the NY/NJ bbox with DuckDB spatial, drops spikes (>50 kn implied, SOG>=45), splits tracks at gaps >20 min, thins stationary samples to one per 5 min, writes binaries. `build_zones.py` builds the zones GeoJSON (general anchorages 33 CFR 110.155, channels dissolved by fairway, six hand-drawn approximate polygons). `validate.py` reads everything back.

Needs `duckdb` in the venv. Run: `python -I build.py && python -I build_zones.py && python -I validate.py`.
