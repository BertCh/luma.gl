# chicago-places / chicago-facilities
Overture Maps places release 2026-09-23.1 queried with DuckDB (overture_query.py, bbox filter) into raw/chicago-places/places.parquet; build.py filters open, confidence >= 0.5, inside city, maps to 14 categories. Licence: CDLA-Permissive-2.0 / Apache-2.0 / CC0 (docs.overturemaps.org/attribution/places).
Facilities: Chicago portal libraries x8fc-8rcq, CPS school locations SY2425 hexd-c4gn, fire stations 28km-gtjn (CSV via Socrata); hospitals from Overture (city file is 2011), deduped within 300 m.
