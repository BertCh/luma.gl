# chicago-nature
Source: iNaturalist API, observations inside place 49906 (City of Chicago) for calendar 2023 with `licensed=true` and `preferred_place_id=49906` (for native/introduced status). Each observation keeps its own licence (CC0, CC BY 4.0 or CC BY-NC 4.0); attribute "iNaturalist contributors".
Download: `python -I fetch.py <RAW>/chicago-nature/inat2023.json`. It pages with `id_above` at about 1 request per second.
Run chicago-tracts/build.py first (writes tracts_full.gpkg), then `geo-venv/bin/python -I build.py`.
