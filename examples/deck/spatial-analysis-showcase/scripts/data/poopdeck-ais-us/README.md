# poopdeck-ais-us

One UTC day (9 January 2023) of US coastal AIS, as vessel trajectories.

Source: the poopdeck.gl archive `https://tiles.poopdeck.gl/data/ais-all-us/manifest.json` (1,289,859 AIS
position reports), itself built from the NOAA / BOEM Marine Cadastre AIS data
(`https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_09.zip`). Public domain (US Government
work); attribute NOAA Office for Coastal Management, BOEM and the U.S. Coast Guard.

Build: `bash build.sh` (about 25 MB of tile range requests, a few seconds; needs Node 18+ and the prebuilt
`@poopdeck.gl/core` of the sibling poopdeck.gl checkout). It runs `../poopdeck/stt-export.mjs`:

- window: lower 48 + Gulf + Great Lakes (`-130,24,-65,49.5`), whole day, tile zoom 0;
- `--group-by mmsi`: point reports become one trajectory per vessel, sorted by time and split where two
  consecutive reports are more than 45 minutes apart (15,132 vessels in the window become 28,254 raw tracks);
- `--still-thin 3600 --still-radius 100`: a vessel that stays within 100 m keeps one report per hour;
- `--min-travel 2000 --keep-dwell 240`: tracks that spread under 2 km are dropped unless they last 4 h or more
  (the parked vessels stay, harbor shuffles go): 15,555 tracks of 13,436 vessels, 432,834 fixes;
- `--max-speed 60`: reports implying more than 60 knots from the previous one are dropped (GPS teleports);
- `--vertex-values speed:uint8*2`: reported speed over ground per fix in half-knots (255 = not available).

Columns: `pathOffsets`, `vertices` (lng, lat float32), `timestamp` (uint32 seconds since 2023-01-09T00:00Z, per
fix), `speed`, per-track `mmsi`, `vesselIndex`, `vessel_type` (category codes), `length`, `width`.

Known limits (also stated in the scene): the archive holds about one report per vessel per 12 minutes, so there
is no 1-2 minute resampling to do; reports come from terrestrial receivers, so the open ocean and busy-harbor
Class B traffic are under-represented; Alaska, Hawaii and Puerto Rico are outside the window; tracks cross land
where a straight line joins two fixes.
