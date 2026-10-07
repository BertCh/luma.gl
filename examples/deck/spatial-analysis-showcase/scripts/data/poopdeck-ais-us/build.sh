#!/usr/bin/env bash
# Builds public/data/poopdeck-ais-us from the poopdeck.gl archive `ais-all-us` (1.29 M AIS position
# reports on 2023-01-09, NOAA / BOEM Marine Cadastre, public domain).
#
# Needs Node 18+ and the prebuilt @poopdeck.gl/core in ~/Documents/GitHub/poopdeck.gl (see
# ../poopdeck/stt-export.mjs). Reads about 25 MB of tiles over HTTP range requests; nothing is
# downloaded into the app tree. Run from anywhere:  bash build.sh [outDir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../../../public/data/poopdeck-ais-us}"

# bbox: lower 48 + Gulf + Great Lakes (no Alaska, Hawaii or Puerto Rico: one continental projection).
# --group-by mmsi        point reports -> one trajectory per vessel, split where reports are > 45 min apart
# --still-thin/--still-radius  a vessel that stays within 100 m keeps one report per hour
# --min-travel/--keep-dwell    drop short harbor shuffles under 2 km unless the track lasts 4 h or more
# --max-speed 60         drop reports implying more than 60 kn (GPS teleports)
# --vertex-values        reported speed over ground, stored as uint8 half-knots (255 = not available)
node "$HERE/../poopdeck/stt-export.mjs" https://tiles.poopdeck.gl/data/ais-all-us/manifest.json \
  --id poopdeck-ais-us --out "$OUT" \
  --bbox -130,24,-65,49.5 \
  --group-by mmsi --max-gap 45 \
  --still-thin 3600 --still-radius 100 \
  --min-travel 2000 --keep-dwell 240 \
  --max-speed 60 \
  --vertex-values 'speed:uint8*2' --vertex-range speed:0:102 \
  --attrs vessel_type,length,width
