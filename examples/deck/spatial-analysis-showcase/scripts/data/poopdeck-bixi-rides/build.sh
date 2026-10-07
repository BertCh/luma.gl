#!/bin/sh
# Rebuild public/data/poopdeck-bixi-rides from the poopdeck.gl bixi-points archive (needs the sibling poopdeck.gl checkout).
# usage: sh build.sh <scratch dir>
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRATCH="${1:?scratch dir for the raw export}"
node "$HERE/../poopdeck/stt-export.mjs" https://tiles.poopdeck.gl/data/bixi-points/manifest.json \
  --id bixi-rides-raw --out "$SCRATCH/rides-raw" \
  --time 2024-08-15T11:30:00Z,2024-08-15T14:00:00Z --stitch trip_id --clip-time \
  --max-features 30000 --zoom 10 --max-tiles 400
node "$HERE/simplify.mjs" "$SCRATCH/rides-raw" "$HERE/../../../public/data/poopdeck-bixi-rides" 2
