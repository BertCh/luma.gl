#!/usr/bin/env bash
# Builds public/data/poopdeck-nyc-taxi-paths: OSRM-routed yellow-taxi paths, Fri 2 Jan 2015 08:00-08:30.
# Run from examples/deck/spatial-analysis-showcase. Needs ~/Documents/GitHub/poopdeck.gl built (see poopdeck/stt-export.mjs).
set -euo pipefail
node scripts/data/poopdeck/stt-export.mjs https://tiles.poopdeck.gl/data/nyc-taxi-paths/manifest.json \
  --id poopdeck-nyc-taxi-paths --out public/data/poopdeck-nyc-taxi-paths \
  --time 2015-01-02T08:00:00Z,2015-01-02T08:30:00Z --bbox -74.03,40.68,-73.90,40.82 \
  --stitch trip_id --max-features 9000 --max-tiles 200 --attrs trip_distance,fare_amount
