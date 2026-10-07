#!/usr/bin/env bash
# Builds public/data/poopdeck-nyc-taxi-paths: OSRM-routed yellow-taxi paths, Fri 2 Jan 2015 08:00-08:30.
# Run from examples/deck/spatial-analysis-showcase. Needs ~/Documents/GitHub/poopdeck.gl built (see poopdeck/stt-export.mjs).
set -euo pipefail
node scripts/data/poopdeck/stt-export.mjs https://tiles.poopdeck.gl/data/nyc-taxi-paths/manifest.json \
  --id poopdeck-nyc-taxi-paths --out public/data/poopdeck-nyc-taxi-paths \
  --time 2015-01-02T08:00:00Z,2015-01-02T08:30:00Z --bbox -74.03,40.68,-73.90,40.82 \
  --stitch trip_id --max-features 9000 --max-tiles 200 --attrs trip_distance,fare_amount

# The upstream archive leaves source.attribution empty: write the credit the data need.
node -e "
const fs = require('node:fs');
const path = 'public/data/poopdeck-nyc-taxi-paths/manifest.json';
const manifest = JSON.parse(fs.readFileSync(path, 'utf8'));
manifest.properties.source.description = 'Yellow-taxi trips routed by OSRM between the recorded pickup and drop-off, with a timestamp on every vertex (routes, not GPS traces).';
manifest.properties.source.attribution = 'NYC Taxi & Limousine Commission trip records via NYC Open Data; poopdeck.gl nyc-taxi-paths archive; routes by OSRM on © OpenStreetMap contributors (ODbL)';
fs.writeFileSync(path, JSON.stringify(manifest, null, 2));
"
