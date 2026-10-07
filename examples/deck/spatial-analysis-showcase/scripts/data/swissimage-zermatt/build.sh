#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$SCRIPT_DIR/../../../public/data/swissimage-zermatt"
WORK="${TMPDIR:-/tmp}/swissimage-zermatt"
mkdir -p "$OUT" "$WORK"

for year in 2018 2024; do
  mkdir -p "$WORK/$year"
  index=0
  for y in 93237 93238 93239 93240; do
    for x in 136711 136712 136713 136714; do
      name="$(printf '%02d.jpeg' "$index")"
      curl --fail --location --silent --show-error \
        "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage-product/default/$year/3857/18/$x/$y.jpeg" \
        --output "$WORK/$year/$name"
      index=$((index + 1))
    done
  done
  ffmpeg -loglevel error -y -pattern_type glob -i "$WORK/$year/*.jpeg" \
    -vf 'tile=4x4,scale=512:512:flags=lanczos' -frames:v 1 "$OUT/$year.png"
done

cp "$SCRIPT_DIR/manifest.json" "$OUT/manifest.json"
