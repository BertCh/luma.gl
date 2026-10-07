# celestrak-ground-tracks

SGP4 ground tracks of 911 satellites over 3 hours (30 s steps, 361 samples each), from current
CelesTrak element sets. poopdeck.gl's `satellites` archive (24 h, 12.7k satellites, 1.87 GB) is too
large for the showcase, so this is rebuilt from the same upstream.

- Source: CelesTrak GP element sets (TLE format), `https://celestrak.org/NORAD/elements/gp.php?GROUP=<group>&FORMAT=tle`.
  Free to use; attribution to CelesTrak. No key.
- Groups used: `stations`, `starlink` (800 random satellites, fixed seed), `gps-ops`, `weather`
  (geostationary ones dropped), and the Earth observation subset of `resource`, `weather` and `science`
  (Landsat, Sentinel, Terra, Aqua, Suomi NPP, NOAA-20/21, Metop) with nominal swath widths.
- Propagation: `satellite.js` 7.1.0 (SGP4). Install it in a scratch directory, not in the app.
- Window: 2026-10-07T00:00:00Z to 03:00:00Z. TLE epochs are 0.3 to 20 days old at the start (median
  0.5 d); SGP4 error grows with TLE age, from about 1 km at epoch to several km per day for LEO.
- Tracks are cut at the antimeridian with interpolated crossing vertices at exactly +-180, because the
  trajectory contributors are planar. A satellite therefore has several tracks (`satellite` column).

```sh
S=<scratch>/downloads/celestrak-ground-tracks
mkdir -p $S && cd $S
for g in stations starlink gps-ops weather resource science; do
  curl -s -o $g.tle "https://celestrak.org/NORAD/elements/gp.php?GROUP=$g&FORMAT=tle"
done
mkdir -p <scratch>/sgp4 && cd <scratch>/sgp4 && npm init -y && npm i satellite.js
cd examples/deck/spatial-analysis-showcase
node scripts/data/celestrak-ground-tracks/build.mjs --tle-dir $S --satellite-js <scratch>/sgp4
```

Output (5.7 MB with the manifest): `pathOffsets`, `vertices` (lng, lat), `timestamp` (uint32 s since the origin),
`altitude` (float32 m), and per track `satellite`, `group`, `orbitType`, `inclination`, `meanAltitude`,
`swathKm`. Per-satellite names, NORAD ids, periods and TLE ages are in `properties.satellites`.
Re-running later gives different TLEs; pass `--start` to match their epoch.
