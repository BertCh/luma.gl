# poopdeck-ecco-currents

Source: NASA/JPL ECCO V4r4 ocean state estimate (PO.DAAC), particle archive `ecco-currents` of poopdeck.gl
(`https://tiles.poopdeck.gl/data/ecco-currents/manifest.json`), 2016-12-15 to 2017-12-11. NASA Earth science
data, open.

Outputs:

- the particle pieces: a 30,000-piece seeded sample (each piece is a 2-vertex, 3.5-day displacement with `speed`);
- `currents.bin`: an annual-mean surface current field derived from ALL 1,443,603 pieces. Each displacement is a
  velocity sample in degrees per day (u: longitude, v: latitude), splatted into 0.5 degree cells with a Gaussian
  weight (sigma 0.5 degree, 5x5 cells). Cells with kernel weight below 2.5 are NaN (land, ice, shelf). Float32,
  depth 3, `[plane][row][col]`, row 0 at the north edge: u, v, weight. 720 x 320 cells over -180..180, -80..80.

Re-run:

    node scripts/data/poopdeck-ecco-currents/build.mjs \
      --raw <scratch>/downloads/poopdeck-ecco-currents-raw \
      --out public/data/poopdeck-ecco-currents

Caveats: the field is a time mean (no seasons, no eddies), ECCO's effective resolution is about 1 degree so western
boundary currents are weak (the peak mean speed here is 0.6 m/s; the Gulf Stream core is about 0.3 m/s), and the
archive is the model's, not observations. Peak and cell counts are recorded in `manifest.properties.field`.
