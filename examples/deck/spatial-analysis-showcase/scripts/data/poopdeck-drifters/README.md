# poopdeck-drifters

Source: NOAA Global Drifter Program (AOML / PMEL), 6-hourly interpolated drifter data, served as the
`drifters` archive of poopdeck.gl (`https://tiles.poopdeck.gl/data/drifters/manifest.json`).
Licence: public domain (NOAA waives copyright).

Window: 2017-01-01 to 2017-12-31 (records are clipped at the window end, so late releases are shorter than 60 days). Every record (stitched by `drifter_id` + `segment`) is clipped to the 60 days
after its first fix inside the window ("release"), thinned to about 12-hourly fixes (the archive is 6-hourly),
and cut where it would cross the antimeridian (none do in this window). 2,811 tracks, 292,112 vertices, 4.5 MB.

`daily.bin` holds the real position at release + 0..30 days, interpolated from the 6-hourly fixes (NaN when the
record has ended or has a gap longer than 36 hours). It is the reference the modelled particles are compared with.
`deployed` is 1 when the first fix is after 5 January (a new deployment or a new record), 0 for drifters already at sea.

Re-run (downloads into an empty directory outside the app tree):

    node scripts/data/poopdeck-drifters/build.mjs \
      --raw <scratch>/downloads/poopdeck-drifters-raw \
      --out public/data/poopdeck-drifters

Caveats: the archive carries no drogue flag, so drifters that lost their drogue (and slip through the water
differently, with more wind slip) are not separated. Time is stored as uint32 seconds since 2017-01-01; scenes
convert to float32 days (exact to a few seconds over a year).
