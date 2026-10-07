# poopdeck-wildfires

Source: poopdeck.gl `wildfires` archive (`https://tiles.poopdeck.gl/data/wildfires/manifest.json`), NIFC interagency fire perimeters,
public domain. Window 2020-01-01 to 2024-01-01, read at the archive's lowest zoom (z3).

```sh
node scripts/data/poopdeck-wildfires/build.mjs --raw <empty dir for the raw export>
```

The script runs `scripts/data/poopdeck/stt-export.mjs` (460 STT features) and regroups them:

- One STT feature is one polygon part. 460 parts belong to 118 distinct NIFC object ids (Dixie alone has 179 parts: the burn plus
  many unburned islands). The build groups parts by object id into one feature per fire.
- Fires with mean longitude east of -100 (prairie and Gulf fires) are dropped, leaving 90 western fires, 421 parts, 499 rings,
  340,813 vertices (2.7 MB).
- Rings are re-oriented (shells counter-clockwise, holes clockwise) and the repeated closing vertex is dropped, so
  `GPUGeometryMeasures` with `holeRule: 'winding'` returns the area with holes subtracted.
- `severity` of the archive is an acreage class (it is not burn severity): kept as `sizeClass`.
- `perimeterTime` is the perimeter date of the NIFC record, uint32 seconds since 2020-01-01Z (use days in float32 on the GPU).

The archive is not the complete NIFC record. For example the August Complex and Caldor fires are absent, and 2020-2023 totals are
about 2.8 million acres for the whole archive.

Daily progression: NIFC's `WFIGS_Daily_Perimeters_Public` layer is key-free but holds only a handful of the large fires of these
years with more than one perimeter (Kelly 72, Pass 16, Newell Road 10; Dixie has 4), so this dataset is final perimeters only.
