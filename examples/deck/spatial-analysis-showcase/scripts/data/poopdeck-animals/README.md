# poopdeck-animals

Open-licence bird tracks from the poopdeck.gl `animals` archive (GBIF/Movebank tracking datasets, multi-year
tracks folded onto calendar 2024), for the movement chapter's `migration-*` scenes.

- Archive: https://tiles.poopdeck.gl/data/animals/manifest.json (366 daily tiles at zoom 0).
- Licence filter: the archive mixes CC0, CC BY and CC BY-NC datasets (the `dataset` attribute names the
  source). Each dataset's licence was looked up with `https://api.gbif.org/v1/dataset/<key>`; only these
  five are kept, all CC0 1.0, all published by the Research Institute for Nature and Forest (INBO):

  | short name | species | GBIF key | DOI |
  |---|---|---|---|
  | MH_WATERLAND | Circus aeruginosus | 66e0553e-75f6-49de-b614-22efd9fbf6e9 | 10.5281/zenodo.10053583 |
  | H_GRONINGEN | Circus aeruginosus | 5124534e-2d9c-46b7-a857-e0012821526b | 10.5281/zenodo.10053658 |
  | MH_ANTWERPEN | Circus aeruginosus | e347ea47-db3f-4c47-8771-ea562330382c | 10.5281/zenodo.10054153 |
  | BOP_RODENT | Circus pygargus (Montagu's harrier) | e2fb42ca-e408-4aa2-a7bd-a9bb4ddcc83a | 10.5281/zenodo.17310324 |
  | SPOONBILL_VLAANDEREN | Platalea leucorodia | 6850e626-46fd-4843-a391-2c06b069a940 | 10.5281/zenodo.23103735 |

  BOP_RODENT holds five raptor species (marsh and Montagu's harrier, buzzard, hen harrier, short-eared owl) but the
  archive labels all its tracks `Circus pygargus`. The build therefore asks the GBIF occurrence API for the species of
  each tag (`facet=organismID` per `scientificName`) and keeps only the marsh and Montagu's harriers.
  The gull datasets (the existing `gull-migration` scene uses LBBG_ZEEBRUGGE) and the curlew dataset were left out.
- Command (needs the sibling `~/Documents/GitHub/poopdeck.gl` checkout built; the raw pieces are cached in `--cache`):

      node scripts/data/poopdeck-animals/build.mjs --out public/data/poopdeck-animals --cache <scratch dir>

- Processing: pieces (the archive cuts tracks at every day) are joined per (species, organism, segment) into one
  animal-year track, sorted by time, thinned to at least 6,600 s between fixes (originally about hourly), and
  fixes implying more than 30 m/s are dropped. Tracks under 60 fixes are dropped. Result: 101 tracks, 42 birds,
  226,492 fixes from 486,298 (55 marsh harrier, 23 Montagu's harrier, 23 spoonbill tracks).
- Columns: `pathOffsets`, `vertices` (lon, lat float32), `timestamp` (uint32 seconds since 2024-01-01 UTC),
  and per track `species`, `individual` (categories are tag ids), `year` (segment number) and `source`.
- Caveat: the archive folds years onto 2024, so the same animal in different years overlaps in time. Float32
  seconds of the year are exact only to 2 s above 2^24 s (194 days); the scenes accept that for 2-hour fixes.
