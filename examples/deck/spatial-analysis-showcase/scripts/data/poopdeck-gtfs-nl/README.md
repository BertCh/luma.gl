# poopdeck-gtfs-nl

Scheduled public-transport trips in the Randstad, morning peak.

- Source: poopdeck.gl archive `gtfs-nl` (`https://tiles.poopdeck.gl/data/gtfs-nl/manifest.json`), built from the OVapi national GTFS feed `https://gtfs.ovapi.nl/nl/gtfs-nl.zip`.
- Licence: CC0 1.0 (OVapi / NDOV). Attribution is kept anyway.
- Window: bbox `4.2,51.85,5.2,52.45`, Friday 3 July 2026, 05:00-07:00 UTC (07:00-09:00 CEST). Time origin is 05:00:00 UTC, columns are `uint32` seconds, so playback runs 0-7200.
- Columns: `pathOffsets`, `vertices` (lng, lat), `timestamp` (per vertex), `startTime`, `endTime`, `route_type` (tram, bus, metro, rail, ferry), `route_short_name`.
- Size: 8,448 trips, 409,211 vertices, about 4.9 MB.

```sh
node scripts/data/poopdeck-gtfs-nl/build.mjs --work <scratch dir outside the app tree>
```

The build runs `scripts/data/poopdeck/stt-export.mjs` (about 28 MB of range requests), clips every trip to the box, then thins vertices with a time-synchronous Douglas-Peucker pass (20 m): a vertex is dropped only when the vehicle position interpolated in time stays within 20 m, so dwell at stops survives.

The positions are the timetable interpolated along the route shape (97.5 % of trips have exact shape-distance timing, the rest are stop-to-stop lines). They are not GPS positions.
