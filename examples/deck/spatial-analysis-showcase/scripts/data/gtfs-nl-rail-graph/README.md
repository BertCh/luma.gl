# gtfs-nl-rail-graph

Directed station-to-station rail graph of the Netherlands with scheduled travel times.

- Source: OVapi national GTFS feed `https://gtfs.ovapi.nl/nl/gtfs-nl.zip` (about 245 MB, CC0), feed version valid 2026-10-06 to 2026-12-12.
- Service day: Friday 2026-10-09 (the feed has `calendar_dates` only). The trips dataset `poopdeck-gtfs-nl` is a July day; the rail timetable is the same pattern, the graph is not tied to the exact trips.
- Nodes: parent stations (`stoparea:*`) inside lon 3.0-7.6, lat 50.6-53.7. Edges: every pair of consecutive stops of a rail trip, one edge per (A, B, train class). Classes: intercity and international, express (Sneltrein), stopping (Sprinter, Stoptrein, other). Replacement buses carried as `route_type 2` ("Drempelvrije bus") are dropped.
- `edgeTravelTime` is the median of arrival(B) - departure(A) over the day, at least 60 s. There is no transfer time: connections are perfect.
- `edgeTripsPerHour` has 24 counts per edge, by local departure hour at A.

```sh
mkdir -p <fresh dir> && cd <fresh dir>
curl -LO https://gtfs.ovapi.nl/nl/gtfs-nl.zip
unzip gtfs-nl.zip agency.txt routes.txt stops.txt trips.txt calendar_dates.txt stop_times.txt
cd <showcase> && node scripts/data/gtfs-nl-rail-graph/build.mjs --gtfs <fresh dir>
```
