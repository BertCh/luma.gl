# chicago-roads
Source: OpenStreetMap via BBBike Chicago extract (https://download.bbbike.org/osm/bbbike/Chicago/Chicago.osm.pbf). Licence ODbL 1.0, "© OpenStreetMap contributors".
Steps: `build.py PBF community-areas.geojson OUT [cache.pkl]` — pyrosm driving network, clip to city, directed graph with oneway handling, osmnx simplification, largest SCC, typed arrays.
The optional cache pickle (kept outside the app, in the scratchpad) is read by the crash-snapping and GPS-trace builders.
