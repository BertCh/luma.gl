# poopdeck-bixi-rides
Source: poopdeck.gl `bixi-points` archive (`https://tiles.poopdeck.gl/data/bixi-points/manifest.json`), every BIXI ride of 15 Aug 2024, routed with OSRM (bicycle profile) on OpenStreetMap.
Licence: ODbL 1.0 for the routes ("© OpenStreetMap contributors"); trip data CC BY, BIXI Montréal.
Window: 11:30-14:00 UTC = 07:30-10:00 local, rides clipped to the window, 8,365 rides, 2 m Douglas-Peucker (1.07 M to 155 k vertices).
Steps: `sh build.sh <scratch dir>` (needs the sibling poopdeck.gl checkout used by stt-export.mjs).
Routes are derived, not GPS. Archive times are real UTC (profile peaks at 12-13 UTC).
