# bixi-flows
Source: BIXI Montreal open data, trip history 2024 (`https://bixi.com/en/open-data/`, zip `DonneesOuvertes2024_010203040506070809101112.zip`, 500 MB, one 2.6 GB CSV with station name, borough, lat/lng and start/end times).
Licence: bixi.com publishes no licence text on the open-data page. The Montreal open data portal listing of the same data (donnees.iriu.ca, "Travel history") states Creative Commons Attribution with attribution to BIXI Montreal. Attribute BIXI Montréal.
Why raw and not the poopdeck archive: `bixi-flowmap(-dense)` are hub-clustered per zoom and drop pairs under 15 rides, so station net flow would be wrong.
Steps: `curl -O <zip>`, then `python3 -I build.py <zip> ../../../public/data/bixi-flows` (about 3 minutes).
Rides under 60 s and rides without station or borough are dropped (1.93 M of 2.01 M kept). Hours are America/Montreal local time. Rare pairs (under 10 rides) are folded into residual rows to and from a sentinel zone (index = stationCount), so every station total is exact.
