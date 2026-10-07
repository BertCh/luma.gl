# openflights

Source: https://github.com/jpatokal/openflights `data/airports.dat`, `data/routes.dat` (ODbL 1.0; attribution OpenFlights.org). Data frozen around 2014.
`build.py` keeps IATA airports with routes and collapses routes to undirected pairs (counts, distinct airlines, per-direction counts, great-circle km).
