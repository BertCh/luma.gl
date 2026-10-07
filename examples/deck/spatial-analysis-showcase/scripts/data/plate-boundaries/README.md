# plate-boundaries

Source: PB2002, "An updated digital model of plate boundaries", Peter Bird, Geochemistry Geophysics Geosystems
4(3), 1027, 2003 (doi:10.1029/2001GC000252), as GeoJSON in https://github.com/fraxen/tectonicplates
(Hugo Ahlenius, Nordpil). Licence: Open Data Commons Attribution License 1.0 (ODC-BY).

5,824 boundary steps (about 100 km each), each with its step class (subduction, spreading ridge, transform...),
the convergence rate and the right-lateral rate. About 0.2 MB.

```sh
mkdir -p <fresh dir>/plate-boundaries
curl -o <fresh dir>/plate-boundaries/steps.json \
  https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_steps.json
node scripts/data/plate-boundaries/build.mjs <fresh dir>/plate-boundaries/steps.json public/data/plate-boundaries
```
