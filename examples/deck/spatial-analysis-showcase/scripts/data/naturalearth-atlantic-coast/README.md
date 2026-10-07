# naturalearth-atlantic-coast

Source: Natural Earth 1:50m coastline (public domain), `ne_50m_coastline.geojson` from
https://github.com/nvkelso/natural-earth-vector. Clipped to 105 W to 15 E, 5 N to 62 N.

```sh
curl -o downloads/ne_50m_coastline.geojson https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson
python3 -I scripts/data/naturalearth-atlantic-coast/build.py downloads/ne_50m_coastline.geojson public/data/naturalearth-atlantic-coast
```
