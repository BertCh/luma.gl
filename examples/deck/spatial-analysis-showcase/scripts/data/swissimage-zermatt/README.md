# swissimage-zermatt

Natural-colour SWISSIMAGE orthophotos of a 424 m square in central Zermatt. The fixed 2018 and
2024 time-travel mosaics use Web Mercator zoom 18 tiles `x=136711..136714`, `y=93237..93240`, then
downsample the 1024 × 1024 tile mosaic to 512 × 512 for browser compute.

- Source: swisstopo SWISSIMAGE Journey through time WMTS
- Licence: swisstopo Open Government Data (OGD), free use with attribution
- Attribution: © swisstopo
- Rebuild: `bash build.sh` (requires `curl` and `ffmpeg`)
- Output: `public/data/swissimage-zermatt/{2018,2024}.png` and `manifest.json`

The years are fixed in the WMTS URLs so a rebuild does not silently replace the target image.
