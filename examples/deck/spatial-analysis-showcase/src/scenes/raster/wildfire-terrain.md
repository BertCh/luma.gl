## Where did these fires burn? {#burn-scars}

A fire perimeter is a polygon; terrain is a grid. To ask what ground a fire burned over we first need to put the two on the same footing. This scene takes the 24 fires of the archive that fall on a 239 m elevation grid of northern and central California (Dixie, the SCU, LNU and CZU complexes, Glass and others), and asks: **do burned areas sit on steeper or south-facing ground than the land around them?**

Red cells are inside a perimeter, blue cells are a ring of unburned land around each fire. **Perimeter year** (below) and **Smallest fire** filter which fires count; the readouts say how many were compared. Click a fire to see only its numbers in the charts later on.

## Turning polygons into cells {#rasterize}

`GPUPolygonRasterization` scan-converts every perimeter into a zone raster on the GPU: each cell takes the id of the fire whose polygon contains the cell center (holes subtract, parts of one fire union, overlapping fires resolve to the smaller id). It counts every (edge, row) crossing, sorts them and fills the spans between pairs, so a 100,000-vertex ring costs the same per vertex as a small one.

The white cells in **Show** = *Perimeter cells* are the rasterizer's `boundary` output: cells that any polygon edge touches. Cells without the flag are entirely inside or outside. The **Rasterizer** readout shows how many crossing records were used against the capacity the scene sized from the data; if it overflowed the zone raster would be left empty rather than half filled.

## Slope and aspect from the elevation grid {#terrain}

`GPUTerrainDerivatives` computes slope, downslope direction (aspect) and a hillshade with Horn's 3x3 method. The grid is Web Mercator, so the contributor runs in `web-mercator` cell-size mode and corrects every row for the real ground size of a pixel (about 239 m here). **Show** = *Slope* paints the result; the hillshade underneath is lit from **Sun azimuth** and **Sun altitude** (they only rewrite the hillshade).

**Vertical exaggeration** multiplies the elevation differences, and so the slopes: leave it at 1 for the analysis. **Border mode** says what the 3x3 window sees beyond the grid and next to missing cells (ocean is no-data), which matters along the coast. Changing it rebuilds the derivatives graph.

## A ring of unburned land {#ring}

To compare, each fire gets a ring. `GPUDistanceField` computes, for every cell, the distance to the nearest burned cell and *which fire that is* (an exact Euclidean transform, or the cheaper jump-flood approximation). The scene keeps unburned cells between **Ring inner distance** and **Ring outer distance**, measured on the ground (the Mercator distance is scaled by the cosine of each row's latitude), and gives each to its nearest fire.

Drag **Ring outer distance** and the blue band grows without re-rasterizing or re-running the distance field: it only rewrites a parameter. Choose `jump-flood` in **Distance algorithm** to see the approximation; **Jump-flood refinement** adds correction passes.

## Do burns sit on steeper ground? {#slope-result}

`GPURasterZonalStatistics` turns the zone rasters into numbers: for every fire, inside and ring, the count, sum and maximum of a band. Four bands are summarised (slope, elevation, a steep-cell flag and "southness"), each in one pass over the grid. **Zonal sum order** chooses between a sorted, reproducible reduction and float atomics.

For **{{scope}}**, mean slope inside versus ring is **{{slopeMean}}**. The share at or above the selected **Steep slope from** threshold is **{{steepShare}}**. The histogram counts fires by their inside-minus-ring mean slope. Move **Steep slope from** and **Ring outer distance**, filter fires or select a fire to examine how the comparison changes.

## Do they face south? {#aspect-result}

Aspect is the direction the ground faces. The charts count the share of sloping cells (**Flat below** removes cells too flat to have an aspect) in eight compass directions. A south-facing preference would show as a bump around S in the inside line and a positive S bar in the difference chart.

For **{{scope}}**, south-leaning cells (SE, S and SW) account for **{{southShare}}**, inside versus ring. Compare all eight directions in the charts. Switch **Show** to *Southness* to see the same thing on the map: red faces south, blue north. Mean elevation inside versus ring is **{{elevation}}**. Careful: the ring includes valley floor, farmland and towns that rarely burn, fires follow fuels, wind and suppression, and 239 m cells smooth the steepest slopes. These are associations, not causes.
