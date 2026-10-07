## Ninety large western fires, one polygon file {#fires}

Between 2020 and 2023 the National Interagency Fire Center mapped the final perimeters of these fires, and the poopdeck.gl archive carries them as polygons with acres, name, year and the date of the perimeter. Dots are colored by year; zoom in and each fire becomes a filled polygon. Many fires are **several polygons**: the Dixie Fire is one huge burn plus 178 small islands, so the 460 polygons of the archive are 118 fires, and 90 of them are in the West, which is what this scene shows.

The question: **do big fires get stringier?** We need the area and perimeter of every fire, and a number for how compact its outline is. Use **Perimeter year** and **Smallest fire** (below) to look at a subset; the readouts count what is left.

## What does the GPU say a fire measures? {#area}

`GPUGeometryMeasures` walks the rings of every fire in parallel and returns area, perimeter, centroid and vertex count. It can do that in four coordinate systems, and the choice matters: **Area system** (below) switches the colors and numbers between planar Web Mercator meters, a sphere, the WGS84 authalic sphere and geodesic edges. All four run in one graph, so switching is just a different column (no rebuild).

Pick `planar` and the areas balloon: Web Mercator stretches lengths by `1 / cos(latitude)`, so areas here come out about 1.6 times too big (1.4 to 2.1 across the fires). With `wgs84` the **Median GPU / NIFC** ratio sits at 1.00. That is a check on the GPU, not a second opinion: the NIFC acres were computed from the same polygons.

## One fire, many polygons {#multipart}

A fire is one *feature* with several polygons, each with its own rings. `GPUGeometryMeasures` takes that as `featureRingOffsets`, and **Hole rule** (below) says how a feature's rings combine. The default, `winding`, sums signed ring areas: counter-clockwise shells add and clockwise holes subtract, which is how the data was prepared, so a second polygon of the same fire adds its area.

`first-ring-exterior` is the rule for one polygon per feature: the first ring is the shell and *every other ring is a hole*. On a multi-polygon fire the other parts are then subtracted instead of added. Most parts here are small islands, so the damage is modest, but **Largest area error** names the fire it hurts most. Changing the rule is a compile-time option: the graphs are rebuilt (the Under the hood drawer counts it).

## Do big fires get stringier? {#compactness}

`GPUShapeDescriptors` adds the shape numbers. **Polsby-Popper compactness** is `4 pi A / P^2`: 1 for a circle, toward 0 for a thin or ragged outline. The map is colored by it (dark is stringy), and the trend chart groups the fires into equal-count size bands: median compactness, with the middle half of the fires as a band.

On this data the rank correlation of area and compactness is about -0.55: fires under 3,000 acres have a median compactness of 0.3 to 0.37, the next band drops to 0.2, and the biggest quarter sits near 0.11. The three fires over 300,000 acres score 0.02 to 0.08. Hover a fire for its compactness, click to select it and read the whole descriptor set. Compactness is computed in Web Mercator meters, which is conformal, so the ratio is safe even though the planar area is not.

## Which way do fires point? {#axes}

Elongation (`1 - sqrt(lambda2 / lambda1)` of the second central moments) is 0 for a circle and tends to 1 for a line; **orientation** is the angle of the major axis. The blue lines are those axes, longer for more elongated fires. In the Coast Ranges and the Sierra the long axes tend to follow ridges and valleys; weather and fuels decide the rest, so read orientation as a descriptor of the polygon, not of the wind.

Switch **Color by** to Convexity (area over the area of the convex hull) for another view of "stringy": a fire that wraps around a lake or a ridge has low convexity even if it is not elongated. There are two algorithms behind **Convexity method**: gift wrapping costs the hull size per vertex, the monotone chain sorts instead, and `auto` picks the chain for large average rings, as here.

## A caveat about perimeters {#detail}

Perimeters depend on the detail of the outline (the coastline paradox): more vertices, longer perimeter, lower compactness. Color by **Vertices**: the biggest fires have a median of about 6,300 vertices, the smallest about 500, and the rank correlation of area and vertex count is about +0.56 (compactness against vertices: about -0.66). Part of "big fires are stringier" is that big fires are mapped in more detail, and the data cannot separate the two.

So read the trend as *what the mapped polygons look like*, not as proof about fire behavior. Try **Large-ring path** too: a ring of more than 512 vertices (Dixie has about 94,000 in its main ring) is measured by a 64-lane workgroup; with `serial` one lane walks it. The numbers agree to float precision; only the latency changes.
