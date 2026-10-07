---
title: Reading the maps
summary: Color ramps, diverging scales, significance classes and how legends stay honest.
order: 3
---

## One table drives the shader and the legend

Every color ramp is defined once, as a list of evenly spaced stops. The WGSL that colors the map is generated from that table, and the legend gradient is drawn from the same table. If you change a stop, the map and the legend change together, so they cannot disagree.

Legend end labels are real numbers. When a legend says *range read from the GPU*, the min and max come from a small buffer the contributor wrote (its `extent` output), read back a few times a second once the camera settles. A square-root scale is noted under the legend, so you know low values are being lifted for visibility.

## Which ramp, and why

| Ramp | Use it for | Notes |
| --- | --- | --- |
| `viridis`, `magma`, `inferno` | Counts, densities, anything that runs from low to high | Perceptually uniform: equal steps in value look like equal steps in color |
| `cividis` | The same, when color-vision deficiency matters most | Designed to look nearly identical under common deficiencies |
| `diverging` | Values with a meaningful midpoint: residuals, change, z-scores | Blue below, near-white at zero, red above. Use a symmetric range so zero is white |
| `grayscale` | Hillshade and masks | Neutral under other layers |
| Categories | Classes, clusters, quadrants | At most eight colors; each entry is labelled in the legend |

## Significance and hot spots

Local statistics such as Getis-Ord Gi* and local Moran test every place against a random-arrangement null. A point is only colored when it passes the test:

- **Hot and cold spots** (Gi*): red means a neighborhood of unusually high values, blue unusually low. Darker means higher confidence (90, 95, 99 percent).
- **Moran quadrants**: **High-High** and **Low-Low** are clusters of similar values. **High-Low** and **Low-High** are outliers that differ from their neighbors.
- **Not significant** points are gray. Their presence matters: it shows how much of the map is indistinguishable from chance.

When thousands of tests run at once, some pass by luck. The *false discovery rate* option raises the bar so the colored points are more trustworthy, at the cost of fewer of them.

## Things to keep in mind

- **Scale changes the answer.** Radii, cell sizes and smoothing widths are analysis choices. Move the slider and watch the pattern; a result that survives several scales is more convincing.
- **The map follows the camera.** Density-style scenes refit their grid to the viewport, so cell size is shown in the readouts and changes with zoom.
- **Synthetic data is labelled.** When a dataset cannot be fetched (offline, or `?data=synthetic`), the panel says so.
