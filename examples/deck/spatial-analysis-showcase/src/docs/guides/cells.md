---
title: Cells and discrete global grids
summary: Keys, tables, topology, covers and pyramids on H3, Quadbin and friends, and how to choose between them.
order: 20
---

## One key per place

A **discrete global grid system** (DGGS) cuts the Earth into cells and gives every cell a 64-bit integer. Two sightings in the same cell share a key, so "how many observations are here" becomes a sort and a count, "what is next to this" becomes arithmetic on the key, and "roll up to a coarser scale" becomes a bit shift. There is no geometry test anywhere.

The cell contributors in luma.gl do this on the GPU, for tens of thousands of points, without reading anything back. Two scenes show them: [Observations on a global grid](#/story/h3-aggregation) and [Nature change between seasons](#/story/cell-pyramid).

## The pipeline, step by step

1. [`GPUPointToCell`](#/reference/GPUPointToCell) turns longitude/latitude into a key. It supports five families: **H3** (hexagons), **Quadbin** (Web Mercator tiles, the CARTO layout), **quadkey**, **geohash** and **S2**. Masked or non-finite rows get the zero key.
2. [`GPUCellAggregation`](#/reference/GPUCellAggregation) sorts the keys and counts runs into a compact table of `(cell, count)` plus optional exact fixed-point sums, minimums and maximums. It only supports H3 and Quadbin.
3. [`GPUCellGeometry`](#/reference/GPUCellGeometry) decodes keys into boundary polygons so a layer can draw the table straight from storage buffers.
4. [`GPUCellSetOutline`](#/reference/GPUCellSetOutline) finds the borders of a cell set, optionally between *groups* of cells (for example density tiers), and can assemble the segments into closed rings with holes.
5. [`GPUCellTopology`](#/reference/GPUCellTopology) expands a cell into its grid **disk**, **ring**, **parent** or **children**.
6. [`GPUCellCover`](#/reference/GPUCellCover) polyfills polygons with cells, and [`GPUCellCompaction`](#/reference/GPUCellCompaction) merges complete sibling groups into parents (or expands them back).

Pre-aggregated tables can be rolled up with [`GPUCellRollup`](#/reference/GPUCellRollup), stacked into a zoom pyramid with [`GPUCellPyramid`](#/reference/GPUCellPyramid) (the level is picked per frame by [`GPUCellLevelSelection`](#/reference/GPUCellLevelSelection)), and compared with [`GPUCellTableCompare`](#/reference/GPUCellTableCompare), which outer-joins two tables into delta, ratio, percent change and a z-score. The recipe [`addPeriodComparisonRecipe`](#/reference/addPeriodComparisonRecipe) wires two aggregations, the compare node and a classified color scale together.

## Which family should I use?

| Question | Pick |
| --- | --- |
| Equal-distance neighbours, no orientation bias, smooth density | **H3**. Every neighbour of a hexagon is the same distance away; areas vary only a few percent. |
| Web maps, tiles, zoom pyramids, easy roll-ups | **Quadbin**. A parent is a bit shift; cells are exact tile squares. Area shrinks towards the poles. |
| Interoperating with a database that stores a particular key | The family that database uses. Quadkey, geohash and S2 only key and decode here, they have no aggregation contributor. |

## Choosing a resolution

Resolution is the trade between **signal and noise**. At fine resolutions most cells hold one or two points, so counts are noisy and the table grows towards the point count (it can overflow its capacity: the readouts say so). At coarse resolutions patterns blur and borders of unrelated places merge, the *modifiable areal unit problem* of every aggregation. A good habit is to look at two or three resolutions before quoting a pattern, and to roll up from the finest table instead of re-aggregating, so the parents are exactly consistent.

Some tips:

- H3 resolution 8 (edge about 460 m) and Quadbin zoom 15 (about 900 m) are good city-scale defaults.
- H3 and S2 keys are computed with f32 sphere math. They are exact at coarse resolutions and degrade past about resolution 10 (H3) or level 20 (S2); the scene shows the error rate for the family you picked. Quadbin, quadkey and geohash keys are bit-exact integers.
- Counts per cell are heavy tailed. Use a square-root or log scale, or classify with `GPUClassBreaks`.

## What is compile-time, and what is not

Per-frame (a buffer write, no recompile): the points and their mask, the sampled share, the group filter, the selected cell, the active pyramid level, class edges, the no-change band, the colors.

Compile-time (the badge in the options panel): the **family**, the **resolution**, a cover's **containment** and a compaction's depth, a topology operation and its radius (it fixes the output stride), the roll-up resolution, and the compared measure. The showcase compiles the variants it needs on first use and keeps them, so you can switch back for free.

## Pitfalls

- **Tables have a capacity.** Aggregation, cover and compaction write capacity-bounded tables and keep the smallest keys on overflow. Always read the overflow flag.
- **Keys of different families or resolutions do not mix.** `GPUCellTableCompare` cannot check that its two tables use the same family and resolution.
- **H3 covers use the cell center only.** For `full` or `intersects` containment use Quadbin.
- **Cover cells are not a perfect copy of the polygon.** The *core* flag marks cells that are provably inside, border cells straddle an edge; coarse covers of small polygons may be empty.
- **Counts of observations are not counts of wildlife.** iNaturalist records follow the observers: parks, trails and the lakefront are watched far more than other places.
- **A change map of a short period is noisy.** Prefer the z-score and equal-length periods, and treat the Poisson score as a screening tool.

## Related reading

The reference pages of [`GPUCellAggregation`](#/reference/GPUCellAggregation), [`GPUCellSetOutline`](#/reference/GPUCellSetOutline) and [`GPUCellCover`](#/reference/GPUCellCover), and the stories [Observations on a global grid](#/story/h3-aggregation) and [Nature change between seasons](#/story/cell-pyramid). Cell tables also serve as the join key of the group-statistics and time chapters.
