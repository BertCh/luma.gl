---
title: Geometry and topology
summary: Measure, generate, cut, snap, simplify, node and dissolve vector geometry on the GPU, and which tool to use when.
order: 20
---

## What this family does

The geometry contributors work on the shapes themselves rather than on values attached to them. They fall into five jobs: **measure** shapes, **generate** new ones, **cut and reshape** lines, **find topology** (crossings, networks, rings) and **generalise** without breaking the map. All of them take GeoArrow-style buffers (vertex rows plus offset columns) and write their results into buffers that the map draws directly, so a slider moves a parameter and the result is redone every frame.

## Measure: `GPUGeometryMeasures`, `GPUShapeDescriptors`, `GPULabelPoint`, `GPUGeometryValidity`, `GPUMapColoring`

`GPUGeometryMeasures` gives area, perimeter, centroid, bounds and vertex count per polygon, and per group. The **coordinate system** is the option that matters: planar for projected data, spherical (turf parity) or WGS84 (authalic area, Vincenty edges). On US counties a Web Mercator planar area overstates Whatcom County, Washington by 2.3 times; at city scale all three agree. `GPUShapeDescriptors` adds compactness (Polsby-Popper, Schwartzberg), elongation, convexity, orientation and a sliver flag whose threshold is a parameter. `GPULabelPoint` finds the pole of inaccessibility, which stays inside concave polygons where a centroid can fall outside. `GPUGeometryValidity` writes a bitmask of defects; check the **orientation convention** before reading orientation flags as errors (Chicago's files wind clockwise). `GPUMapColoring` colours touching polygons differently using adjacency from `GPUContiguityWeights`.

Story: [#/story/polygon-measures](#/story/polygon-measures).

## Generate and clip: `GPUOutlineGeometry`, `GPUShapeGenerator`, `GPUGridGenerator`, `GPUHilbertKeys`, `GPURectangleClip`

Buffers (`GPUOutlineGeometry`) are a **picture**: round-join triangles that overlap and are not unioned, so use a distance query for numbers. `GPUShapeGenerator` makes circles, sectors and ellipses per feature; segments and radius are parameters, while shape, coordinate system and ellipse spacing are compile-time. Equal-arc ellipse spacing keeps edges visually even on flat ellipses. `GPUGridGenerator` makes square, hexagon, triangle or point grids of fixed size whose origin and cell width change every frame; square and hex grids can also emit their unique shared corners and an extent mask for those points. `GPUHilbertKeys` orders items along a space-filling curve: low orders give coarse blocks, order 16 the finest. `GPURectangleClip` cuts lines (Liang-Barsky) or polygons (Sutherland-Hodgman) to a rectangle read per frame.

Story: [#/story/buffers-and-shapes](#/story/buffers-and-shapes).

## Lines: `GPULineSegmentize`, `GPULineChunk`, `GPULineLocate`, `GPULinearReferencing`, `GPULineSimplification`, `GPULineSmooth`

Densify to add vertices, chunk to cut into pieces or take a substring between two measures, locate to place events by distance or fraction, linear referencing to snap points to lines and read the measure and side. Simplification computes importance once and re-selects at a per-frame tolerance; use *time-ratio* for tracks where speed matters. Chaikin smoothing doubles vertices per iteration. Output sizes are capacities fixed at compile time: watch the overflow flag.

Story: [#/story/line-operations](#/story/line-operations).

## Geodesics: `GPUGreatCircleArcs`, `GPUGeodesicPairs`, `GPUGeodesicDestination`

Arcs for many pairs with at least a minimum number of segments (the shortest path bends on a flat map), distance and bearing on a sphere or the ellipsoid, and a destination from an origin, bearing and distance for range rings.

Story: [#/story/great-circles](#/story/great-circles).

## Topology: `GPUSegmentIntersection`, `GPULineSplit`, `GPULineMerge`, `GPUNetworkNoding`, `GPUSegmentRingAssembly`, `GPUCoverageSimplification`

Exact intersection with a kind for each pair (proper, touch, collinear, overlap). Splitting cuts lines at every intersection; merging joins chains where exactly two ends meet; noding snaps end points and builds a network. Ring assembly chains directed segments into shells and holes, which is a dissolve when opposing segments cancel within a group. Coverage simplification simplifies shared arcs once so neighbours stay gap-free; compare it with simplifying each ring on its own.

Story: [#/story/noding-and-coverage](#/story/noding-and-coverage).

## Which tool when

- Need an area or length that is correct: `GPUGeometryMeasures` with the right coordinate system.
- Need a label position: `GPULabelPoint`, not the centroid.
- Data from a messy source: `GPUGeometryValidity`, then `GPUSegmentIntersection`.
- A catchment picture: `GPUOutlineGeometry` or `GPUShapeGenerator`; a catchment number: a distance join.
- Thin a track: `GPULineSimplification`; thin a polygon map: `GPUCoverageSimplification`.

## Pitfalls

- Planar measures are only as good as the projection; the planar column is wrong at national scale.
- Buffers are not unioned. Grids are not clipped. Rectangle clips are planar.
- Capacities (pairs, pieces, rings, candidates) are compile-time; a small one truncates and sets an overflow flag.
- Noding connects end points only, within a snap cell; it does not connect a line ending near another's middle.
- Simplification can cross itself unless the topology-preserving coverage tool is used.
