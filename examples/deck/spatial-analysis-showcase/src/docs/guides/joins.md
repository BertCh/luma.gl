---
title: Joins and overlay
summary: Point-in-polygon, nearest-feature, select-by-distance, topological and raster joins, and which one answers which question about how two layers relate.
order: 5.5
---

## What this family does

A join attaches one layer to another. Almost every geospatial question starts with one: which tract is each report in, which hospital is nearest to each address, which roads cross a boundary. The contributors in this chapter answer those questions on the GPU, for hundreds of thousands of features at once, and leave the results in buffers that layers draw directly.

1. **Which polygon is each point in, and how many points per polygon?** [`GPUPointInPolygonJoin`](#/reference/GPUPointInPolygonJoin) assigns every point its containing polygon; [`GPUZonalStatistics`](#/reference/GPUZonalStatistics) reduces counts, sums, means, extremes and densities per polygon; [`addPointsInPolygonsChoroplethRecipe`](#/reference/addPointsInPolygonsChoroplethRecipe) chains it into class breaks and colours. [`GPUSpatialJoinPrepared`](#/reference/GPUSpatialJoinPrepared) builds the polygon index once. Story: [Nature observations by census tract](#/story/observations-by-tract).
2. **What is the nearest feature, and how far?** [`GPUNearestFeatureJoin`](#/reference/GPUNearestFeatureJoin) snaps points to the nearest point, line or polygon within a radius, or returns the k nearest with exact distances and foot points. [`GPUNearestFeatureWeights`](#/reference/GPUNearestFeatureWeights) turns those neighbours into spatial weights, and [`GPUSpatialJoinCandidates`](#/reference/GPUSpatialJoinCandidates) exposes the bounding-box stage of a distance join. Story: [How far is the nearest hospital?](#/story/nearest-facility).
3. **Which points lie within a distance of these features?** [`GPUBufferSelection`](#/reference/GPUBufferSelection) is select-by-distance with a per-frame distance and an indirect draw. Story: [Select by distance from the L](#/story/near-transit).
4. **How do two geometries relate, exactly?** [`GPUSpatialPredicateJoin`](#/reference/GPUSpatialPredicateJoin) evaluates every OGC predicate and any DE-9IM pattern between points, lines and polygons, with anti joins. Story: [Topological relations between layers](#/story/topological-relations).
5. **Can I join without testing polygons at all?** [`GPUPolygonRasterization`](#/reference/GPUPolygonRasterization) paints polygons into a zone raster and [`GPURasterJoin`](#/reference/GPURasterJoin) bins points into it in O(1) per point, with a measurable error bound. Story: [Raster join](#/story/raster-join).

## Which join for which question

| Question | Use | Cost grows with | Exact? |
| --- | --- | --- | --- |
| Count or average points per polygon | `GPUPointInPolygonJoin` + `GPUZonalStatistics` | points x candidate polygons | yes |
| Per-polygon counts for a changing viewport | `GPURasterJoin` | points only | no: boundary cells |
| Nearest feature within a radius | `GPUNearestFeatureJoin` (nearest-feature mode) | candidates inside the radius | yes |
| k nearest, any geometry, with foot points | `GPUNearestFeatureJoin` (neighbors mode) | k and the feature index depth | yes |
| Points within d of features | `GPUBufferSelection` | candidates inside d | yes |
| Overlap, contains, touches, crosses, equals | `GPUSpatialPredicateJoin` | candidate pairs x vertices | yes |

## Key options

- **Prepare a static right-hand side.** Every join builds bounds and a BVH over its right-hand features. When those features do not change, `GPUSpatialJoinPrepared` builds the index once and every later run reuses it. It reuses a build, never results. If the features move and you do not call `invalidate()`, the joins keep querying the stale index: that is the contract of a static right-hand side.
- **Hilbert-sort shuffled features.** `spatialSort` reorders features along a space-filling curve before the BVH build. Results are identical; traversal cost drops sharply for thousands of spatially shuffled features and does nothing for a few coherent polygons.
- **Capacity and overflow.** Candidate pairs, matched pairs and BVH leaves have compile-time capacities. Every join reports an `overflow` flag on the GPU; read it. An undersized candidate buffer gives a partial result, never an error.
- **Parameters versus topology.** Search radii, buffer distances, `dwithin` distances and DE-9IM patterns are per-frame buffers. Predicates, geometry kinds, `k`, tie rules and engines are compile-time: the panel marks them with a rebuild badge.
- **Engines.** `GPUSpatialPredicateJoin` has a short-circuiting fast kernel for intersects, contains, within and dwithin and a DE-9IM relate engine for everything else. `auto` picks by vertex counts.
- **Anti joins.** `how: 'anti'` returns the left features with no match, the complement of the inner join, which is how you find the unmatched.

## Pitfalls

- **Rates from counts.** A per-tract count follows where people go as much as where they live. Divide by the right denominator, and be careful with tracts that hold few residents.
- **Points on edges.** Many point sets are rounded to street midpoints or block centroids, and many boundaries follow streets. A point within a metre of an edge can fall either side depending on coordinate precision, so two correct pipelines disagree on a few percent of points. Report the disagreement and where it sits.
- **Straight-line distance.** The nearest and buffer joins measure planar distance. Rivers, rail yards and expressways make a close feature far away on foot.
- **Different boundary vintages.** Census tracts and city community areas come from different sources and generalisations. Exact predicates such as `within` expose slivers that the eye ignores; use a distance predicate when you need tolerance.
- **Approximate means bounded.** A raster join is wrong only in boundary cells. Look at the boundary-cell share before trusting a count, and raise the resolution until it matches your tolerance.

## Where to go next

Joined counts are the input to the rate and class chapters ([Rates, classes and indices](#/guide/statistics)); k-nearest and contiguity weights feed the autocorrelation scenes ([Spatial weights and autocorrelation](#/guide/weights)). Every scene in this chapter exposes the code for its current options under **Code**, and the full option list of each contributor is on its reference page.
