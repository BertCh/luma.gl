---
title: "GPU Spatial Analysis concepts"
description: "Ontology, shared ports, conventions, coordinate precision and how analysis contributors run."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Spatial Analysis concepts

<ExperimentalDocsTabs active="gpu-spatial-analysis-concepts" />

## Ontology and capability catalogue

The entry point exports a small ontology over the flat contributor API. It gives applications,
documentation tools and future planners stable names for the data flowing between contributors:
points, lines, polygons, trajectories, DGGS cells, spatial weights, feature pairs, regions, sampled
surfaces, summaries and render geometry. Capabilities then group public exports by domain and
pipeline stage and state their logical inputs, outputs, coordinate assumptions and output-bounding
contract.

The catalogue stores export names, not constructor references, so using it for discovery does not
eagerly retain every implementation in a bundle. Every public `GPU...` contributor and `add...`
recipe belongs to exactly one available capability. Deliberate gaps are `status: 'opportunity'` and
have no operators, which makes coverage and missing connective tissue queryable with the same API.

```ts
import {
  getGPUSpatialAnalysisCapabilityForOperator,
  queryGPUSpatialAnalysisCapabilities
} from '@luma.gl/experimental/gpu-spatial-analysis';

const producers = queryGPUSpatialAnalysisCapabilities({
  status: 'available',
  output: 'spatial-weights'
});
const gaps = queryGPUSpatialAnalysisCapabilities({status: 'opportunity'});
const localMoranFamily = getGPUSpatialAnalysisCapabilityForOperator('GPULocalMoran');
```

The exported vocabulary is:

- `GPU_SPATIAL_ANALYSIS_DATA_ONTOLOGY` and `GPU_SPATIAL_ANALYSIS_DATA_KINDS`: logical data nouns.
- `GPU_SPATIAL_ANALYSIS_DOMAINS` and `GPU_SPATIAL_ANALYSIS_STAGES`: subject and pipeline roles.
- `GPU_SPATIAL_ANALYSIS_CAPABILITIES`: available capability families and named opportunities.
- `queryGPUSpatialAnalysisCapabilities(query)`: filters by status, domain, stage, input, output,
  coordinate model or search text.
- `getGPUSpatialAnalysisCapability(id)` and
  `getGPUSpatialAnalysisCapabilityForOperator(name)`: direct lookup from a concept or public export.
- `getGPUSpatialAnalysisCapabilityConnections(id)`: resolves the curated adjacent capabilities
  used to assemble workflows.

Operator-level discovery is available through `GPU_SPATIAL_ANALYSIS_OPERATORS`,
`getGPUSpatialAnalysisOperator(name)` and `queryGPUSpatialAnalysisOperators(query)`. Descriptors
contain export names rather than constructors and distinguish primary capability, secondary tags,
logical ports, spatial support, topology, cardinality, numerical semantics, determinism, status and
explicit readback. Unknown details are reported as `unspecified`, never inferred as a stronger
guarantee. `GPU_SPATIAL_ANALYSIS_AUDIT` and `formatGPUSpatialAnalysisAuditMarkdown()` expose the
same inventory used by package verification.

## Shared ports, status and planning

The entry point exports non-owning structural ports for point, line, polygon and trajectory
geometry, compact pairs, generated geometry, spatial weights, cell tables, sampled surfaces and
classified values. Ports only group existing graph views; they do not allocate, pack, submit or
read back buffers. Polygons always use the four-level layout of `GPUPolygonGeometryPort`
(`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`; rows are features, the first ring of
each polygon is its shell), lines use `positions` and `lineOffsets`, and tracks `trackOffsets`.
Contributors with one geometry take these fields flat (`GPUPolygonGeometryFields`,
`GPULineGeometryFields`, `GPUTrajectoryGeometryFields`); contributors with two geometries, such as
the joins, take whole tagged ports (`GPUPackedGeometryPort<...>`, one packed `positions` view), so a
generated geometry output feeds them directly. `GPUSpatialPredicateJoin` `output.pairs` is a
`GPUCompactPairPort` and feeds `GPUPairGather` directly. Every GPU-written status scalar lives in the
`status` of the bag that owns it: capacity-bounded outputs publish `status.count`, `status.requiredCount`
when exact, and the applicable overflow, convergence, invalid or uncertainty fields of the shared status
ports (`GPUCardinalityPort`, `GPUOverflowStatusPort`, `GPUIterationStatusPort`, ...). A nested list owns
its own `status` (`output.pieces.status.requiredVertexCount`), stage-prefixed words keep their prefix
(`output.status.pairOverflow`), and model fits report their solver `status.code`.

`GPUSpatialContext` separates coordinate space, metric, units and sphere/ellipsoid parameters; omitted context means native planar coordinates.
Planar native contexts may declare `units: 'meters'` (local frames do); `'kilometers'` is only for geographic metrics.
Planar `'native'` units are undeclared and compatible with planar `'meters'`; meters and kilometers are incompatible.
The `frame` identity has kind `'local-azimuthal-equidistant'`, `'local-lambert-equal-area'` or `'crs'`;
`assertCompatibleGPUSpatialContexts` rejects contexts with different frame ids.
`datum` declares the horizontal datum of longitude/latitude positions or of a planar frame's source (for example `'EPSG:4326'`, `'EPSG:4269'`, `'GCJ-02'`); `createGPULocalFrame({datum})` copies it to the frame's context.
Two declared datums must match by `getGPUDatumKey` (trimmed, upper-cased; only WGS84 spellings alias to `'EPSG:4326'`), and an undeclared datum matches any; the library never converts datums.
The check runs where two contexts meet; `GPUProjection` CRS ids already imply their datum.
Two-input operators (joins, zonal and region statistics, raster join, pairwise point-in-polygon, line clip, overlay, zone events and the choropleth, change-of-support and hot-spot recipes) take an optional context for their second input, named after that input (`polygonsSpatialContext`, `rightSpatialContext`, `featuresSpatialContext`, `regionSpatialContext`, `zonesSpatialContext`, ...).
When given, `resolveGPUSecondarySpatialContext` validates it, applies the operator's edge-model requirement to it and checks it against `spatialContext`, so a datum, frame, units or metric mismatch between the two sides throws `<id> <side>SpatialContext incompatible spatial contexts: <reason>`; omitted, `spatialContext` describes both inputs and nothing is checked between them.
`edgeModel` (longitude/latitude only) says how polygon and line edges run between vertices: `'linear'` (default, straight in degrees as in RFC 7946, GEOS and turf) or `'great-circle'` (minor arcs, as in PostGIS `geography`, BigQuery `GEOGRAPHY` and S2); ellipsoidal contexts use `ellipsoidalEdgeModel` instead.
Polygon consumers declare `edgeModels: GPU_LINEAR_EDGE_MODELS` and reject great-circle input with a pointer to `GPULonLatPolygonPreparation`, which splits rings at the antimeridian, caps pole rings and densifies great-circle edges; their optional `status: GPULonLatPolygonStatusPort` counts unprepared rings (`antimeridianUncertainCount`, `poleUncertainCount`).
Point-pattern statistics and kernel density take a `studyWindow`: `'rectangle'` (default, per-frame bounds) or a `GPUPolygonStudyWindow` built by `GPUStudyWindowPreparation` in the same planar frame, checked by `resolveGPUStudyWindow`.
Support declarations use `measure: 'distance' | 'area'`: distance operators reject Lambert equal-area frames,
while area/density operators reject azimuthal-equidistant frames, with an error naming the frame to project into.
This measure check accepts `'crs'` frames and frameless contexts.
Spherical distances use `getGPUSphereRadius` (mean radius 6371008.8 m); spherical areas use `getGPUAuthalicSphereRadius` (WGS84 authalic radius 6371007.1809 m).
Both honour `sphereRadius` in the context's units; `units: 'kilometers'` divides default radii by 1000, giving areas in km².
Ellipsoidal metrics use `getGPUEllipsoid` (`context.ellipsoid` or `GPU_WGS84_ELLIPSOID`).
Terrain heights declare a vertical datum the same way (`GPUTerrainHeightReference`, `GPU_ELLIPSOIDAL_VERTICAL_DATUM`, `assertCompatibleGPUVerticalDatums`); see [GPU Terrain](../gpu-terrain/README.md).
`getGPULengthUnits` gives the length unit reported by `GPUMeasureDescription`.
Compatibility helpers reject invalid workflow connections during graph construction and contain no
CRS parsing or reprojection; that remains the responsibility of GPU Project.

`GPUSpatialParameterSchema` describes packed layouts without changing their WGSL representation.
Schemas are published for geometry predicates, neighbor search, line segmentizing, great-circle
arcs, spatial autocorrelation and line density. The generic packer validates defaults, offsets,
ranges and units; contributor-specific packers retain any documented normalization that is not yet
expressed by the generic schema.

Capacity plans (`GPUCapacityPlan` per stage, `GPUContributorCapacityPlan` per contributor) report
stage capacities, work items, caller-owned output bytes and transient bytes. They are computed by
the contributor that allocates, from the same constants as its allocation, and stay pure:
applications own allocations. The weights grow loop is `getGPUSpatialWeightsCapacityPlan` plus the
producers' `getCapacityPlan` (see [Spatial weights](./operators-weights.md)). Candidate and output stages
grow independently: a candidate overflow (for example `output.status.candidateOverflow` of
`GPUContiguityWeights`) means the final `requiredCount` may describe only the candidates that fit,
so growing only the output is invalid.

```ts
const plan = contiguity.getCapacityPlan({requiredCount, candidateOverflow: candidateOverflow === 1});
// Reallocate plan.slots.capacity slots and rebuild with pairCapacity: plan.pairs.capacity.
```

`GPUPartitionDescriptor` retains contiguous source-batch cores, read halos and deterministic seam
ownership without repacking `GPUVector` chunks. `GPUTilePartitionDescriptor` carries spatial core
and halo bounds for tiled consumers. `GPUSpatialJoinPrepared` validates its `partitioning` against the
indexed features. The Arrow
analytics adapter publishes its record batches through this contract, including empty batches.
`iterateGPUAnalyticsPartitionsFromArrowTable()` lazily allocates one source batch at a time while
retaining its global batch ID and row offset; destroying each yielded result before advancing bounds
GPU residency to one partition. The tile-LOD explorer declares explicit core, halo and cross-tile
ownership rules with the tile contract. Execution remains consumer-owned.

Recipe results use a required common envelope with `outputs`, `intermediates` and stage-owned
`status` in addition to `contributors`. Result-level status words sit in `result.status` next to
`result.status.stages` (for example `result.status.weightsOverflow`), and sub-results own a `status`
(`result.permutation.status.overflow`); the `outputs` overrides mirror those paths.

## Conventions

- **Graph views in, graph views out.** Inputs and outputs are `GraphDataView`, `GraphVectorView`, or
  `GraphTextureView` objects created on the target graph. Outputs are always caller-owned; scratch
  storage is graph transients that die with `compiled.destroy()`. Contributors never compile, encode,
  submit, or read back.
- **Per-frame values never recompile.** Viewports, thresholds, radii, observer positions, budgets,
  and region shapes are read from storage views, usually a `GPUParameterBuffer` that the
  application rewrites with `write()`. Lengths, capacities, grid sizes, and which optional views
  exist are compile-time topology; each prop's TSDoc says which category it belongs to.
- **Bounded results report overflow on the GPU.** Compact ID lists use `GPUCompactOutput`
  (`ids` and `status: {count, overflow, requiredCount?}`). `status.count` is clamped to `ids.length` and
  can be an indirect draw instance count; `status.overflow` is rewritten every encoding.
- **Stable IDs.** Result IDs are the caller's `sourceIds[row]` (or tile IDs) when given and zero-based
  rows otherwise. Node and transient IDs are `${id}-<step>`, so two instances in one graph need
  different `id` props.
- **One import per buffer.** Import a buffer once (for example with `graph.importGPUData()` or
  `GPUParameterBuffer.importToGraph()`) and pass the returned view to every contributor that reads
  it, so the graph tracks hazards on one logical handle.
- **Shared contracts.** Four shapes mean the same on every analysis page:
  - **Status.** Every GPU-written status word sits in the `status` member of the bag that owns it:
    `output.status.overflow`, `output.status.converged`, and `output.status.code` for model fits.
  - **Grid.** Raster and terrain contributors take `grid: GPURasterMetadata`, and their packers read
    the cell size from it (see [GPU Terrain conventions](../gpu-terrain/concepts.md#conventions)).
  - **Network.** Network contributors take `topology: GPUNetworkTopology` (`offsets`, `neighbors`,
    `weights`, optional `nodePositions`). Its CSR part can be passed as `GPUSpatialWeights` (see
    [GPU Network conventions](../gpu-network/concepts.md#conventions)).
  - **Polygons.** Polygons use the four-level `GPUPolygonGeometryPort`: `kind: 'polygons'`,
    `positions`, `featureOffsets`, `polygonOffsets` and `ringOffsets`. Contributors with a single
    geometry take the same fields flat, without `kind` (`GPUPolygonGeometryFields`). See
    [Shared ports](#shared-ports-status-and-planning).

## Precision of longitude/latitude positions

Longitude/latitude positions are float32 degrees, never split into high/low parts. At
`|longitude| >= 128°`, the worst-case ground quantum is `Q = 2⁻¹⁶° · π/180 · R ≈ 1.70 m`
for the mean Earth radius `R = 6371008.8 m` (the equatorial worst case). Representative combined
east-west/north-south quanta are:

| Location | Ground quantum |
| --- | --- |
| Chicago | 0.76 m |
| NYC | 0.77 m |
| San Francisco | 0.79 m |
| London | 0.42 m |
| Sydney | 1.47 m |
| Equator near ±180° | 1.70 m |

Great-circle distance kernels use haversine, so short-distance arithmetic is not the limit;
the rounded inputs are. Use `createGPULocalFrame` for sub-17 m work, projecting before rounding
positions to float32 degrees. When a host-known distance is finite, greater than zero and less
than `10·Q` (16.97 m; 0.01697 km with the default sphere), operators warn once per operator class:
`<Operation> distance is too short for float32 lon/lat; use createGPULocalFrame`.
This precision check never throws; zero, NaN and non-finite distances do not warn.

The exported helpers are `getGPULongitudeLatitudeQuantum(context)`,
`isGPUShortLongitudeLatitudeDistance(context, distance)` and
`warnGPUShortLongitudeLatitudeDistance(operation, context, distance)`, with
`GPU_SHORT_LONGITUDE_LATITUDE_DISTANCE_FACTOR = 10`. The quantum uses the context's sphere radius
(or ellipsoid semi-major axis) and length unit; it is zero for planar contexts or metric `'none'`.

- **Tier 1 — construction.** `GPUSpatialPredicateJoin` checks numeric `dwithin` distances;
  `GPUTrajectoryEncounters` checks its derived `maximumDistance`.
- **Tier 2 — per-frame writers.** Pass optional `spatialContext` in the settings passed to the
  parameter writer to check neighbor-search `radius`, DBSCAN `epsilon`, IDW `searchRadius`,
  kriging `searchRadius` and `range`, Douglas-Peucker `tolerance`, line segmentize / great-circle
  arcs `maximumSegmentLength`, line chunk `chunkLength`, outline `distance`, map matching
  `sigma` and `searchRadius`, and GWR fixed `bandwidths`. Packed parameter layouts are unchanged;
  omitting the settings context disables this warning.
- **Tier 3 — GPU views.** No warning is possible for `GPUBufferSelection.distance`,
  `GPUNearestFeatureJoin.radius`, a view-valued `GPUSpatialPredicateJoin.distance`,
  `GPUTrajectoryEncounters.distance`, `GPUTrackSimilarity.frechetEpsilon`,
  `GPULinearReferencing.radius`, `GPUNetworkSnapping.maximumSnapDistance`,
  `GPUNetworkKFunction` distance views, the drive-time catchment recipe's `maximumSnapDistance`, or
  `GPUGeodesicDestination.distances`: the host does not read these GPU values. Trajectory-metrics
  stop detection also cannot warn: the affected distance is speed × sampling interval, so float32
  position noise means about ±0.8 m/s at 1 Hz in Chicago; use a local frame.

## Running contributors

This applies to every contributor page (spatial analysis, network, terrain, raster, dataframe
analysis and geospatial). Applications compile with `await graph.compileAsync()`. It creates every
GPU pipeline asynchronously and concurrently, so the page keeps responding while a graph with many
kernels compiles. `graph.compile()` gives the same result but blocks the main thread until every
pipeline exists; keep it for tests and scripts. Either call freezes the graph, and a graph compiles
once, including when `compileAsync()` rejects.

Compile once per topology and rewrite parameter buffers every frame. A compile-time option (a grid
size, a method that adds nodes, an optional output) needs a new graph. Build the new graph and await
its compile while the old compiled graph keeps encoding. Then swap the new graph in, and only then
destroy the old one. If a later request arrives before the compile finishes, destroy the stale
result instead of swapping it in.

```ts
type Options = {gridSize: [number, number]};
declare function buildGraph(options: Options): GPUCommandGraph;

let compiled = await buildGraph(options).compileAsync();
let latestRequest = 0;

async function rebuild(nextOptions: Options): Promise<void> {
  const request = ++latestRequest;
  const next = await buildGraph(nextOptions).compileAsync(); // the old graph keeps encoding
  if (request !== latestRequest) {
    next.destroy(); // a later rebuild superseded this one
    return;
  }
  const previous = compiled;
  compiled = next; // swap first
  previous.destroy(); // then release transients and node state
}

// every frame: record into the device's default encoder, then submit it
compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();
```

`encode()` only records commands; the application submits (`device.submit()` for the device's
default encoder, or `device.submit(encoder.finish())` for its own encoder). The second argument's
`parameters` is the graph's typed CPU channel, which node conditions and encode callbacks receive.
Analysis contributors do not use it: every per-frame value they read is packed into a GPU view (a
`GPUParameterBuffer` written by a `get*ParameterValues` packer, see each class's "Parameters"
subsection), so changing it needs no re-encode logic and no CPU-side state. The graph is therefore a
`GPUCommandGraph<void>` and the call passes `{parameters: undefined}`.

The gpu-raster base operators (`GPURasterStatistics`, the filters, morphology and region operators)
predate `GPUCommandNodeProducer`: they declare their work with `operator.addToGraph(graph)` instead
of `graph.add(operator)`. Compiling, encoding and submitting are the same.

Destroy a compiled graph before the caller-owned buffers it imports. Never compile, submit or read
back synchronously inside the frame loop.


## GPU execution model

How spatial analysis contributors run on the device: the command graph they declare, the kernels it
compiles to, and the resources each pass reads and writes, drawn from a recorded real run.

{/* lens:execution-model */}

Each contributor adds nodes to a `GPUCommandGraph`. `compile()` orders them by the buffers they read and write,
`describe()` records that structure without running anything, and the lenses below draw it from recorded runs.

::lens{scenario=gpu-spatial-analysis.kernel-density rung=graph caption="Kernel density: grid index over the points, in-cell sort for determinism, then one gather pass."}

::lens{scenario=gpu-spatial-analysis.kernel-density rung=kernel caption="The gather kernel: one invocation per output cell; the binding table names every buffer it reads."}

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=graph caption="A join: bounding-box candidate pairs first, then one exact containment test per candidate pair."}
