import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';
import {DeferredGPUCrossfilterExample} from '@site/src/components/docs/deferred-gpu-crossfilter-example';
import {GPUExampleCard} from '@site/src/components/docs/gpu-example-card';

# GPU Crossfilter

<ExperimentalDocsTabs active="gpu-crossfilter" />

## Overview

`@luma.gl/experimental/gpu-crossfilter` coordinates linked selections across GPU-resident maps,
scatterplots, histograms, categorical summaries, and application-defined views. Brush one view,
update its selection, and encode the same command graph again: source rows, selection masks,
aggregations, and stable visible-row identifiers remain on the GPU.

GPU Crossfilter is an experimental, renderer-independent WebGPU controller. It does not ship a dashboard
framework, dataframe importer, or charting library; applications retain ownership of their source
vectors, output buffers, user interface, rendering, command submission, and optional readback.

## When to use it

Use GPU Crossfilter when several views must share interactive selections over the same GPU-resident rows.
Use GPU Dataframe directly for a one-shot query without self-excluding linked views.

## Try the million-row explorer

The embedded [Million-Row Crossfilter Explorer](/examples/showcase/million-row-crossfilter) links a
GPU-rendered map, scatterplot, three self-excluding histograms, and categorical cohorts across
1,048,576 synthetic rows. Start it when you are ready to allocate a WebGPU device; ordinary visits
to this documentation do not import the example registry or initialize the dashboard.

<GPUExampleCard
  demonstrates={['linked selections', 'self-excluding histograms', 'stable compaction']}
  input="1,048,576 GPU-resident synthetic rows"
  gpuOutput="Per-view masks, distributions, cohorts, and visible row IDs"
  cpuReadback="One compact dashboard summary"
  execution="Deferred startup and invalidation-driven updates"
  compatibility="WebGPU"
  fullPageHref="/examples/showcase/million-row-crossfilter"
  sourceHref="https://github.com/visgl/luma.gl/tree/master/examples/showcase/million-row-crossfilter"
  inspectorHref="/examples/showcase/million-row-crossfilter?panel=graph"
/>

<DeferredGPUCrossfilterExample embeddedHeight={900} />

Brush the map or scatterplot, adjust a histogram range, and try the preset scenarios. The source
columns and linked masks stay on the GPU; the example explicitly reads only one compact summary
buffer to update its visible counts and distributions.

## Quick start

```ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUCrossfilter} from '@luma.gl/experimental/gpu-crossfilter';

const graph = new GPUCommandGraph(device);
const longitude = graph.importGPUVector('longitude', longitudeVector);
const latitude = graph.importGPUVector('latitude', latitudeVector);
const value = graph.importGPUVector('value', valueVector);
const category = graph.importGPUVector('category', categoryVector);

const filter = new GPUCrossfilter(graph, {
  id: 'linked-dashboard',
  dimensions: [
    {id: 'map', kind: 'bounds', x: longitude, y: latitude},
    {id: 'value', kind: 'range', input: value},
    {id: 'category', kind: 'range', input: category}
  ],
  views: [
    {
      id: 'distribution',
      kind: 'histogram',
      dimension: 'value',
      input: value,
      domain: [0, 100],
      output: histogramCounts
    },
    {id: 'cohorts', kind: 'group', keys: category, output: categoryCounts},
    {
      id: 'visible-rows',
      kind: 'visibility',
      output: visibleSourceIds,
      count: visibleCount
    }
  ],
  outputMask: combinedSelectionMask
});

filter.addToGraph(graph);
const compiled = graph.compile();

filter.setBounds('map', [minimumLongitude, minimumLatitude, maximumLongitude, maximumLatitude]);
filter.setRange('value', [20, 40]);
filter.setRange('category', [categoryId, categoryId]);

const encoder = device.createCommandEncoder({id: 'linked-dashboard-update'});
compiled.encode(encoder, {parameters: undefined});
device.submit(encoder.finish());
```

`range` selects inclusive scalar endpoints. `bounds` selects an inclusive axis-aligned rectangle.
All active dimensions intersect; exact categorical selection can be expressed as an identical
minimum and maximum unsigned category identifier.

Histogram and group views exclude their own associated selection by default, retaining the available
distribution while that view is brushed. Set `includeOwnSelection: true` when a view should include
its own predicate. Visibility views publish stable compacted row identifiers and a visible count;
mask views publish source-aligned selection flags for custom rendering or compute work.

Selections accept packed `float32`, `sint32`, and `uint32` scalar inputs. Chunked `GPUVector` inputs
retain their original ordered chunk boundaries; chunking is not distributed or multi-GPU execution.
Call `filter.clear(dimensionId)`, `filter.clearAll()`, or `filter.destroy()` as the application
updates or releases its selections.

## Live rows, missing values, and brush intervals

Tables whose rows come and go, such as a time window or a residency mask, need three options beyond
the defaults. All keep the graph topology fixed, so their contents or ranges can change every frame
without recompiling.

```ts
const filter = new GPUCrossfilter(graph, {
  dimensions: [
    {id: 'speed', kind: 'range', input: speed, rejectNonFinite: true, exclusiveMaximum: true},
    {id: 'hour', kind: 'range', input: hour, exclusiveMaximum: true}
  ],
  // nonzero = live; dead rows never reach any view
  liveMask,
  views: [
    // 'auto' infers the domain from live, finite rows only, independent of every brush
    {id: 'speed-histogram', kind: 'histogram', dimension: 'speed', input: speed, domain: 'auto', output: speedBins},
    {id: 'selected-count', kind: 'count', output: selectedCount}
  ]
});
```

- `liveMask` (controller prop) removes dead rows from every histogram, group, count, visibility,
  and mask output, including self-excluding views, and from `'auto'` histogram domains.
- `rejectNonFinite` (range and bounds dimensions) treats NaN and infinite `float32` values as
  missing: such rows always fail that dimension, even while it is inactive, so they leave every
  other linked view too. By default they are accepted while the dimension is inactive.
- `exclusiveMaximum` (range dimensions) makes the brush half-open, `minimum <= value < maximum`,
  so adjacent brushes partition a domain. The default interval is inclusive on both ends.
- The `count` view publishes a one-row `uint32` count of live rows that pass the view's effective
  selection, without compacting row identifiers.

Mask composition splits into intermediate masks when the dimension count exceeds the device's
storage-buffer binding limit, so the number of dimensions is not bounded by one compute pass.

## Skipping unchanged dimensions

By default every node runs on every encoding, because the contract lets you rewrite source buffers
between encodings. For large tables set `skipUnchangedDimensions: true`: each node then records which
selections (and which source contents) it depends on and runs only when one changed, so a brush costs
work proportional to what depends on it, as in the original crossfilter.js.

```ts
const filter = new GPUCrossfilter(graph, {dimensions, views, skipUnchangedDimensions: true});

filter.setRange('value', [20, 40]); // re-runs the value predicate, the composed masks that include
// it, and the views that include its brush; the value histogram (self-excluding) is not recomputed
filter.invalidate('value'); // after rewriting the value input buffer
filter.invalidate(); // after rewriting any other source: group keys, liveMask, other view inputs
```

- Source buffers, group keys and values, `liveMask`, and view domains are assumed constant until
  `invalidate()`. A dimension id refreshes that dimension's predicate and its own histogram.
- Masks become persistent buffers, and view outputs are rewritten only when they can change.
- Change tracking is recorded when the graph evaluates each node's condition, that is, at encode
  time. If an encoding is never submitted, the skipped nodes' outputs stay stale until the next
  change; call `invalidate()` to refresh everything.

## Histogram cubes for very large tables

For tens of millions of rows, set `histogramCubes: true` and make one histogram view active. This
follows the Falcon system (Moritz, Howe, Heer, "Falcon: Balancing Interactive Latency and Resolution
Sensitivity for Scalable Linked Visualizations", CHI 2019).

```ts
const filter = new GPUCrossfilter(graph, {dimensions, views, histogramCubes: true});

filter.activateView('value-histogram'); // its dimension becomes the active dimension
filter.setRange('value', [20, 40]); // every other histogram now reads a cube: O(bins), not O(rows)
filter.deactivateView(); // all views return to their exact path
```

While a view is active, the GPU builds for every other histogram view `V` an exact integer cube of
the rows that pass every selection except the active one (and except `V`'s own when `V` excludes it),
by (active bin, `V` bin), prefix-summed over the active bins. A brush on the active dimension is then
`cube[last] - cube[first - 1]` per bin. Cubes are rebuilt when the activation or a non-active dimension
changes, and an unbrushed active view keeps the exact path.

- The active view must be a histogram of a `range` dimension, with that dimension's own `input`
  object, and not `exclusiveMaximum`. Views that are not histograms keep the exact path.
- The brush snaps outward to the active view's bins. Results equal the exact path for edge-aligned
  brushes: the lower endpoint is a bin's lower boundary and the upper endpoint lies in the last
  included bin (for integer data, one below the next boundary). Rows outside the active view's domain or
  edges are not counted.
- Memory is `activeBins * linkedBins * 4` bytes per linked view; construction throws when a cube
  exceeds the device's storage-binding limit. GPU-resident `edges` must be valid and increasing.
- Source rows are assumed constant until `invalidate()`.
- When every view is a histogram and no public mask exists, the masks that depend on the brushed
  dimension stop updating while cubes serve its brush (`getDimensionMask()` and `mask` are stale until
  `deactivateView()` or a clear). With other views, those passes keep running and only the histogram
  passes are saved.

## Core concepts and data model

- Dimensions own active range or rectangular selection state.
- Views consume every applicable dimension and may exclude their own dimension while brushed.
- Masks remain source-row aligned; compact visibility outputs retain stable source identifiers.
- Chunked inputs preserve their ordered source topology.
- The application owns graph compilation, encoding, submission, rendering, and compact readback.

## Operations and API index

| Surface | Responsibility |
| --- | --- |
| `GPUCrossfilter` | Declares linked dimensions and views in one command graph |
| `setRange()` / `setBounds()` | Updates control state without rebuilding graph topology |
| `clear()` / `clearAll()` | Removes one or all active predicates |
| `invalidate()` | Reports changed source contents when `skipUnchangedDimensions` or `histogramCubes` is on |
| `activateView()` / `deactivateView()` | Starts or stops serving linked histograms from Falcon-style cubes (`histogramCubes`) |
| Histogram view | Publishes a dense self-excluding distribution |
| Group view | Publishes dense grouped count, sum, minimum, maximum, or mean |
| Visibility view | Publishes stable compacted source IDs and a count |
| Mask view | Publishes source-aligned selection flags |
| Count view | Publishes the number of live rows passing the effective selection |

## Limits and compatibility

- GPUCrossfilter is experimental and WebGPU-only.
- Scalar dimensions currently use packed `float32`, `sint32`, or `uint32` values.
- Group keys and capacities are explicit; preserving chunks is not distributed execution.
- Dashboard layout, rendering, data import, and synchronization are outside the module.

## Attribution and feature comparison

GPUCrossfilter is inspired by [NVIDIA RAPIDS cuXfilter](https://github.com/rapidsai/cuxfilter), whose
contributors demonstrated how GPU-resident dataframes, coordinated visualizations, and linked
selections can make large-scale exploratory analysis feel immediate. cuXfilter itself acknowledged
the original [JavaScript Crossfilter](https://github.com/crossfilter/crossfilter); GPUCrossfilter brings
that family of ideas back into the browser with modern WebGPU execution.

We gratefully acknowledge NVIDIA and the RAPIDS contributors for that pioneering work. cuXfilter
is distributed under the
[Apache License 2.0](https://github.com/rapidsai/cuxfilter/blob/main/LICENSE). GPUCrossfilter is an
independently written, [MIT-licensed](https://github.com/visgl/luma.gl/blob/master/LICENSE) vis.gl
implementation; it does not copy or translate cuXfilter source code. It is not a CUDA port,
compatible Python API, feature-parity claim, or NVIDIA/RAPIDS successor project, and is neither
affiliated with nor endorsed by NVIDIA.

The comparison below reflects the official cuXfilter
[26.06 documentation](https://docs.rapids.ai/api/cuxfilter/stable/),
[DataFrame and dashboard API](https://docs.rapids.ai/api/cuxfilter/stable/api_reference/dataframe/),
[chart integrations](https://docs.rapids.ai/api/cuxfilter/stable/user_guide/charts/),
[multi-GPU guide](https://docs.rapids.ai/api/cuxfilter/stable/user_guide/dask-cudf-support/), and
[RAPIDS sunset notice](https://docs.rapids.ai/notices/rsn0060/).

| Capability | NVIDIA RAPIDS cuXfilter | luma.gl GPUCrossfilter |
| --- | --- | --- |
| Primary interface | Python `cuxfilter.DataFrame` and `dashboard()` in notebooks or Bokeh applications | TypeScript `GPUCrossfilter` controller inside a browser application |
| GPU platform | CUDA, cuDF, and NVIDIA GPUs | A browser-supported WebGPU adapter and luma.gl GPU command graphs |
| Source data | cuDF or Dask-cuDF dataframes; Arrow files and tables through `from_arrow()` | Caller-owned typed graph views and `GPUVector` chunks; no built-in dataframe, file, or Arrow importer |
| Linked interaction | Coordinated chart and widget selections across dataframe dimensions | Inclusive scalar ranges and rectangular brushes; active dimensions intersect |
| Input value types | cuDF-supported numeric, string, and datetime columns, subject to chart support | Packed `float32`, `sint32`, and `uint32`; categorical keys use `uint32` |
| Aggregation | GPU dataframe filtering, grouping, and chart-specific aggregation | GPU histograms plus dense grouped count, sum, minimum, maximum, and mean |
| Selection outputs | Filtered cuDF exports and queried dataframe indices | GPU-resident masks, stable compacted source identifiers, and visible-row counts |
| Visualization ecosystem | Bokeh, Datashader, deck.gl, Panel widgets, and table integrations | Renderer-independent GPU outputs; applications provide charts, maps, and scatterplots |
| Dashboard composition | Built-in chart collections, widgets, layouts, themes, and notebook/server presentation | Filtering primitives only; applications own layout, controls, rendering, and lifecycle |
| GPU data boundary | cuDF data lives with the Python/CUDA application; charts consume their required results | Uploaded source rows, masks, and aggregates stay on the local GPU; compact display readback is explicit |
| Scaling model | Single-GPU cuDF or distributed, multi-GPU Dask-cuDF | One WebGPU device; preserves source chunks without claiming multi-GPU execution |
| Deployment | Jupyter notebooks, Bokeh-backed applications, and documented multi-user deployments | A browser application; no Python process or dashboard server is required |
| Project status | Final RAPIDS release 26.06; subsequently sunset and archived | Experimental optional `@luma.gl/experimental/gpu-crossfilter` entry point |

The projects solve related interaction problems in different environments. GPUCrossfilter does not
provide cuXfilter feature parity, a migration layer, built-in chart integrations, or distributed
GPU execution.

See [GPU Primitives and Command Graphs](/docs/api-reference/experimental/gpu-core) for the
underlying compute infrastructure and [GPU Coordinate Projection](/docs/api-reference/experimental/gpu-project)
for another browser-native GPU data workflow.

## External mask dimensions

Register `{id, kind: 'external', mask}` to consume caller-written `uint32` flags, for example from
`GPUTransferMask`. The mask must match the other dimensions' row count and chunk topology. Zero rejects
and any nonzero value selects a row, including count-valued transfer masks. No predicate pass or source
upload is added. The caller retains ownership of the mask.

External dimensions start inactive and pass every row. Call `setExternalActive(id, true)` to include
one in the intersection and in other dimensions' self-excluding views; `setExternalActive(id, false)`,
`clear(id)`, or `clearAll()` bypasses it again. `setRange` and `setBounds` reject external dimensions.
After GPU work rewrites a mask, call `invalidate(id)` so `skipUnchangedDimensions` and histogram cubes
refresh their dependent results. Schedule the writer before the controller's composition nodes.
