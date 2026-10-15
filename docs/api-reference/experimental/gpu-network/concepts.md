---
title: "GPU Network concepts"
description: "Conventions and execution model of GPU Network contributors."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Network concepts

<ExperimentalDocsTabs active="gpu-network-concepts" />

## Spatial contexts and length units

Distance operators (`GPUNetworkSnapping`, `GPUMapMatching`, `GPUNetworkKFunction`,
`GPUNetworkLineGraph`, `GPUNetworkNoding`, `GPUNetworkIsochrones` and
`addDriveTimeCatchmentRecipe`) accept planar azimuthal-equidistant (AEQD) frames from
`createGPULocalFrame(...).spatialContext`, including `units: 'meters'`, and reject Lambert
equal-area frames. Project positions into the declared frame before passing them to an operator.
Where supported, longitude/latitude degrees with `metric: 'great-circle'` are an alternative.
`GPUEdgeBundling` declares no measure: lengths are fractions of its work box, so either local
frame is accepted.

| Contributor | Longitude/latitude and kilometers |
| --- | --- |
| `GPUNetworkSnapping` | Great-circle distances in the context's length unit; `units: 'kilometers'` selects kilometers. |
| `GPUMapMatching` | Great-circle distances and distance parameters in the context's length unit, including kilometers. |
| `GPUNetworkKFunction` | Great-circle snapping; weights, distance thresholds, network length, K values and envelope in the context's length unit, including kilometers. |
| `GPUNetworkLineGraph` | Great-circle turn bearings accept meters or kilometers; angles and caller-owned travel costs are unchanged. |
| `GPUNetworkNoding` | Planar only; snap tolerance and output lengths are in the context's length unit. Longitude/latitude is rejected. |
| `GPUNetworkIsochrones` | Raster is planar; buffer radius and extent are in the context's length unit. Cell outlines accept longitude/latitude, including kilometer contexts, but only index positions in degrees. |
| `addDriveTimeCatchmentRecipe` | Longitude/latitude with metric `'none'` or `'great-circle'` uses great-circle snapping in the context's length unit, including kilometers. Travel costs keep the units of `topology.weights`. |
| `GPUEdgeBundling` | Longitude/latitude selects a degree-aware work box; kilometer contexts are accepted without physical distance measurements. |

Geographic distance units default to meters. `sphereRadius`, when supplied, is in the context's
length unit. Kilometer inputs must use kilometer-valued radii, thresholds and distance weights;
positions remain longitude/latitude degrees. Map matching's default `routeSlack` is 50 in the
context's length unit; supply `0.05` for a 50-meter allowance in a kilometer context.

Great-circle snapping searches all edges regardless of `candidateCapacity`, in meters or
kilometers alike.

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
  (`ids` and `status`: `count`, `overflow`, optional `requiredCount`). `status.count` is clamped to
  `ids.length` and can be an indirect draw instance count; `status.overflow` is rewritten every encoding.
- **Stable IDs.** Result IDs are the caller's `sourceIds[row]` (or tile IDs) when given and zero-based
  rows otherwise. Node and transient IDs are `${id}-<step>`, so two instances in one graph need
  different `id` props.
- **One import per buffer.** Import a buffer once (for example with `graph.importGPUData()` or
  `GPUParameterBuffer.importToGraph()`) and pass the returned view to every contributor that reads
  it, so the graph tracks hazards on one logical handle.
- **CSR convention.** Undirected graphs list both directions in the CSR. A reverse CSR, built with
  `GPUCOOToCSR`, gives "cost to reach" semantics.
- **One network input.** Every contributor takes the network as `topology: GPUNetworkTopology`
  (`offsets`, `neighbors`, `weights`, optional `nodePositions`); connectivity-only contributors
  accept a `GPUNetworkAdjacency`, the same object with optional `weights`. Its CSR part has the names
  and formats of `GPUSpatialWeights`, so a topology can be passed wherever spatial weights are read.
  Contributors that write a network (line graph, noding, induced subgraph) emit the same shape, so
  their output feeds the next contributor's `topology` directly.
- **Outputs in one bag.** Every GPU-written view is a field of `output`, and every status word
  (`converged`, `iterationCount`, `overflow`, `count`, `requiredCount`, and prefixed forms such as
  `pairOverflow`) lives in the bag's `status` member (`output.status.overflow`). Graph start
  nodes are `startNodes`/`startCosts`/`startCount`, cost bands are `costBreaks`, iteration caps are
  `maximum*Iterations`, and an active count is `number | GraphDataView<'uint32'>` (a number is
  compiled in, a view is read every frame).
- **Shared contracts.** Four shapes mean the same on every analysis page:
  - **Status.** Every GPU-written status word sits in the `status` member of the bag that owns it:
    `output.status.overflow`, `output.status.converged`, and `output.status.code` for model fits.
  - **Grid.** Raster and terrain contributors take `grid: GPURasterMetadata`, and their packers read
    the cell size from it (see [GPU Terrain conventions](../gpu-terrain/concepts.md#conventions)).
  - **Network.** Network contributors take `topology: GPUNetworkTopology` (`offsets`, `neighbors`,
    `weights`, optional `nodePositions`). Its CSR part can be passed as `GPUSpatialWeights` (see
    [GPU Network conventions](#conventions)).
  - **Polygons.** Polygons use the four-level `GPUPolygonGeometryPort`: `kind: 'polygons'`,
    `positions`, `featureOffsets`, `polygonOffsets` and `ringOffsets`. Contributors with a single
    geometry take the same fields flat, without `kind` (`GPUPolygonGeometryFields`). See
    [Shared ports](../gpu-spatial-analysis/concepts.md#shared-ports-status-and-planning).


## GPU execution model

How network contributors run on the device: the command graph they declare, the kernels it
compiles to, and the resources each pass reads and writes, drawn from a recorded real run.

{/* lens:execution-model */}

Inspect the recorded single-hop search, then contrast its rounds with workgroup-local chaining.

::lens{scenario=gpu-network.network-reachability rung=graph caption="Inspect the declared relax rounds, dependencies and retained observation buffers."}

::lens{scenario=gpu-network.network-reachability rung=kernel caption="Inspect atomic cost minima and the indirect frontier dispatch."}

::lens{scenario=gpu-network.network-reachability.chained rung=iterations caption="Chained rounds span multiple hops; queue stamps record global rounds, not the hops within them."}
