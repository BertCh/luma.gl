# Computation record

## Overview

A `GPUComputationRecord` describes one recorded real execution. It identifies the adapter,
implementation variant, run, and creation time. Graphs and semantic metadata describe structure.
Metrics and captures are separate sidecars joined by `runId`, graph id, and node or resource id.
This lets measurements from two runs share a structural comparison without mixing their timings.

The schema is fork-local. It does not promise external compatibility. Record helpers are exported
from `@luma.gl/gpgpu`. Record construction and export operate on plain data without a GPU or DOM;
the capture and mask-transfer helpers below execute GPU work explicitly.

## Graph forms and identity

Graphs appear from most abstract to most concrete: `intent`, `lowered`, `compiled`, then `encoded`.
A record can omit forms. Each graph has a unique id without a colon. Node ids are unique within
that graph and may contain colons.

A node's `derivedFrom` array links to nodes in earlier graphs using `<graphId>:<nodeId>`.
Fusion can reference several origins. Unrolled copies can share an origin. Use
`formatGPUComputationNodeRef(graphId, nodeId)` and `parseGPUComputationNodeRef(ref)` to handle these
references. Parsing splits at the first colon and throws if either part is empty.

`namespace` is a slash-separated hierarchy. The empty string means top level. `loop` identifies
an unrolled copy by loop id and zero-based iteration. `dedupName` names the operation shared by
those copies. Consumers do not need to infer loops from node names.

Compiled resources carry ownership, byte size, allocation identity, and scheduled lifetimes.
Transient resources may share allocations when their inclusive lifetimes do not overlap.
Retained resources describe observation runs that preserve intermediate storage.

## Axes, key spaces and relations

Axes describe navigation dimensions, including lane, workgroup, row, and iteration. An axis can
name a coarser `parent` and a `keySpace`. Relations map one key space to another, with explicit
cardinality. Function relations describe known arithmetic mappings. Buffer relations identify
stored mappings. A mapping that was not captured is `unknown`; consumers must never guess it.
A node's `datum` binds its invocations to a key space and an optional relation.

## Refs, decisions, choices and workflow

The `refs` tables deduplicate formulas and source modules. Nodes reference their ids. Formula
symbols can bind to resources or shader variables. Formulas state whether they are derived or
curated. Source text is optional; its id and hash can remain when text retention is disabled.

`decisions` preserve the chosen implementation and every candidate, including rejected and
unsupported candidates with their reasons. `choices` describe analyst-selected parameters and
alternatives in analysis-stage order. A `workflow` lists graph invocations in execution order,
including the state or reason that selected them.

## Recording a command graph

`GPUCommandGraph` and `CompiledGPUCommandGraph.describe()` are GPU Core integration APIs. Enable
recording before compiling. Use `withRecordScope` to attach metadata to nodes contributed inside
the callback. The scope's namespace joins enclosing namespaces.

```ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  createGPUComputationRecord,
  getGPUComputationAdapter,
  validateGPUComputationRecord
} from '@luma.gl/gpgpu';

const graph = new GPUCommandGraph(device, {
  recording: {sourceText: true, retain: ['result']}
});
graph.withRecordScope({namespace: 'solve', op: 'relax'}, () => {
  // Contribute the real computation's nodes here.
});
const compiled = graph.compile();
const description = compiled.describe();
const record = createGPUComputationRecord({
  id: 'relaxation',
  adapter: getGPUComputationAdapter(device, 'portable'),
  domains: ['gpu-core'],
  graphs: [description],
  datasetVersion: 'example-v1',
  seed: 42
});
const report = validateGPUComputationRecord(record);
```

For programs, the recording integration is
`GPUProgramCompiler.compile(program, bindings, {recording: {}})`. The compiler and command graph
APIs produce the graph descriptions consumed by these pure helpers.

`createGPUComputationRecord` accepts graphs directly or `{graph, sources}` descriptions. It fills
empty axes, refs, decisions, and choices. Sources with the same id are merged; different retained
text for one id throws. An entry without text does not erase retained text.

`getGPUComputationRunId` returns `run-<8 hex>` from ordered graph and node ids, adapter backend and
variant, dataset version, and seed. Pass `runId` to override it. `createdAt` defaults to the current
ISO timestamp; pass it explicitly for reproducible fixtures. The id hashes those identity inputs,
not every field of the record.

`validateGPUComputationRecord` accepts unknown JSON data and returns `{valid, issues}` without
throwing. Issues contain a level, field path, and message. It checks references, loop bounds,
resource lifetimes, and required structure. Missing compiled resources, unbound formula symbols,
and mappings attached to unknown relations produce warnings.

## Sidecars and observer effects

`createGPUComputationMetrics({runId, graph, timing, encoding})` converts a timing report to
nanosecond metrics. GPU timestamps are relative to the earliest begin timestamp in that report.
Nodes without timestamps keep their CPU encode time. Conditional outcomes distinguish executed,
skipped, and GPU-gated indirect work. The helper does not read timestamps from the GPU.

The required `encoding` flags describe observer effects: `coalesced`, `timestamped`, `taps`, and
`lensTwin`, plus optional retained resource ids. Per-node timestamps disable compute-pass
coalescing. Retention and taps can also change memory use and scheduling. Compare measurements
with these differences visible.

`createGPUComputationCaptures({runId, snapshots, stamps, replays})` fills omitted arrays with empty
arrays. Captures contain caller-supplied values, component counts, key spaces, and readback or
replay provenance. This helper performs no readback or replay.

`validateGPUComputationSidecars(record, {metrics, captures})` checks schemas, run identity, graph
and resource references, timestamp order, and whole component counts. If a capture omits its graph,
resource and node references are resolved across the record's graphs.

## Capture

`captureGPUCommandGraph(compiled, props)` executes an observation run and returns
`{captures, stepCount, snapshotCount}`. Supply `encodeOptions` and a `views` object mapping capture
names to packed `GraphDataView`s or whole `GraphBufferHandle`s. Capture resources use the underlying
handle id; `graphId` defaults to the same `<compiled.id>-compiled` id as `describe()`.

```ts
const {captures} = await captureGPUCommandGraph(compiled, {
  encodeOptions: {parameters: undefined},
  views: {residual: residualView},
  at: 'publication',
  runId: record.runId
});
const sidecar = createGPUComputationCaptures({runId: record.runId, snapshots: captures});
```

`at: 'publication'` is the default: it captures after steps with declared publications.
`at: 'step'` defaults to one node per step; `budget` overrides the execution budget.
`at: 'end'` copies once after the last step. All modes use progressive publication scheduling.
`maximumSnapshots` sets arena capacity; its default is the planned step count, plus one for `end`.
Insufficient capacity fails before submission. `elementTypes` overrides inferred `f32`, `i32`, or
`u32` types; formats without a 32-bit scalar interpretation default to raw `u32` words. Interleaved
views must be packed explicitly before capture.

List observed transient ids in `recording.retain`. `getPhysicalBuffer` rejects aliased transients
that were not retained. Imported source buffers need `Buffer.COPY_SRC`. A retained or tapped
transient without it gets `COPY_SRC` on its physical allocation, and `describe()` marks the resource
`attrs.observerUsage: 'copy-src'` as an observer effect; graphs without retention or taps are
unchanged. The helper splits queue submissions,
which prevents coalescing across steps, and inserts readback copies. State these observer effects
alongside `GPUComputationEncodingFlags`; do not present capture timings as production timings.
`runId` belongs on the enclosing sidecar, not on each capture.

For manual observation, `new GPUSnapshotArena(device, {id, slots, capacity, maximumByteLength})`
allocates one `MAP_READ | COPY_DST` buffer. Each slot declares `name`, `byteLength`, `elementType`,
and optional `components`. `encodeSnapshot(encoder, sources, label?)` copies all slots to frame
`count` and returns that frame index. Sources are `{buffer, byteOffset?}` by slot name. Sizes and
offsets must be multiples of four. Missing sources, exhausted capacity, and an exceeded byte budget
throw errors naming the arena. The default byte budget is 64 MiB; decimate map-scale histories.

Submit all frames before calling `read({graph, resources, keySpaces})`. The arena maps once and
returns independent typed-array copies with `gpu-readback` provenance and ids `<arena>-<slot>-<frame>`.
Labels accept `iteration` (defaults to frame index), `afterNode`, and `label` (stored as capture
`meaning`). Call `destroy()` after reading; returned values remain valid. The graph capture helper
handles this cleanup automatically. For one intermediate value, `graph.addObservationTap(source,
{afterNode, sink})` inserts a hazard-ordered copy into an imported sink.

## transferMask

`new GPUTransferMask({id, source, relation, target, mode?})` transfers a GPU selection without
readback. Add its `getCommandNodes(graph)` result with `addGPUCommandNodes(graph, nodes)`.
The two nodes, `<id>-clear` and `<id>-scatter`, clear the target and scatter selected source keys.
All views must be packed `uint32`; the target must use storage separate from source and relation.

- `{layout: 'index', targets}` maps source key `i` to `targets[i]`. Target and source lengths match.
  `0xffffffff` means no target; all indices outside the target mask are ignored.
- `{layout: 'csr', offsets, neighbors}` maps source key `i` to
  `neighbors[offsets[i]..offsets[i + 1])`. Offsets have `source.length + 1` entries. Use one entry
  per source-to-target pair when counting selected keys; duplicate entries count separately.

Nonzero source flags are selected. `mode: 'any'` (the default) writes one to every reached target;
`mode: 'count'` counts selected incoming relation entries using atomic addition. Both modes use
bounded three-dimensional dispatch. Nodes declare resource hazards and a `GPUTransferMask` workload
with variant `<layout>-<mode>`. Recording adds operator attributes and hashed WGSL source refs;
`recording.sourceText` additionally retains shader text.

## Exports

- `toChromeTraceEvents(record, metrics, {flows})` returns Chrome Trace Event JSON for Perfetto.
  Events use microseconds even though `displayTimeUnit` is `ns`. Namespace depth selects a thread.
  RAW dependencies become flow pairs by default. Counter series use the earliest timestamp of a
  matching loop iteration when the series axis is `iteration` or the loop id. If no such timestamp
  exists, the counter event uses timestamp zero. Metadata includes the schema, run id, adapter,
  and encoding flags. With several metrics sidecars, adapter and encoding metadata are arrays in
  input order, and each sidecar retains its relative time origin.
- `toModelExplorerGraph(record)` returns one Model Explorer graph per form, with namespaces,
  incoming dependencies, and string attributes.
- `toGraphvizDOT(record, {graph, includeResources})` returns monochrome DOT. It selects the last
  graph by default. Namespace segments become nested clusters. Optional note nodes show resources.
- `printGPUComputationGraph(graph)` and `printGPUComputationRecord(record)` produce deterministic
  text for golden tests and logs. Their text formats are not an API promise.

All exporters preserve deterministic ordering and add no measurements or GPU work. Serialize the
Chrome trace object as JSON and load it into Perfetto. Serialize the Model Explorer collection
inside the collection array expected by its JSON loader.

## Production cost

Instrumentation is a compile-time opt-in. Omit `recording` for production graphs. Recording then
adds no nodes, passes, allocations, per-node metadata objects, or string building. The pure helpers
on this page run only when the application calls them. Timestamp collection, captures, retained
resources, and observation taps remain explicit choices.
