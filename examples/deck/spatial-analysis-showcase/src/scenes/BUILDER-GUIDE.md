# Scene builder guide

You add a **scene**: a real dataset, a guided story, a full options panel, legends, readouts, a code
snippet and links into the reference docs, all driving one or more luma.gl analysis contributors.
You never edit a shared file. Scenes are found by file name, chapters are fixed in `chapters.ts`,
datasets are found by file name.

Reference implementations: `points/nature-density.scene.ts` + `nature-density.compute.ts` and
`weights/hot-spots.scene.ts` + `hot-spots.compute.ts`. Read them first.

**Cartography and the story card** are documented in [`CARTOGRAPHY-GUIDE.md`](./CARTOGRAPHY-GUIDE.md):
the twelve rules every story follows, ground presets, exact class tables and the hue registry,
gazetteers, annotations, legends, furniture, structured tooltips, charts, control kinds, step
headlines and live `{{readout}}` numbers, compare swipes, the time bar, flow/trail/relief layers
and the geometry helpers. Lint a chapter with `node scripts/check-cartography.mjs --strict
<chapter>` and colours with `node scripts/check-colours.mjs`.

## 1. Files

```
src/scenes/<chapter-id>/<scene-id>.scene.ts       metadata, options, story, legends, snippet, create()
src/scenes/<chapter-id>/<scene-id>.compute.ts     the GPU work (graphs, buffers, layers) - imported by create()
src/scenes/<chapter-id>/<scene-id>.md             optional narrative, see "Story in markdown"
src/data/datasets/<dataset-id>.dataset.ts         only when you add a dataset (see src/data/README.md)
scripts/data/<dataset-id>/                        build script for that dataset
```

- `<scene-id>` is kebab-case and **must equal** the `id` in the file (the registry warns otherwise).
- The folder is your chapter id from `chapters.ts` (points, joins, geometry, weights, statistics,
  regression, interpolation, cells, networks, flows, movement, time, terrain, hydrology, raster,
  dataframe). The scene's `chapter` field must match it.
- The scene file is loaded by the gallery to read titles and summaries, so keep it light: **no static
  imports of luma.gl or engine code** in `*.scene.ts` (type-only imports are fine). Import the compute
  module inside `create`: `create: async ctx => (await import('./foo.compute')).createFoo(ctx)`.
- Scenes never touch the DOM. Everything the user sees comes from the declarative fields.

## 2. The contract (`scene.ts`)

```ts
export default defineScene<MyOptions>({
  id, title, chapter, order, summary,   // order sorts within the chapter
  contributors: ['GPUFoo', 'GPUBar'],   // class names; linked to #/reference/<Name> and back
  datasets: [{id: 'chicago-nature', role: 'points'}],
  initialView: {longitude, latitude, zoom, pitch?, bearing?},
  options: [...],                       // see section 4
  story: [...],                         // see section 3
  legends: state => [...],              // see section 5
  readouts: [...],                      // see section 6
  snippet: state => `...`,              // see section 7
  about: {what, why, howToRead},        // optional markdown, recommended
  create: async ctx => instance
});
```

`create(ctx)` receives `ctx.device`, `ctx.options` (live), `ctx.datasets.get(id)`, `ctx.signal`,
`ctx.theme()`, `ctx.getViewport()`, `ctx.requestLayers()`, `ctx.setReadout(id, value)`,
`ctx.setLegendExtent(legendId, [min, max])`, `ctx.setStatus(text)`, `ctx.setMapDragEnabled(bool)`,
`ctx.forceSynthetic`, plus (section 6b) `ctx.setOptions`, `ctx.flyTo`, `ctx.getViewState`,
`ctx.refreshLegends`, `ctx.setLegendData`, `ctx.setChart`, `ctx.refreshTooltip`, `ctx.limits`.
It returns an instance:

```ts
{
  getCompiledGraphs(): CompiledGPUCommandGraph[],   // every graph you encode (drives Under the hood)
  encode(commandEncoder, frame),                    // write parameter buffers + compiled.encode(...)
  getLayers(): Layer[],                             // layers bound to your output buffers
  setOption?(id, value, state),                     // an option changed
  onAction?(id, state),                             // a button was pressed
  onThemeChange?(theme),
  getTooltip?, onClick?, onDragStart?, onDrag?, onDragEnd?,
  destroy()                                         // resources.destroy()
}
```

Rules that keep the "compile once" proof honest:

- Never compile, submit or synchronously read back inside `encode`. Compile in `create` or in
  `setOption` for a compile-time option. Prefer compiling every variant up front and switching.
- A `compile` option that really rebuilds returns new graph objects from `getCompiledGraphs()`; the
  shell counts them as rebuilds. A `param` option must never change graph identity.
- Create every GPU resource through `SpatialAnalysisResources` so `destroy()` frees it in order.
- Read small values back with `SummaryReader` (ring tickets), never with a blocking read.
- Project lon/lat with `dataset.projectColumn('position')` and use `dataset.defaultOrigin` as the
  layer `coordinateOrigin`. Positions are planar meters.
- `ctx.options` is a live view. Read it in `encode`/`getLayers`; use `setOption` to react (write a
  buffer, rebuild, `ctx.requestLayers()`).

Engine pieces you can import (relative paths from your scene folder):
`../../engine/layers` (`SpatialAnalysisPointLayer`, `SegmentLayer`, `RasterLayer`, `PolygonLayer`),
`../../engine/polygon-buffers`, `../../engine/flow-layer`, `../../engine/trail-layer`,
`../../engine/relief`, `../../engine/stage-fader`, `../../engine/draw-order`, `../../cartography/*`
(grounds, class-table, hue-registry, gazetteer, anchors, breaks, live-text, ... see
CARTOGRAPHY-GUIDE section 16),
`../../engine/resources`, `../../engine/graph-buffers` (`importGraphBuffer`),
`../../engine/summary-reader`, `../../engine/vector-timing`, `../../engine/mode-kernels`,
`../../engine/projection` (`LocalMetricProjection`, `createSeededRandom`),
`../../engine/builders` (road networks, polygon sets, elevation rasters), `../../engine/ramps`.
If you need a bespoke layer, put it next to your compute file.

## 3. Story

Four to six steps. Each step:

```ts
{
  id: 'kebab-id',                 // used in the URL
  title: 'A sentence-style title',
  body: 'Markdown. `GPUFoo` mentions auto-link to the reference.',
  evidence: 'What the reader can see or measure: **{{hotCount}}** hot cells.',
  caveat: 'The claim stops at the sampled dates; this is not a forecast.',
  camera?: {zoom, latitude, longitude, pitch, bearing, transitionMs},  // flies when the step opens
  options?: {radius: 900},        // set when the step opens
  controls?: ['radius', 'ramp'],  // controls shown inside the step card, under the text
  readouts?: ['hotCount'],        // readouts shown inside the step card
  callout?: {coordinate: [lng, lat], text: 'Short label'},
  highlight?: {readout: 'id'}     // emphasises a readout row
}
```

How steps apply: opening step N resets every option to its default, then applies the `options` of
steps 0..N in order, so any step is a valid deep link and "Back" restores the right state. Cameras
accumulate the same way. User tweaks last until the next step.

Narrative quality bar (this is what reviewers check):

1. **What** the tool computes, in plain words, with the class name linked.
2. **Why** an analyst cares: the decision or question it supports.
3. **How to read** the map: what a bright cell, a red point or a gap means; point at the legend.
4. **What each option does**, introduced when a step touches it ("Slide **Neighbourhood radius**
   below: ..."). List those options in the step's `controls` (one to four ids, max five, in the order
   the text uses them; `[]` for a pure "look" step): the shell renders exactly these controls under
   the step text, and every other option lives in the "All controls" tab. Refer to a control by its
   exact label in bold and say "below", never "in the Options panel". A listed control must not be
   disabled in the step's state. Put the numbers the text cites in `readouts`.
5. **Evidence before interpretation**: use `evidence` to name the observation that supports the
   headline, preferably with a live `{{readoutId}}` or a chart shown in `readouts`. Use `caveat` to
   state the most important uncertainty, data boundary or plausible alternative explanation.
6. Real data, honest limits: broader dataset or method context belongs in `about`.

Open with the question, end with something to try. Keep the body to two short paragraphs; the
structured evidence and caveat keep claims scannable without burying them in prose.

### Story in markdown

For long narratives, write `<scene-id>.md` with one `## Step title {#step-id}` per step and import it:

```ts
import narrative from './nature.md?raw';
import {storyFromMarkdown} from '../story-markdown';
story: storyFromMarkdown<MyOptions>(narrative, {
  'step-id': {camera: {zoom: 13}, options: {radius: 900}}
})
```

## 4. Options

Every showcase-worthy option of every contributor you use must be reachable. Check the TSDoc and
the reference page of each contributor and expose: statistics and variants, radii, thresholds,
significance levels, seeds, permutations, kernels, weight transforms, units, classification schemes,
colour ramps, and display toggles. Skip only options that make no visual sense.

```ts
{kind: 'slider', id: 'radius', label: 'Neighbourhood radius', apply: 'param',
 min: 100, max: 1500, step: 25, default: 400, unit: 'm', help: 'What it does and why you would change it.'}
{kind: 'select', id: 'ramp', label: 'Color ramp', apply: 'param', default: 'viridis',
 options: [{value: 'viridis', label: 'Viridis', help: 'optional per-choice help'}]}
{kind: 'toggle', id: 'fdr', label: 'FDR correction', apply: 'compile', default: false}
{kind: 'range', id: 'window', label: 'Time window', apply: 'param', min: 0, max: 24, step: 1, default: [0, 24]}
{kind: 'button', id: 'measure', label: 'Time both strategies'}
```

- `apply: 'param'` is a buffer write (no recompile). `apply: 'compile'` rebuilds or switches a
  compiled graph; the panel shows a "rebuild" badge. Be accurate: the badge is a promise.
- `help` is mandatory in practice: one or two sentences saying what the option does.
- `group` puts controls under a heading. `disabledWhen: state => ...` greys out options that do not
  apply. Option state is `O`; ids are keys of `O` (buttons carry no state).
- Option state is serialised into the URL (`?o=radius:900~fdr:1`); only non-default values appear.
- Slider values you need formatted specially: `format: value => ...`.

## 5. Legends

`legends(state, data?)` returns one entry per visual encoding, recomputed from the state:

```ts
{kind: 'ramp', id: 'density', title: 'Trips per cell', ramp: state.ramp, extent: 'gpu', sqrtScale: true, unit: 'trips'}
{kind: 'ramp', title: 'Residual', ramp: 'diverging', extent: [-3, 3], labels: ['below', 'above']}
{kind: 'categories', title: 'Class', entries: [{color: [r, g, b, a], label: 'High-High'}], note: '...'}
{kind: 'size', title: 'Radius ~ spaces', entries: [{radiusPixels: 3, label: '2'}, {radiusPixels: 8, label: '24'}]}
```

- A ramp legend and the layer's `colormap` must use the same ramp name. Both come from one table
  (`engine/ramps.ts`): `grayscale viridis magma inferno cividis diverging`. Use `diverging` with a
  symmetric `valueRange` for values with a meaningful zero. Do not hard-code RGB gradients.
- `extent: 'gpu'` shows what you report with `ctx.setLegendExtent(id, [min, max])` after reading the
  contributor's `extent` output with a `SummaryReader` (see `density.compute.ts`). Report once the
  camera settles, not every frame.
- Categorical colors: use the same RGB list for the layer `palette` and the legend entries.
- Match `sqrtScale` between legend and layer. Always give a `unit` or a meaningful title.
- `legends(state, data)` also receives `data`, everything you stored with
  `ctx.setLegendData(key, value)` (class breaks read back from the GPU, a loaded range, labels).
  Storing data, `ctx.refreshLegends()` and `ctx.setLegendExtent` all re-run `legends`. Do not abuse
  `setLegendExtent` to force a refresh.
- Ramp legends take `scale: 'log'` (decade ticks `1`, `10`, `10^2` from a positive `extent`, used
  with `extent: 'gpu'` or a fixed extent) and `colors: [[r,g,b], ...]` for a custom gradient
  (for example wind speed) when no named ramp fits. Custom `colors` must match the layer's own colors.

## 6. Readouts and charts

Declare `readouts: [{id, label, format?, help?, kind?, layout?}]` and call
`ctx.setReadout(id, value)`. `format` is `text | integer | decimal | percent | milliseconds | bytes |
meters` for numbers; strings are shown as given. Readouts are for numbers an analyst would quote:
counts per class, global statistics, overflow flags, timings. Fill them from `SummaryReader`
results; do not block.

- **Multi-line text.** A string containing `\n` switches the row to a block: label on top, value in a
  monospace block below that keeps line breaks. Force it with `layout: 'block'` (or `'inline'`).
- **Charts.** Declare `{id: 'kcurve', label: 'K(d)', kind: 'chart'}` and call
  `ctx.setChart('kcurve', data)` (`null` clears). Charts are inline SVG that follow the light and
  dark theme (`--chart-1`..`--chart-6` tokens); hover a line chart to read values. Data types live in
  `scenes/chart-types.ts` (re-exported from `scene.ts`); `x` defaults to the index, `NaN` breaks a line.

```ts
// Line chart with an envelope band and an expected curve.
ctx.setChart('kcurve', {
  kind: 'line',
  xLabel: 'distance (km)', yLabel: 'L(d) - d',
  series: [{label: 'observed', x: distances, y: observed}, {label: 'expected', x: distances, y: zero, dashed: true}],
  band: {x: distances, low: envelopeLow, high: envelopeHigh, label: '99% envelope'},
  markers: [{x: peakDistance, label: 'peak'}]
});
// Histogram: bins span xDomain; markers draw the observed statistic.
ctx.setChart('perm', {kind: 'histogram', values: binCounts, xDomain: [-3, 3], markers: [{x: observed, label: 'observed'}]});
// Categories: `labels`, `highlight` indexes in the accent color, the rest muted.
ctx.setChart('classes', {kind: 'bars', values: counts, labels: classNames, highlight: [2]});
// Sparkline: tiny axis-free trend, `highlight` is the index of the current value.
ctx.setChart('trend', {kind: 'sparkline', values: series, highlight: frame});
```

Charts re-render on each `setChart` call: call it when a summary arrives, not every frame.

## 6b. More services on `ctx`

- **`ctx.setOptions(partial, {notify?})`** writes option values from code (an animated playhead
  slider, click-to-select that fills a `select`). The panel and the URL update (coalesced per frame,
  so writing every frame is fine). `instance.setOption` is **not** called for scene writes unless
  `notify: true`, so a write cannot re-trigger the compile or rewrite you are in. Unknown ids are
  ignored; values equal to the current one are no-ops (a story step that sets an option already at
  that value does not notify `setOption`, so do not rely on step options to re-run work).
- **`ctx.flyTo(view, {transitionMs?})`** moves the camera (`view` is a partial `ViewState`; default
  is an instant jump, give `transitionMs` for a flight). `ctx.getViewState()` reads the camera.
- **`ctx.refreshTooltip()`** re-runs `getTooltip` for the last hover position. Call it when an
  async probe resolves while the pointer rests on the map.
- **Pointer modifiers.** `ScenePointerEvent` has `shiftKey`, `altKey`, `ctrlKey`, `metaKey` for
  `getTooltip`, `onClick`, `onDragStart`, `onDrag` and `onDragEnd` (shift-click to multi-select).
- **`ctx.limits`** `{maxStorageBuffersPerShaderStage, maxStorageBufferBindingSize, maxBufferSize}`
  as granted to the device. The shell requests `min(adapter, 16)` storage buffers per stage and the
  adapter's maximum binding and buffer sizes, so GWR grid indexes (10 buffers) and
  `GPURegionStatistics` with a grid and an id list work on most adapters. Disable an option and say
  why when `ctx.limits.maxStorageBuffersPerShaderStage` is still too small.
- **`SummaryReader`** callbacks that throw are logged with the scene id (`console.error`) instead of
  being swallowed; you still need to fix them, but you will see them.

### Large rasters

`SpatialAnalysisRasterLayer` draws one quad whose corners are projected exactly but whose in-quad
position is interpolated linearly in screen space; Web Mercator's scale varies with latitude, so over
about 100 km the cells drift away from the vectors drawn on top (about 40 km at 1,800 km). Pass
`tessellation: 48` (sub-quads per side, each corner projected exactly; 32-64 for a continental
extent; fixed at layer creation, use a new `id` to change it). Works for grid and hexagon binning.
The scene-local `B7TessellatedRasterLayer` can be replaced by `new SpatialAnalysisRasterLayer({...,
tessellation: 48})`.

### Tooling

`node scripts/check-story-steps.mjs [chapter ...]` checks that every story step lists explicit
`controls`, that listed controls and readouts exist, and that the first listed control is not
disabled when the step opens. Run it before you hand off a scene.

`node scripts/check-wgsl-identifiers.mjs [dir]` scans template strings for WGSL declarations that
use a reserved word or keyword as an identifier (`let target`, `fn f(from: u32)`, a struct member
named `filter`, `flat`). WGSL reports these only at runtime in the browser, so run it before you hand
off. Also: `earcut` is an app dependency with shared typings (`src/earcut.d.ts`); do not add a
local `declare module 'earcut'`.

## 7. Snippet

`snippet` is a string or `state => string`: a short, real, copy-pasteable excerpt of how the
contributor is wired for the current options. It should compile if pasted into an app that has the
named variables. Show the contributor constructor, the key options, and the per-frame write.

## 8. Datasets

Real data first. Declare the datasets you use in `datasets`; they are loaded (with the session memo
cache) before `create`. Read `src/data/README.md` for the manifest format and the `LoadedDataset`
API. To add a dataset:

1. Write `scripts/data/<dataset-id>/build.py` (+ README: source, licence, steps) that produces
   `public/data/<dataset-id>/manifest.json` and its files within the size budget. Raw downloads go
   outside the app tree.
2. Write `src/data/datasets/<dataset-id>.dataset.ts` (`DatasetInfo`: id, title, description, license,
   attribution, sourceUrl, approxBytes, bbox).
3. Declare it in your scene. The `#/data` page and the scene panel pick it up.

Beyond `columns`, `geojson` and `raster`, a loaded dataset gives you `loadRaster(name)` (any entry of
`manifest.rasters`, such as the Sentinel bands of `dixie-fire`), `loadRasterFile(file)` (more frames
with the primary raster's encoding, such as the hourly `gfs-wind` PNGs), `rasterSpecs` and
`loadTable()` (the dataset's CSV: `openflights` airports, `chicago-tracts` attributes). Raster
encodings: `terrarium`, `mapbox`, `uint8-classes`, `rg-uv-8bit` (decoded to interleaved `u, v`
floats), `float32-bin`, `uint8-bin`, `uint16-bin`, `int16-bin` (raw arrays: apply `raster.scale`
and `raster.offset`, `raster.noData` marks missing; stacks are `[depth][row][col]`, see
`raster.depth`). See `src/data/README.md`.

Datasets that are fetched from a remote host need a deterministic synthetic fallback registered in
`src/data/builtin-datasets.ts`; shipped datasets do not.

## 9. Typecheck, run, verify

From `examples/deck/spatial-analysis-showcase/`:

```sh
npx tsc -p . --noEmit                                          # must be clean
../../../node_modules/.bin/vite --port 5312 --strictPort --open false   # dev server (may already be running)
```

Open `http://localhost:5312/?scene=<scene-id>` (add `&data=synthetic` to skip the network). A scene
is working when `document.body.dataset.showcaseReady === 'true'` (20 frames encoded) and
`document.body.dataset.showcaseError` is unset. `globalThis.spatialAnalysisShowcase` exposes
`{sceneId, ready, error, rebuildCount, selectScene(id)}`. Check, in a real browser or headless Chrome
with `--enable-unsafe-webgpu`: every step, every option (param options keep **Rebuilds** at 0), both
themes, and a narrow phone viewport. Do not run `yarn build`, `yarn install` or commit.

## 10. Style and review checklist

- TypeScript, single quotes, semicolons, descriptive names, kebab-case files, Biome formatting.
- Colors work on light and dark basemaps (`ctx.theme()`, `onThemeChange`); avoid pure black or white.
- `destroy()` frees everything; switching scenes repeatedly must not leak buffers.
- Every contributor you list in `contributors` is used, and every one used is listed (exact class name
  as in the reference, so the backlink resolves).
- Summary is one or two sentences; story steps explain what, why, how to read, what to try.
- The scene works with `?data=synthetic`, or says clearly why not.
