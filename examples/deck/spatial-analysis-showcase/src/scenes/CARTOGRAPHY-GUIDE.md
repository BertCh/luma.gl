# Cartography guide

The shared toolkit and the rules for turning a scene into a published-quality map: something a
cartographer at a newsroom, a national mapping agency or a GIS textbook would publish. Read
`BUILDER-GUIDE.md` first (the scene contract). Everything here is opt-in: a scene that declares
nothing still renders, and only the global restyle (fonts, tokens, legends, tooltips) changes it.

**Reference implementations** (open them before designing): `points/nature-density` (night
ground, additive points, continuous density, kernel ring, bandwidth lesson) and `weights/hot-spots`
(paper ground, exact class tables, interactive Gi* legend, compare step, national paper sheet).

**Lint:** `node scripts/check-cartography.mjs [--strict] <chapter ...>` checks the mechanical rules
below. Findings are warnings by default; a redesigned chapter must pass `--strict`. Waive a rule
only with a reason: `// cartography-allow: pitch (satellite altitude is real z)`.
`node scripts/check-colours.mjs` checks every ramp and class table for luminance order and
colour-vision distinctness.

Contents: 0 the twelve rules, 1 grounds, 2 colour, 3 hierarchy and layers, 4 typography and labels,
5 legends, 6 furniture, 7 the card and the story, 8 controls, 9 charts, 10 tooltips and linking,
11 time, compare and stages, 12 geometry and data helpers, 13 shared datasets, 14 consistency rules,
15 recipes, 16 API index.

---

## 0. The twelve rules (every story, every step)

1. **Ground follows the data, on purpose.** Pick one of six grounds per step (section 1). Labels
   are drawn above the data from a curated list; basemap labels are never left under data.
2. **Hue answers a question, and the question is the same everywhere** (the hue registry, 2.2).
   Harm and intensity warm; access and nature counts cool; deviation from expected PuOr (orange
   above); z-scores RdBu; change over time BrBG; direction and clock time cyclic `romao`; water blue
   only in terrain, hydrology and earth.
3. **No viridis, jet, spectral or rainbow as a default.** `magma`/`inferno`/`fire` only on dark
   grounds, trimmed `rampRange: [0.15, 1]`. A rainbow appears only as a labelled "before".
4. **Classed by default, with exact published ColorBrewer tables** (`makeClassTable`), 5 classes
   (7 max; diverging 5 or 7 with a real neutral class). Layer, legend, tooltip and histogram read one
   table. Show the classification at least once per chapter; keep breaks fixed across toggles.
5. **Normalise, and say the denominator in the legend** (`basis: 'per km²'`). Raw counts on
   unequal areas only as the deliberate "wrong map" step.
6. **Annotation is the main event.** At most 3 finding notes and 6 place labels per step (2 and 4
   on phones; the overlay enforces it). Places come from a gazetteer, snapped to the data.
7. **Hierarchy in three tiers**: subject (full ramp, opacity 0.85-1), supporting input (one hue,
   0.35-0.6, thin), context (`#8A949E`, 0.12-0.25, 0.5-1 px). Demotion ladder 1 / 0.5 / 0.25 / 0.15.
8. **Step grammar**: step 1 is the unanalysed establishing shot with a question; each middle step
   changes one thing; one step shows the parameter failing; the last step summarises and unlocks
   exploration. No text-only "limits" step: fold the limit into the step where it is visible. 5-6
   steps.
9. **Show the intermediate product and the comparison**: stage chips, swipe or hold-to-compare,
   before/after on a fixed legend. Never ask the reader to compare from memory.
10. **Teach the computation in the card**: the cost chip row and pipeline strip; engine readouts go
    in "Under the hood" (`hood: true`). 3-4 readouts per step.
11. **Pitch 0 unless the layer has real z** (satellite or flight altitude, arcs). A pitched flat
    raster is a tilted picture. Displaced terrain is not available this round.
12. **Every number in prose comes from a readout**: write `{{readoutId}}` in step markdown, or
    `liveText` for annotations. No literal counts, dates, percentages or coordinates.

---

## 1. Grounds (`cartography/grounds.ts`)

```ts
import {ground} from '../../cartography/grounds';
basemap: ground('paperCity'),                                   // scene default
story: [{..., basemap: ground('paperSheet')}]                   // a step switches ground (cross-fades 600 ms)
basemap: ground('night', {labels: 'above', labelPreset: 'places-only', suppressNames: ['Chicago']}),
```

| Preset | Use | Light theme | Dark theme |
| --- | --- | --- | --- |
| `paperCity` | classed, diverging, significance maps at city scale | positron recoloured: land `#F2F0EA`, water `#D6E0E6`, parks `#E5EBDF`, roads off below z12 then hairline; labels above (`orientation`); dim 0.3, desaturate 0.4 | dark-matter recoloured: land `#14181C`, water `#0A0E13`, parks `#1A2420` |
| `paperSheet` | national choropleths, matrices, diagrams | no tiles, sheet `#F4F1EA`, labels above | sheet `#14171C` |
| `night` | additive points, flows, trails, glow | dark-matter in BOTH themes, labels none, dim 0.2 | same |
| `abyss` | oceans, storms, drifters, great circles | no tiles, ocean `#0A1322`, Natural Earth land `#1A2231`, coast `#3B475E`, 10° graticule | same (forced dark) |
| `space` | satellites | `#04070D`, land `#10151E`, 30° graticule, equator, tropics and polar circles | same |
| `relief` | terrain, hydrology, raster with a DEM | paper `#F3EFE6`, no tiles, labels ours; draw the relief underlay (12.4) | paper `#14171C` |

- `ground(name, overrides)` merges; palettes merge per theme and field. `GROUNDS[name]` is the raw spec.
- `BasemapSpec` fields: `style`, `labels` (`basemap | above | none`), `labelPreset`
  (`orientation | places-only | water-only | streets | all | none`), `labelScale` (0.6-1.4; < 1
  also caps city names at 15 px above z11), `suppressNames` (hide basemap names you annotate
  yourself), `dim`, `desaturate`, `palette` (`background land water park road roadOpacity boundary
  boundaryOpacity graticule`), `world: 'land' | 'coast'` (Natural Earth on a tile-less ground),
  `graticule: true | {stepDegrees, opacity, widthPixels}`, `referenceLines: ['equator', 'tropics',
  'polar-circles']`, `transitionMs`.
- **`ctx.ground()`** is the luminance of the ground actually shown (`abyss` is dark in the light page
  theme). Choose ramp direction, outline and halo colours from it, never from `ctx.theme()`;
  implement `onGroundChange` and call `ctx.requestLayers()`. Map labels, annotations, legends and
  furniture follow the ground automatically (`data-ground` swaps the `--map-*` tokens). When the
  ground differs from the page theme the map shows a "Map: dark ground" chip.
- Switching ground inside a story is fine when the representation changes (points to choropleth);
  say why in one card line.
- Additive and multiply blending work only inside the Deck canvas: there is no multiply over the
  basemap. Relief and analysis-over-relief live in the same Deck pass (12.4).

---

## 2. Colour

### 2.1 Class tables (`cartography/class-table.ts`) - the default for anything classed

```ts
import {getClassTableLayerProps, getClassTableLegend, makeClassTable} from '../../cartography/class-table';
import {getClassBreaks, getGoodnessOfVarianceFit} from '../../cartography/breaks';

const breaks = getClassBreaks(values, 5, 'natural-breaks');           // compute ONCE, reuse across toggles
const table = makeClassTable({
  breaks, scheme: 'YlGnBu', ground: ctx.ground(), unit: 'observations', extent: [min, max],
  method: `Natural breaks (Jenks), GVF ${getGoodnessOfVarianceFit(values, breaks).toFixed(2)}`,
  noData: {label: 'No observations'}
});
new SpatialAnalysisPolygonLayer({...mesh, values, colormap: 'greys', ...getClassTableLayerProps(table)});
ctx.setLegendData('table', table);
// scene.legends(state, data):
getClassTableLegend(data.table, {title: 'Observations', basis: 'per km²', counts, interactive: true})
```

- `getClassPalette(scheme, n, {reverse, alpha, ground})`: the **published** ColorBrewer n-class
  table (sequential 3-9, diverging 3-11, qualitative), not a resample. Diverging tables run **low
  class first** (RdBu: blue low, red high). `ground: 'dark'` returns the authored dark table
  (magma/inferno families for YlOrRd, YlOrBr, YlGnBu; the dark-neutral RdBu-7) or a derived one whose
  lowest class keeps >= 3:1 against the ground.
- `makeClassTable({breaks, scheme | colors, reverse, ground, alpha, transparent, hatched, labels,
  unit, extent, method, noData, format})`; `transparent: [0]` makes the lowest class "nothing to
  say". `getClassIndexOf(table, value)`, `getClassLabel(table, i)` for tooltips.
- Significance: `makeGiStarClassTable({ground})` (`GI_STAR_BREAKS` ±1.65/1.96/2.58,
  `GI_STAR_LABELS`, translucent not-significant class), `getDivergingBreaks(center, steps)`.
- Layers take `classColors` (exact colours incl. alpha) with `classBreaks`; `classTable` on a layer
  is the same in one prop. The round-1 `classBreaks` + ramp path still works (it samples the
  9-stop ramp, darker at both ends; prefer tables).
- Classed rules: lowest class never near-white on paper nor black on dark (unless transparent on
  purpose); zero is its own class or transparent, never "no data"; always a no-data swatch; round
  break labels to data precision; unit in the title; method named; GVF shown for natural breaks.

### 2.2 Hue registry (`cartography/hue-registry.ts`)

One row per question, light and dark tables, exact hexes from SYNTHESIS 1.2:

```ts
import {getRegistryColors, HUE_REGISTRY} from '../../cartography/hue-registry';
const colors = getRegistryColors('harm', ctx.ground(), 5);   // PaletteColor[], low class first
HUE_REGISTRY.deviation.midpointMeaning                      // say it in the legend
```

Ids: `harm fireEffect people nature natureWeights share socioEconomic access remoteness load
similarity inequality rainfall elapsedTime waiting vegetationChange deviation zScore coefficient
direction party windSpeed hurricane slope slopeGentle visibility warnings`. Constants: `MAP_INK`
(ink, inkMuted, halo, water, waterLabel, context, signal, rule, noData, notSignificant per ground),
`OKABE_ITO_LIGHT` / `OKABE_ITO_DARK` (+ `_HEXES`), `OTHER_GREY`, `RACE_GROUP_COLORS` / `_LABELS`,
`HURRICANE_CLASS` / `_LABELS`, `SELECTION_INK`, `TERRAIN_EYE_GOLD`, `NO_DATA_COLOR`,
`NOT_SIGNIFICANT_COLOR`, `EFFORT_CAVEAT`, `getGroundColor(pair, ground)`, `hexToRgba`.
`check-colours.mjs` lists the registry entries that are weak under colour-vision deficiency
(harm, similarity, rainfall (tritan), elapsed time, vegetation change...): give those maps a second
cue (labels, outlines, order).

Hue guards: blue means water only in terrain, hydrology and earth; red never means "you" or
"error" and "the fire" at once; "other / between groups" is neutral grey; brightest = highest on
dark grounds, darkest = highest on light grounds.

### 2.3 Ramps (`engine/ramps.ts`) - for genuinely continuous surfaces

- Sequential on light: ColorBrewer single/multi-hue, `batlow`. On dark: `inferno`, `magma`, `fire`
  (CET-L3), `mako`, trimmed with layer `rampRange: [0.15, 1]` and legend `range: [0.15, 1]`.
- Diverging: `rdbu brbg puor piyg prgn` (+ `vik roma`); dark-centre `berlin`, `vanimo`. Always a
  symmetric domain and a `midpoint` + `midpointLabel` in the legend ("1.0 = expected").
- Cyclic: `romao`, `twilight` (start at north / 00:00). Over relief: `isolum` (CET-I1) or low chroma.
- New ramps: `fire lajolla bamako berlin vanimo mako twilight isolum`.
- `sampleRamp(name, t, reverse?, range?)`, `getClassColors(name, n, reverse?, alpha?, range?)`,
  `getRampCssGradient(name, {sqrtScale, reverse, range})`: order is normalise, sqrt, reverse, trim.
- `directionFor(ground, ramp)` returns `{reverse}` so "more" reads right on that ground;
  `RAMP_LOW_END[name]` says whether a ramp starts light or dark.
- Bivariate: `BIVARIATE_PALETTES.tealPink`, `.redBlue`, `.dkBlue`, `.brown` (Stevens, light grounds)
  and `.darkGround` (a proposal: simulate first). 3x3 only, always with a toggle back to the two
  univariate maps.
- Categorical: Okabe-Ito (light / dark lifted) for marks, Set2 for fills (alpha 0.70-0.78), at most
  7 hues + grey "other". Keep identity stable across steps and k: `matchStableColors(previous, next,
  paletteSize)` (centres, Hungarian) or `matchByOverlap(previousLabels, nextLabels, paletteSize)`
  (`cartography/stable-hues.ts`). Never colour nominal data by rank.

### 2.4 Uncertainty, no data, dark grounds

Preference order: mask or ghost the unreliable (alpha 0.15-0.25), value-by-alpha (layer channel
`alpha`, `alphaOutput: [0.25, 1]`, legend `kind: 'alpha'`), hatch (`hatchClasses`, `hatchNoData`),
bivariate estimate x reliability. One suppression rule per story ("fewer than 20 events"), stated
with the count hidden. Edge effects: fade the outer band (`getEdgeFadeWeights`) and draw a `frame`
"Data ends here". On dark grounds: additive points at alpha 0.05-0.15, clip at p98 and say so, no
halos on additive density, lowest visible class >= 3:1 against the ground or omitted.

---

## 3. Hierarchy and layers (`engine/layers.ts`)

| Element | Spec |
| --- | --- |
| Fill opacity | 1.0 on `paperSheet`; 0.88 on `paperCity`; 0.9 on dark; continuous surfaces 0.82 / 0.9; masked 0.15 |
| Polygon hairline | 0.3-0.5 px white at 0.63 (dark: `rgba(14,17,22,.5)`); never 0.8 px dark over thousands of polygons |
| Zone boundary | 0.9-1 px `#3A3F4B` 0.55-0.9 over a 1.6 px white casing (`outlineColor`/`outlineWidthPixels`) |
| Context lines | 0.5-1 px grey 0.12-0.3 |
| Analysis lines | 1.2-3.4 px by class or rank, low first, hot on top (`getSortedOrder` as `ids`) |
| Routes | ink core 5 px with a 9 px ground casing; a second route `#5E3C99` dashed |
| Flows | `SpatialAnalysisFlowLayer`: 0.5 + k·sqrt(w/wmax) px, one global max, heaviest last |
| Trails | `SpatialAnalysisTrailLayer`: 1.4-2.2 px, alpha (1 - age/tail)^2 |
| Dense points | `radiusPixels: DENSE_POINT_RADIUS_STOPS` (z<=10 1.2, z11 1.8, z12-13 2.6, z>=14 4.0) |
| Parameter geometry | dashed achromatic `ring` with the value on it + scale-bar tick at that distance |
| Z-order | ground, graticule, raster/choropleth, masks, context, subject, selected, heads, labels, annotations, furniture |

Props on every `SpatialAnalysis*Layer` (all optional, defaults render as before):

```ts
classColors / classTable          // exact class colours (alpha 0 = hidden class)
rampRange: [0.15, 1]              // trim the ramp (match the legend's `range`)
opacityStops: [[10, 0.4], [14, 0.9]]          // zoom stops (also radius/width/outline props)
highlightClasses: [5, 6], dimOpacity: 0.12    // legend isolate / class brushing (uniform, no buffer)
instanceChannels, channelStride, channels: {alpha, highlight, heading, width}   // ONE packed float buffer
alphaDomain, alphaOutput          // value-by-alpha mapping of the `alpha` channel
hatchClasses: [3], hatchNoData, hatchColor, hatchSpacingPixels, hatchWidthPixels
compareSide: 'a' | 'b'            // swipe compare (section 11)
blending: 'normal' | 'additive' | 'multiply'  // multiply = into what Deck already drew (relief)
colormap: 'rgba'                  // packed RGBA8 per row (pre-composited images)
```

- **Points**: `shape: circle | square | hexagon | triangle | diamond | star | cross | ring | chevron
  | arrow` (`chevron`/`arrow` rotate by the `heading` channel or `angleDegrees`), `fillOpacity`
  (hollow marks with an outline), `radiusMeters` + clamps, `sizeValues` (proportional symbols).
- **Segments**: `dashArray: [6, 4]` (phase restarts per segment), `cap: 'butt' | 'square' |
  'round'`, `widthMeters`, casing, per-row `width` channel (`widthDomain` -> `widthRange`, sqrt).
- **Raster**: `outlineClasses: {color, widthPixels}` draws class edges (basins, HAND, travel-time
  bands); `tessellation` for large extents (changing it no longer needs a new id).
- **Polygons**: `SpatialAnalysisPolygonLayer` + `buildPolygonMesh` + `createPolygonMeshBuffers`;
  outlines with `createPolygonOutlineBuffers`.
- **Draw order**: `getSortedOrder(values, 'ascending')` and `getShuffledOrder(count, seed)`
  (`engine/draw-order.ts`; "last drawn wins" bias in dot maps), uploaded as `ids`.
- **Binding ceiling**: devices may allow 8 storage buffers per stage. Defaults use point 6, segment
  7, polygon 5, raster 5; `instanceChannels` adds exactly one. Pack every per-row channel into it.
- Chapter subclasses that patch shader text keep working (all anchors verified); the WGSL now lives
  in `engine/layer-wgsl.ts`.

---

## 4. Typography and labels

### 4.1 Type (`cartography/typography.ts`, `styles.css`)

Source Sans 3 for everything, Source Serif 4 for cartouche titles, water and landforms (Google
Fonts, `display=swap`). Scale 11 / 12 / 13 / 14 / 16 / 20 / 28 / 40 (`--fs-xs .. --fs-hero`).
Tabular numerals everywhere; mono only for code and literal GPU values. Minimum 11 px for anything
the reader must read (10 for credits). Weight carries emphasis, tracking extent, case class, italic
hydrography and landform only; no italic uppercase. `MAP_TEXT_ROLES`, `getRoleFont(role)` and
`whenFontsReady()` serve canvas/SVG code.

### 4.2 Annotations (`scene.annotations`, `step.annotations`, `ctx.setAnnotations(key, list)`)

Drawn above the data with ground-coloured halos, placed by priority with Imhof candidates (NE, SE,
NW, SW), avoiding the map furniture and a 12 px edge, with hysteresis during flights. Budget per
frame: 3 notes + 6 place labels (2 + 4 under 600 px), lowest priority culled first.

```ts
{kind: 'point', coordinate, text: 'Montrose Point', rank: 'subject' | 'context', anchor: 'auto', detail: '412 records'}
{kind: 'area', coordinate, text: 'The Loop', size: 'small'}                  // spaced capitals
{kind: 'water', coordinate, text: 'Lake Michigan', size: 'large'}            // serif italic, water blue
{kind: 'landform', coordinate, text: 'Matterhorn', elevationMeters: 4478, marker: 'peak'}
{kind: 'note', coordinate, title: liveText('{share:percent} of trips', {share}), text: 'the three busiest segments'}
{kind: 'marker', coordinate, number: 1, text: 'Gold Coast'}                 // numbered; list them in the card
{kind: 'outline', rings, text: 'Loop', dashed: false}                        // district highlight, no fill
{kind: 'ring', coordinate, radiusMeters: 150, text: 'epsilon 150 m', dashed: true, geodesic: false}
{kind: 'dimension', from, to}                                                // two ticks + "2.4 km"
{kind: 'bracket', from, to, text: 'rupture', side: 'left'}
{kind: 'star', coordinate, text: 'M7.8 mainshock'}
{kind: 'line', coordinates, text: 'great circle', dashed: true}
{kind: 'frame', bounds: [w, s, e, n], text: 'Data ends here'}
{kind: 'arrow', from, to, text: 'prevailing wind'}
{kind: 'callout', coordinate, text}                                          // the single "look here" pointer
```

Every kind takes `priority`, `minZoom`, `maxZoom`, `tone` (`ink accent muted water signal`), `id`
(stable DOM: update a moving label every frame) and `timeRange` (shown while
`ctx.setAnnotationTime(t)` is inside). Tiers: orientation (basemap labels, 3-6), subject
(annotations, `priority` high), detail (`minZoom`). Copy: notes are sentences with a number and a
unit from a readout. Accent = the story's darkest data class or `--map-signal`. Over dense or
saturated data call `ctx.setAnnotationHalo('heavy')` (3 px halo; reset per scene).

### 4.3 Gazetteers and anchors (`cartography/gazetteer`, `cartography/anchors.ts`)

Never type coordinates into a scene. Every place is in a verified gazetteer (`CHICAGO`, `NYC`,
`MONTREAL`, `RANDSTAD`, `GRAND_CANYON`, `ALPS`, `DIXIE`, `US`, `WORLD`):

```ts
import {CHICAGO, CITY_FRAMES, labelsFor, nearestPlaceLabel, placeToAnnotation} from '../../cartography/gazetteer';
annotations: labelsFor(CHICAGO, ['lake-michigan', 'loop', 'montrose-point'], {loop: {minZoom: 10.2}}),
initialView: CITY_FRAMES.chicago,
// a finding note near the peak the GPU found:
{kind: 'note', coordinate: peak, title: `${formatCount(count)} records`, text: nearestPlaceLabel(CHICAGO, peak)}
```

Snap labels to the data: `snapToHighestCell(dem, place.lngLat, 160)` (peaks),
`snapToNearestPoint(positions, lngLat)`, `snapToNearestRow(rows, lngLat)`, `resolveAnchors(places,
resolver)`, `getPolygonLabelPoint(rings)`. Use `suppressNames` on the basemap for places you
annotate.

---

## 5. Legends (`scene.legends(state, data)`)

Map chrome (translucent, follows the ground), title `Name (unit) basis`, classes as butted swatches
with break labels at boundaries, counts, method note, histogram strip, no-data swatch, hatched
swatches, a value `marker` (the hovered feature). On phones the legend is a pill with a thumbnail.

```ts
{kind: 'ramp', title: 'Observation density', unit: 'per km²', ramp: 'inferno', range: [0.15, 1],
 extent: 'gpu', id: 'density', note: 'Top 2 % clipped', histogram, marker: hoveredValue}
{kind: 'ramp', title: 'Residual', ramp: 'puor', extent: [-3, 3], midpoint: 0, midpointLabel: '0 = as expected'}
getClassTableLegend(table, {title, basis, counts, interactive: true, layout: 'list'})
{kind: 'categories', title, entries: [{color, label, detail: '~ every 10 min', count, shape: 'dot'}], interactive: true, layout: 'list'}
{kind: 'matrix', title: 'Aspect x slope', rows, columns, colors, rowTitle, columnTitle}
{kind: 'cyclic', title: 'Hour of day', ramp: 'romao', labels: ['0', '6', '12', '18'], marks: [{at: 0.27, label: 'dawn'}]}
{kind: 'alpha', title: 'Rate, by reliability', colors, ends: ['unreliable', 'reliable'], steps: 4}
{kind: 'bivariate', ...}, {kind: 'line', ...}, {kind: 'size', layout: 'nested', ...}
```

- **Interactive** (`interactive: true` + an `id`): hover isolates, click locks (Shift adds), Esc
  clears; the scene gets `instance.onLegendFilter(id, classes | null)` and sets the layer's
  `highlightClasses` (and `ctx.requestLayers()`).
- Every `classes` legend has a unit, numeric breaks (or a `table`) and a no-data entry; every
  diverging ramp a `midpoint` (the lint checks).
- `formatLegendNumber(value, {maxSignificantDigits, thinSpace, unit})` from `shell/legend` for
  consistent numbers in tooltips.

---

## 6. Furniture (`scene.furniture`, `step.furniture`)

```ts
furniture: {
  title: {title: 'Where does Chicago notice nature?', subtitle: 'Observations per km², 2023',
          sample: '43,557 iNaturalist records', chips: ['Observer effort']},
  scaleBar: {units: 'metric', ticks: [bandwidthMeters]},     // tick at the step's parameter
  credit: true,                                               // dataset attributions + basemap credit
  caveat: EFFORT_CAVEAT,
  clock: {option: 'time', time: 'epoch-seconds', zones: ['UTC', 'America/Chicago']}
}
```

- Data- or camera-derived fields: `ctx.setFurniture({title: {...cartouche, sample:
  \`${formatCount(rows)} iNaturalist records\`}, scaleBar: {ticks: [bandwidthMeters]}})` merges last
  over the scene and step furniture (kept until replaced; `null` clears). Pass whole objects
  (`title`, `scaleBar`): the merge is per field.
- Cartouche on every step: line 1 the claim or question (<= 9 words), line 2 variable, unit,
  method, N, vintage; a standing `sample` line per dataset; `chips` when data are simulated,
  modelled or sampled. Phones show the title only.
- Scale bar on every step with a metric parameter, ticked at it; `units: 'nautical'` for ships and
  aircraft; hidden above 40° pitch. **National and global Web Mercator maps** (decision: no
  projection system this round; @deck.gl/core 9.4.0 has no CustomProjectionView):
  `...NATIONAL_FURNITURE` (scale bar at the mid latitude, "Scale at 37° N", hidden below z3.5,
  caveat) or `...GLOBAL_FURNITURE` (no scale bar, caveat, draw geodesic rings instead);
  `mercatorCaveat({latitudes: [25, 49], kind: 'area'})` computes the sentence; use hex or point
  symbols for any area comparison at national scale.
- North arrow only when the bearing is not 0 (none planned). Credits from `CREDITS`
  (`cartography/credits.ts`, licences included) via `joinCredits`.
- Clock: 40 px tabular, first zone large, UTC always named, optional progress strip.

---

## 7. The card and the story (`StoryStep`)

```ts
{
  id: 'bandwidth', title: 'Bandwidth is a choice',
  headline: 'A wider kernel merges the lakefront hot spots',   // <= 9 words, says something
  textAlternative: 'Density map of Chicago with a 600 m kernel: the lakefront reads as one ridge.',
  body: 'At **{{bandwidth}}** the three peaks merge into one ridge holding {{peakShare}} of records...',
  evidence: '**{{peakShare}}** of records now fall in the connected lakefront ridge.',
  caveat: 'The ridge depends on bandwidth and on where observers recorded points.',
  options: {sigma: 2.5}, optionsMode: 'fresh',                 // no hidden state from earlier steps
  controls: ['sigma'], readouts: ['bandwidth', 'peakShare'],
  camera: {bounds: [-87.75, 41.85, -87.58, 41.99]},            // fitted to the free map area
  basemap: ground('night'), furniture: {scaleBar: {ticks: [600]}},
  annotations: [...], compare: {labels: ['200 m', '600 m']}, stage: 'gaussian', diagram: {...}
}
```

- Card anatomy: progress, title, **headline**, cost chip row (`ctx.setCost({records, passes})` +
  GPU time + rebuilds), pipeline strip (`scene.pipeline`, step `stage`), body (<= 70 words, one bold
  control reference per control), structured `evidence` and `caveat` notes (both support
  `{{readoutId}}` live numbers), 1-2 controls (3 in the final explore step), consequence readout
  (one number with unit, a chart or a delta), Back/Next.
- Action links in markdown: `[Lincoln Park](action:fly?lng=-87.636&lat=41.921&z=14)`,
  `[top 5 %](action:set?threshold=0.95)`, `(action:reset)`, `(action:camera)`,
  `(action:step?id=summary)`, `(action:highlight?lng=..&lat=..)`.
- Camera: frame the subject per step (`bounds` or centre/zoom); the phone sheet is kept free
  automatically; 1200-1800 ms between subjects; pitch 0; never move the camera while playback runs.
  `ctx.fitBounds(bounds)` after a readback. Reduced motion jumps.
- Keys: `[`/`]` step, arrows step when the panel has focus, `R` resets options to the step's,
  `Esc` returns the camera, `?` shortcut sheet. "Modified · Reset to step" chip after reader changes.
- Pace: establishing shot, first intermediate product, one change per step, the parameter failing,
  sensitivity, summary + unlock. One step names the cartographic principle in the text and shows it
  in a comparison; the principle owner table is in section 14.

---

## 8. Controls (`scene.options`)

```ts
{kind: 'select', id: 'binning', display: 'segmented', apply: 'compile', options: [...]}     // 2-4 choices
{kind: 'select', id: 'stat', display: 'chips', ...}
{kind: 'preset', id: 'traveller', label: 'Traveller', presets: [
  {label: 'Walk', values: {speed: 1.4, mode: 'walk'}}, {label: 'Bike', values: {speed: 4.5, mode: 'bike'}}],
  resets: ['maxTime']}
{kind: 'slider', id: 'sigma', ..., describe: (v, s) => `${v} cells = ${Math.round(v * s.cellMeters)} m`,
 marks: [{value: 1.5, label: 'default'}], danger: [0.5, 0.75],
 autoSweep: {durationMs: 8000}}                                 // Play sweeps the parameter ladder
{kind: 'slider', id: 'k', display: 'stepper', format: v => `k = ${v}`, ...}
{kind: 'button', id: 'shuffle', label: 'New seed'}               // one-shot
{..., expert: true}                                               // folded under "Expert" in All controls
```

Readouts: `unit`, `emphasis: 'tile'` for the key figure, `hood: true` for engine detail,
`kind: 'chart'` (+ `placement: 'map'` for a screen-anchored inset).

---

## 9. Charts (`ctx.setChart(readoutId, data)`, `shell/chart.ts`)

No unicode sparklines or block characters anywhere. All kinds share `title`, `xLabel`, `yLabel`,
domains, formatters, `xScale/yScale: 'log'`, `markers`, `guides`, `bands`, `table` ("Show values"),
`description`, and `link: {option, label}` (the option value is the chart marker; clicking or
dragging the chart writes the option).

```ts
{kind: 'line', series: [{label: 'Observed', x, y}, {label: 'CSR', x, y: expected, dashed: true}],
 band: {low, high, label: '99 simulations'}, xLabel: 'Distance (m)', link: {option: 'radius', label: v => `r = ${v} m`}}
{kind: 'histogram', values: counts, xDomain: [min, max], breaks, classColors, now: observed, nowLabel: 'Observed I'}
{kind: 'bars', values, labels, horizontal: true, onBarClick: i => ctx.setHighlight(...)}
{kind: 'scatter', x, y, colorIndex, palette, quadrants: {x: 0, y: 0, labels: ['HH', 'LH', 'LL', 'HL']},
 fit: {slope: moranI, intercept: 0, label: `slope = I = ${moranI.toFixed(2)}`}, onBrush: indices => ...}
{kind: 'timeline', x: days, y: counts, window: [from, to], playhead, events: [{at, label: 'Landfall'}], bands, onScrub}
{kind: 'matrix', values, rows: 5, columns: 5, rowLabels, columnLabels, ramp: 'blues', marginals: 'both',
 diagonal: true, highlight: {row: 2}, onCellClick: (row, column) => ...}
{kind: 'lorenz', x: cumulativePopulation, y: cumulativeShare}               // Gini computed
{kind: 'forest', rows: [{label, estimate, low, high}], reference: 1, referenceLabel: 'No effect'}
{kind: 'dumbbell' | 'slope', rows: [{label, a, b}], aLabel: '2010', bLabel: '2020'}
{kind: 'rose', values: hourly, labels: ['0', '6', '12', '18'], colors, baseline: citywide}
{kind: 'stacked', segments: [{label: 'Forest', value: 25, color}]}
{kind: 'multiples', charts: [a, b, c, d], titles: ['100 m', '200 m', '400 m', '800 m'], columns: 2, highlight: 1}
{kind: 'diagram', width: 320, height: 120, description: '...', svg: '<line class="diagram-muted" .../>'}
{kind: 'sparkline', values, highlight: last}
```

Scatter quadrant labels are in quadrant order I-IV: top-right, top-left, bottom-left, bottom-right.
A step `diagram` (any chart, typically `kind: 'diagram'`) renders under the step text.

---

## 10. Tooltips, highlighting, picking

```ts
getTooltip: event => ({
  title: areaName, subtitle: 'Community area',
  rows: [{label: 'Density', value: formatCount(value), unit: 'per km²', swatch: table.colors[k], emphasis: true},
         {label: 'Rank', value: `${formatOrdinal(percentile)} percentile`}],
  spark: monthly, note: count < 20 ? 'Suppressed: fewer than 20 events' : undefined,
  anchor: labelPoint,                                  // polygons: anchor at the label point
  highlight: {kind: 'polygon', rings}                  // achromatic outline while hovered
})
```

- The shell renders its own card (theme-aware, flips at edges, tap-to-pin on touch, Esc). Strings
  still work. First row = the mapped value with its class swatch; second = rank or percentile.
- `ctx.setHighlight(highlights | null)`: link a chart bar, legend class or list row to map
  features (`{kind: 'point' | 'circle' | 'polygon' | 'line' | 'box', tone: 'signal', pulse: true}`).
  Selection is achromatic (ink core + ground halo); reader-placed inputs use `--map-signal`.
- CPU picking for custom or compacted data (`cartography/picking.ts`): `createNearestIndex(positions)`
  (`.nearest(lngLat, maxMeters)`, `.within(lngLat, radius)`), `getRasterValueAt(raster, lngLat)`,
  `getCellBounds`, `createFeatureLocator(geojson)`, `pointInPolygon`.

---

## 11. Time, compare and stages

- **Time bar** (`scene.timeline: {time, play, speed, window, format, bands, ticks}`) docks under the
  map, bound to the same options `engine/playback.ts` uses; feed it with
  `ctx.setTimelineData({domain, histogram, events})`. Add `furniture.clock`. Fixed class breaks
  across frames. Animation shows "that it moves"; a static small multiple or time strip proves the
  claim.
- **Compare** (`step.compare: {mode: 'swipe' | 'toggle', labels: [a, b], position}`): draw both
  versions as layers with `compareSide: 'a'` / `'b'` (often the same buffers with different breaks or
  ramps); the divider clips them; `ctx.getCompare()` / `onCompareChange` for anything else. Freeze
  the legend (one table) so a colour change is a data change.
- **Stages** (`scene.pipeline` chips with `show: {option, value}`, step `stage`): keep both stages'
  layers mounted and cross-fade with `createStageFader({stages, initial, onFrame:
  ctx.requestLayers})` (`engine/stage-fader.ts`) so Back is instant.
- **Flows and trails** (`engine/flow-layer.ts`, `engine/trail-layer.ts`): `SpatialAnalysisFlowLayer`
  (curved, width by sqrt value with one global max, arrowheads, end offsets, halo) and
  `SpatialAnalysisTrailLayer` (age-faded tails as a draw-time uniform, head dots, taper).

---

## 12. Geometry and data helpers (`cartography/*`, pure TypeScript)

| Module | Exports |
| --- | --- |
| `zoom.ts` | `getMetersPerPixel(zoom, lat)` (also `ctx.getMetersPerPixel()`), `metersToPixels`, `getZoomForMeters`, `evaluateZoomStops`, `ZoomStops`, `DENSE_POINT_RADIUS_STOPS` |
| `reference-geometry.ts` | `geodesicCircle` (antimeridian split, pole closure), `greatCircleArc`, `rhumbLine`, `graticuleLines`, `latitudeLine`, `REFERENCE_LATITUDES`, `metricGridLines`, `getNiceGridSpacing`, `squareMileFrame`, `haversineDistance`, `initialBearing`, `geodesicDestination` |
| `boundaries.ts` | `dissolveBoundaries(geojson, getGroup)` (interior group lines + exterior), `getStateBoundaries(counties)`, `toSegmentPairs` |
| `segments.ts` | `getLocalProjector(origin)` (same metres as `projectColumn`), `projectLinesToSegments`, `projectRingsToSegments`, `projectSegmentRows` |
| `masks.ts` | `rasterizePolygonMask(geojson, grid)`, `getEdgeFadeWeights(mask, grid, cells)`, `getBoundsFrame(bounds)` |
| `projection-notes.ts` | `getMercatorScaleFactor`, `getMercatorAreaFactor`, `mercatorCaveat`, `CONUS_LATITUDES`, `getScaleBarLatitude`, `NATIONAL_FURNITURE`, `GLOBAL_FURNITURE`, `mercatorMetersToGroundMeters`, `groundMetersToMercatorMeters` |
| `live-text.ts` | `liveText(template, values)` (`{name:percent}`, `:integer`, `:km`, `:area`, `:duration`, `:ordinal`, `:signed`), `formatCount`, `formatPercent`, `formatDistance`, `formatArea`, `formatRate`, `formatDuration`, `formatSigned`, `formatOrdinal` |
| `credits.ts` | `CREDITS`, `joinCredits` |
| `picking.ts` | section 10 |
| `breaks.ts`, `proportional.ts`, `polygon-mesh.ts` | round 1 (classification, proportional symbols, triangulation) |

### 12.4 Relief (`engine/relief.ts`)

`buildReliefImage(dem, {tints: 'alpine' | 'canyon' | table, ground})` composites a
multi-directional hillshade (cool shade `#5C6B8A`, warm light `#FFF4DE`, never grey) with stepped
hypsometric tints (`ALPINE_TINTS`, `CANYON_TINTS`) and aerial perspective on the CPU; draw it with
`createReliefLayerProps` as the first raster layer (`colormap: 'rgba'`), then the analysis above it
stepped at alpha 0.62-0.88 (transparent class for "nothing to say") or with `blending: 'multiply'`.
Contours: `CONTOUR_STYLE` (USGS browns). Pitch stays 0.

---

## 13. Shared datasets (new this round)

| Id | Contents | Licence |
| --- | --- | --- |
| `chicago-parks` | 1,139 parks, nature reserves, forest preserves, woodlands (`name`, `kind`, `areaKm2`); `green-mask.geojson` (dissolved) | ODbL, "© OpenStreetMap contributors (ODbL)" |
| `chicago-boundary` | `city.geojson` (dissolved city limit), `lake.geojson` (Lake Michigan, NE 10m), `land-mask.geojson` | City of Chicago terms; public domain |
| `natural-earth` | `ne_110m_{land,coastline,admin_0_countries,admin_0_boundary_lines,lakes}.geojson`, `ne_10m_{land,coastline,boundary_lines,lakes,rivers}_nl.geojson` | public domain |
| `dixie-perimeter` | the 2021 Dixie Fire final perimeter (from `poopdeck-wildfires`) | as its source |

Also already there: `us-states` (state outlines dissolved from counties). Grounds `abyss` and
`space` read `natural-earth` themselves. `catalog.load(id, signal, onProgress)` reports download
progress (the shell shows it).

---

## 14. Consistency rules (check these in every review)

1. One hue per question (registry); a different ramp needs a written reason.
2. Diverging registry: PuOr deviation (orange above), RdBu z-score or coefficient (red high), BrBG
   change (teal gain, brown loss), party colours only in election. Each diverging legend says in
   words what both ends and the centre mean.
3. Selection achromatic; reader-placed inputs ring-and-dot `--map-signal`; terrain eye gold.
   Parameters drawn the same everywhere: dashed achromatic ring in true metres with the value on
   the ring and a scale-bar tick.
4. Basemap declared per scene and per step; both themes defined; ground flips cross-fade.
5. Labels above data from a gazetteer; 3 notes + 6 places; snapped to data; duplicates suppressed.
6. Cartouche on every step (claim + variable/unit/method + standing sample line); credits; scale bar
   wherever a metric parameter exists.
7. Fixed breaks across toggles; legend frozen per step; rate and density maps on one set of breaks.
8. Numbers from readouts (`{{id}}`, `liveText`).
9. Cameras: same city frame across sibling stories (`CITY_FRAMES`: Chicago (-87.69, 41.84, z10.25),
   NYC (-73.97, 40.74, z10.9)); pitch 0; 1200-1800 ms; no motion during playback.
10. 5-6 steps; no text-only limits step; first step unanalysed; one change per step; a sensitivity
    step; summary + unlock; hidden state reset explicitly (`optionsMode: 'fresh'`).
11. 3-4 readouts per step, units on every number, engine readouts under the hood; no unicode charts.
12. Symbol sizing by zoom band; context thinner and lighter than subject.
13. Time: UTC plus local, zone named; fixed breaks across frames; animation for motion, static for
    the claim.
14. Sample line first: sampled or modelled data say so from step 1 (cartouche `sample` + `chips`).
15. Observer effort stated once per chapter with `EFFORT_CAVEAT`; taught at length only in
    nature-density.
16. Honesty chips for approximations ("Numerical note" block in the card, not a step).
17. Dark and light tables authored, never auto-inverted; both theme switches work.
18. Chapter reorders are `order` fields (SYNTHESIS 3.3).
19. Cross-chapter helper files get additive exports only; switch importers to this toolkit.
20. Shared datasets get additive columns or sibling datasets only.
21. Wholesome data only: no crime, police, enforcement, injury-victim or redlining data.
22. No performance caches as fixes; cost that teaches is shown as such ("compiled once, 6 cached").
23. Copy voice: the headline states the finding; `evidence` names what is visible or measured;
    `caveat` bounds the claim; the principle is named in the explanatory body.
24. Chapter summaries as decided in SYNTHESIS 3.1 rule 24.

Principle owners (teach once, cite elsewhere) are listed in SYNTHESIS 3.2: e.g. observer effort,
bandwidth and MAUP zoning in `points/nature-density`; classification in `statistics/choropleth-
classes`; small numbers in `statistics/rate-smoothing`; Gi*, LISA, FDR in `weights/hot-spots`.

---

## 15. Recipes

- **Dense point cloud to density (night):** `ground('night')`, additive dots
  (`DENSE_POINT_RADIUS_STOPS`, alpha 0.05-0.15), density in `inferno` trimmed `[0.15, 1]`, p98 clip
  in the legend note, kernel `ring` + scale-bar tick, peaks as `note`s with `nearestPlaceLabel`.
- **Classed choropleth (paper):** `ground('paperCity')` or `paperSheet` nationally, `makeClassTable`
  with a registry scheme, fill 0.88 / 1.0, white hairlines 0.4 px, state or district boundaries from
  `dissolveBoundaries` / `us-states` as a cased line, interactive legend, tooltip with rank.
- **Significance:** `makeGiStarClassTable`, not-significant translucent or hatched, `EFFORT_CAVEAT`
  where counts measure observers, scatter or histogram with the critical-value guides.
- **Diverging around a claim:** symmetric domain, `midpoint` + words, dark-centre table on dark
  grounds, `compare` swipe between the wrong midpoint and the right one.
- **Network / routing:** `ground('paperCity', {labelPreset: 'streets'})`, ink route with ground
  casing, origin as a `signal` ring-and-dot, isochrones as access classes (YlGnBu reversed),
  `outlineClasses` on travel-time rasters.
- **Ocean and global:** `ground('abyss')`, `GLOBAL_FURNITURE`, `geodesicCircle` rings, great-circle
  `line`s, ocean names from `WORLD`.
- **Terrain:** `ground('relief')`, `buildReliefImage`, stepped analysis above it, landform labels
  snapped with `snapToHighestCell`, pitch 0, profile chart instead of a pitched view.
- **Playback:** `ground('night')` or `abyss`, `SpatialAnalysisTrailLayer`, `scene.timeline` time bar,
  `furniture.clock`, no camera motion while playing, a static timeline chart for the claim.

---

## 16. API index (import paths from a scene folder)

- `../../cartography/grounds` - `ground`, `GROUNDS`, `getGroundTone`
- `../../cartography/class-table` - `makeClassTable`, `getClassPalette`, `getClassTableLayerProps`,
  `getClassTableLegend`, `makeGiStarClassTable`, `GI_STAR_BREAKS`, `getDivergingBreaks`,
  `getClassIndexOf`, `getClassLabel`
- `../../cartography/hue-registry` - `HUE_REGISTRY`, `getRegistryColors`, `MAP_INK`, `EFFORT_CAVEAT`, ...
- `../../cartography/stable-hues` - `matchStableColors`, `matchByOverlap`
- `../../cartography/gazetteer` - gazetteers, `labelsFor`, `placeToAnnotation`, `nearestPlace`,
  `nearestPlaceLabel`, `findPlace`, `CITY_FRAMES`; `../../cartography/anchors` - snapping
- `../../cartography/{zoom,reference-geometry,boundaries,segments,masks,projection-notes,live-text,credits,picking,typography,breaks,proportional,polygon-mesh}`
- `../../engine/ramps` - ramps, palettes, `directionFor`
- `../../engine/layers` - `SpatialAnalysis{Point,Segment,Polygon,Raster}Layer`;
  `../../engine/flow-layer`, `../../engine/trail-layer`, `../../engine/relief`,
  `../../engine/stage-fader`, `../../engine/draw-order`, `../../engine/polygon-buffers`
- Contracts: `../scene` (`StoryStep`, `TooltipContent`, `LegendSpec`, `OptionSpec`, ...),
  `../chart-types`.
