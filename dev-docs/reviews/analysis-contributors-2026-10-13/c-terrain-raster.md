# Stream C: correctness of gpu-terrain/** and the gpu-raster subdirs (2026-10-13)

Paths are relative to `modules/experimental/src/` unless stated. Method: three Sonnet slice reviews (C1 terrain
derivatives, curvature, geomorphons, TPI, relief, illumination, decode, features; C2 hydrology, flow field,
point horizon, viewshed and LOS, cost distance, distance field; C3 the raster subdirs), then a fourth sweep (C4)
of every iteration-capped loop plus RVT and GRASS reference checks. Every finding in section 2 was re-read in
code by the coordinator. Slice notes are in `work/C/C1.md` to `C4.md`; oracles are `work/C/c1/az.py` and
`flor.py` and `work/C/c3/tp.py`.

## 1. Verdict

The core formulas are right. These were all checked against references and found correct:
- Horn, Evans-Young, Zevenbergen-Thorne and Florinsky stencils (Florinsky 5x5 checked numerically), all curvature
  signs, hillshade, aspect convention, and the WGS84 N/M ground-scale maths.
- The geomorphon table and its GRASS tie rules, the Weiss classes, the exact int64 SAT, NOAA solar position,
  SVF and openness.
- D8, D-inf and MFD routing, Planchon-Darboux fill, Barnes flats, Strahler, HAND, the tiled relaxation's
  barriers and stamps, the sight-line curvature model, the Felzenszwalb EDT, and the isolines and isobands
  saddle rule with bit-identical seams.
- Zonal ordered-float min/max, and the Student-t p-value (max error 6e-6 against scipy).

The defects sit at the edges:
- One real geometric bug: azimuth is applied in pixel space on anisotropic grids.
- One budget-truncation result that impersonates a legitimate value: a cost-distance back-link of 0 means
  "source".
- One NaN-to-valid collapse: unresolved accumulation becomes "not a stream".
- One hash quantizer that saturates on projected coordinates.
- A library-wide pattern: every capped loop writes partial results into the same outputs as converged ones,
  with the flag optional.

I found no P0. No readback for control flow, no >65535 dispatch (every kernel goes through
`getBoundedDispatchLayout`, or a 2D 16x16 dispatch in rgb-decode), and no filtering of r32float (all field
sampling is manual bilinear on storage buffers).

## 2. Findings (verified)

| ID | Sev | Kind | Where (path:line) | Finding | Evidence | Fix (principled) | Effort |
|---|---|---|---|---|---|---|---|
| C-01 | P1 | correctness | `gpu-terrain/terrain-illumination/gpu-terrain-horizon.ts:287`, march `:1263-1264` (and `:1398`); `terrain-illumination/gpu-terrain-cast-shadow.ts:111`; digital lines in `terrain-horizon-sweep.ts` | A sector or sun azimuth becomes the pixel-space direction `(sin az, ±cos az)`. Only the step LENGTH is rescaled to ground (`length(direction * groundCell)`); the DIRECTION is not. So whenever ground cell x ≠ y the ray follows ground azimuth `atan2(sin az·cx, cos az·cy)`. That happens on every `'geographic'` grid off the equator (ratio ≈ cos lat) and on `'uniform'` grids with a non-square `cellSize`. Cast shadows, the shadow-mask and irradiance lookups, and the horizon, SVF and openness sectors all rotate toward north-south. Web Mercator is unaffected (N/M ≤ 1.007). No test covers an anisotropic horizon. | `return [snap(Math.sin(azimuth)), snap(northRowSign * Math.cos(azimuth))];` and `let groundStep = length(direction * groundCell);`. Oracle `az.py`: a 45° sector actually points at 41.0° at lat 30, 35.4° at lat 45, 26.6° at lat 60 and 18.9° at lat 70. | Build the pixel step from the ground direction: `normalize((sin az / cx, ±cos az / cy))`. The march kernels can do it per row from `groundCell`. The sweep and cast shadow take one global slope, so use the reference-row ratio (the same approximation the sweep already makes for `referenceStep`). Add a lat-60 geographic oracle spec. | M |
| C-02 | P1 | correctness | `gpu-raster/cost-distance/gpu-cost-distance.ts:45` (`DEFAULT_MAXIMUM_TIE_ITERATIONS = 8`), `:665-672` (tieProps, no `sweepAfterIteration`), back-link body ~`:497-505`; consumer `gpu-cost-distance-path.ts:155-165` | When `backLinks` is requested, the zero-friction tie phase runs 8 plain tiled iterations with no sweeps (about one 16x16 tile per iteration). A plateau more than about 128 cells deep from its strict entry keeps tie level +Inf. The back-link kernel then keeps `link = 0u`, which the contract defines as "reached as a source". `GPUCostDistancePath` stops at code 0 and returns a short, valid-looking path with `overflow` 0. `converged` exists only if the caller binds it, and the spec checks only the flag. | `link = 0u; ... if (isFiniteValue(level) && level > 0.5) {...}`, with no else branch for a non-finite level | Write `GPU_COST_DISTANCE_NONE` (or a distinct unresolved code) when the level is non-finite, never 0. Give the tie relaxation the cost phase's sweep schedule, and derive its default from `maxIterations`. | S |
| C-03 | P1 | correctness | `gpu-terrain/hydrology/terrain-flow-passes.ts:381-384` (streams kernel); docs `hydrology/gpu-terrain-flow.ts:257-263` | `streams` maps accumulation that is unresolved (NaN, when the default 64-round accumulation cap is hit) to 0, which means "not a stream". Unresolved cells are the downstream trunk, so the main river silently vanishes from `streams` and from what HAND and Strahler consume. The roadmap's "NaN tails on long rivers" (P2 tree contraction) is worse than stated, because the NaN becomes a valid-looking 0 downstream. | `isStream = (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u && value >= threshold; streams[...] = select(0u, 1u, isStream);` | Write a third state (e.g. `0xffffffff` unresolved) for NaN accumulation on valid cells, and have HAND and Strahler propagate it. Alternatively make `accumulationConverged` mandatory whenever `streams` is bound. | S |
| C-04 | P1 | correctness + perf | `gpu-raster/isolines/gpu-isoband-rings.ts:134` (default); quantizer `gpu-spatial-analysis/ring-assembly/gpu-segment-ring-assembly.ts:421-422`, bucket scan `:556` | `GPUIsobandRings` defaults `vertexTolerance` to an absolute 1e-4 world units. (a) Ring hashing uses `i32(floor(v * 1e4))`, which saturates for \|coord\| > 214,748, so every Web Mercator or UTM vertex of a band shares one bucket. The `while (sortedKeys[pos] == hash)` scan is uncapped, so successor and reverse lookups become O(n²), a watchdog risk. (b) On rasters with cells near 1e-4 (10 m lon/lat, unit-space grids), distinct crossings fall inside the tolerance and merge. The ring spec uses about 1-unit cells, so neither case is covered. | `vertexTolerance: props.vertexTolerance ?? 1e-4`; `fn ringQuantize(value: f32) -> i32 { return i32(floor(value * INVERSE_CELL)); }` | The isoband crossings are bit-identical by construction, so add an exact mode to `GPUSegmentRingAssembly` (`vertexTolerance: 0` hashes `bitcast<u32>`) and use it here. Independently, quantize relative to a per-input origin so the i32 cannot saturate (this affects every ring-assembly caller). | M |
| C-05 | P2 | correctness / API trap | `gpu-terrain/point-horizon/point-horizon-parameters.ts:181`; `gpu-point-horizon-visibility.ts:257-258` | `targetIgnoreDistance` defaults to 150 m. Every target within 150 m is classed VISIBLE (`stop <= 0`, no occlusion march), even behind a ridge. Viewshed and sight-line default to 0, so the two visibility families disagree on identical input. The doc says only "Meters before the target where the occlusion test stops". | `target[8] = settings.targetIgnoreDistance ?? 150;` and `let stop = distance - max(settings[8], settings[9]*distance); let ray = marchRay(..., stop, stop > 0.0);` | Default to 0, and offer the mt-image behaviour as an explicit preset. Document that targets inside the stretch are not tested. | S |
| C-06 | P2 | correctness | `gpu-raster/isolines/isolines-utils.ts:68-70` (`readCell` keeps ±inf); `gpu-isolines.ts:813` `t = (level - a)/(b - a)`; `isobands-wgsl.ts` likewise | ±inf count as valid samples. An edge with one infinite corner gives `t = NaN`, which emits NaN vertices, and an infinite saddle centre `(inf + -inf)` is NaN, which silently flips the saddle. `GPUIsobands` documents "unspecified geometry"; `GPUIsolines` says infinities "are values". | `// Compare bits: NaN is nodata, and infinities are kept as values.` | Extend "NaN means missing" to non-finite in every interpolating kernel (`isFiniteValue` in `readCell`), one rule library-wide (see pattern P-4). | S |
| C-07 | P2 | correctness / API trap | `gpu-raster/isolines/isobands-parameters.ts:65-79` vs `isolines-parameters.ts` | The isobands packer checks only that the extent is finite. A y-down extent `[minX, maxY, maxX, minY]` gives a negative `cellHeight`, which mirrors the geometry and flips ring winding: shells become CW and holes CCW in `GPUIsobandRings`. A zero-size extent collapses all vertices. Isolines rejects both. | `if (![minX, minY, maxX, maxY].every(Number.isFinite)) throw` is the only extent check; `target[9] = (maxY - minY) / settings.height;` | Share one `validateExtent` (`max > min`) across the isolines, isobands and sampling packers. | S |
| C-08 | P2 | correctness (precision) | `gpu-raster/distance-field/distance-field-kernels.ts:217-218` with `distance-field-parameters.ts:78-83`; `gpu-raster/isolines/gpu-isolines.ts:816-821`; `isolines/isobands-wgsl.ts:61-62`; `raster-sampling/raster-sampling-utils.ts:158-160` | Absolute world coordinates are handled in f32. Distance-field seeds snap with `floor((position - origin) * invCell)` after both operands are already f32-rounded. Contour and sample vertices are emitted as `minX + (gridX + 0.5) * cellWidth`. At 5e5 to 2e7 m (UTM, 3857) the ulp is 0.03 to 2 m, so seeds snap to the wrong cell on metre-scale grids and contour vertices are quantized by metres. Only `GPUPolygonRasterization` documents "use extent-local coordinates". | quoted lines | Make the extent origin a CPU float64 concern. Emit cell-space or origin-relative coordinates (vertices `gridX + t`), and take seeds origin-relative. Document the same rule everywhere. | S (doc) / M |
| C-09 | P2 | correctness (silent truncation) | Shared cause `gpu-raster/cost-distance/raster-relaxation.ts:142-235`. Instances: cost phase `gpu-cost-distance.ts:44,160-163`; fill `hydrology/gpu-terrain-flow.ts:218,242`; flats `terrain-flow-flats.ts:44-50`; streamlines `flow-texture/gpu-streamlines.ts:63-64,137`; profile peaks `terrain-features/gpu-profile-peaks.ts:283-291`; spike repair `terrain-decode/gpu-terrain-spike-repair.ts:107-119`; watersheds and HAND (`maxIterations` lowered or cyclic input); patch guards `raster-patches/patch-labels.ts:30-35` | Every capped or gated loop writes its partial result into the same outputs as a converged run, and the converged or overflow flag is an optional binding. Examples: finite upper-bound costs and fill heights; an unresolved flat that reads as a legitimate outlet-less `flat`; dropped streamlines and peaks; spike repair "nothing to repair"; patch fragments measured as real patches. Isolines (budget derived from capacity), isobands, contours, polygon rasterization and profile (mandatory overflow) are fine. | C4 table rows 1, 4, 5, 9, 10, 12-14, 20 | One rule (P-1): a capped loop must either make its converged or overflow output mandatory, or write a sentinel into the data that no converged run can produce. | M (across files) |
| C-10 | P2 | perf trap | `gpu-terrain/terrain-analysis/gpu-terrain-cumulative-viewshed.ts:132` | The default `observersPerDispatch = floor(2^22 / pixelCount)` targets about 4M invocations per dispatch regardless of work. Each march takes up to `max(w, h)` samples, so a 1024² march dispatch is about 4e9 samples, the watchdog regime the viewshed doc itself warns about. | `props.observersPerDispatch ?? Math.max(1, Math.floor(2 ** 22 / pixelCount))` | Size the batch by estimated samples (`observers · pixels · max(w,h) ≤ ~2^28`). The pyramid default is already a roadmap P1 item. | S |
| C-11 | P2 | ergonomics (binding budget) | `gpu-terrain/point-horizon/point-horizon-march.ts:1127` | `elevationValidity` is bound for every point-horizon kernel but never read: the WGSL tests NaN in `bilinearRelative`, and `elevationValidity` appears only in the binding list and docs. It still counts toward the 8-binding check, so `GPUPointHorizonProfile` with the pyramid and all four outputs throws even though 8 would suffice. | `grep elevationValidity point-horizon/*.ts` gives only the doc lines 702 and 1105 and the binding at 1127 | Drop the binding from `createPointHorizonSource` and update the doc at :702. | S |
| C-12 | P2 | perf trap | `gpu-raster/raster-sampling/gpu-raster-profile.ts` (`measure` and `cumulative` kernels) | Gain/loss, min/max and the vertex-distance walk run serially, one lane per path. One fine transect (1e5 to 1e6 samples) becomes a single-lane dependent loop near the watchdog. It is documented but unbounded. | `for (var sample = firstSample; sample < endSample; sample++) { ... gain += delta ... }` | Segmented scans (the float segmented scan is already wanted in the roadmap's GPU Core promotions); min/max via segmented reduce. | M |
| C-13 | P2 | correctness (reference parity) | `gpu-terrain/geomorphons/gpu-geomorphons.ts:378-379` (validation), `:334-335` (use) | GRASS `r.geomorphon` (`main.c:360-366`) ignores `flat_distance` when it is ≤ the skip distance or ≥ the search distance. Ours applies any `flatDistance > 0`. For `0 < flatDistance ≤ skip` it lowers the flatness threshold on every line of sight, so forms diverge from GRASS. | `flatDistance >= 0.0` is the only check | Reject, or treat as 0, `flatDistance ≤ skipRadius·cell` in the packer, matching GRASS, and document it. | S |
| C-14 | P3 | correctness (precision) | `gpu-terrain/terrain-illumination/gpu-texture-shading.ts:318` | The pyramid premultiply blurs absolute elevation in f32 and then differences levels, which gives about 1 mm of noise at 4000 m. The library's own `relief-visualization-utils.ts:92` anchors to avoid exactly this. | `numerator[...] = select(0.0, elevationValues[...], isValid);` | Subtract a per-grid anchor before premultiplying; it cancels in the band differences. | S |
| C-15 | P3 | correctness (u32 overflow) | `gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts:753-755` | `size * 100u <= validCount * MAXIMUM_COMPONENT_PERCENT` and `agree * 100u >= seams * P` wrap for components over 42.9M px or `validCount · P > 2^32` (> 43M valid px at P = 100), giving a wrong shift decision. | quoted | Compare in f32 ratio form, or bound the grid size in the constructor. | S |
| C-16 | P3 | numerics | `gpu-terrain/topographic-position/gpu-terrain-topographic-position.ts:312` | DEV variance is `d2/n − (d1/n)²` formed in f32 after the exact 64-bit sums. It cancels when the centre is far from the window mean at large n (about 6% error at R = 500). The class doc claims only the final division rounds. | `unsigned64ToFloat(centredSquare) / count - meanOffset * meanOffset` | Form `n·d2 − d1²` in wide integer arithmetic before converting, or document the R limit. | M |
| C-17 | P3 | perf | `gpu-terrain/terrain-features/gpu-terrain-summits.ts:357` | When the radius is clamped, every pixel does `atomicMax` on one address. Peak snap guards this with `isLeader`. | `if (isRadiusClamped(requested, cell)) { atomicMax(&overflowValues[...], 1u); }` | Evaluate once (row-0 cell plus settings) or reduce per workgroup. | S |
| C-18 | P3 | latent compile error | `gpu-raster/cost-distance/raster-relaxation.ts:278,294` | `groupSize` is capped at 32, but the bit is `${1 << groupOffset}u`, which for offset 31 emits `-2147483648u`. No caller passes `groupSize` above 4 today. | `Math.min(props.groupSize ?? RASTER_GATE_INTERVAL, 32)` | Cap at 31 or emit `((1 << g) >>> 0)`. | S |
| C-19 | P3 | docs | `gpu-raster/distance-field/distance-field-kernels.ts:106-123` | The `'exact'` mode is exact only for isotropic cells. Anisotropic keys are f32 `(sx·dx)² + (sy·dy)²`, so envelope crossings and tie order can be off by a cell near ties. | `getAnisotropicKey` | Document "exact on isotropic grids", or use integer keys scaled by a rational ratio. | S |
| C-20 | P3 | consistency | `gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight.ts:240` vs `gpu-terrain-viewshed.ts:418` | LOS uses the bilinear target elevation while viewshed uses the raw pixel; these differ for fractional targets. Only the cumulative doc states it. | quoted in C2 | Document it in the LOS TSDoc. | S |
| C-21 | P3 | perf (watchdog on bad input) | `gpu-raster/cost-distance/gpu-cost-distance-path.ts:157` | A single thread walks up to `CELL_COUNT` links, so a cyclic or corrupt link field spins for the full cell count. | `for (var walkIndex = 0u; walkIndex < CELL_COUNT; walkIndex++)` | Bound the walk by capacity and report `overflow`. | S |
| C-22 | P3 | correctness (degenerate) | `gpu-raster/isolines/gpu-isolines.ts:364-394` | When a corner equals the level exactly, zero-length segments are emitted and counted, and they become duplicate polyline vertices. Isobands drops the same case (`emitPieceEdge` `a == b`). | no counterpart in the isolines count and scatter | Drop segments whose two crossings coincide. Add a spec with an integer raster and integer levels. | S |
| C-23 | P3 | consistency (layering) | `gpu-raster/flow-texture/gpu-streamlines.ts:31`, `gpu-line-integral-convolution.ts:23`, `particle-advection/gpu-particle-advection.ts:21` | Shipped GPU files import their RNG purpose constants from the `*-cpu.ts` oracles, which puts the CPU code in the production import graph. | `import {STREAMLINES_SEED_PURPOSE} from './streamlines-cpu';` | Move the constants to `*-parameters.ts`; the oracles import from there. | S |
| C-24 | P3 | fragility | `raster-stretch/gpu-raster-stretch.ts`, `raster-zonal-statistics/gpu-raster-zonal-statistics.ts` (`guardIndex: false`, no `workgroupSize`); `raster-algebra/gpu-weighted-overlay.ts` and `gpu-raster-reclassify.ts` (barrier variants) | Workgroup arrays and strides hard-code 256 (or 128), relying on the implicit `DEFAULT_WORKGROUP_SIZE` in `utils/wgsl-kernel-nodes.ts`. | `array<u32, 256>`, `slot += 256u`, no `workgroupSize` prop | Pass `workgroupSize` explicitly wherever WGSL depends on it. | S |
| C-25 | P3 | ergonomics | `gpu-raster/raster-algebra/gpu-weighted-overlay.ts` (score body) | A table-mode layer on an overlay built without remap tables evaluates to NaN, so every cell silently becomes nodata. | `: 'remapped = getNaN();'` | Reject it in the packer (pass `hasTables`). | S |
| C-26 | P3 | ergonomics | `gpu-raster/raster-stretch/gpu-raster-stretch.ts:105,646` | `gamma` is `t ** gamma` (> 1 darkens), the inverse of the rio-color, GDAL and GIMP `t ** (1/gamma)`. It is documented. | `t = pow(t, gamma)` | Rename it to `exponent`, or invert it (breaking is fine). | S |
| C-27 | P3 | perf | `raster-zonal-statistics/gpu-raster-zonal-statistics.ts:264-275`, `raster-pyramid/gpu-raster-extrema-pyramid.ts` (`getTerrainElevationNodes(..., true)`) | `canonicalize = true` always adds an identity neighbourhood copy (read plus two writes), even for an already-canonical float32 buffer band with validity. | `terrain-analysis-utils.ts:139` passthrough only when `!canonicalize` | Pass through bands with no scale, offset or noData. | S |
| C-28 | P3 | docs | `gpu-terrain/terrain-illumination/gpu-terrain-horizon.ts:76-80,152-155` | The RVT conversion `(360 − A) mod 360` is correct: RVT direction angles run CCW from north, derived from `horizon_shift_vector` plus `np.roll`, and corroborated by `sky_illumination`'s `360 - shadow_az`. But the default 315 here equals RVT's 45, not RVT's default of 315. Also, `abs(cos)` differs from RVT's signed `cos^level` for odd levels. | RVT `vis.py` (C4) | Say explicitly that the default is not RVT's, or default to 45 for parity. Restrict `anisotropyLevel` to RVT's presets, or document the difference. | S |

## 3. Patterns (library-wide) and the rule that prevents each

- **P-1: silent loop truncation.** Partial output is written into valid outputs and the flag is optional. 10
  contributors in this slice (C4 table); the same shared loop is used outside the slice by `map-coloring` and
  `line-simplification`. Rule: *a capped or gated loop either makes its converged or overflow output mandatory,
  or encodes "unresolved" in the data with a value no converged run produces*. C-02 and C-03 are the two cases
  where the partial value is an actual legitimate code.
- **P-2: compass bearings applied as pixel steps.** 3 sites in this slice (`gpu-terrain-horizon.ts:287`,
  `gpu-terrain-cast-shadow.ts:111`, `terrain-horizon-sweep.ts`). Lighting vectors in derivatives, relief shading
  and solar irradiance are in ground space and are fine. Rule: *convert a bearing to a pixel step as
  `(sin/cx, cos/cy)` using the row's ground cell*.
- **P-3: absolute world coordinates in f32.** 5 sites here (distance-field seeds, isolines, isobands,
  sampling, ring quantizer); polygon rasterization documents it. Rule: *the extent origin lives on the CPU in
  f64; GPU geometry is cell-space or origin-relative, and any `i32(floor(v * k))` hash subtracts an origin*.
- **P-4: two definitions of "missing".** `0x7f800000u` NaN-only tests appear in 11 files (isolines, sampling,
  vector-field, raster-algebra, cost-distance grid, change detection, hydrology flats and passes, terrain
  features, distance field), while zonal and stretch use `isFiniteValue`. Interpolating kernels break on inf.
  Rule: *non-finite means missing for any kernel that interpolates or averages; one shared WGSL predicate*.
- **P-5: absolute literal tolerances and defaults that change semantics** (`vertexTolerance 1e-4`,
  `targetIgnoreDistance 150`). Rule: *a default tolerance is derived from cell size or ulp; a non-zero
  semantic default must match the sibling contributors or be a named preset*.
- **P-6: binding-budget pressure.** 11 kernels in the raster subdirs sit at exactly 8 storage buffers, and the
  headless device uses featureLevel `'max'`. C-11 burns a slot for nothing. Rule: *a node spec builds every
  contributor against default limits*; also grep bindings that are never read.
- **P-7: JS shifts interpolated into WGSL.** `${1 << n}u` appears at 4 sites (relaxation, streamlines,
  isolines x2), all with n < 31 today. Rule: *emit `>>> 0` for any bit constant*.

## 4. Roadmap claims checked

- `gpu-analysis-algorithms-roadmap.md` P2, "Tree contraction for `'cells'` accumulation … 64-round cap and NaN
  tails on long rivers": accurate, but **worse than stated**. The NaN tail is converted into valid-looking
  `streams = 0` and propagates into HAND and Strahler (C-03).
- `gpugraph-roadmap.md` Cross-cutting, "`GPUTerrainFlow` reports the iterations it used (SF terrain: fill 14 of
  256, accumulation 2 of 128)": **stale numbers**. The defaults are now `maxFillIterations` 512 and
  `maxAccumulationIterations` 64 (`hydrology/gpu-terrain-flow.ts:366,526`).
- `gpu-analysis-algorithms-roadmap.md` P1, "SLRM/MSRM via the existing int64 SAT (verify current code first)":
  verified **still open**. SLRM and MSRM use the anchored O(radius) separable mean filter
  (`relief-visualization-utils.ts`). Its doc says "a prefix sum cannot be made precise on the GPU", which the
  exact int64 SAT in `topographic-position/` contradicts; fix the comment when the item lands.
- P1 viewshed (`'march'` default, pyramid passed in) and P2 summits (O(R²) disc loop at
  `gpu-terrain-summits.ts:381-382`): **still accurate**.
- `gpugraph-roadmap.md` "`rowOrigin: 'north'` for … the distance field": still accurate. The distance field has
  row 0 at min-y while every terrain contributor is north-up; it is not re-reported.

## 5. Unverified suspicions

- **D-inf/MFD accumulation cost** (`hydrology/terrain-flow-routing.ts:285-318`): it recomputes
  `getFlowFraction` (8 facets, atan2/sqrt) for 8 neighbours per round per unresolved cell, which may be 10-50x
  D8. Confirm with one A/B of `'d8'` against `'d-infinity'` at 2048².
- **Strict back-link tolerance** `bestCandidate <= centerCost * 1.00001` (cost-distance): it can accept a
  slightly worse "strict" predecessor next to a zero plateau. Confirm with a CPU Dijkstra oracle on tiny
  friction adjoining zero friction.
- **Isoband `isBorderShared`** re-reads 4 neighbours plus `isSliverBand` per border edge in both the count and
  scatter passes; this may cost 2-3x on the rings path. Confirm with an A/B of `edgeCount` against triangles at
  4k².
- **`GPURasterSieve` `'merge'`** never merges into background (label 0), while `gdal_sieve` does. The doc
  wording suggests this is intended; confirm the intent.
- **Large grids against `maxStorageBufferBindingSize`**: `validateTerrainGrid` bounds only pixelCount < 2^32,
  and a 6000² float32 buffer exceeds the default 128 MiB binding limit at bind time rather than at
  construction. Confirm whether `createWGSLKernelNode` (trusted core) rejects it early.
- **Geomorphon empty line of sight after the skip**: ours gives digit 0, GRASS `anglev2` gives 1. This is
  reachable only with nulls beyond the first neighbour; ours is arguably better. Decide and document.
