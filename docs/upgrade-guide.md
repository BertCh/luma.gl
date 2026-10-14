# Upgrade Guide

The upgrade guide lists breaking changes in each major and minor version of the luma.gl API, and provides information on how to update applications.

Upgrade instructions assume that you are upgrading from the immediately previous release.
If you are upgrading across multiple releases you will want to consider the release notes for all
intermediary releases.

luma.gl largely follows [SEMVER](https://semver.org) conventions. Breaking changes are typically only done in major versions, minor version bumps bring new functionality but few breaking changes, and patch releases typically contain only low-risk fixes.

*For detailed commit level logs that include alpha and beta releases, see the [CHANGELOG](https://github.com/visgl/luma.gl/blob/master/CHANGELOG.md) in the github repository.*

## Upgrading to v10.0

### Render pass draw offsets and counts

Indexed WebGL draws now use `firstIndex` in index elements. Replace uses of
`firstVertex` as an indexed byte offset with `firstIndex: byteOffset / indexByteSize`.
`firstVertex` only applies to non-indexed draws. WebGL now rejects `indexCount`
unless the selected vertex array has an index buffer; attribute-less non-indexed
draws should use `vertexCount`.
WebGPU render passes and render bundles now honor explicit zero draw and instance
counts instead of drawing an implicit instance.

### Model parameters on WebGL

On WebGL, `Model.setParameters()` no longer replaces `model.pipeline`. `Model.draw()` passes
the model's current parameters to every draw call, as before. Code that draws
`model.pipeline` directly without passing `parameters` gets the parameters the retained
pipeline was created with; pass `model.parameters` to the draw call instead.

### CPU geometry validation

Engine geometry now delegates CPU storage and tessellation to `@math.gl/geometry`.
Indices must reference valid vertices across the entire supplied index array, including entries
after an explicit draw count. Remove invalid unused indices before constructing a geometry.
Uint8 indices supplied to the engine compatibility adapter are widened to Uint16.
`TruncatedConeGeometry` defaults to a bottom radius of 1; provide explicit nonzero radii for
frusta. Primitive tessellation order can change, so derive face metadata from normals rather
than relying on vertex or index ordering.

**@luma.gl/gpgpu**

### GPUFFT2D graph lifecycle

`GPUFFT2D` now takes `{input, output, width, height, batchCount?, direction?}` graph views.
Replace `new GPUFFT2D(device, dimensions)` and `transform.encode(encoder, buffers)` with
`graph.add(new GPUFFT2D(props))`, `graph.compile()`, and `compiled.encode(encoder, {parameters: undefined})`.
Destroy the compiled graph instead of the primitive. `GPUFFT2DEncodeOptions` is removed.
Forward and inverse transforms are separate graph operations; pass parameters are immutable.
See [GPUFFT2D](./api-reference/experimental/gpu-core/gpu-fft2d) for the migration example.

- `GPULZByteDecompressor` descriptors now contain five uint32 words per record:
  `[outputOffset, byteLength, literalSourceOffset, literalPeriod, matchOffset]`. Code that creates
  descriptor buffers directly must add `literalPeriod` and use `GPU_LZ_BYTE_DESCRIPTOR_WORDS` when
  allocating and indexing them. Prefer `planGPULZByteDescriptors()` to convert the unchanged
  four-word parser spans into GPU descriptors.

### GPUSort radix tiling

- Radix `GPUSort` processes `elementsPerThread` keys per thread, default `8`, so one workgroup
  covers 2048 keys. Dispatch sizes, histogram lengths and scratch sizes shrink accordingly, and
  ranges that previously exceeded a small `maxComputeWorkgroupsPerDimension` now fit. Pass
  `elementsPerThread: 1` for the previous tiling. `GPUSort.digitBits` is typed `4 | 8`.

**@luma.gl/shadertools**

- `ShaderPassPipeline`, `ShaderPassPipelineStep`, and `ShaderPassComputeOptimization` have been
  renamed to `CompositeShaderPass`, `CompositeShaderPassStep`, and
  `CompositeShaderPassComputeOptimization`. Effect factories and values likewise replace their
  `ShaderPassPipeline` suffix with `CompositeShaderPass`.

**@luma.gl/webgpu**

- `WebGPUDevice.adapter` is now typed `GPUAdapter | null`. It is `null` for devices wrapped with `webgpuAdapter.attach()` / `luma.attachDevice()`, because a `GPUDevice` does not reference its `GPUAdapter`. Devices from `luma.createDevice()` still have an adapter; TypeScript code that reads `device.adapter` needs a null check or non-null assertion. Use `device.adapterInfo` for adapter metadata.

**@luma.gl/splats**

- `SPLAT_DEPTH_KEY_BITS` dropped from `24` to `16`. `packSplatDepthKey(depth, options)` now takes
  `{mode, keyBits, depthMin, depthMax, tileId}`; `mode` defaults to `'linear'` (the old behavior)
  but keys are 16 bits and the largest visible key is `2 ** keyBits - 2`, because the all-ones key
  is reserved for culled rows (`getSplatInvalidDepthKey()`). A `tileId` now sits directly above
  `keyBits`. `'linear'` and `'ndc'` keys are clamped to at most 24 bits.
- `GPU_SPLAT_PROJECTED_RECORD_BYTE_LENGTH` is `32` (was `48`): a projected record is a `vec4<f32>`
  clip center followed by half-precision axes and color (`packedAxis0`, `packedAxis1`,
  `packedColorRG`, `packedColorBA`). `GPU_SPLAT_GRAPH_UNIFORM_BYTE_LENGTH` is `160` (was `128`)
  with a new `GraphSplatUniforms` layout. Custom shaders reading renderer buffers should include
  `GPU_SPLAT_GRAPH_SHARED_WGSL` and use its accessors rather than hard-coding offsets.
- Visual defaults changed in `GPUSplatGraphRenderer` and `GPUPagedSplatRenderer`, and the screen
  filter default in `SplatRenderer`:
  - The screen-space filter adds `0.3` px² of variance (the reference rasterizer), not
    `0.3² = 0.09`. `kernel2DSize` (a standard deviation) is deprecated in favor of
    `screenSpaceFilterVariance`; pass `screenSpaceFilterVariance: 0.09` for the old look.
    `projectSplatCovarianceToScreen()` still adds no filter unless one is passed.
  - Mip-Splatting opacity compensation is on (`antialiasing: 'mip-splatting'`; use `'none'` for
    base 3DGS), as are `dynamicSupportRadius` and `compensateScreenSpaceClamp`.
  - The GPU sort key defaults to `depthKeyMode: 'float16'` (view-space distance) instead of
    hyperbolic device depth. Under an orthographic (affine) projection every mode keys on device
    depth instead, since clip `w` is constant there.
  - `pickingAlphaThreshold` defaults to `0`, so picking uses `alphaCutoff` like rendering.
  - `alphaMode: 'stochastic'` sorts a one-bit visibility key (`renderer.depthKeyBits === 1`), and
    `GPUSplatGraphMixedRenderer` always enables depth writes and disables blending for it.
- `getSplatDilationCompensation()` returns `0` rather than `1` when the pre-dilation covariance is
  degenerate, so point- or line-like projections no longer render at full opacity.
- `GPUPagedSplatRendererProps` no longer accepts graph-renderer props the paged renderer ignored
  (`clipRegion`, `antialiasing`, `fragmentKernel`, `dynamicSupportRadius`,
  `compensateScreenSpaceClamp`, `depthKeyMode`, `depthKeyBits`, `depthRange`, `alphaMode`,
  `pickingAlphaThreshold`, `batchParams`, `renderPath`, `presentation`).
- `packSplatClipUniforms()` drops degenerate planes before enforcing the 8-plane limit.
- `SplatHierarchyFrontierEntry` gained required `filterVariance` and `fadeOpacity` fields, and
  `SplatHierarchyStats` gained required `budgetedSplatCount`, `budgetExhausted` and
  `framesSinceBudgetPlan`. Code that constructs these objects itself must supply them.
- `getSplatHierarchyFoveatedPriority()` now measures to the nearest edge of a page's projected
  bounds rather than its center, so large or nearby pages are relaxed less than before.

**@luma.gl/experimental**

- `sumOrder` now defaults to `'sorted'`, so float sums and means are bitwise reproducible, for
  `GPURasterZonalStatistics`, `GPUFlowAggregation` and `GPUSpatialClustering`. `GPUZonalStatistics`
  defaults to `'sorted'` when every point-rate input is packed and to `'atomic'` when chunked.
  Last-bit differences against previous atomic output are expected. Pass `sumOrder: 'atomic'` for the
  old behavior, which is faster only with many scattered zones or nearly unique flow pairs.
- Iteration defaults rose: `GPUCostDistance.maxIterations` 64 to 512 (new `sweepAfterIteration`,
  default 96), `GPUTerrainFlow.maxFillIterations` and `maxFlatIterations` 128 to 512 (sweeps after 96).
  Converged results are unchanged; unconverged graphs now unroll more nodes per phase.
- `GPUTextureShading` computes coarse cascade levels on a decimated grid by default
  (`downsampleLevels: true`). Output differs from before by at most 1% of the output range (measured
  0.1-0.27%); pass `downsampleLevels: false` for the exact cascade.
- `GPUGeographicallyWeightedRegression` bisquare-fixed fits sum in grid cell order, shifting results
  by about 1e-5 relative to before; they stay deterministic.
- Network defaults changed. `GPUNetworkReachability.localIterations` 16 to 32 (costs are identical;
  rounds needed drop). `GPUNetworkCostMatrix.laneCount` is `recommendLaneCount()` (at least 32, about
  1M expanded nodes per batch, capped by 128 MB scratch and `rowCount`) instead of 32; results are
  identical, scratch memory grows. `GPUNetworkServiceAreas` labels with a frontier pass bounded by new
  `labelIterations`.
- OIT fullscreen resolution is now exposed as `createABufferResolveCompositeShaderPass()` and
  `createWBOITResolveCompositeShaderPass()`. `WBOITRenderer.capture()` returns the accumulation and
  revealage bindings for inserting the WBOIT resolve into a larger shader-pass stack.

**@luma.gl/experimental/gpu-spatial-analysis**

The correctness pass changes several defaults to the statistically principled choice. Results differ
from earlier alphas even where the API is unchanged.

- Coordinates: every class that reads positions takes an optional `spatialContext`. Omitting it means
  planar. A longitude/latitude context throws at construction unless the class has a lon/lat path, so
  project to a planar frame first (`createGPULocalFrame`). The `coordinateSystem`, `radius`,
  `sphereRadius` and ring-assembly `geographic` props are removed from `GPULineSegmentize`,
  `GPULineChunk`, `GPUGreatCircleArcs`, `GPULinearReferencing`, `GPULineLocate`, `GPUShapeGenerator`,
  `GPUOutlineGeometry`, `GPULineLengthPerPolygon` and `GPUSegmentRingAssembly`; pass
  `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` instead.
  `GPUSegmentRingAssembly` is planar by default. `GPU_SHAPE_GENERATOR_EARTH_RADIUS` and
  `GPU_CELL_MEASURES_EARTH_RADIUS_KM` are removed; use the constants in `contracts/earth`.
  `GPUCellMeasures` reports meters and m² unless `units: 'kilometers'`.
- Inference: the default `alternative` is `'two-sided'` (was `'directed'`, which is anti-conservative).
  `falseDiscoveryRate` is a deprecated alias of `multipleTesting: 'benjamini-hochberg'`, and
  `getFalseDiscoveryRateNodes` is removed. The global permutation test draws with a new shuffle, so a
  given seed gives different draws.
- Spatial weights: rows without neighbors are excluded by default (`islands: 'exclude'`; pass
  `islands: 'keep'` for esda parity). Zero-weight slots are not neighbors. The weights summary
  `counts` output grows from 5 to 6 values and `globalStatistics` from 4 to 5. Hot spot `neighborCounts`
  no longer counts the focal row. `GPUSpatialWeightsTransform` with `'kernel'` requires `bandwidth`,
  and `'adaptive'` requires kNN weights. `GPUGlobalSpatialStatistics` accepts at most 2^24 rows.
- Recipes: `addHotSpotAnalysisRecipe` points sources need `studyArea` and `cellCapacity`;
  `neighborSearchParameters`, `gridSize` and `tableCapacity` are removed. It analyzes density per km²,
  requires 8 neighbors and applies Benjamini-Hochberg by default, as does `addRateClusterMapRecipe`.
  `addSpaceTimeHotSpotsRecipe` takes absolute `{step, interval, origin, count}` slices; cyclic fields
  and `GPUSpaceTimeSliceField` are removed.
- `GPUEmergingHotSpots` uses 95% confidence, Benjamini-Hochberg and a minimum of 10 trend slices by
  default, and adds category 17 (`UNDETERMINED`), so palettes need 18 entries.
- `GPUSpatialScanStatistic`: `GPU_SCAN_STATISTIC_MAXIMUM_WINDOW_ZONES` is removed. Windows default to
  every zone, and time windows default to 50% of the period.
- `GPUEmpiricalBayesRates` clamps the prior variance by default (`clampPriorVariance: false` for esda
  parity). `GPULISAMarkov` is ungated by default and writes `0xffffffff` for rows outside the analysis.
  `GPULocalOutlierFactor` has no default threshold. Knox and Mantel counts change for asymmetric pair
  lists. Knox and Mantel permute with the swap-or-not shuffle, so a given seed gives different draws,
  and they accept at most 2^24 - 1 events.
- Point patterns and density: border-corrected Ripley's K is lower by a factor of (n−1)/n. Clark–Evans
  defaults to the Donnelly correction and a two-sided test, and its output and the quadrat output grow
  from 6 to 7 values. The IDW parameter length grows from 8 to 9.
- Kriging averages duplicate positions and writes NaN for ill-conditioned cells and materially
  negative variances. Intensive areal interpolation defaults to `denominator: 'overlap'`.
  `GPUPycnophylactic.iterations` is now a maximum. `GPUZonalStatistics` densities in lon/lat need
  caller-supplied `areas`.
- `GPUPointInPolygonJoin` defaults to `boundaryRule: 'half-open'`: a point on a shared edge belongs to
  exactly one polygon, and points on the outer right or top edge belong to none. Pass
  `boundaryRule: 'closed'` for the old inclusive behavior.
- Regression summaries grow and some slots change meaning: OLS 16 to 22 values, and
  `BREUSCH_PAGAN` is now the classic Breusch-Pagan test (Koenker-Bassett moved to slots 9/10 as
  `KOENKER_BASSETT`); regression diagnostics 12 to 13 (a failed OLS reports status 2, not 4); 2SLS 7 to 16;
  `GPUSpatialErrorGM` 6 to 13; GWR 6 to 12. Size summaries with the exported `*_LENGTH` constants.
- `GPUSpatialErrorGM` defaults to `estimator: 'homoskedastic'` (spreg `GM_Error_Hom`); pass
  `'kelejian-prucha'` for the previous `GM_Error`. GWR runs bandwidth refinement passes by default, so
  `selectedBandwidth` may lie between ladder entries, and its non-stationarity test draws with the
  swap-or-not shuffle.
- `GPUKMeans` initializes with `'kmeans++'` by default. `GPUMaxPRegions`, `GPUAZPRegions` and
  `GPUWardRegions` standardize attributes by default (`standardize: true`) and symmetrize the adjacency;
  the max-p status grows from 4 to 6 words and the partition evaluation summary from 8 to 9.
- `GPULocationAllocation` runs a Teitz-Bart swap phase by default (`maximumSwapIterations` 100), its status
  grows from 4 to 7 words, and unreachable demand is excluded from the objective and counted in
  `uncoveredDemandCount`, with `assignedCosts` of +Infinity.
- `GPUSegregation` writes NaN, not 0, for undefined indices. Relate outputs may carry
  `GPU_SPATIAL_RELATE_UNCERTAIN_BIT`. Rings that enclose a pole or span about 180° measure as NaN.
- `addDriveTimeCatchmentRecipe` with longitude/latitude positions snaps great-circle, so
  `maxSnapDistance` is in meters (was degrees). `GPUMapMatching` `breaks` can be 2 (a route-budget
  break), and `overflow` also reports route-table drops through the new `routeOverflow` output.

## Upgrading to v9.4

**@luma.gl/core**

- WebGPU device creation now defaults to the portable `DeviceProps.featureLevel: 'core'`. Applications that relied on luma.gl requesting every adapter feature and supported limit should pass `featureLevel: 'max'`.
- Render draw state is now owned by `RenderPass`. `RenderPipelineProps.bindings`, `RenderPipelineProps.bindGroups`, `RenderPipeline.setBindings()`, and `RenderPipeline.draw()` are deprecated compatibility APIs. Migrate low-level rendering code to `renderPass.setPipeline()`, `renderPass.setBindings()`, `renderPass.setVertexArray()`, and `renderPass.draw()`.
- `CommandEncoder.finish()` no longer accepts command-buffer properties, and the `CommandBufferProps` type has been removed. Set `id` and `userData` on the command encoder; the finished command buffer inherits them.

**@luma.gl/engine**

- `BufferTransform.run()` now creates its render pass with `discard: true` by default, avoiding
  unnecessary attachment stores for transform-feedback-only workloads. Applications that attach a
  framebuffer and consume rasterized fragment output must pass `discard: false` to `run()`.
- `Model.predraw(commandEncoder)` now requires an explicit command encoder. Call it with the encoder that will be submitted when ordered pre-draw uploads must be shared across multiple draws or viewports. Normal `Model.draw(renderPass)` calls continue to perform their own pre-draw work.
- `makeGPUGeometry()` now interleaves CPU geometry attributes into a single vertex buffer by default. Callers that require separate attribute buffers should create those buffers and construct `GPUGeometry` explicitly with the corresponding `bufferLayout`.

**@luma.gl/webgl**

- WebGLDeveloperTools and Spector integration now require `import '@luma.gl/webgl/debug'` before enabling `debugWebGL` or `debugSpectorJS`. This keeps debug-only code and the full GL enum out of normal adapter application bundles.

**@luma.gl/webgpu**

- `getShaderLayoutFromWGSL()` now uses lightweight interface scanning and returns `null` when WGSL is ambiguous or outside the supported subset. Raw render and compute pipelines must provide an explicit `shaderLayout` in that case. Uniform-buffer member reflection is no longer included in the returned layout.

**@luma.gl/shadertools**

- `ShaderAssembler` is now abstract and can no longer be constructed directly. Replace
  `new ShaderAssembler()` with `new GLSLShaderAssembler()` for GLSL or
  `new WGSLShaderAssembler()` for WGSL.
- `ShaderAssembler.getDefaultShaderAssembler()` now requires an explicit shader language.
  Replace calls without an argument with `ShaderAssembler.getDefaultShaderAssembler('glsl')`
  or `ShaderAssembler.getDefaultShaderAssembler('wgsl')`.
- `assembleGLSLShaderPair()` is available only on `GLSLShaderAssembler`, and
  `assembleWGSLShader()` is available only on `WGSLShaderAssembler`. Narrow existing
  `ShaderAssembler` references with `instanceof GLSLShaderAssembler` or
  `instanceof WGSLShaderAssembler` before assembling shader source.

## Upgrading to v9.3

**Potentially breaking behavior**
- `AsyncTexture` has been renamed to `DynamicTexture`.
- Scenegraph creation API has been improved, see [`createScenegraphsFromGLTF()`](/docs/api-reference/gltf).
- gltf module now creates `DynamicTexture` instances rather than raw `Texture`s.
- glTF texture sampling now defaults to linear filtering when a glTF sampler omits explicit filter settings. Applications relying on the previous nearest-neighbor default should verify visual output and set sampler filters explicitly when nearest sampling is required.
- The legacy feature flag `timer-query-webgl` has been removed. Replace checks for `timer-query-webgl` with `timestamp-query` for GPU timestamp/query support on both WebGPU and WebGL.
- `PipelineFactory` and `ShaderFactory` now import from `@luma.gl/core` instead of `@luma.gl/engine`.

## Upgrading to v9.2

v9.2 brings full WebGPU support. Some additional deprecations and breaking changes have been necessary, but apart from the `Texture` -> `AsyncTexture` split, impact on most applications should be minimal.

**New VertexFormats**
- `VertexFormat` Replace `'unorm8-webgl'` with `'unorm8'`.

**Texture and AsyncTexture**
- The `Texture` class has been simplified to the minimum API required for GPU portability. The `AsyncTexture` texture class provides a higher-level API and is recommended for most applications.
- `device.createTexture()` no longer accepts `props.data`: Use `AsyncTexture` or call `texture.setImageData()`
- `device.createTexture()` no longer accepts `props.mipmaps`: Use `AsyncTexture` (or call `texture.generateMipmapsWebGL()`)
- On WebGPU, mipmap generation now lives in `AsyncTexture.generateMipmaps()`, not in core `Texture`.
- WebGPU `AsyncTexture` uses render passes for `2d`, `2d-array`, `cube`, and `cube-array`, and a compute path for `3d`.
- Unsupported WebGPU formats now fail explicitly when mipmap generation is requested, instead of silently acting as a no-op.
- `TextureFormat` Correct the PVRTC 2bpp RGB format spelling from `pvrtc-rbg2unorm-webgl` to `pvrtc-rgb2unorm-webgl`.

**Removal of WebGL uniform support**
- The transition from uniforms to uniform buffers is complete, and remaining support for non-buffer uniforms has been removed.
- `core`: `RenderPipeline.setUniformsWebGL()` dropped, use uniform buffer bindings
- `engine`: `Model.setUniformsWebGL()` deprecated, use uniform buffer bindings
- `shadertools`: WebGL1 shader modules have been removed, use the new modules uniform buffer-based counterparts.

**`CanvasContext` simplifications**
- `canvasContext.devicePixelWidth` and `canvasContext.devicePixelHeight` are now kept updated to exact device pixel size of underlying canvas.
- Instead `canvasContext.setDrawingBufferSize()` to explicitly control drawing buffer size, if not using `CanvasContextProps.autoResize`
- A new `DeviceProps.onResize` callback can be used to react to changes.

**Minor changes**
- `core`: The shader types has been refactored, some shader type names have changed. These are typically not used directly by applications.

## Upgrading to v9.1

v9.1 continues to build out WebGPU support. Some additional deprecations and breaking changes have been necessary, but impact on most applications should be minimal.

**Major change: Adapters**

- When initializing luma.gl, applications now import an `Adapter` singleton from the WebGPU and/or the WebGL module, and passes the adapter object(s) to `luma.createDevice()`, `makeAnimationLoop` etc.
- `luma.registerDevices()` can be replaced with `luma.registerAdapters()` if global registration is still desired.

**Major change: Texture and AsyncTextures**

- The texture API is being streamlined to work symmetrically across WebGPU and WebGL.
- `Texture.copyExternalImage()` and `Texture.copyImageData()` replaces `Texture.setImageData()` when initializing texture memory with image data.
- `Textures` no longer accept promises when setting data (e.g. from `loadImageBitmap(url)`.
- Instead, a new `AsyncTexture` class does accept promises and creates actual `Textures` once the promise resolves and data is available.
- The `Model` class now accepts `AsyncTextures` as bindings and defers rendering until the underlying texture has been created.

**@luma.gl/core**

| Updated API                   | Status     | Replacement                                  | Comment                                                         |
| ----------------------------- | ---------- | -------------------------------------------- | --------------------------------------------------------------- |
| `luma.registerDevices()` | Deprecated | [`luma.registerAdapters()`][adapters]. | Adapters provide a cleaner way to work with GPU backends. |
| `DeviceProps.canvas` | Moved | [`DeviceProps.createCanvasContext`][canvas]. | Move canvas related props to `props.createCanvasContext: {}`. |
| `DeviceProps.<webgl options>` | Moved | [`DeviceProps.webgl.<options>`][webgl]. | Move canvas related props to `props.webgl: {}`. |
| `DeviceProps.break` | Removed | — | Use an alternative [debugger][debugging] |
| `TextureProps.data` (Promise) | Removed | `AsyncTexture` class | `Texture` no longer accept promises. Use `AsyncTexture` |
| `Parameters.blend` | New | — | Explicit activation of color blending |
| `triangle-fan-webgl` topology | Removed | `triangle-strip`. | Reorganize your geometries |
| `line-loop-webgl` topology | Removed | `line-list`. | Reorganize your geometries |
| `glsl` shader template string | Removed | `/* glsl */` comment | Enable syntax highlighting in vscode using before shader string |
| `depth24unorm-stencil8` | Removed | `depth24plus-stencil8` | The `TextureFormat` was dropped from the WebGPU spec |
| `rgb8unorm-unsized` | Removed | `rgb8unorm` | Drop support for unsized WebGL1 `TextureFormat` |
| `rgba8unorm-unsized` | Removed | `rgba8unorm` | Drop support for unsized WebGL1 `TextureFormat` |

[adapters]: /docs/api-reference/core/luma#lumaregisteradapters
[canvas]: /docs/api-reference/core/canvas-context#canvascontextprops
[webgl]: https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/getContext#contextattributes
[debugging]: /docs/developer-guide/debugging

**@luma.gl/shadertools**

| Updated API                          | Status  | Replacement                             | Comment                                            |
| ------------------------------------ | ------- | --------------------------------------- | -------------------------------------------------- |
| `ShaderModuleInstance` | Removed | Use `ShaderModule` instead. | Type has been removed. |
| `initializeShaderModule()` | Changed | — | Initializes the original shader module object |
| `ShaderModuleInstance.getUniforms()` | Removed | `getShaderModuleUniforms(module, ...)`. | Interact directly with the shader module |
| `getDependencyGraph()` | Removed | `getShaderModuleDependencies(module)` . | Interact directly with the shader module |
| `glsl` template string | Removed | `/* glsl */` comment | Enable syntax highlighting in vscode using comment |

**@luma.gl/effects**

New module. All postprocessing effects that were previously in `@luma.gl/shadertools` are now exported from `@luma.gl/effects`.

**@luma.gl/webgl**

- `WebGLDeviceContext` - Note that luma.gl v9.1 and onwards set `DeviceProps.webgl.preserveDrawingBuffers` to `true` by default. This can be disabled for some (potential) memory savings and a (potential) minor performance boost on resource limited devices, such as mobile phones, at the cost of not being able to take screenshots or rendering to the screen without clearing it.

## Upgrading to v9.0

luma.gl v9 is a major modernization of the luma.gl API, with many breaking changes, so the upgrade notes for this release are unusually long. To facilitate porting to the v9 release we have also provided a
[Porting Guide](/docs/legacy/porting-guide) that also provides more background information and discusses porting strategies.

## Upgrading to v8 and earlier releases

This page only covers luma.gl v9 and later releases.
For information on upgrading to from v8 and earlier releases, see the [Legacy Upgrade Guide](/docs/legacy/legacy-upgrade-guide).

## GPU Core composition

GPU Core primitives now expose `getCommandNodes(graph)` instead of `addToGraph(graph)`. Schedule
a primitive with `graph.add(primitive)`, which calls `getCommandNodes(graph)` internally. To inspect
or modify nodes before scheduling, use `addGPUCommandNodes(graph, nodes)`. Graph-independent
primitives can also be added to `GPUProgram`. `GPUCommandGraphContributor` and the compiler's legacy mutation fallback were
removed. The three range/scatter primitives that also publish scratch views expose
`getCommands(graph)` returning `{nodes, ...views}`; schedule `nodes` explicitly.

`GPUProgramBindings.vectors` now accepts typed `GPUData`, readonly `GPUData[]`, or `GPUVector`
instead of raw buffers. Compiler vector resolution returns `GraphVectorView`; access physical
chunks through `.data` or canonical logical descriptors through `.chunks`.
