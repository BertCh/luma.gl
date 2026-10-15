var e=`import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Terrain

<ExperimentalDocsTabs active="gpu-terrain" />

:::caution Experimental
\`@luma.gl/experimental/gpu-terrain\` is an initial, experimental entry point. APIs may change between
releases without a deprecation period.
:::

## Overview

\`@luma.gl/experimental/gpu-terrain\` provides analysis contributors for elevation tiles: Horn slope,
aspect and hillshade, marching-squares contours, viewshed visibility, hydrology (depression
filling, flat resolution, D8, D-infinity and multiple-flow-direction routing, flow accumulation,
streams, height above drainage, watersheds, stream order, hydrologic indices), batched line of sight
and cumulative viewsheds, point horizon profiles, horizon angles, solar position, shadow masks and
irradiance, relief shading, texture shading, archaeological relief visualizations (RVT), terrain-RGB
decoding, summits and critical points, curvature, geomorphons, ruggedness, topographic position,
Weiss landforms, and terrain-following wind fields. Each contributor is an algorithm that declares its
resources and command nodes into a caller-owned
[\`GPUCommandGraph\`](../gpu-core/gpu-command-graph.md) through \`getCommandNodes(graph)\`
(\`GPUCommandNodeProducer\`). It never compiles, submits, encodes, or reads back.

The contributors compose public primitives from [GPU Raster](../gpu-raster/README.md) and
[GPU Core](../gpu-core/concepts.md); new WGSL is written only where no primitive fits.

## When to use it

Use GPU Terrain when the elevation values themselves are the product: slope and aspect for
suitability models, contour or viewshed geometry for analysis, flow accumulation for hydrology, or
analytic shading textures that stay on the GPU. Per-frame observers, light directions, contour
levels, and cell-size settings are read from storage views, so interaction never recompiles.

Prefer the generic GPU Raster operations when the field is not an elevation model or you need a
different kernel:

- \`GPUTerrainDerivatives\` is Horn's method (two Sobel \`GPURasterGradient\` passes plus a fused shade
  kernel) with terrain cell-size modes, a z factor, and latitude handling. Use \`GPURasterGradient\`
  directly for a single signed derivative of any raster.
- \`GPUTerrainContours\` runs one \`GPURasterContours\` pipeline per elevation level and reports one
  overflow flag. Use \`GPUIsolines\` and \`GPUIsobands\` from GPU Raster when you need stitched
  polylines or filled bands, and \`GPURasterContours\` for a single level.
- The solar and shading contributors write analytic values and textures. If you only need lit
  terrain for presentation, the lighting of a terrain renderer is simpler.

## Quick start

\`\`\`ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUParameterBuffer,
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';
import {makeGPURasterMetadata} from '@luma.gl/experimental/gpu-raster';

const graph = new GPUCommandGraph(device, {id: 'terrain'});
const parameters = new GPUParameterBuffer(device, {
  id: 'derivatives-parameters',
  format: 'float32',
  length: GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
});
// 30 m cells, top-left corner at (originX, originY), rows advancing south.
const grid = makeGPURasterMetadata({width, height, cellSize: 30, origin: [originX, originY]});

// \`elevation\`, \`slope\`, and \`hillshade\` are graph views created on \`graph\`.
graph.add(new GPUTerrainDerivatives({
  grid, elevation, parameters: parameters.importToGraph(graph), output: {slope, hillshade}
}));
const compiled = await graph.compileAsync(); // once

// every frame: rewrite parameters (cell size and latitude band come from \`grid\`)
parameters.write(getGPUTerrainDerivativesParameterValues({azimuthDegrees: 315}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
\`\`\`

The application compiles, encodes, submits, and decides whether any bounded result is read back.
[Running contributors](../gpu-spatial-analysis/concepts.md#running-contributors) covers \`compileAsync()\`,
rebuilding without a stall, submission and why \`parameters\` is \`undefined\`.

## Concepts and operators

- [GPU Terrain concepts](./concepts.md): conventions, units and the GPU execution model.
- [GPU Terrain operators](./operators.md): every contributor, its inputs, outputs, parameters and how it
  computes.

## Catalog

One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.

::lens-catalog{entry=gpu-terrain}

## Limits and compatibility

- GPU Terrain is experimental and WebGPU-only.
- Inputs are planar \`float32\` elevation values; projection is done upstream (for example with
  [\`GPUProjection\`](../gpu-project.md)).
- Each contributor's TSDoc lists its non-goals. The GPU Core maintainer roadmap
  (\`dev-docs/roadmaps/gpugraph-roadmap.md\`) tracks what is still open.

## Related modules

- [GPU Raster](../gpu-raster/README.md) supplies the raster operators that terrain contributors compose.
- [GPU Core](../gpu-core/concepts.md) defines contributors, composition levels, and graph ownership.
- [GPU Network](../gpu-network/README.md) shares the conventions above for road and flow analysis.
`;export{e as default};