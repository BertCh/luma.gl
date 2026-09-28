// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// Spark-compatible RAD opacity and support behavior is adapted from Spark:
// https://github.com/sparkjsdev/spark (MIT, Copyright © 2025 WORLD LABS TECHNOLOGIES, INC.)

import type {ShaderLayout} from '@luma.gl/core';
import {
  GPU_SPLAT_FEATURE_SHADER,
  GPU_SPLAT_PROJECTION_SHADER,
  GPU_SPLAT_RENDER_SHADER
} from './gpu-splat-graph-shaders';

/**
 * Paged Gaussian shaders, derived from the shared graph shaders by explicit substitution.
 *
 * Only the *structural* differences live here: sparse source rows reached through an active-row
 * index, a separate global depth domain, and visibility counted in the feature pass rather than
 * during projection. Everything appearance-related - radiance-field opacity, the circular support
 * profile, sRGB decoding - is a feature flag in the shared shaders, so the two paths cannot drift
 * apart in how a Gaussian looks.
 *
 * {@link replacePagedShaderSource} throws when an anchor no longer matches, which turns any
 * upstream shader edit into a build failure here rather than a silent behavior change.
 */

/** Sparse source projection retains every original column within eight storage bindings. */
export const GPU_PAGED_SPLAT_PROJECTION_SHADER_LAYOUT = {
  attributes: [],
  bindings: [
    {name: 'positions', type: 'read-only-storage', group: 0, location: 0},
    {name: 'scales', type: 'read-only-storage', group: 0, location: 1},
    {name: 'rotations', type: 'read-only-storage', group: 0, location: 2},
    {name: 'colors', type: 'read-only-storage', group: 0, location: 3},
    {name: 'opacities', type: 'read-only-storage', group: 0, location: 4},
    {name: 'projectedRecords', type: 'storage', group: 0, location: 5},
    {name: 'depthKeys', type: 'storage', group: 0, location: 6},
    {name: 'activeRows', type: 'read-only-storage', group: 0, location: 7},
    {name: 'graphUniforms', type: 'uniform', group: 0, location: 8}
  ]
} satisfies ShaderLayout;

/** Sparse SH, semantics, and global visibility also honor the portable eight-buffer limit. */
export const GPU_PAGED_SPLAT_FEATURE_SHADER_LAYOUT = {
  attributes: [],
  bindings: [
    {name: 'positions', type: 'read-only-storage', group: 0, location: 0},
    {name: 'sphericalHarmonics', type: 'read-only-storage', group: 0, location: 1},
    {name: 'semanticIds', type: 'read-only-storage', group: 0, location: 2},
    {name: 'semanticSelections', type: 'read-only-storage', group: 0, location: 3},
    {name: 'projectedRecords', type: 'storage', group: 0, location: 4},
    {name: 'depthKeys', type: 'storage', group: 0, location: 5},
    {name: 'drawCommands', type: 'storage', group: 0, location: 6},
    {name: 'activeRows', type: 'read-only-storage', group: 0, location: 7},
    {name: 'graphUniforms', type: 'uniform', group: 0, location: 8},
    {name: 'featureUniforms', type: 'uniform', group: 0, location: 9}
  ]
} satisfies ShaderLayout;

/** Already gathered global-order segments require one uniform and one projected storage buffer. */
export const GPU_PAGED_SPLAT_RENDER_SHADER_LAYOUT = {
  attributes: [],
  bindings: [
    {name: 'graphUniforms', type: 'uniform', group: 0, location: 0},
    {name: 'projectedRecords', type: 'read-only-storage', group: 0, location: 1}
  ]
} satisfies ShaderLayout;

/** Resolves one sparse source row, either densely offset or through the active-row index. */
const PAGED_SPARSE_ROW_RESOLUTION = `  let projectedRowIndex = globalInvocationId.x;
  if (projectedRowIndex >= graphUniforms.rowCount) {
    return;
  }
  var batchRowIndex = projectedRowIndex + graphUniforms.sourceRowOffset;
  if (graphUniforms.hasActiveRows != 0u) {
    batchRowIndex = activeRows[projectedRowIndex];
  }`;

const DENSE_ROW_RESOLUTION = `  let batchRowIndex = globalInvocationId.x;
  if (batchRowIndex >= graphUniforms.rowCount) {
    return;
  }

  let projectedRowIndex = graphUniforms.batchOffset + batchRowIndex;`;

/**
 * Original Gaussian projection over sparse source rows in a separate global depth domain.
 *
 * Projected records are indexed page-locally while depth keys are indexed globally, because the
 * sort spans every resident page while each page owns its own record range.
 */
const GPU_PAGED_SPLAT_SPARSE_PROJECTION_SHADER = [
  // Binding 7 carries the active-row index instead of the indirect draw command: a paged renderer
  // counts visible rows once, in its feature pass, after semantic filtering has run.
  [
    '@group(0) @binding(7) var<storage, read_write> drawCommands: array<atomic<u32>>;',
    '@group(0) @binding(7) var<storage, read> activeRows: array<u32>;'
  ],
  ['@group(0) @binding(9) var<uniform> clipUniforms: GraphSplatClipUniforms;\n', ''],
  [DENSE_ROW_RESOLUTION, PAGED_SPARSE_ROW_RESOLUTION],
  [
    '  depthKeys[rowIndex] = graphUniforms.maximumDepthKey + 1u;',
    '  depthKeys[graphUniforms.batchOffset + rowIndex] = graphUniforms.maximumDepthKey + 1u;'
  ],
  [
    '  depthKeys[projectedRowIndex] = packSplatDepthKey(',
    '  depthKeys[graphUniforms.batchOffset + projectedRowIndex] = packSplatDepthKey('
  ],
  [
    `  if (hasSplatFlag(graphUniforms.featureFlags, SPLAT_FLAG_CLIPPING)) {
    alpha = alpha * getSplatClipCoverage(
      clipUniforms,
      position,
      worldAxis0,
      worldAxis1,
      worldAxis2
    );
    if (alpha < graphUniforms.alphaCutoff) {
      clearProjectedSplat(projectedRowIndex);
      return;
    }
  }

`,
    ''
  ],
  ['  atomicAdd(&drawCommands[1u], 1u);\n', '']
].reduce(
  (source, [search, replacement]) => replacePagedShaderSource(source, search, replacement),
  GPU_SPLAT_PROJECTION_SHADER
);

/** Analytic perspective covariance preserves Spark RAD appearance without extra bindings. */
export const GPU_PAGED_SPLAT_PROJECTION_SHADER = makeCalibratedPagedProjectionShader(
  GPU_PAGED_SPLAT_SPARSE_PROJECTION_SHADER
);

/** Sparse source feature evaluation publishes the exact globally visible indirect count. */
export const GPU_PAGED_SPLAT_FEATURE_SHADER = [
  [
    `@group(0) @binding(7) var<uniform> graphUniforms: GraphSplatUniforms;
@group(0) @binding(8) var<uniform> featureUniforms: GraphSplatFeatureUniforms;`,
    `@group(0) @binding(7) var<storage, read> activeRows: array<u32>;
@group(0) @binding(8) var<uniform> graphUniforms: GraphSplatUniforms;
@group(0) @binding(9) var<uniform> featureUniforms: GraphSplatFeatureUniforms;`
  ],
  [
    DENSE_ROW_RESOLUTION,
    `${PAGED_SPARSE_ROW_RESOLUTION}
  let globalRowIndex = graphUniforms.batchOffset + projectedRowIndex;`
  ],
  [
    '  let invalidDepthKey = graphUniforms.maximumDepthKey + 1u;\n  if (depthKeys[projectedRowIndex] == invalidDepthKey) {',
    '  let invalidDepthKey = graphUniforms.maximumDepthKey + 1u;\n  if (depthKeys[globalRowIndex] == invalidDepthKey) {'
  ],
  [
    `    depthKeys[projectedRowIndex] = invalidDepthKey;
    atomicSub(&drawCommands[1u], 1u);`,
    '    depthKeys[globalRowIndex] = invalidDepthKey;'
  ],
  // The visible count is published here, once, after both harmonics and semantic filtering.
  [
    `    projectedRecords[projectedRowIndex].packedColorRG = packedColor.x;
    projectedRecords[projectedRowIndex].packedColorBA = packedColor.y;
  }
}
`,
    `    projectedRecords[projectedRowIndex].packedColorRG = packedColor.x;
    projectedRecords[projectedRowIndex].packedColorBA = packedColor.y;
  }
  atomicAdd(&drawCommands[1u], 1u);
}
`
  ]
].reduce(
  (source, [search, replacement]) => replacePagedShaderSource(source, search, replacement),
  GPU_SPLAT_FEATURE_SHADER
);

/** Final gathered records already occupy exact global painter order. */
export const GPU_PAGED_SPLAT_RENDER_SHADER = [
  ['@group(0) @binding(2) var<storage, read> sortedIds: array<u32>;\n', ''],
  [
    '  let projected = projectedRecords[sortedIds[instanceIndex]];',
    '  let projected = projectedRecords[instanceIndex];'
  ]
].reduce(
  (source, [search, replacement]) => replacePagedShaderSource(source, search, replacement),
  GPU_SPLAT_RENDER_SHADER
);

/**
 * Applies the exact homogeneous-coordinate projection Jacobian without extra source bindings.
 *
 * Finite differencing the projection along each Gaussian axis is accurate for small, near-axis
 * Gaussians and increasingly wrong for large ones near the edge of a wide field of view.
 * Differentiating the perspective divide directly removes that error for the same instruction
 * count, which matters for the coarse parent pages a paged hierarchy spends most of its time in.
 */
function makeCalibratedPagedProjectionShader(source: string): string {
  return [
    [
      `fn getProjectedScreenPosition(position: vec3<f32>) -> vec2<f32> {
  let clipPosition = graphUniforms.modelViewProjectionMatrix * vec4<f32>(position, 1.0);`,
      `fn getProjectedScreenPosition(clipPosition: vec4<f32>) -> vec2<f32> {`
    ],
    [
      `}

fn getProjectedRotation(quaternion: vec4<f32>)`,
      `}

fn getProjectedScreenAxis(clipCenter: vec4<f32>, worldAxis: vec3<f32>) -> vec2<f32> {
  let clipAxis = graphUniforms.modelViewProjectionMatrix * vec4<f32>(worldAxis, 0.0);
  let inverseClipW = 1.0 / clipCenter.w;
  let normalizedAxis =
    (clipAxis.xy - clipCenter.xy * (clipAxis.w * inverseClipW)) * inverseClipW;
  return vec2<f32>(
    normalizedAxis.x * graphUniforms.viewportSize.x * 0.5,
    -normalizedAxis.y * graphUniforms.viewportSize.y * 0.5
  );
}

fn getProjectedRotation(quaternion: vec4<f32>)`
    ],
    [
      `  let center = getProjectedScreenPosition(position);
  let delta0 = getProjectedScreenPosition(position + worldAxis0) - center;
  let delta1 = getProjectedScreenPosition(position + worldAxis1) - center;
  let delta2 = getProjectedScreenPosition(position + worldAxis2) - center;`,
      `  let center = getProjectedScreenPosition(clipCenter);
  let delta0 = getProjectedScreenAxis(clipCenter, worldAxis0);
  let delta1 = getProjectedScreenAxis(clipCenter, worldAxis1);
  let delta2 = getProjectedScreenAxis(clipCenter, worldAxis2);`
    ]
  ].reduce(
    (calibrated, [search, replacement]) =>
      replacePagedShaderSource(calibrated, search, replacement),
    source
  );
}

/** Prevent upstream shader edits from silently removing sparse-source projection invariants. */
function replacePagedShaderSource(source: string, search: string, replacement: string): string {
  if (!source.includes(search)) {
    throw new Error('Paged Gaussian shader source no longer matches its shared graph shader');
  }
  return source.replace(search, replacement);
}
