// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Viewport} from '@deck.gl/core';
import type {Buffer, Device} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import {GPUIndexPickingTarget, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../graph-buffers';
import type {SpatialAnalysisResources} from '../spatial-analysis-resources';

/** Edge length of the square index-picking target, in pixels (the map viewport is stretched to it). */
export const PICK_TARGET_SIZE = 1024;
/** Index-picking pairs the region reduction can store. */
export const PICK_RESULT_PAIRS = 4096;
/** Half-size of each point's picking quad in target pixels. */
const PICK_POINT_RADIUS = 5;
/** Edge of the pick window around a click, in target pixels. */
export const PICK_WINDOW_SIZE = 44;

/** Buffers owned by one point set's picking path. */
export type RackPickBuffers = {
  /** 20 floats: column-major meters-to-clip matrix (16) padded to 20. */
  matrix: Buffer;
  /** `uint32x4` `[x, y, width, height]` pick window in target pixels. */
  region: Buffer;
  /** `[count, overflow, (object, batch)...]` written by the region pass. */
  result: Buffer;
  /** Source-aligned 0/1 mask written by `GPUPickRegionMask`. */
  mask: Buffer;
  /** One-word overflow flag written by `GPUPickRegionMask`. */
  overflow: Buffer;
};

/** Creates the buffers of the picking path for `count` source rows. */
export function createRackPickBuffers(
  resources: SpatialAnalysisResources,
  count: number
): RackPickBuffers {
  return {
    matrix: resources.createBuffer('pick-matrix', new Float32Array(20)),
    region: resources.createBuffer('pick-region', new Uint32Array(4)),
    result: resources.createBuffer('pick-result', (2 + PICK_RESULT_PAIRS * 2) * 4),
    mask: resources.createBuffer('pick-mask', count * 4),
    overflow: resources.createBuffer('pick-overflow', 4)
  };
}

const PICK_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> matrixData: array<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) pointIndex: u32,
};
struct FragmentOutputs {
  @location(0) color: vec4<f32>,
  @location(1) indices: vec2<i32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let column0 = vec4<f32>(matrixData[0], matrixData[1], matrixData[2], matrixData[3]);
  let column1 = vec4<f32>(matrixData[4], matrixData[5], matrixData[6], matrixData[7]);
  let column3 = vec4<f32>(matrixData[12], matrixData[13], matrixData[14], matrixData[15]);
  let point = positions[instanceIndex];
  let clip = column0 * point.x + column1 * point.y + column3;
  var output: VertexOutput;
  output.pointIndex = instanceIndex;
  if (!(clip.w > 0.0)) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return output;
  }
  let offset = corners[vertexIndex] * ${PICK_POINT_RADIUS}.0 * 2.0 / ${PICK_TARGET_SIZE}.0;
  output.position = vec4<f32>(clip.xy + offset * clip.w, 0.5 * clip.w, clip.w);
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> FragmentOutputs {
  var output: FragmentOutputs;
  output.color = vec4<f32>(0.0);
  output.indices = vec2<i32>(i32(input.pointIndex), 0);
  return output;
}
`;

/**
 * Adds the index-picking render pass over the points and `GPUIndexPickingTarget.addRegionPass` to
 * `graph`, and returns the region-result view for `GPUPickRegionMask`. The render node draws one
 * small quad per point whose color-attachment index is the point row.
 */
export function addRackPickPasses(
  graph: GPUCommandGraph<void>,
  device: Device,
  props: {id: string; positions: Buffer; count: number; buffers: RackPickBuffers}
) {
  const {id, positions, buffers} = props;
  const target = new GPUIndexPickingTarget<void>(graph, {
    id: `${id}-target`,
    width: PICK_TARGET_SIZE,
    height: PICK_TARGET_SIZE
  });
  const positionsView = importGraphBuffer(graph, `${id}-positions`, positions, 'float32x2');
  const matrixView = importGraphBuffer(graph, `${id}-matrix`, buffers.matrix, 'float32', 20);
  const renderId = `${id}-render`;
  graph.addRenderPass({
    id: renderId,
    attachments: target.attachments,
    resources: [
      {buffer: positionsView, usage: 'storage-read'},
      {buffer: matrixView, usage: 'storage-read'}
    ],
    compile: () => {
      const model = new Model(device, {
        id: `${id}-model`,
        source: PICK_SHADER,
        topology: 'triangle-list',
        isInstanced: true,
        vertexCount: 6,
        instanceCount: props.count,
        bufferLayout: [],
        colorAttachmentFormats: ['rgba8unorm', 'rg32sint'],
        depthStencilAttachmentFormat: 'depth24plus',
        shaderLayout: {
          attributes: [],
          bindings: [
            {name: 'positions', type: 'read-only-storage', group: 0, location: 0},
            {name: 'matrixData', type: 'read-only-storage', group: 0, location: 1}
          ]
        },
        parameters: {depthCompare: 'always', depthWriteEnabled: false}
      });
      return {
        getRenderPassProps: () => target.renderPassProps,
        encode: ({renderPass, getBuffer}) => {
          model.setBindings({
            positions: getBuffer(positionsView),
            matrixData: getBuffer(matrixView)
          });
          model.draw(renderPass);
        },
        destroy: () => model.destroy()
      };
    }
  });
  const regionView = importGraphBuffer(graph, `${id}-window`, buffers.region, 'uint32', 4);
  const resultView = importGraphBuffer(
    graph,
    `${id}-result`,
    buffers.result,
    'uint32',
    2 + PICK_RESULT_PAIRS * 2
  );
  target.addRegionPass({after: renderId, region: regionView, result: resultView});
  return {result: resultView};
}

/**
 * Meters-around-origin to clip-space matrix (column-major, 20 floats with trailing zeros): the
 * chain Deck applies for `METER_OFFSETS`, `viewProjection * translate(commonOrigin) *
 * scale(unitsPerMeter)`.
 */
export function getPickMatrix(viewport: Viewport, origin: readonly [number, number]): Float32Array {
  const commonOrigin = viewport.projectPosition([origin[0], origin[1], 0]);
  const unitsPerMeter = viewport.distanceScales.unitsPerMeter;
  const viewProjection = viewport.viewProjectionMatrix;
  const matrix = new Float32Array(20);
  for (let row = 0; row < 4; row++) {
    matrix[row] = viewProjection[row] * unitsPerMeter[0];
    matrix[4 + row] = viewProjection[4 + row] * unitsPerMeter[1];
    matrix[8 + row] = viewProjection[8 + row] * unitsPerMeter[2];
    matrix[12 + row] =
      viewProjection[row] * commonOrigin[0] +
      viewProjection[4 + row] * commonOrigin[1] +
      viewProjection[8 + row] * commonOrigin[2] +
      viewProjection[12 + row];
  }
  return matrix;
}

/** Pick window in target pixels around a CSS pixel position of a viewport. */
export function getPickWindow(
  viewport: Viewport,
  pixel: readonly [number, number]
): Uint32Array<ArrayBuffer> {
  const targetX = (pixel[0] / viewport.width) * PICK_TARGET_SIZE;
  const targetY = (pixel[1] / viewport.height) * PICK_TARGET_SIZE;
  const half = PICK_WINDOW_SIZE / 2;
  const x = Math.max(0, Math.min(PICK_TARGET_SIZE - 1, Math.round(targetX - half)));
  const y = Math.max(0, Math.min(PICK_TARGET_SIZE - 1, Math.round(targetY - half)));
  return Uint32Array.of(x, y, PICK_WINDOW_SIZE, PICK_WINDOW_SIZE);
}
