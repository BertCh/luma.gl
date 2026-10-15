// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createFrontierKernelNode} from '../../engine/frontier-kernel';

export const FRONTIER_RESIDUAL_PARAMETER_LENGTH = 1;

/** Packs the stable explicit-diffusion step size. */
export function getFrontierResidualDiffusionParameterValues(
  settings: {stepScale?: number} = {},
  target = new Float32Array(FRONTIER_RESIDUAL_PARAMETER_LENGTH)
): Float32Array {
  target[0] = Math.max(0, Math.min(1, settings.stepScale ?? 0.8));
  return target;
}

/** Inputs and outputs of one cellular residual-diffusion step. */
export type FrontierResidualDiffusionProps = {
  id?: string;
  width: number;
  height: number;
  state: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  /** Three-channel prior matched to current-day colour and contrast. */
  prior: GraphDataView<'float32'>;
  nextState: GraphDataView<'float32'>;
  /** Packed RGBA8 output, with red in the low byte. */
  display: GraphDataView<'uint32'>;
  boundary?: 'clamp' | 'wrap';
};

/** One deterministic Laplacian update over a four-channel cellular field. */
export class FrontierResidualDiffusion {
  readonly id: string;
  readonly props: FrontierResidualDiffusionProps;

  constructor(props: FrontierResidualDiffusionProps) {
    this.id = props.id ?? 'frontier-residual-diffusion';
    this.props = props;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const props = this.props;
    const edge =
      (props.boundary ?? 'clamp') === 'wrap'
        ? 'return (value % size + size) % size;'
        : 'return clamp(value, 0, size - 1);';
    return [
      createFrontierKernelNode(graph, {
        id: `${this.id}-step`,
        operation: 'FrontierResidualDiffusion',
        variant: props.boundary ?? 'clamp',
        bindings: [
          {name: 'state', view: props.state, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'prior', view: props.prior, type: 'f32', access: 'read'},
          {name: 'nextState', view: props.nextState, type: 'f32', access: 'read_write'},
          {name: 'display', view: props.display, type: 'u32', access: 'read_write'}
        ],
        invocationCount: props.width * props.height,
        declarations: `const WIDTH: i32 = ${props.width};
const HEIGHT: i32 = ${props.height};
fn edge(value: i32, size: i32) -> i32 { ${edge} }
fn loadResidual(column: i32, row: i32) -> vec3<f32> {
  let cell = u32(edge(row, HEIGHT) * WIDTH + edge(column, WIDTH));
  let base = stateOffset + cell * 4u;
  return vec3<f32>(state[base], state[base + 1u], state[base + 2u]);
}`,
        body: `let column = i32(index % u32(WIDTH));
  let row = i32(index / u32(WIDTH));
  let stateBase = stateOffset + index * 4u;
  let observed = state[stateBase + 3u];
  let center = loadResidual(column, row);
  var next = center;
  if (observed < 0.5) {
    let laplacian = loadResidual(column - 1, row) + loadResidual(column + 1, row) +
      loadResidual(column, row - 1) + loadResidual(column, row + 1) - 4.0 * center;
    next += 0.24 * parameters[parametersOffset] * laplacian;
  }
  let base = nextStateOffset + index * 4u;
  nextState[base] = next.x;
  nextState[base + 1u] = next.y;
  nextState[base + 2u] = next.z;
  // Never turn repaired pixels into observations as diffusion converges.
  nextState[base + 3u] = observed;
  let priorBase = priorOffset + index * 3u;
  let red = u32(round(clamp(prior[priorBase] + next[0], 0.0, 1.0) * 255.0));
  let green = u32(round(clamp(prior[priorBase + 1u] + next[1], 0.0, 1.0) * 255.0));
  let blue = u32(round(clamp(prior[priorBase + 2u] + next[2], 0.0, 1.0) * 255.0));
  display[displayOffset + index] = red | (green << 8u) | (blue << 16u) | (255u << 24u);`
      })
    ];
  }
}
