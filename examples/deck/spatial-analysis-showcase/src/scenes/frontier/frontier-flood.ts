// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createFrontierKernelNode, type FrontierKernelBinding} from './frontier-kernel';

/** Float32 parameter count consumed by {@link FrontierFloodSimulation}. */
export const FRONTIER_FLOOD_PARAMETER_LENGTH = 14;

/** Live physical and interaction settings for the shallow-water prototype. */
export type FrontierFloodSettings = {
  cellSize: readonly [number, number];
  timeStep: number;
  gravity?: number;
  manningCoefficient?: number;
  rainfallRate?: number;
  infiltrationRate?: number;
  minimumDepth?: number;
  maximumCourantNumber?: number;
  maximumDepth?: number;
  source?: readonly [number, number];
  sourceRadius?: number;
  sourceRate?: number;
};

/** Packs the live flood settings into the graph parameter buffer. */
export function getFrontierFloodParameterValues(
  settings: FrontierFloodSettings,
  target = new Float32Array(FRONTIER_FLOOD_PARAMETER_LENGTH)
): Float32Array {
  const source = settings.source ?? [-1e6, -1e6];
  target.set([
    settings.cellSize[0],
    settings.cellSize[1],
    settings.timeStep,
    settings.gravity ?? 9.81,
    settings.manningCoefficient ?? 0.04,
    settings.rainfallRate ?? 0,
    settings.infiltrationRate ?? 0,
    settings.minimumDepth ?? 0.002,
    settings.maximumCourantNumber ?? 0.2,
    settings.maximumDepth ?? 30,
    source[0],
    source[1],
    settings.sourceRadius ?? 0,
    settings.sourceRate ?? 0
  ]);
  return target;
}

/** Persistent caller-owned flood state. */
export type FrontierFloodState = {
  depth: GraphDataView<'float32'>;
  xFlux: GraphDataView<'float32'>;
  yFlux: GraphDataView<'float32'>;
};

/** Inputs and outputs of {@link FrontierFloodSimulation}. */
export type FrontierFloodProps = {
  id?: string;
  width: number;
  height: number;
  terrain: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  state: FrontierFloodState;
  /** Optional display field. The last solver step writes depth into it. */
  display?: GraphDataView<'float32'>;
  stepsPerEncoding?: number;
};

/**
 * Conservative local-inertial shallow-water steps for the Frontier Lab flood story.
 *
 * The graph owns no resources, never submits and never reads back. Face discharge and water depth
 * stay resident and the final depth is rendered directly by deck.gl.
 */
export class FrontierFloodSimulation {
  readonly id: string;
  readonly props: FrontierFloodProps;
  readonly stepsPerEncoding: number;

  constructor(props: FrontierFloodProps) {
    this.id = props.id ?? 'frontier-flood';
    this.props = props;
    this.stepsPerEncoding = props.stepsPerEncoding ?? 2;
    if (props.width < 2 || props.height < 2 || !Number.isInteger(this.stepsPerEncoding)) {
      throw new Error('FrontierFloodSimulation requires a valid grid and integer step count');
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    for (let step = 0; step < this.stepsPerEncoding; step++) {
      nodes.push(
        this.createFaceNode(graph, 'x', step),
        this.createFaceNode(graph, 'y', step),
        this.createCellNode(graph, step)
      );
    }
    return nodes;
  }

  private createFaceNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    axis: 'x' | 'y',
    step: number
  ): GPUCommandNode<Parameters> {
    const {width, height, terrain, parameters, state} = this.props;
    const horizontal = axis === 'x';
    const faceWidth = horizontal ? width + 1 : width;
    const faceHeight = horizontal ? height : height + 1;
    const flux = horizontal ? state.xFlux : state.yFlux;
    return createFrontierKernelNode(graph, {
      id: `${this.id}-${step}-${axis}-flux`,
      operation: 'FrontierFloodSimulation',
      variant: `${axis}-flux`,
      bindings: [
        {name: 'terrain', view: terrain, type: 'f32', access: 'read'},
        {name: 'depth', view: state.depth, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'flux', view: flux, type: 'f32', access: 'read_write'}
      ],
      invocationCount: faceWidth * faceHeight,
      declarations: /* wgsl */ `
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const FACE_WIDTH: u32 = ${faceWidth}u;

fn updateFlux(previous: f32, firstCell: u32, secondCell: u32, spacing: f32) -> f32 {
  let firstTerrain = terrain[terrainOffset + firstCell];
  let secondTerrain = terrain[terrainOffset + secondCell];
  let firstDepth = depth[depthOffset + firstCell];
  let secondDepth = depth[depthOffset + secondCell];
  let firstSurface = firstTerrain + firstDepth;
  let secondSurface = secondTerrain + secondDepth;
  let flowDepth = max(max(firstSurface, secondSurface) - max(firstTerrain, secondTerrain), 0.0);
  let minimumDepth = parameters[parametersOffset + 7u];
  if (flowDepth < minimumDepth) { return 0.0; }
  let timeStep = parameters[parametersOffset + 2u];
  let gravity = parameters[parametersOffset + 3u];
  let manning = parameters[parametersOffset + 4u];
  let slope = (secondSurface - firstSurface) / spacing;
  let denominator = 1.0 + gravity * timeStep * manning * manning * abs(previous) /
    max(pow(flowDepth, 7.0 / 3.0), 1.0e-12);
  let unconstrained = (previous - gravity * flowDepth * timeStep * slope) / denominator;
  let maximumFlux = parameters[parametersOffset + 8u] * spacing * flowDepth / timeStep;
  return clamp(unconstrained, -maximumFlux, maximumFlux);
}`,
      body: horizontal
        ? `let faceColumn = index % FACE_WIDTH;
  let row = index / FACE_WIDTH;
  if (faceColumn == 0u || faceColumn == WIDTH) { flux[fluxOffset + index] = 0.0; return; }
  let left = row * WIDTH + faceColumn - 1u;
  flux[fluxOffset + index] = updateFlux(flux[fluxOffset + index], left, left + 1u, parameters[parametersOffset]);`
        : `let column = index % WIDTH;
  let faceRow = index / WIDTH;
  if (faceRow == 0u || faceRow == HEIGHT) { flux[fluxOffset + index] = 0.0; return; }
  let south = (faceRow - 1u) * WIDTH + column;
  flux[fluxOffset + index] = updateFlux(flux[fluxOffset + index], south, south + WIDTH, parameters[parametersOffset + 1u]);`
    });
  }

  private createCellNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    step: number
  ): GPUCommandNode<Parameters> {
    const {width, height, parameters, state, display} = this.props;
    const bindings: FrontierKernelBinding[] = [
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'xFlux', view: state.xFlux, type: 'f32', access: 'read'},
      {name: 'yFlux', view: state.yFlux, type: 'f32', access: 'read'},
      {name: 'depth', view: state.depth, type: 'f32', access: 'read_write'}
    ];
    if (display) bindings.push({name: 'display', view: display, type: 'f32', access: 'read_write'});
    return createFrontierKernelNode(graph, {
      id: `${this.id}-${step}-depth`,
      operation: 'FrontierFloodSimulation',
      variant: 'depth',
      bindings,
      invocationCount: width * height,
      declarations: `const WIDTH: u32 = ${width}u;`,
      body: `let column = index % WIDTH;
  let row = index / WIDTH;
  let xFaceWidth = WIDTH + 1u;
  let west = xFlux[xFluxOffset + row * xFaceWidth + column];
  let east = xFlux[xFluxOffset + row * xFaceWidth + column + 1u];
  let south = yFlux[yFluxOffset + row * WIDTH + column];
  let north = yFlux[yFluxOffset + (row + 1u) * WIDTH + column];
  let cellSizeX = parameters[parametersOffset];
  let cellSizeY = parameters[parametersOffset + 1u];
  let timeStep = parameters[parametersOffset + 2u];
  let delta = vec2<f32>(f32(column), f32(row)) - vec2<f32>(
    parameters[parametersOffset + 10u], parameters[parametersOffset + 11u]
  );
  let radius = max(parameters[parametersOffset + 12u], 0.0001);
  let source = max(1.0 - length(delta) / radius, 0.0) * parameters[parametersOffset + 13u];
  let forcing = parameters[parametersOffset + 5u] + source - parameters[parametersOffset + 6u];
  let nextDepth = clamp(
    depth[depthOffset + index] + timeStep *
      (forcing + (west - east) / cellSizeX + (south - north) / cellSizeY),
    0.0,
    parameters[parametersOffset + 9u]
  );
  depth[depthOffset + index] = nextDepth;
  ${display ? 'display[displayOffset + index] = nextDepth;' : ''}`
    });
  }
}
