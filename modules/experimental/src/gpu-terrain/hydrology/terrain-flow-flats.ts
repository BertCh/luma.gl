// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  createContributorTransientView,
  getRasterGridWGSL
} from '../../gpu-raster/cost-distance/raster-grid-utils';
import {
  createRasterTiledRelaxation,
  createRasterTiledRelaxationNodes,
  getRasterRelaxationTileWGSL
} from '../../gpu-raster/cost-distance/raster-relaxation';
import {TERRAIN_FLOW_SWEEP_AFTER_ITERATION, type TerrainFlowGrid} from './terrain-flow-passes';

const OPERATION = 'GPUTerrainFlow';

/** WGSL bit-test helpers for kernels that do not include the raster grid declarations. */
const FLOAT_HELPERS_WGSL = /* wgsl */ `
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }
fn getInfinity() -> f32 { var bits = 0x7f800000u; return bitcast<f32>(bits); }`;

/**
 * Returns the flat-resolution nodes (Barnes, Lehman and Mulla 2014) that run after the D8
 * direction node and rewrite `classes`, `directions` and `receivers` of drainable flat cells.
 *
 * Terminology on the routing `surface`: a cell "has flow" when its class is `draining` or
 * `outlet`; a flat cell has class `flat`. A low edge is a has-flow cell next to an equal-surface
 * flat cell, a high edge is a flat cell next to a strictly higher valid cell.
 *
 * Three exact integer hop-count fields are computed by GPU tiled min-relaxations
 * (`createRasterTiledRelaxation`) restricted to 8-neighbors of exactly equal surface:
 *
 * - `towardLower`: hops from low edges (low edge = 1) through flat cells; unreached = +infinity.
 * - `awayFromHigher`: hops from high edges (high edge = 1) through flat cells; unreached = 0.
 * - `flatMaximum`: the maximum `awayFromHigher` over the equal-surface 8-connected component of
 *   all valid cells (also draining ones, as RichDEM labels all equal-elevation cells), computed as
 *   a minimum over negated values.
 *
 * A mask `2 * towardLower + (awayFromHigher > 0 ? flatMaximum - awayFromHigher : 0)` is written for
 * flat cells with finite `towardLower` (2 for low edges, +infinity elsewhere). Each such flat
 * cell then routes to the equal-surface neighbor with the smallest mask strictly below its own.
 * Ties between equal masks keep the lowest luma D8 direction index (E, SE, S, SW, W, NW, N, NE),
 * which differs from RichDEM's neighbor order; the resulting flow field is equally valid. Cells
 * with infinite `towardLower` (flats without an outlet) stay `flat`. Each pass runs at most
 * `maxIterations` gated relaxations; `converged` is 1 only when all three reached their fixpoint.
 * Cell size mode does not matter: hop counts are unitless.
 *
 * @internal
 */
export function createTerrainFlowFlatNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TerrainFlowGrid & {
    id: string;
    maxIterations: number;
    /** Canonical elevation, NaN for invalid cells. */
    elevation: GraphDataView<'float32'>;
    /** Routing surface (filled elevation or `elevation`), NaN for invalid cells. */
    surface: GraphDataView<'float32'>;
    settings: GraphDataView<'float32'>;
    /** D8 classes written by the direction node; flat cells that get routed become `draining`. */
    classes: GraphDataView<'uint32'>;
    directions?: GraphDataView<'uint32'>;
    receivers?: GraphDataView<'uint32'>;
    converged?: GraphDataView<'uint32'>;
    iterationCount?: GraphDataView<'uint32'>;
    cellClass: {draining: number; flat: number; pit: number; outlet: number; invalid: number};
  }
): GPUCommandNode<Parameters>[] {
  const cellCount = props.width * props.height;
  const {cellClass} = props;
  const makeView = (suffix: string) =>
    createContributorTransientView<'float32', Parameters>(
      graph,
      OPERATION,
      props.id,
      `${props.id}-flats-${suffix}`,
      'float32',
      cellCount
    );
  const lowerValues = makeView('lower-values');
  const lowerAuxiliary = makeView('lower-aux');
  const higherValues = makeView('higher-values');
  const higherAuxiliary = makeView('higher-aux');
  const maximumValues = makeView('maximum-values');
  const mask = makeView('mask');

  const prepareEdges = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-flats-prepare`,
    operation: OPERATION,
    variant: 'flats-prepare',
    bindings: [
      {name: 'surface', view: props.surface, type: 'f32', access: 'read'},
      {name: 'classes', view: props.classes, type: 'u32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'lowerValues', view: lowerValues, type: 'f32', access: 'read_write'},
      {name: 'lowerAuxiliary', view: lowerAuxiliary, type: 'f32', access: 'read_write'},
      {name: 'higherValues', view: higherValues, type: 'f32', access: 'read_write'},
      {name: 'higherAuxiliary', view: higherAuxiliary, type: 'f32', access: 'read_write'}
    ],
    invocationCount: cellCount,
    declarations: getRasterGridWGSL(props),
    // Low edge: has-flow cell beside an equal-surface flat. High edge: flat beside a higher cell.
    body: `_ = settings[settingsOffset];
  let centerClass = classes[classesOffset + index];
  let centerSurface = surface[surfaceOffset + index];
  let isFlat = centerClass == ${cellClass.flat}u;
  let hasFlow = centerClass == ${cellClass.draining}u || centerClass == ${cellClass.outlet}u;
  var isLowEdge = false;
  var isHighEdge = false;
  if (isFlat || hasFlow) {
    for (var direction = 0u; direction < 8u; direction++) {
      let neighbor = getD8Neighbor(index, direction);
      if (neighbor == GRID_NONE) { continue; }
      let neighborClass = classes[classesOffset + neighbor];
      if (neighborClass == ${cellClass.invalid}u) { continue; }
      let neighborSurface = surface[surfaceOffset + neighbor];
      if (hasFlow && neighborClass == ${cellClass.flat}u && neighborSurface == centerSurface) {
        isLowEdge = true;
      }
      if (isFlat && neighborSurface > centerSurface) { isHighEdge = true; }
    }
  }
  lowerAuxiliary[lowerAuxiliaryOffset + index] = select(getQuietNaN(), centerSurface, isFlat || isLowEdge);
  lowerValues[lowerValuesOffset + index] = select(getInfinity(), 1.0, isLowEdge);
  higherAuxiliary[higherAuxiliaryOffset + index] = select(getQuietNaN(), centerSurface, isFlat);
  higherValues[higherValuesOffset + index] = select(getInfinity(), 1.0, isHighEdge);`
  });

  const buildRelaxation = (
    suffix: string,
    values: GraphDataView<'float32'>,
    auxiliary: GraphDataView<'float32'>,
    candidate: string
  ) => {
    const relaxationProps = {
      id: `${props.id}-flats-${suffix}`,
      operation: OPERATION,
      width: props.width,
      height: props.height,
      cellSizeMode: props.cellSizeMode,
      maxIterations: props.maxIterations
    };
    const {relaxation, resetNodes} = createRasterTiledRelaxation<Parameters>(
      graph,
      relaxationProps,
      true
    );
    const relaxNodes = createRasterTiledRelaxationNodes<Parameters>(graph, {
      ...relaxationProps,
      relaxation,
      sweepAfterIteration: TERRAIN_FLOW_SWEEP_AFTER_ITERATION,
      values,
      auxiliary,
      settings: props.settings,
      declarations: `
fn getRelaxationCandidate(neighborValue: f32, neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  if (neighborAuxiliary == centerAuxiliary) { return ${candidate}; }
  return getInfinity();
}`
    });
    return {relaxation, resetNodes, relaxNodes};
  };
  const lower = buildRelaxation('lower', lowerValues, lowerAuxiliary, 'neighborValue + 1.0');
  const higher = buildRelaxation('higher', higherValues, higherAuxiliary, 'neighborValue + 1.0');
  const maximum = buildRelaxation('maximum', maximumValues, props.surface, 'neighborValue');

  const tileWGSL = getRasterRelaxationTileWGSL(props.width);
  // Marks the tiles of the lower and higher hop-count seeds (finite starting values).
  const seedEdges = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-flats-seed-edges`,
    operation: OPERATION,
    variant: 'flats-seed-edges',
    bindings: [
      {name: 'lowerValues', view: lowerValues, type: 'f32', access: 'read'},
      {name: 'higherValues', view: higherValues, type: 'f32', access: 'read'},
      {
        name: 'lowerStamps',
        view: lower.relaxation.tileStamps,
        type: 'atomic<u32>',
        access: 'read_write'
      },
      {
        name: 'higherStamps',
        view: higher.relaxation.tileStamps,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ],
    invocationCount: cellCount,
    declarations: `${FLOAT_HELPERS_WGSL}\n${tileWGSL}`,
    body: `if (isFiniteValue(lowerValues[lowerValuesOffset + index])) {
    atomicMax(&lowerStamps[lowerStampsOffset + getRelaxationTile(index)], 1u);
  }
  if (isFiniteValue(higherValues[higherValuesOffset + index])) {
    atomicMax(&higherStamps[higherStampsOffset + getRelaxationTile(index)], 1u);
  }`
  });

  // Negated awayFromHigher so a min-relaxation computes the component maximum; unreached = 0. The
  // finite (negative) starting values are the seeds of the maximum relaxation.
  const prepareMaximum = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-flats-prepare-maximum`,
    operation: OPERATION,
    variant: 'flats-prepare-maximum',
    bindings: [
      {name: 'higherValues', view: higherValues, type: 'f32', access: 'read'},
      {name: 'maximumValues', view: maximumValues, type: 'f32', access: 'read_write'},
      {
        name: 'maximumStamps',
        view: maximum.relaxation.tileStamps,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ],
    invocationCount: cellCount,
    declarations: `${FLOAT_HELPERS_WGSL}\n${tileWGSL}`,
    body: `let higher = higherValues[higherValuesOffset + index];
  let reached = isFiniteValue(higher);
  maximumValues[maximumValuesOffset + index] = select(0.0, -higher, reached);
  if (reached) {
    atomicMax(&maximumStamps[maximumStampsOffset + getRelaxationTile(index)], 1u);
  }`
  });

  const maskNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-flats-mask`,
    operation: OPERATION,
    variant: 'flats-mask',
    bindings: [
      {name: 'lowerValues', view: lowerValues, type: 'f32', access: 'read'},
      {name: 'higherValues', view: higherValues, type: 'f32', access: 'read'},
      {name: 'maximumValues', view: maximumValues, type: 'f32', access: 'read'},
      {name: 'mask', view: mask, type: 'f32', access: 'read_write'}
    ],
    invocationCount: cellCount,
    declarations: FLOAT_HELPERS_WGSL,
    body: `let towardLower = lowerValues[lowerValuesOffset + index];
  if (!isFiniteValue(towardLower)) {
    mask[maskOffset + index] = getInfinity();
    return;
  }
  let awayFromHigher = higherValues[higherValuesOffset + index];
  var awayTerm = 0.0;
  if (isFiniteValue(awayFromHigher)) {
    awayTerm = -maximumValues[maximumValuesOffset + index] - awayFromHigher;
  }
  mask[maskOffset + index] = 2.0 * towardLower + awayTerm;`
  });

  const routeBindings: WGSLKernelBinding[] = [
    {name: 'surface', view: props.surface, type: 'f32', access: 'read'},
    {name: 'mask', view: mask, type: 'f32', access: 'read'},
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
    {name: 'classes', view: props.classes, type: 'u32', access: 'read_write'}
  ];
  if (props.directions) {
    routeBindings.push({
      name: 'directions',
      view: props.directions,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.receivers) {
    routeBindings.push({
      name: 'receivers',
      view: props.receivers,
      type: 'u32',
      access: 'read_write'
    });
  }
  const routeNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-flats-route`,
    operation: OPERATION,
    variant: 'flats-route',
    bindings: routeBindings,
    invocationCount: cellCount,
    declarations: getRasterGridWGSL(props),
    // Reads only 'surface' and 'mask' of neighbors; writes only this cell, so there are no races.
    body: `_ = settings[settingsOffset];
  if (classes[classesOffset + index] != ${cellClass.flat}u) { return; }
  let centerMask = mask[maskOffset + index];
  if (!isFiniteValue(centerMask)) { return; }
  let centerSurface = surface[surfaceOffset + index];
  var bestMask = centerMask;
  var bestDirection = 8u;
  var bestNeighbor = GRID_NONE;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE || surface[surfaceOffset + neighbor] != centerSurface) { continue; }
    let neighborMask = mask[maskOffset + neighbor];
    // Strictly smaller: ties keep the lowest direction index.
    if (isFiniteValue(neighborMask) && neighborMask < bestMask) {
      bestMask = neighborMask;
      bestDirection = direction;
      bestNeighbor = neighbor;
    }
  }
  if (bestDirection < 8u) {
    classes[classesOffset + index] = ${cellClass.draining}u;
    ${props.directions ? 'directions[directionsOffset + index] = getD8Code(bestDirection);' : ''}
    ${props.receivers ? 'receivers[receiversOffset + index] = bestNeighbor;' : ''}
  }`
  });

  const nodes: GPUCommandNode<Parameters>[] = [
    prepareEdges,
    ...lower.resetNodes,
    ...higher.resetNodes,
    ...maximum.resetNodes,
    seedEdges,
    ...lower.relaxNodes,
    ...higher.relaxNodes,
    prepareMaximum,
    ...maximum.relaxNodes,
    maskNode,
    routeNode
  ];
  if (props.converged || props.iterationCount) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-flats-finalize`,
        operation: OPERATION,
        variant: 'flats-finalize',
        bindings: [
          {name: 'lowerStatus', view: lower.relaxation.state.status, type: 'u32', access: 'read'},
          {name: 'higherStatus', view: higher.relaxation.state.status, type: 'u32', access: 'read'},
          {
            name: 'maximumStatus',
            view: maximum.relaxation.state.status,
            type: 'u32',
            access: 'read'
          },
          ...(props.converged
            ? [
                {
                  name: 'converged',
                  view: props.converged,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : []),
          ...(props.iterationCount
            ? [
                {
                  name: 'iterationCount',
                  view: props.iterationCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: 1,
        // Status word 3 is 1 only when a relaxation iteration changed nothing.
        body: `${
          props.converged
            ? `converged[convergedOffset] = select(0u, 1u,
    lowerStatus[lowerStatusOffset + 3u] == 1u &&
    higherStatus[higherStatusOffset + 3u] == 1u &&
    maximumStatus[maximumStatusOffset + 3u] == 1u);`
            : ''
        }
  ${
    props.iterationCount
      ? `iterationCount[iterationCountOffset] = lowerStatus[lowerStatusOffset + 2u] +
    higherStatus[higherStatusOffset + 2u] + maximumStatus[maximumStatusOffset + 2u];`
      : ''
  }`
      })
    );
  }
  return nodes;
}
