// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  createRasterIterationState,
  getRasterIterationCondition
} from '../../gpu-raster/cost-distance/raster-relaxation';

const OPERATION = 'GPUTerrainFlow';

/** Sentinels written by {@link createTerrainFlowUnresolvedNodes}. @internal */
export type TerrainFlowUnresolvedCodes = {
  /** `flowDirections` word of a valid cell. */
  direction: number;
  /** `cellClasses` word of a valid cell. */
  cellClass: number;
  /** `streams` word of a valid cell. */
  stream: number;
};

/**
 * Returns the nodes that mark a non-converged depression fill or flat resolution (library rule 9):
 * a one-thread node reads the relaxation status words and arms an indirect dispatch only when one
 * of them stopped at its iteration limit, then a gated pass over the grid overwrites the bound
 * outputs on every valid cell. A converged run pays one tiny dispatch and its outputs are
 * untouched.
 *
 * A truncated fill leaves interior cells above the fixpoint (or at +infinity), and a truncated
 * flat relaxation leaves hop counts short, so directions, classes, receivers and everything routed
 * from them would look like a legitimate result. `filledElevation` depends only on the fill and is
 * overwritten only when the fill did not converge. Invalid cells (non-finite canonical elevation)
 * keep their codes. Pass views only for outputs that are bound or consumed downstream.
 *
 * @internal
 */
export function createTerrainFlowUnresolvedNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    /** Canonical elevation, NaN for invalid cells. */
    elevation: GraphDataView<'float32'>;
    /** Status word views (word 3 is 1 once converged) of the fill relaxation, if filling runs. */
    fillStatus?: GraphDataView<'uint32'>;
    /** Status word views of every flat-resolution relaxation, if flats are resolved. */
    flatStatuses: readonly GraphDataView<'uint32'>[];
    filledElevation?: GraphDataView<'float32'>;
    directions?: GraphDataView<'uint32'>;
    classes?: GraphDataView<'uint32'>;
    accumulation?: GraphDataView<'float32'>;
    streams?: GraphDataView<'uint32'>;
    codes: TerrainFlowUnresolvedCodes;
  }
): GPUCommandNode<Parameters>[] {
  const statuses = [...(props.fillStatus ? [props.fillStatus] : []), ...props.flatStatuses];
  if (statuses.length === 0) {
    return [];
  }
  const outputs: WGSLKernelBinding[] = [];
  for (const [name, view] of [
    ['filledElevation', props.filledElevation],
    ['directions', props.directions],
    ['classes', props.classes],
    ['accumulation', props.accumulation],
    ['streams', props.streams]
  ] as const) {
    if (view) {
      outputs.push({name, view, type: 'u32', access: 'read_write'});
    }
  }
  if (outputs.length === 0) {
    return [];
  }
  const gate = createRasterIterationState(
    graph,
    `${props.id}-unresolved`,
    OPERATION,
    props.cellCount,
    props.id
  );
  const {x, y, z} = gate.layout;
  const {condition, extraResources} = getRasterIterationCondition<Parameters>(
    gate,
    `${props.id}-unresolved-cells`
  );
  const statusBindings: WGSLKernelBinding[] = statuses.map((view, index) => ({
    name: `status${index}`,
    view,
    type: 'u32',
    access: 'read'
  }));
  const convergedWGSL = statuses.map(
    (_, index) => `status${index}[status${index}Offset + 3u] != 0u`
  );
  const write = (name: string, view: unknown, value: string) =>
    view ? `${name}[${name}Offset + index] = ${value};` : '';
  const armNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-unresolved-arm`,
    operation: OPERATION,
    variant: 'unresolved-arm',
    bindings: [
      ...statusBindings,
      {name: 'dispatch', view: gate.dispatch, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    // Word 3 of a relaxation status is 1 only when an iteration changed nothing.
    body: `let converged = ${convergedWGSL.join(' && ')};
  dispatch[dispatchOffset] = select(${x}u, 0u, converged);
  dispatch[dispatchOffset + 1u] = ${y}u;
  dispatch[dispatchOffset + 2u] = ${z}u;`
  });
  const fillBinding: WGSLKernelBinding[] =
    props.fillStatus && props.filledElevation
      ? [{name: 'fillStatus', view: props.fillStatus, type: 'u32', access: 'read'}]
      : [];
  const cellNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-unresolved-cells`,
    operation: OPERATION,
    variant: 'unresolved-cells',
    bindings: [
      {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
      ...fillBinding,
      ...outputs
    ],
    invocationCount: gate.invocationCount,
    workgroupSize: 256,
    body: `if ((bitcast<u32>(elevation[elevationOffset + index]) & 0x7fffffffu) >= 0x7f800000u) { return; }
  ${write('directions', props.directions, `${props.codes.direction}u`)}
  ${write('classes', props.classes, `${props.codes.cellClass}u`)}
  ${write('accumulation', props.accumulation, '0x7fc00000u')}
  ${write('streams', props.streams, `${props.codes.stream}u`)}
  ${fillBinding.length > 0 ? `if (fillStatus[fillStatusOffset + 3u] == 0u) { ${write('filledElevation', props.filledElevation, '0x7fc00000u')} }` : ''}`,
    condition,
    extraResources
  });
  return [armNode, cellNode];
}
