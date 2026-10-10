// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUPolygonGeometryPort} from '../contracts/index';
import {GPUPolygonize, type GPUPolygonizeProps} from './gpu-polygonize';
import type {GPUPolygonizeOutput, GPUTopologyPrecisionPolicy} from './topology-types';

const OPERATION = 'GPUMakeValid';

/** Properties for {@link GPUMakeValid}. */
export type GPUMakeValidProps = {
  id?: string;
  /** Polygon boundaries to node and rebuild into bounded faces. */
  polygons: Omit<GPUPolygonGeometryPort, 'positions' | 'sourceIds'> & {
    positions: GraphDataView<'float32x2'>;
    sourceIds?: GraphDataView<'uint32'>;
  };
  intersectionCapacity: number;
  precision: GPUTopologyPrecisionPolicy;
  output: GPUPolygonizeOutput;
  uncertainCount?: GraphDataView<'uint32'>;
  spatialSort?: boolean;
  leafCapacity?: number;
};

/**
 * First topology repair operation: nodes all polygon boundary intersections and rebuilds valid
 * bounded faces independently for each source feature. Bow ties, self-crossing rings and mutually
 * crossing rings are split into polygon features; open/cut remnants remain in diagnostics.
 */
export class GPUMakeValid implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUMakeValidProps;

  constructor(props: GPUMakeValidProps) {
    this.id = props.id ?? 'make-valid';
    this.props = props;
    if (props.polygons.kind !== 'polygons') {
      throw new Error(`${this.id} polygons.kind must be polygons`);
    }
    if (props.polygons.ringOffsets.length < 2) {
      throw new Error(`${this.id} polygons require at least one ring`);
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {polygons} = props;
    const ringCount = polygons.ringOffsets.length - 1;
    const ringSourceIds = createTransientView(graph, `${id}-ring-source-ids`, 'uint32', ringCount);
    const sourceBinding = polygons.sourceIds
      ? [
          {
            name: 'inputSourceIds',
            view: polygons.sourceIds,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]
      : [];
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ring-sources`,
        operation: OPERATION,
        variant: 'ring-sources',
        bindings: [
          {name: 'featureOffsets', view: polygons.featureOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: polygons.polygonOffsets, type: 'u32', access: 'read'},
          ...sourceBinding,
          {name: 'ringSourceIds', view: ringSourceIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: ringCount,
        declarations: `const FEATURE_COUNT: u32 = ${polygons.featureOffsets.length - 1}u;
const POLYGON_COUNT: u32 = ${polygons.polygonOffsets.length - 1}u;`,
        body: `var polygonLow = 0u;
  var polygonHigh = POLYGON_COUNT;
  while (polygonLow + 1u < polygonHigh) {
    let middle = (polygonLow + polygonHigh) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= index) { polygonLow = middle; } else { polygonHigh = middle; }
  }
  var featureLow = 0u;
  var featureHigh = FEATURE_COUNT;
  while (featureLow + 1u < featureHigh) {
    let middle = (featureLow + featureHigh) / 2u;
    if (featureOffsets[featureOffsetsOffset + middle] <= polygonLow) { featureLow = middle; } else { featureHigh = middle; }
  }
  ringSourceIds[ringSourceIdsOffset + index] = ${polygons.sourceIds ? 'inputSourceIds[inputSourceIdsOffset + featureLow]' : 'featureLow'};`
      })
    ];
    const polygonizeProps: GPUPolygonizeProps = {
      id: `${id}-polygonize`,
      lines: {
        kind: 'lines',
        positions: polygons.positions,
        lineOffsets: polygons.ringOffsets,
        sourceIds: ringSourceIds
      },
      groupIds: ringSourceIds,
      intersectionCapacity: props.intersectionCapacity,
      precision: props.precision,
      output: props.output,
      uncertainCount: props.uncertainCount,
      spatialSort: props.spatialSort,
      leafCapacity: props.leafCapacity
    };
    nodes.push(...new GPUPolygonize(polygonizeProps).getCommandNodes(graph));
    return nodes;
  }
}
