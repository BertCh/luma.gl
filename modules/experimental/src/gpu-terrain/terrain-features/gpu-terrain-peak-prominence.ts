// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, type Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '../../gpu-raster';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterGrid
} from '../../gpu-raster/raster-band-utils';
import {
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {getSpillCommonWGSL} from '../hydrology/terrain-flow-spill-utils';
import {getPeakProminenceMergeNodes} from './terrain-peak-prominence-tree';

const OPERATION = 'GPUTerrainPeakProminence';

/** Dense, caller-owned outputs of {@link GPUTerrainPeakProminence}. */
export type GPUTerrainPeakProminenceOutput = {
  /** Height minus key saddle. Zero for nonpeaks; +Infinity for each component's elder peak; NaN for nodata. */
  prominence: GraphDataView<'float32'>;
  /** Optional key-saddle elevation. -Infinity for component elders; NaN for nodata. */
  keySaddle?: GraphDataView<'float32'>;
  /** Optional elder peak cell at the merge, not the union-find parent. 0xffffffff for component elders and nodata. */
  parentPeak?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUTerrainPeakProminence}. No packed per-frame parameters are needed. */
export type GPUTerrainPeakProminenceProps = {
  /** Node prefix. Defaults to 'terrain-peak-prominence'. */
  id?: string;
  /** Raster dimensions. Connectivity is the spill graph's eight-neighbor connectivity. */
  grid: GPURasterMetadata;
  /** Per-frame elevation; invalid cells are excluded, not connected to an ocean. */
  elevation: GPURasterBand;
  /** Dense outputs, one value per cell. */
  output: GPUTerrainPeakProminenceOutput;
};

/** Maximum cells: two child words per tree node, five nodes per cell, 40 bytes per binding. */
export function getGPUTerrainPeakProminenceMaximumCellCount(device: Device): number {
  return Math.min(0x19999999, Math.floor(device.limits.maxStorageBufferBindingSize / 40));
}

/**
 * Computes upper-excursion-set (elder-rule, zero-dimensional persistence) peak prominence entirely
 * on the GPU. The key saddle of a peak is the highest level at which its component joins one
 * containing an elder peak. Higher elevation wins; equal-height peaks use the lowest row-major
 * index. Thus a plateau has one representative with positive prominence; other plateau cells
 * have zero prominence. A disconnected component's surviving peak has infinite prominence.
 * Nodata is a barrier and the raster boundary is closed: no artificial sea-level reference.
 *
 * Uses the depression-fill spill graph's unordered neighbor edges and ordered float keys.
 * The graph is refined to one label per cell: collapsing a tile to its
 * boundary outlets would discard peaks born inside that tile. Edge saddle = min(z(a), z(b)).
 * A GPU radix sort orders edges by descending saddle. Parallel atomic-min parent splicing
 * constructs their join tree; cycle edges are unary nodes. Gated pointer doubling propagates
 * the elder peak to every ancestor in logarithmic dispatch depth, then one lane per binary
 * merge records the losing peak. Equal saddle edges preserve the stable edge order (cell index,
 * then E, SE, S, SW), including parentPeak ties.
 *
 * Construction and reduction are parallel and use O(cellCount) scratch. A contended splice
 * lane can traverse a long increasing chain; performance measurements remain pending. There
 * is no CPU fallback, readback, elevation quantization, or data-dependent iteration cap.
 */
export class GPUTerrainPeakProminence implements GPUCommandNodeProducer {
  /** Node prefix. */
  readonly id: string;
  /** Validated construction properties. */
  readonly props: GPUTerrainPeakProminenceProps;

  constructor(props: GPUTerrainPeakProminenceProps) {
    this.id = props.id ?? 'terrain-peak-prominence';
    this.props = props;
    const count = validateRasterGrid(this.id, props.grid);
    for (const [name, view, format] of [
      ['prominence', props.output.prominence, 'float32'],
      ['keySaddle', props.output.keySaddle, 'float32'],
      ['parentPeak', props.output.parentPeak, 'uint32']
    ] as const) {
      if (!view) continue;
      validatePackedView(view, [format], `${this.id} ${name}`);
      // Every output has one element per raster cell.
      assert(view.length === count);
    }
    validateRasterBuffersDistinct(
      this.id,
      Object.values(props.output),
      getRasterBandViews(props.elevation)
    );
  }

  /** Canonicalization, spill-edge sort, parallel join tree, and optional output publication. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props.grid;
    const count = width * height;
    // The child table must fit one binding; no silently truncated graph.
    assert(count <= getGPUTerrainPeakProminenceMaximumCellCount(graph.device));
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, Object.values(props.output));
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const keys = createTransientView(graph, `${id}-edge-keys`, 'uint32', count * 4);
    const edges = createTransientView(graph, `${id}-edges`, 'uint32', count * 4);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', count * 4);
    const sortedEdges = createTransientView(graph, `${id}-sorted-edges`, 'uint32', count * 4);
    const result = createTransientView(graph, `${id}-result`, 'uint32', count * 2);
    const common = `${getSpillCommonWGSL(width, height)}\nconst CELL_COUNT: u32 = ${count}u;`;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...source.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        bindings: [
          {name: 'values', view: source.band.storage.values, type: 'f32', access: 'read'},
          {name: 'validity', view: source.band.validity!, type: 'u32', access: 'read'},
          {name: 'result', view: result, type: 'u32', access: 'read_write'}
        ],
        invocationCount: count,
        declarations: common,
        body: `let valid = validity[validityOffset + index] != 0u && isSpillFinite(values[valuesOffset + index]);
          result[resultOffset + index] = select(0x7fc00000u, 0xff800000u, valid);
          result[resultOffset + CELL_COUNT + index] = 0xffffffffu;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-spill-edges`,
        operation: OPERATION,
        bindings: [
          {name: 'values', view: source.band.storage.values, type: 'f32', access: 'read'},
          {name: 'validity', view: source.band.validity!, type: 'u32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'edges', view: edges, type: 'u32', access: 'read_write'}
        ],
        invocationCount: count * 4,
        declarations: common,
        body: `let cell = index / 4u;
          let neighbor = getSpillNeighbor(cell, index % 4u);
          var key = 0u;
          if (neighbor != SPILL_NONE && validity[validityOffset + cell] != 0u && validity[validityOffset + neighbor] != 0u) {
            let first = values[valuesOffset + cell]; let second = values[valuesOffset + neighbor];
            if (isSpillFinite(first) && isSpillFinite(second)) {
              // Normalize signed zero so equal numerical saddles have one stable key.
              let saddle = min(first, second);
              key = getSpillKey(select(saddle, 0.0, saddle == 0.0));
            }
          }
          keys[keysOffset + index] = key; edges[edgesOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-sort`,
        keys,
        values: edges,
        outputKeys: sortedKeys,
        outputValues: sortedEdges,
        direction: 'descending',
        algorithm: 'radix'
      }).getCommandNodes(graph),
      ...getPeakProminenceMergeNodes(graph, {
        id,
        width,
        height,
        values: source.band.storage.values,
        validity: source.band.validity!,
        sortedKeys,
        sortedEdges,
        result
      })
    ];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        bindings: [
          {name: 'values', view: source.band.storage.values, type: 'f32', access: 'read'},
          {name: 'result', view: result, type: 'u32', access: 'read'},
          {name: 'prominence', view: props.output.prominence, type: 'f32', access: 'read_write'},
          ...(props.output.keySaddle
            ? [
                {
                  name: 'saddle',
                  view: props.output.keySaddle,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : []),
          ...(props.output.parentPeak
            ? [
                {
                  name: 'parent',
                  view: props.output.parentPeak,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: count,
        declarations: common,
        body: `let saddleBits = result[resultOffset + index];
        var prominenceBits = 0x7fc00000u;
        if (saddleBits == 0xff800000u) { prominenceBits = 0x7f800000u; }
        else if (saddleBits != 0x7fc00000u) { prominenceBits = bitcast<u32>(values[valuesOffset + index] - bitcast<f32>(saddleBits)); }
        prominence[prominenceOffset + index] = bitcast<f32>(prominenceBits);
        ${props.output.keySaddle ? 'saddle[saddleOffset + index] = bitcast<f32>(saddleBits);' : ''}
        ${props.output.parentPeak ? 'parent[parentOffset + index] = result[resultOffset + CELL_COUNT + index];' : ''}`
      })
    );
    return nodes;
  }
}
