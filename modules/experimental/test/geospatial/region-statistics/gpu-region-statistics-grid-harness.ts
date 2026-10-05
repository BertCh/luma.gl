// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, GPUGridIndex, type GPUGridIndexView} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  decodeGPURegionStatistics,
  getGPURegionStatisticsSummaryLength,
  GPURegionStatistics,
  type GPURegionStatisticsProps,
  type GPURegionStatisticsResult
} from '../../../src/geospatial/region-statistics';
import {
  createInputBuffer,
  createOutputBuffer,
  readCompactIds,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Options for {@link GridStatisticsHarness}. */
export type GridStatisticsHarnessOptions = {
  device: Device;
  id: string;
  positions: Float32Array;
  values: Float32Array;
  sourceIds?: Uint32Array;
  selectionKind: 'rectangle' | 'radius' | 'polygon';
  vertexCapacity?: number;
  candidateCapacity: number;
  binCount?: number;
  histogramDomain?: 'selection' | readonly [number, number];
  gridSize?: readonly [number, number];
  domain?: readonly [number, number, number, number];
  withOutput?: boolean;
  withOutputMask?: boolean;
  /** `'separate-graph'` builds the index once in its own graph, see {@link buildIndex}. */
  indexMode: 'same-graph' | 'separate-graph';
};

/** Everything read back from one contributor encoding. */
export type ContributorReadback = {
  words: Uint32Array;
  result: GPURegionStatisticsResult;
  ids?: number[];
  outputOverflow?: number;
  outputTotal?: number;
  mask?: number[];
};

type ContributorBuffers = {
  summary: Buffer;
  ids?: Buffer;
  count?: Buffer;
  overflow?: Buffer;
  total?: Buffer;
  mask?: Buffer;
};

type Compiled = ReturnType<GPUCommandGraph['compile']>;

/**
 * Builds a brute-force `GPURegionStatistics` graph and a grid-index graph over the same inputs and
 * per-frame query buffers so tests and benchmarks can compare them.
 */
export class GridStatisticsHarness {
  readonly rowCount: number;
  readonly binCount: number;
  readonly bruteCompiled: Compiled;
  readonly gridCompiled: Compiled;
  readonly indexCompiled?: Compiled;
  readonly device: Device;
  private readonly options: GridStatisticsHarnessOptions;
  private readonly inputBuffers: Buffer[] = [];
  private readonly indexBuffers: Buffer[] = [];
  private readonly brute: ContributorBuffers;
  private readonly grid: ContributorBuffers;
  private readonly bounds: GPUParameterBuffer<'float32'>;
  private readonly circle: GPUParameterBuffer<'float32'>;
  private readonly vertices: GPUParameterBuffer<'float32'>;
  private readonly vertexCount: GPUParameterBuffer<'uint32'>;
  private readonly vertexCapacity: number;

  constructor(options: GridStatisticsHarnessOptions) {
    const {device, id} = options;
    this.options = options;
    this.device = device;
    this.rowCount = options.positions.length / 2;
    this.binCount = options.binCount ?? 0;
    this.vertexCapacity = options.vertexCapacity ?? 64;
    const gridSize = options.gridSize ?? [64, 64];
    const domain = options.domain ?? [0, 0, 1000, 1000];
    const cellCount = gridSize[0] * gridSize[1];

    this.bounds = new GPUParameterBuffer(device, {
      id: `${id}-bounds`,
      format: 'float32',
      length: 4
    });
    this.circle = new GPUParameterBuffer(device, {
      id: `${id}-circle`,
      format: 'float32',
      length: 3
    });
    this.vertices = new GPUParameterBuffer(device, {
      id: `${id}-vertices`,
      format: 'float32',
      length: this.vertexCapacity * 2
    });
    this.vertexCount = new GPUParameterBuffer(device, {
      id: `${id}-vertex-count`,
      format: 'uint32',
      length: 1
    });

    const positionsBuffer = createInputBuffer(device, options.positions);
    const valuesBuffer = createInputBuffer(device, options.values);
    const sourceIdsBuffer = options.sourceIds
      ? createInputBuffer(device, options.sourceIds)
      : undefined;
    this.inputBuffers.push(
      positionsBuffer,
      valuesBuffer,
      ...(sourceIdsBuffer ? [sourceIdsBuffer] : [])
    );

    const cellOffsetsBuffer = createOutputBuffer(device, cellCount + 1);
    const objectIdsBuffer = createOutputBuffer(device, this.rowCount);
    const indexCountBuffer = createOutputBuffer(device, 1);
    const indexOverflowBuffer = createOutputBuffer(device, 1);
    this.indexBuffers.push(
      cellOffsetsBuffer,
      objectIdsBuffer,
      indexCountBuffer,
      indexOverflowBuffer
    );

    const createGraph = (tag: 'brute' | 'grid') => {
      const graph = new GPUCommandGraph(device, {id: `${id}-${tag}`});
      return {
        graph,
        positions: importGraphBuffer(
          graph,
          'positions',
          positionsBuffer,
          'float32x2',
          this.rowCount
        ),
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', this.rowCount),
        sourceIds: sourceIdsBuffer
          ? importGraphBuffer(graph, 'source-ids', sourceIdsBuffer, 'uint32', this.rowCount)
          : undefined
      };
    };
    const createContributor = (
      tag: 'brute' | 'grid',
      inputs: ReturnType<typeof createGraph>
    ): {props: GPURegionStatisticsProps; buffers: ContributorBuffers} => {
      const {graph} = inputs;
      const buffers: ContributorBuffers = {
        summary: createOutputBuffer(device, getGPURegionStatisticsSummaryLength(this.binCount))
      };
      const props: GPURegionStatisticsProps = {
        id: `${id}-${tag}`,
        selection:
          options.selectionKind === 'rectangle'
            ? {kind: 'rectangle', bounds: this.bounds.importToGraph(graph)}
            : options.selectionKind === 'radius'
              ? {kind: 'radius', circle: this.circle.importToGraph(graph)}
              : {
                  kind: 'polygon',
                  vertices: importGraphBuffer(
                    graph,
                    'vertices',
                    this.vertices.buffer,
                    'float32x2',
                    this.vertexCapacity
                  ),
                  vertexCount: this.vertexCount.importToGraph(graph)
                },
        positions: inputs.positions,
        values: inputs.values,
        sourceIds: inputs.sourceIds,
        histogram: this.binCount
          ? {binCount: this.binCount, domain: options.histogramDomain}
          : undefined,
        summary: importGraphBuffer(
          graph,
          'summary',
          buffers.summary,
          'uint32',
          getGPURegionStatisticsSummaryLength(this.binCount)
        )
      };
      if (options.withOutput) {
        buffers.ids = createOutputBuffer(device, this.rowCount);
        buffers.count = createOutputBuffer(device, 1);
        buffers.overflow = createOutputBuffer(device, 1);
        buffers.total = createOutputBuffer(device, 1);
        props.output = {
          ids: importGraphBuffer(graph, 'ids', buffers.ids, 'uint32', this.rowCount),
          count: importGraphBuffer(graph, 'count', buffers.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1),
          totalCount: importGraphBuffer(graph, 'total', buffers.total, 'uint32', 1)
        };
      }
      if (options.withOutputMask) {
        buffers.mask = createOutputBuffer(device, this.rowCount);
        props.outputMask = importGraphBuffer(graph, 'mask', buffers.mask, 'uint32', this.rowCount);
      }
      return {props, buffers};
    };

    const bruteInputs = createGraph('brute');
    const bruteContributor = createContributor('brute', bruteInputs);
    this.brute = bruteContributor.buffers;
    bruteInputs.graph.add(new GPURegionStatistics(bruteContributor.props));
    this.bruteCompiled = bruteInputs.graph.compile();

    const gridInputs = createGraph('grid');
    const gridContributor = createContributor('grid', gridInputs);
    this.grid = gridContributor.buffers;
    let index: GPUGridIndexView;
    if (options.indexMode === 'same-graph') {
      const gridIndex = new GPUGridIndex({
        id: `${id}-index`,
        positions: gridInputs.positions,
        gridSize,
        bounds: domain,
        cellOffsets: importGraphBuffer(
          gridInputs.graph,
          'cell-offsets',
          cellOffsetsBuffer,
          'uint32',
          cellCount + 1
        ),
        objectIds: importGraphBuffer(
          gridInputs.graph,
          'object-ids',
          objectIdsBuffer,
          'uint32',
          this.rowCount
        ),
        count: importGraphBuffer(gridInputs.graph, 'index-count', indexCountBuffer, 'uint32', 1),
        overflow: importGraphBuffer(
          gridInputs.graph,
          'index-overflow',
          indexOverflowBuffer,
          'uint32',
          1
        )
      });
      gridInputs.graph.add(gridIndex);
      index = gridIndex;
    } else {
      const indexGraph = new GPUCommandGraph(device, {id: `${id}-index-graph`});
      const indexPositions = importGraphBuffer(
        indexGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        this.rowCount
      );
      indexGraph.add(
        new GPUGridIndex({
          id: `${id}-index`,
          positions: indexPositions,
          gridSize,
          bounds: domain,
          cellOffsets: importGraphBuffer(
            indexGraph,
            'cell-offsets',
            cellOffsetsBuffer,
            'uint32',
            cellCount + 1
          ),
          objectIds: importGraphBuffer(
            indexGraph,
            'object-ids',
            objectIdsBuffer,
            'uint32',
            this.rowCount
          ),
          count: importGraphBuffer(indexGraph, 'index-count', indexCountBuffer, 'uint32', 1),
          overflow: importGraphBuffer(
            indexGraph,
            'index-overflow',
            indexOverflowBuffer,
            'uint32',
            1
          )
        })
      );
      this.indexCompiled = indexGraph.compile();
      index = {
        gridSize,
        bounds: domain,
        cellOffsets: importGraphBuffer(
          gridInputs.graph,
          'cell-offsets',
          cellOffsetsBuffer,
          'uint32',
          cellCount + 1
        ),
        objectIds: importGraphBuffer(
          gridInputs.graph,
          'object-ids',
          objectIdsBuffer,
          'uint32',
          this.rowCount
        ),
        count: importGraphBuffer(gridInputs.graph, 'index-count', indexCountBuffer, 'uint32', 1),
        overflow: importGraphBuffer(
          gridInputs.graph,
          'index-overflow',
          indexOverflowBuffer,
          'uint32',
          1
        )
      } as GPUGridIndexView;
    }
    gridInputs.graph.add(
      new GPURegionStatistics({
        ...gridContributor.props,
        spatialIndex: {kind: 'grid', index, candidateCapacity: options.candidateCapacity}
      })
    );
    this.gridCompiled = gridInputs.graph.compile();
  }

  /** Builds the index once. Only for `'separate-graph'` mode (same-graph rebuilds every encoding). */
  buildIndex(): void {
    if (this.indexCompiled) {
      submitGraph(this.device, this.indexCompiled, undefined);
    }
  }

  setRectangle(bounds: ArrayLike<number>): void {
    this.bounds.write(Float32Array.from(bounds));
  }

  setCircle(circle: ArrayLike<number>): void {
    this.circle.write(Float32Array.from(circle));
  }

  /** Writes `[x, y, ...]` vertices and an explicit active vertex count. */
  setPolygon(vertices: ArrayLike<number>, vertexCount: number = vertices.length / 2): void {
    const padded = new Float32Array(this.vertexCapacity * 2);
    padded.set(Array.from(vertices).slice(0, padded.length));
    this.vertices.write(padded);
    this.vertexCount.write(Uint32Array.of(vertexCount));
  }

  /** Encodes both contributors (and the index when it lives in the grid graph). */
  encodeBoth(): void {
    submitGraph(this.device, this.bruteCompiled, undefined);
    submitGraph(this.device, this.gridCompiled, undefined);
  }

  encodeBrute(): void {
    submitGraph(this.device, this.bruteCompiled, undefined);
  }

  encodeGrid(): void {
    submitGraph(this.device, this.gridCompiled, undefined);
  }

  /** Small readback that waits for the most recent submission of either contributor. */
  async sync(which: 'brute' | 'grid'): Promise<void> {
    await (which === 'brute' ? this.brute : this.grid).summary.readAsync(0, 4);
  }

  /** Reads summary, optional IDs, optional mask. */
  async read(which: 'brute' | 'grid'): Promise<ContributorReadback> {
    const buffers = which === 'brute' ? this.brute : this.grid;
    const length = getGPURegionStatisticsSummaryLength(this.binCount);
    const words = Uint32Array.from(await readUint32(buffers.summary, length));
    const readback: ContributorReadback = {words, result: decodeGPURegionStatistics(words)};
    if (buffers.ids && buffers.count && buffers.overflow && buffers.total) {
      readback.ids = await readCompactIds(buffers.ids, buffers.count);
      readback.outputOverflow = (await readUint32(buffers.overflow, 1))[0];
      readback.outputTotal = (await readUint32(buffers.total, 1))[0];
    }
    if (buffers.mask) {
      readback.mask = await readUint32(buffers.mask, this.rowCount);
    }
    return readback;
  }

  /** Reads `[count, overflow]` of the index. */
  async readIndexCount(): Promise<number> {
    return (await readUint32(this.indexBuffers[2], 1))[0];
  }

  destroy(): void {
    this.bruteCompiled.destroy();
    this.gridCompiled.destroy();
    this.indexCompiled?.destroy();
    for (const parameters of [this.bounds, this.circle, this.vertices, this.vertexCount]) {
      parameters.destroy();
    }
    for (const buffers of [this.brute, this.grid]) {
      for (const buffer of Object.values(buffers)) buffer?.destroy();
    }
    for (const buffer of [...this.inputBuffers, ...this.indexBuffers]) buffer.destroy();
  }
}

/** Seeded xorshift32 generator returning floats in `[0, 1)`. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}
