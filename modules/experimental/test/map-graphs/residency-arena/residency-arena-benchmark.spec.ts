// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  createTransientView,
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUVisibilityWorkflow,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {GPUResidentRowSelection} from '../../../src/map-graphs/residency-arena/gpu-resident-row-selection';
import {
  getGPUTimeWindowParameterValues,
  GPUTimeWindowFilter
} from '../../../src/map-graphs/time-window-filter';
import {getTimeWindowClassifyNodes} from '../../../src/map-graphs/time-window-filter/time-window-classify-node';
import {createInputBuffer, createOutputBuffer, createVectorView} from '../map-graph-test-utils';

const ROW_COUNT = 1_048_576;
const CHUNK_COUNTS = [1, 8, 32];
/** Time window end for each visible fraction; times are uniform in [0, 1000), both exact in f32. */
const FRACTIONS = [
  {label: '1%', windowEnd: 10},
  {label: '10%', windowEnd: 100}
];
const TIME_DOMAIN = 1000;
const LIVE_PROBABILITY = 0.95;
const TILE_COUNT = 1000;
const WARMUP_COUNT = 3;
const SAMPLE_COUNT = 10;
const PIPELINE_DEPTH = 20;
const PIPELINE_BATCHES = 5;

type Variant =
  | 'a chunked-vector'
  | 'b chunked-in packed-out'
  | 'd1 arena time filter'
  | 'd2 arena row selection';
const VARIANTS: Variant[] = [
  'a chunked-vector',
  'b chunked-in packed-out',
  'd1 arena time filter',
  'd2 arena row selection'
];

/** Seeded mulberry32 generator. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function getMedian(samples: number[]): number {
  const sorted = [...samples].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

type SharedBuffers = {
  times: Buffer;
  live: Buffer;
  slots: Buffer;
  tileMask: Buffer;
  window: Buffer;
  ids: Buffer;
  count: Buffer;
  overflow: Buffer;
};

/** One packed view per buffer, or `chunkCount` offset views into it wrapped in a vector. */
function importChunked<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  id: string,
  buffer: Buffer,
  format: Format,
  chunkCount: number
): GraphVectorView<Format> {
  const handle = graph.importBuffer(
    {id, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  const chunkLength = ROW_COUNT / chunkCount;
  const chunks: GraphDataView<Format>[] = [];
  for (let chunk = 0; chunk < chunkCount; chunk++) {
    chunks.push(
      graph.createDataView(handle, {
        format,
        length: chunkLength,
        byteOffset: chunk * chunkLength * 4
      })
    );
  }
  return createVectorView(id, format, chunks);
}

/** Transient chunked uint32 vector with the chunking of the inputs. */
function createTransientChunked(
  graph: GPUCommandGraph,
  id: string,
  chunkCount: number
): GraphVectorView<'uint32'> {
  const chunks = Array.from({length: chunkCount}, (_, chunk) =>
    createTransientView(graph, `${id}-chunk-${chunk}`, 'uint32', ROW_COUNT / chunkCount)
  );
  return createVectorView(id, 'uint32', chunks);
}

type BuiltVariant = {
  compiled: ReturnType<GPUCommandGraph['compile']>;
  compileMilliseconds: number;
  /** The destination of the published count: the count buffer or the draw record. */
  readCount: () => Promise<number>;
  destroy: () => void;
};

function buildVariant(
  device: Device,
  buffers: SharedBuffers,
  variant: Variant,
  chunkCount: number
): BuiltVariant {
  const graph = new GPUCommandGraph(device, {
    id: `bench-${variant}-${chunkCount}`.replace(/ /g, '-')
  });
  const packed = <Format extends 'uint32' | 'float32'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, name, buffer, format, length);
  const window = packed('window', buffers.window, 'float32', 8);
  let drawCommands: DrawCommandBuffer | undefined;

  if (variant === 'a chunked-vector' || variant === 'b chunked-in packed-out') {
    const timeMask = createTransientChunked(graph, 'time-mask', chunkCount);
    const live = importChunked(graph, 'live', buffers.live, 'uint32', chunkCount);
    const workflow = new GPUVisibilityWorkflow({
      id: 'visibility',
      predicates: [
        {kind: 'time-range', mask: timeMask},
        {kind: 'selection', mask: live}
      ],
      output:
        variant === 'a chunked-vector'
          ? importChunked(graph, 'ids', buffers.ids, 'uint32', chunkCount)
          : packed('ids', buffers.ids, 'uint32', ROW_COUNT),
      count: packed('count', buffers.count, 'uint32', 1)
    });
    graph.add({
      getCommandNodes: graph => [
        ...getTimeWindowClassifyNodes(graph, {
          id: 'classify',
          timestamps: importChunked(graph, 'times', buffers.times, 'float32', chunkCount),
          window,
          mask: timeMask
        }),
        ...workflow.getCommandNodes(graph)
      ]
    });
  } else {
    drawCommands = new DrawCommandBuffer(device, {
      id: 'bench-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    });
    const drawInstanceCount = graph.importGPUData(
      'draw-count',
      drawCommands.getInstanceCountData(0)
    );
    const output = {
      ids: packed('ids', buffers.ids, 'uint32', ROW_COUNT),
      count: packed('count', buffers.count, 'uint32', 1),
      overflow: packed('overflow', buffers.overflow, 'uint32', 1)
    };
    const liveMask = packed('live', buffers.live, 'uint32', ROW_COUNT);
    if (variant === 'd1 arena time filter') {
      graph.add(
        new GPUTimeWindowFilter({
          timestamps: packed('times', buffers.times, 'float32', ROW_COUNT),
          window,
          additionalPredicates: [{kind: 'selection', mask: liveMask}],
          output,
          drawInstanceCount
        })
      );
    } else {
      // The time mask comes from the same classify kernel GPUTimeWindowFilter uses, then
      // GPUResidentRowSelection ANDs live, the tile gate, and that mask.
      const timeMask = createTransientView(graph, 'time-mask', 'uint32', ROW_COUNT);
      const selection = new GPUResidentRowSelection({
        liveMask,
        tileVisibility: {
          rowTileSlots: packed('slots', buffers.slots, 'uint32', ROW_COUNT),
          tileMask: packed('tile-mask', buffers.tileMask, 'uint32', TILE_COUNT)
        },
        additionalPredicates: [{kind: 'time-range', mask: timeMask}],
        output,
        drawInstanceCount
      });
      graph.add({
        getCommandNodes: graph => [
          ...getTimeWindowClassifyNodes(graph, {
            id: 'classify',
            timestamps: packed('times', buffers.times, 'float32', ROW_COUNT),
            window,
            mask: timeMask
          }),
          ...selection.getCommandNodes(graph)
        ]
      });
    }
  }

  const compileStart = performance.now();
  const compiled = graph.compile();
  const compileMilliseconds = performance.now() - compileStart;
  const draw = drawCommands;
  return {
    compiled,
    compileMilliseconds,
    readCount: async () => {
      if (draw) {
        const bytes = await draw.buffer.readAsync(draw.getInstanceCountByteOffset(0), 4);
        return new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
      }
      const bytes = await buffers.count.readAsync(0, 4);
      return new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
    },
    destroy: () => {
      compiled.destroy();
      draw?.destroy();
    }
  };
}

it('residency arena versus chunked visibility benchmark', {timeout: 150000}, async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(7);
  const times = new Float32Array(ROW_COUNT);
  const live = new Uint32Array(ROW_COUNT);
  const slots = new Uint32Array(ROW_COUNT);
  for (let row = 0; row < ROW_COUNT; row++) {
    times[row] = Math.fround(random() * TIME_DOMAIN);
    live[row] = random() < LIVE_PROBABILITY ? 1 : 0;
    slots[row] = Math.floor(random() * TILE_COUNT);
  }
  const buffers: SharedBuffers = {
    times: createInputBuffer(device, times),
    live: createInputBuffer(device, live),
    slots: createInputBuffer(device, slots),
    tileMask: createInputBuffer(device, new Uint32Array(TILE_COUNT).fill(1)),
    window: createInputBuffer(device, new Float32Array(8)),
    ids: createOutputBuffer(device, ROW_COUNT),
    count: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1)
  };
  const expectedIds = FRACTIONS.map(({windowEnd}) => {
    const ids: number[] = [];
    for (let row = 0; row < ROW_COUNT; row++) {
      if (live[row] && times[row] >= 0 && times[row] <= windowEnd) ids.push(row);
    }
    return ids;
  });

  const lines: string[] = [];
  for (const chunkCount of CHUNK_COUNTS) {
    for (const variant of VARIANTS) {
      const built = buildVariant(device, buffers, variant, chunkCount);
      const cells: string[] = [];
      for (const [fractionIndex, fraction] of FRACTIONS.entries()) {
        buffers.window.write(getGPUTimeWindowParameterValues({start: 0, end: fraction.windowEnd}));
        // Correctness gate: count and the stable ID prefix match the CPU oracle.
        submitGraph(device, built.compiled, undefined);
        const count = await built.readCount();
        expect(count).toBe(expectedIds[fractionIndex].length);
        const idBytes = await buffers.ids.readAsync(0, count * 4);
        const ids = new Uint32Array(idBytes.buffer, idBytes.byteOffset, count);
        expect(Array.from(ids)).toEqual(expectedIds[fractionIndex]);

        const synced: number[] = [];
        for (let sample = 0; sample < WARMUP_COUNT + SAMPLE_COUNT; sample++) {
          const start = performance.now();
          submitGraph(device, built.compiled, undefined);
          await built.readCount();
          if (sample >= WARMUP_COUNT) synced.push(performance.now() - start);
        }
        const piped: number[] = [];
        for (let batch = 0; batch < PIPELINE_BATCHES; batch++) {
          const start = performance.now();
          for (let encoding = 0; encoding < PIPELINE_DEPTH; encoding++) {
            submitGraph(device, built.compiled, undefined);
          }
          await built.readCount();
          piped.push((performance.now() - start) / PIPELINE_DEPTH);
        }
        cells.push(
          `${String(count).padStart(7)} ${getMedian(synced).toFixed(2).padStart(8)} ${getMedian(piped).toFixed(2).padStart(8)}`
        );
      }
      lines.push(
        `${variant.padEnd(24)} ${String(chunkCount).padStart(3)} ${String(built.compiled.stats.nodeOrder.length).padStart(6)} ` +
          `${built.compileMilliseconds.toFixed(0).padStart(8)}  ${cells.join('  |  ')}`
      );
      built.destroy();
    }
  }
  for (const buffer of Object.values(buffers)) {
    buffer.destroy();
  }
  // eslint-disable-next-line no-console
  console.log(
    [
      `residency arena benchmark, ${ROW_COUNT} resident rows, live ${LIVE_PROBABILITY}, ${device.info.gpu}; synced = median of ${SAMPLE_COUNT} (warmup ${WARMUP_COUNT}) encode+submit+count readback; piped = median per encoding with ${PIPELINE_DEPTH} queued per readback`,
      `${' '.repeat(24)} ${'C'.padStart(3)} ${'nodes'.padStart(6)} ${'compile'.padStart(8)}  ${FRACTIONS.map(f => `${f.label} visible: count synced ms piped ms`).join('  |  ')}`,
      ...lines
    ].join('\n')
  );
});
