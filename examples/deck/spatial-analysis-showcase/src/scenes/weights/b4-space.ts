// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, CommandEncoder, Device} from '@luma.gl/core';
import {
  GPU_NEIGHBORHOOD_SUMMARY_NO_MODE,
  GPUNeighborhoodSummary
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, GPUReduction, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {runGuarded} from './b4-format';
import {
  createWeightsKit,
  FOCUS_SLOTS,
  type WeightsCsrBuffers,
  type WeightsKit,
  type WeightsKitSummary,
  type WeightsMatrixChoice
} from './b4-weights-kit';

/** Neighbourhood statistics `GPUNeighborhoodSummary` writes, in table column order. */
export const SUMMARY_STATISTICS = [
  'count',
  'weightSum',
  'sum',
  'mean',
  'min',
  'max',
  'standardDeviation',
  'median'
] as const;

/** A statistic the display can map: the table columns plus the categorical entropy. */
export type SummaryChoice = (typeof SUMMARY_STATISTICS)[number] | 'entropy';

/** What the fill shows. */
export type SpaceDisplay = 'value' | 'neighbors' | 'lag' | 'oneWay' | 'summary' | 'focus';

/** Per-row CPU copy of the buffers tooltips need, refreshed after a change. */
export type SpaceSnapshot = {
  cardinality: Float32Array;
  oneWay: Float32Array;
  lag: Float32Array;
  /** Weighted dominant category of the neighbourhood, or `GPU_NEIGHBORHOOD_SUMMARY_NO_MODE`. */
  modes: Uint32Array;
  /** Value currently shown by the fill. */
  shown: Float32Array;
  /** Neighbour row IDs of the focus row per matrix (the focus row itself where unused). */
  focusIds: Record<WeightsMatrixChoice, Uint32Array>;
  /** `[min, max]` of the shown value over valid rows. */
  extent: [number, number];
};

/** Compile-time and per-frame inputs of {@link AnalysisSpace.encode}. */
export type SpaceEncodeOptions = {
  normalizeLag: boolean;
  matrix: WeightsMatrixChoice;
  display: SpaceDisplay;
  summary: SummaryChoice;
  includeFocal: boolean;
  focalWeight: number;
};

/** Weights analysis of one row space (polygons or lattice cells). */
export type AnalysisSpace = {
  rows: number;
  /** The CSR every analysis reads. Producers or algebra write it. */
  csr: WeightsCsrBuffers;
  slots: number;
  kit: WeightsKit;
  /** Float32 shown by the fill; NaN rows are no data. */
  shown: Buffer;
  /** `[min, max]` of `shown` over valid rows. */
  shownExtent: Buffer;
  /** Uint32 1 for rows without neighbours. */
  isolate: Buffer;
  /** Uint32: 0 other, 1 neighbour of the focus row, 2 focus row. */
  focusClass: Buffer;
  /** Choice of the display kernel (a buffer write). */
  setDisplayChoice: (display: SpaceDisplay, summary: SummaryChoice) => number;
  /** Compiles the neighbourhood variant a configuration needs; true when it compiled. */
  prepare: (includeFocal: boolean, focalWeight: number) => boolean;
  /** Encodes summary, transpose, lag, focus links, neighbourhood summary and the display kernels. */
  encode: (commandEncoder: CommandEncoder, options: SpaceEncodeOptions) => void;
  /** Reads the weights summary and the tooltip snapshot; call after `encode` when stale. */
  requestReadback: (
    commandEncoder: CommandEncoder,
    onKit: (summary: WeightsKitSummary) => void,
    onSnapshot: (snapshot: SpaceSnapshot) => void
  ) => void;
  /** True while a readback is in flight. */
  isReading: () => boolean;
  /** Every compiled graph. */
  getGraphs: () => CompiledGPUCommandGraph<never>[];
  stop: () => void;
};

const NEIGHBORHOOD_COLUMNS = SUMMARY_STATISTICS.length;
const MAXIMUM_NEIGHBORHOOD_MEMBERS = 64;

const CLASSIFY_BODY = /* wgsl */ `
  let start = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  let degree = end - start;
  isolate[isolateOffset + index] = select(0u, 1u, degree == 0u);
  cardinality[cardinalityOffset + index] = f32(degree);
  var oneWayCount = 0u;
  for (var slot = start; slot < end; slot++) {
    let other = neighbors[neighborsOffset + slot];
    if (other >= ROW_COUNT) { continue; }
    var low = offsets[offsetsOffset + other];
    var high = offsets[offsetsOffset + other + 1u];
    let rowEnd = high;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (neighbors[neighborsOffset + middle] < index) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    if (!(low < rowEnd && neighbors[neighborsOffset + low] == index)) {
      oneWayCount++;
    }
  }
  oneWay[oneWayOffset + index] = f32(oneWayCount);
  let focus = focusRow[focusRowOffset];
  var focusValue = 0u;
  if (index == focus) {
    focusValue = 2u;
  } else {
    let focusStart = offsets[offsetsOffset + focus];
    let focusEnd = offsets[offsetsOffset + focus + 1u];
    for (var slot = focusStart; slot < focusEnd; slot++) {
      if (neighbors[neighborsOffset + slot] == index) { focusValue = 1u; }
    }
  }
  focusClass[focusClassOffset + index] = focusValue;`;

const SELECT_BODY = /* wgsl */ `
  let choice = choiceBuffer[choiceBufferOffset];
  var value = 0.0;
  if (choice == 0u) {
    value = values[valuesOffset + index];
  } else if (choice == 1u) {
    value = cardinality[cardinalityOffset + index];
  } else if (choice == 2u) {
    value = lag[lagOffset + index];
  } else if (choice == 3u) {
    value = oneWay[oneWayOffset + index];
  } else if (choice < ${4 + NEIGHBORHOOD_COLUMNS}u) {
    value = table[tableOffset + index * ${NEIGHBORHOOD_COLUMNS}u + (choice - 4u)];
  } else {
    value = entropy[entropyOffset + index];
  }
  shown[shownOffset + index] = value;`;

const VALID_BODY = /* wgsl */ `
  let value = shown[shownOffset + index];
  let finite = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
  let included = mask[maskOffset + index] != 0u && finite;
  valid[validOffset + index] = select(0u, 1u, included);
  var nanBits = 0x7fc00000u;
  shown[shownOffset + index] = select(bitcast<f32>(nanBits), value, included);`;

/**
 * The analysis of one row space: summary, transpose and union, spatial lag, neighbourhood summary
 * (`GPUNeighborhoodSummary`, lazily compiled per `includeFocal` / `focalWeight`), the focus row and
 * the kernels that prepare what the map fill shows. Polygons and lattice cells share it.
 */
export function createAnalysisSpace(props: {
  device: Device;
  resources: SpatialAnalysisResources;
  id: string;
  rows: number;
  slots: number;
  unionSlots: number;
  positions: Buffer;
  values: Buffer;
  /** Uint32 per row: zero rows are skipped by the lag and the neighbourhood summary. */
  mask: Buffer;
  /** Uint32 per row: zero rows are not drawn. Defaults to `mask`. */
  displayMask?: Buffer;
  /** Category per row for the dominant neighbour and the entropy (state, community area). */
  categories: Uint32Array;
  drawAllLinks: boolean;
  producer: {overflow: Buffer; total: Buffer};
}): AnalysisSpace {
  const {device, resources, id, rows, slots, positions, values, mask} = props;
  const displayMask = props.displayMask ?? mask;
  const create = (name: string, data: number | Float32Array | Uint32Array) =>
    resources.createBuffer(`${id}-${name}`, data);
  const csr: WeightsCsrBuffers = {
    offsets: create('analysis-offsets', (rows + 1) * 4),
    neighbors: create('analysis-neighbors', slots * 4),
    weights: create('analysis-weights', slots * 4)
  };
  const kit = createWeightsKit({
    device,
    resources,
    id: `${id}-kit`,
    rowCount: rows,
    slots,
    unionSlots: props.unionSlots,
    positions,
    values,
    mask,
    csr,
    producer: props.producer,
    drawAllLinks: props.drawAllLinks
  });
  const categoriesBuffer = create('categories', props.categories);
  const table = create('neighborhood-table', rows * NEIGHBORHOOD_COLUMNS * 4);
  const modes = create('neighborhood-modes', rows * 4);
  const entropy = create('neighborhood-entropy', rows * 4);
  const neighborhoodOverflow = create('neighborhood-overflow', 4);
  const cardinality = create('cardinality-float', rows * 4);
  const oneWay = create('one-way', rows * 4);
  const isolate = create('isolate', rows * 4);
  const focusClass = create('focus-class', rows * 4);
  const shown = create('shown', rows * 4);
  const valid = create('shown-valid', rows * 4);
  const shownExtent = create('shown-extent', 8);
  const choiceBuffer = create('display-choice', new Uint32Array([0]));

  // Neighbourhood summary variants, compiled on first use.
  const neighborhoodGraphs = new Map<string, CompiledGPUCommandGraph<void>>();
  const getNeighborhoodKey = (includeFocal: boolean, focalWeight: number) =>
    `${includeFocal ? 'focal' : 'plain'}-${focalWeight}`;
  const prepare = (includeFocal: boolean, focalWeight: number): boolean => {
    const key = getNeighborhoodKey(includeFocal, focalWeight);
    if (neighborhoodGraphs.has(key)) return false;
    const graph = new GPUCommandGraph<void>(device, {id: `${id}-neighborhood-${key}`});
    graph.add(
      new GPUNeighborhoodSummary({
        id: 'neighborhood',
        weights: {
          offsets: importGraphBuffer(graph, 'offsets', csr.offsets, 'uint32', rows + 1),
          neighbors: importGraphBuffer(graph, 'neighbors', csr.neighbors, 'uint32', slots),
          weights: importGraphBuffer(graph, 'weights', csr.weights, 'float32', slots)
        },
        values: importGraphBuffer(graph, 'values', values, 'float32', rows),
        mask: importGraphBuffer(graph, 'mask', mask, 'uint32', rows),
        categories: importGraphBuffer(graph, 'categories', categoriesBuffer, 'uint32', rows),
        includeFocal,
        focalWeight,
        statistics: SUMMARY_STATISTICS,
        output: importGraphBuffer(graph, 'table', table, 'float32', rows * NEIGHBORHOOD_COLUMNS),
        maximumNeighbors: MAXIMUM_NEIGHBORHOOD_MEMBERS,
        overflow: importGraphBuffer(graph, 'overflow', neighborhoodOverflow, 'uint32', 1),
        modes: importGraphBuffer(graph, 'modes', modes, 'uint32', rows),
        entropy: importGraphBuffer(graph, 'entropy', entropy, 'float32', rows)
      })
    );
    neighborhoodGraphs.set(key, resources.track(graph.compile()));
    return true;
  };
  prepare(false, 1);

  // Classification kernels (isolates, one-way links, focus class) and the display select.
  const displayGraph = new GPUCommandGraph<void>(device, {id: `${id}-display`});
  {
    const importCsr = () => ({
      offsets: importGraphBuffer(displayGraph, 'offsets', csr.offsets, 'uint32', rows + 1),
      neighbors: importGraphBuffer(displayGraph, 'neighbors', csr.neighbors, 'uint32', slots)
    });
    const view = importCsr();
    const cardinalityView = importGraphBuffer(
      displayGraph,
      'cardinality',
      cardinality,
      'float32',
      rows
    );
    const oneWayView = importGraphBuffer(displayGraph, 'one-way', oneWay, 'float32', rows);
    addKernelPass(displayGraph, {
      id: `${id}-classify`,
      invocationCount: rows,
      bindings: [
        {name: 'offsets', view: view.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: view.neighbors, type: 'u32', access: 'read'},
        {
          name: 'focusRow',
          view: importGraphBuffer(displayGraph, 'focus-row', kit.focusRow, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'isolate',
          view: importGraphBuffer(displayGraph, 'isolate', isolate, 'uint32', rows),
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'focusClass',
          view: importGraphBuffer(displayGraph, 'focus-class', focusClass, 'uint32', rows),
          type: 'u32',
          access: 'read_write'
        },
        {name: 'cardinality', view: cardinalityView, type: 'f32', access: 'read_write'},
        {name: 'oneWay', view: oneWayView, type: 'f32', access: 'read_write'}
      ],
      declarations: `const ROW_COUNT: u32 = ${rows}u;`,
      body: CLASSIFY_BODY
    });
    const shownView = importGraphBuffer(displayGraph, 'shown', shown, 'float32', rows);
    const validView = importGraphBuffer(displayGraph, 'valid', valid, 'uint32', rows);
    addKernelPass(displayGraph, {
      id: `${id}-display-select`,
      invocationCount: rows,
      bindings: [
        {
          name: 'choiceBuffer',
          view: importGraphBuffer(displayGraph, 'choice', choiceBuffer, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'values',
          view: importGraphBuffer(displayGraph, 'values', values, 'float32', rows),
          type: 'f32',
          access: 'read'
        },
        {name: 'cardinality', view: cardinalityView, type: 'f32', access: 'read'},
        {
          name: 'lag',
          view: importGraphBuffer(displayGraph, 'lag', kit.lag, 'float32', rows),
          type: 'f32',
          access: 'read'
        },
        {name: 'oneWay', view: oneWayView, type: 'f32', access: 'read'},
        {
          name: 'table',
          view: importGraphBuffer(
            displayGraph,
            'table',
            table,
            'float32',
            rows * NEIGHBORHOOD_COLUMNS
          ),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'entropy',
          view: importGraphBuffer(displayGraph, 'entropy', entropy, 'float32', rows),
          type: 'f32',
          access: 'read'
        },
        {name: 'shown', view: shownView, type: 'f32', access: 'read_write'}
      ],
      body: SELECT_BODY
    });
    addKernelPass(displayGraph, {
      id: `${id}-display-valid`,
      invocationCount: rows,
      bindings: [
        {name: 'shown', view: shownView, type: 'f32', access: 'read_write'},
        {
          name: 'mask',
          view: importGraphBuffer(displayGraph, 'mask', displayMask, 'uint32', rows),
          type: 'u32',
          access: 'read'
        },
        {name: 'valid', view: validView, type: 'u32', access: 'read_write'}
      ],
      body: VALID_BODY
    });
    displayGraph.add(
      new GPUReduction({
        id: 'shown-extent',
        input: shownView,
        mask: validView,
        output: importGraphBuffer(displayGraph, 'shown-extent', shownExtent, 'float32', 2),
        operation: 'extent'
      })
    );
  }
  const displayCompiled = resources.track(displayGraph.compile());

  // Tooltip snapshot: a handful of row-sized buffers, read only when something changed.
  let snapshotHandler: ((snapshot: SpaceSnapshot) => void) | null = null;
  const snapshotReader = new SummaryReader(
    resources,
    `${id}-snapshot`,
    [
      {buffer: cardinality, size: rows * 4},
      {buffer: oneWay, size: rows * 4},
      {buffer: kit.lag, size: rows * 4},
      {buffer: modes, size: rows * 4},
      {buffer: shown, size: rows * 4},
      {buffer: shownExtent, size: 8},
      {buffer: kit.focusNeighbors.weights, size: FOCUS_SLOTS * 4},
      {buffer: kit.focusNeighbors.transpose, size: FOCUS_SLOTS * 4},
      {buffer: kit.focusNeighbors.union, size: FOCUS_SLOTS * 4}
    ],
    bytes =>
      runGuarded('weights snapshot', () => {
        const slice = (index: number) => rows * 4 * index;
        snapshotHandler?.({
          cardinality: new Float32Array(bytes, slice(0), rows),
          oneWay: new Float32Array(bytes, slice(1), rows),
          lag: new Float32Array(bytes, slice(2), rows),
          modes: new Uint32Array(bytes, slice(3), rows),
          shown: new Float32Array(bytes, slice(4), rows),
          extent: [
            new Float32Array(bytes, slice(5), 2)[0],
            new Float32Array(bytes, slice(5), 2)[1]
          ],
          focusIds: {
            weights: new Uint32Array(bytes, slice(5) + 8, FOCUS_SLOTS),
            transpose: new Uint32Array(bytes, slice(5) + 8 + FOCUS_SLOTS * 4, FOCUS_SLOTS),
            union: new Uint32Array(bytes, slice(5) + 8 + FOCUS_SLOTS * 8, FOCUS_SLOTS)
          }
        });
      })
  );
  let kitPending = false;

  const choiceFor = (display: SpaceDisplay, summary: SummaryChoice): number => {
    switch (display) {
      case 'neighbors':
        return 1;
      case 'lag':
        return 2;
      case 'oneWay':
        return 3;
      case 'summary':
        return summary === 'entropy'
          ? 4 + NEIGHBORHOOD_COLUMNS
          : 4 + SUMMARY_STATISTICS.indexOf(summary);
      default:
        return 0;
    }
  };

  return {
    rows,
    csr,
    slots,
    kit,
    shown,
    shownExtent,
    isolate,
    focusClass,
    prepare,

    setDisplayChoice(display, summary) {
      const choice = choiceFor(display, summary);
      choiceBuffer.write(new Uint32Array([choice]));
      return choice;
    },

    encode(commandEncoder, options) {
      kit.encodeAnalysis(commandEncoder, {
        normalizeLag: options.normalizeLag,
        matrix: options.matrix
      });
      prepare(options.includeFocal, options.focalWeight);
      neighborhoodGraphs
        .get(getNeighborhoodKey(options.includeFocal, options.focalWeight))!
        .encode(commandEncoder, {parameters: undefined});
      displayCompiled.encode(commandEncoder, {parameters: undefined});
    },

    requestReadback(commandEncoder, onKit, onSnapshot) {
      snapshotHandler = onSnapshot;
      if (!kitPending) {
        kitPending = true;
        void kit
          .readSummary(commandEncoder)
          .then(summary => {
            if (summary) runGuarded('weights summary', () => onKit(summary));
          })
          .catch(error => {
            // biome-ignore lint/suspicious/noConsole: a failed readback must be visible while authoring
            console.warn('weights readback failed', error);
          })
          .finally(() => {
            kitPending = false;
          });
      }
      snapshotReader.request(commandEncoder);
    },

    isReading: () => kitPending || snapshotReader.isPending,

    getGraphs: () =>
      [
        ...kit.compiledGraphs(),
        ...neighborhoodGraphs.values(),
        displayCompiled
      ] as CompiledGPUCommandGraph<never>[],

    stop: () => snapshotReader.stop()
  };
}

/** Marker for rows without a dominant neighbour category. */
export const NO_MODE = GPU_NEIGHBORHOOD_SUMMARY_NO_MODE;
