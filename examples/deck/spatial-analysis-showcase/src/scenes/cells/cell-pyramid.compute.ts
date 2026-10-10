// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Chicago observations in a Quadbin cell pyramid, rolled up, classified and compared across periods.
 *
 * - Pyramid view: `GPUCellPyramid` keys every observation into Quadbin cells at the finest resolution (17)
 *   and rolls the table up level by level (to 4). It is encoded only when the observation mask changes. A
 *   tiny second graph (`GPUCellLevelSelection`) runs every frame: the map zoom is written into a
 *   one-word buffer and the node publishes that level's cell count and an indirect draw record.
 * - Roll-up view: `GPUCellRollup` rolls the finest table (with fixed-point research-grade sums) up to a
 *   chosen parent resolution. Counts and sums are integers, so the roll-up conserves them exactly.
 *   A resolution is a kernel constant, so each choice is its own small graph, compiled on first use.
 * - Class view: the roll-up table becomes a value column and a mask (`addCellTableColumnsNode`
 *   adapter), `GPUClassBreaks` classifies it and `GPUColorScale` colors it, all on the GPU.
 * - Compare view: `addPeriodComparisonRecipe` aggregates two periods (two masks over the same observation
 *   rows), outer-joins them with `GPUCellTableCompare` into delta, ratio, percent change and a
 *   z-score, then classifies the delta around zero. The layer draws either the classes or any
 *   compared column.
 */

import {COORDINATE_SYSTEM} from '@deck.gl/core';
import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  GPUClassBreaks,
  GPUColorScale,
  packGPUColor,
  type GPUClassBreaksMethod
} from '@luma.gl/experimental/gpu-dataframe';
import {
  addPeriodComparisonRecipe,
  GPUCellLevelSelection,
  GPUCellPyramid,
  GPUCellRollup,
  GPU_CELL_DEFAULT_SUM_SCALE
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {sampleRamp, type RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {findArea, readAreaPolygons} from './b8-areas';
import {addCellTableColumnsNode} from './b8-cell-table-columns';
import {
  decodeQuadbin,
  fillObservationMask,
  formatMonthWindow,
  getCategoryIndex,
  getQuadbinCenter,
  readNatureCells,
  hashToUnit
} from './b8-common';
import {QuadbinTableLayer} from './b8-quadbin-layer';

/** Option state of the cell-pyramid scene. */
export type CellPyramidOptions = {
  view: 'pyramid' | 'rollup' | 'classes' | 'compare';
  category: string;
  sampledPercent: number;
  showAreas: boolean;
  ramp: RampName;
  // Pyramid
  resolutionOffset: number;
  manualLevel: boolean;
  manualResolution: number;
  densityCeiling: number;
  // Roll-up
  rollupResolution: number;
  rollupColor: 'density' | 'researchShare';
  rateCeiling: number;
  // Classes
  classValue: 'observations' | 'confirmed';
  classMethod:
    | 'quantile'
    | 'equal-interval'
    | 'natural-breaks'
    | 'standard-deviation'
    | 'head-tail'
    | 'box-plot';
  classCount: number;
  // Compare
  periodA: readonly [number, number];
  periodB: readonly [number, number];
  compareResolution: number;
  compareMeasure: 'count' | 'sum';
  zScoreKind: 'poisson' | 'standardized';
  changeColumn: 'classes' | 'delta' | 'ratio' | 'percentChange' | 'zScore';
  changeClassing: 'band' | 'standard-deviation';
  stableBand: number;
  colorLimit: number;
};

/** Finest and coarsest Quadbin resolution of the pyramid (inclusive). */
const FINEST_RESOLUTION = 17;
const COARSEST_RESOLUTION = 4;
const LEVEL_COUNT = FINEST_RESOLUTION - COARSEST_RESOLUTION + 1;
/** Rows per level table; rounded so every level slab starts on a 256-byte boundary. */
const MAXIMUM_TABLE_CAPACITY = 1 << 18;
const READBACK_INTERVAL_FRAMES = 15;
const COMPARE_TABLE_CAPACITY = 1 << 14;
const COMPARE_UNION_CAPACITY = 1 << 15;
const CLASS_CAPACITY = 8;
const REBUILD_DEBOUNCE_MILLISECONDS = 250;
const DEFAULT_ROLLUP_RESOLUTION = 14;

/** Diverging class colors of the compare recipe, fewer to more. */
export const CHANGE_CLASS_COLORS = [
  [33, 102, 172],
  [103, 169, 207],
  [210, 210, 210],
  [239, 138, 98],
  [178, 24, 43]
] as const;

/** Resolution of pyramid level `levelIndex` (level 0 is the finest). */
const getLevelResolution = (levelIndex: number) => FINEST_RESOLUTION - levelIndex;

/** Natural scale of each compared column, mapped to the ramp ends at a color limit of 100%. */
function getColumnScale(column: CellPyramidOptions['changeColumn'], maximumDelta: number): number {
  if (column === 'delta') return Math.max(5, maximumDelta);
  if (column === 'percentChange') return 100;
  if (column === 'zScore') return 4;
  return 2;
}

type RollupVariant = {graph: CompiledGPUCommandGraph<void>};
type ClassVariant = {
  graph: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
};
type CompareVariant = {
  key: string;
  resolution: number;
  resources: SpatialAnalysisResources;
  graph: CompiledGPUCommandGraph<void>;
  classBreaks: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
  colorScale: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
  cells: Buffer;
  count: Buffer;
  colors: Buffer;
  columns: Record<'delta' | 'ratio' | 'percentChange' | 'zScore', Buffer>;
  reader: SummaryReader;
};

export async function createCellPyramid(
  ctx: SceneContext<CellPyramidOptions>
): Promise<SceneInstance<CellPyramidOptions>> {
  const {device} = ctx;
  const columns = readNatureCells(ctx.datasets.get('chicago-nature'));
  const areas = readAreaPolygons(ctx.datasets.get('chicago-community-areas'));
  const pointCount = columns.count;
  const resources = new SpatialAnalysisResources(device, 'cell-pyramid');
  const capacity = Math.min(Math.ceil(pointCount / 1024) * 1024, MAXIMUM_TABLE_CAPACITY);
  // A roll-up may not read and write the same buffer, so even and odd levels alternate between
  // two slabs (and two scalar buffers); each slab holds half of the levels.
  const slabRows = capacity * Math.ceil(LEVEL_COUNT / 2);

  const positionsBuffer = resources.createBuffer('lng-lat', columns.lngLat);
  const researchGradeBuffer = resources.createBuffer('confirmed', columns.researchGrade);
  const maskValues = new Uint32Array(pointCount).fill(1);
  const maskBuffer = resources.createBuffer('mask', maskValues);
  const beforeMaskValues = new Uint32Array(pointCount);
  const afterMaskValues = new Uint32Array(pointCount);
  const beforeMaskBuffer = resources.createBuffer('before-mask', beforeMaskValues);
  const afterMaskBuffer = resources.createBuffer('after-mask', afterMaskValues);
  const areaOutline = resources.createBuffer('area-outline', areas.outlineSegments);

  const cellsSlabs = [0, 1].map(parity =>
    resources.createBuffer(`cells-slab-${parity}`, slabRows * 8)
  );
  const countsSlabs = [0, 1].map(parity =>
    resources.createBuffer(`counts-slab-${parity}`, slabRows * 4)
  );
  const tableCountBuffers = [0, 1].map(parity =>
    resources.createBuffer(`table-counts-${parity}`, LEVEL_COUNT * 4)
  );
  const levelOverflowBuffers = [0, 1].map(parity =>
    resources.createBuffer(`level-overflow-${parity}`, LEVEL_COUNT * 4)
  );
  // Fixed-point research-grade sums of the finest level only; coarser pyramid levels do not need them.
  const finestSumsBuffer = resources.createBuffer('finest-sums', capacity * 8);
  const rollupCellsBuffer = resources.createBuffer('rollup-cells', capacity * 8);
  const rollupCountsBuffer = resources.createBuffer('rollup-counts', capacity * 4);
  const rollupSumsBuffer = resources.createBuffer('rollup-sums', capacity * 8);
  const rollupSumValuesBuffer = resources.createBuffer('rollup-sum-values', capacity * 4);
  const rollupOverflowBuffer = resources.createBuffer('rollup-overflow', 4);
  const zeroWordBuffer = resources.createBuffer('zero-word', 4);
  const rollupDrawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'cell-rollup-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const levelCountsBuffer = resources.createBuffer('level-counts', LEVEL_COUNT * 4);
  const levelTotalsBuffer = resources.createBuffer('level-totals', 4);
  const activeLevel = resources.createParameterBuffer('active-level', 'uint32', 1);
  const activeFirstRow = resources.createBuffer('active-first-row', 4);
  const drawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'cell-pyramid-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );

  // --- Pyramid and level selection --------------------------------------------------------------

  const pyramidGraph = new GPUCommandGraph<void>(device, {id: 'cell-pyramid'});
  const importedHandles = new Map<Buffer, ReturnType<typeof pyramidGraph.importBuffer>>();
  const importSlice = (
    buffer: Buffer,
    format: 'uint32' | 'uint32x2',
    rowByteLength: number,
    firstRow: number,
    length: number
  ): GraphDataView<'uint32'> & GraphDataView<'uint32x2'> => {
    const handle =
      importedHandles.get(buffer) ??
      pyramidGraph.importBuffer(
        {id: buffer.id, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
    importedHandles.set(buffer, handle);
    return pyramidGraph.createDataView(handle, {
      format,
      length,
      byteOffset: firstRow * rowByteLength
    }) as GraphDataView<'uint32'> & GraphDataView<'uint32x2'>;
  };
  const levels = Array.from({length: LEVEL_COUNT}, (_, levelIndex) => {
    const parity = levelIndex & 1;
    const slabRow = (levelIndex >> 1) * capacity;
    return {
      resolution: getLevelResolution(levelIndex),
      output: {
        cells: importSlice(cellsSlabs[parity], 'uint32x2', 8, slabRow, capacity),
        counts: importSlice(countsSlabs[parity], 'uint32', 4, slabRow, capacity),
        count: importSlice(tableCountBuffers[parity], 'uint32', 4, levelIndex, 1),
        overflow: importSlice(levelOverflowBuffers[parity], 'uint32', 4, levelIndex, 1),
        ...(levelIndex === 0
          ? {
              requiredCount: importSlice(levelTotalsBuffer, 'uint32', 4, 0, 1),
              sums: importSlice(finestSumsBuffer, 'uint32x2', 8, 0, capacity)
            }
          : {})
      }
    };
  });
  const pyramid = new GPUCellPyramid({
    id: 'nature-pyramid',
    family: 'quadbin',
    positions: importGraphBuffer(
      pyramidGraph,
      'positions',
      positionsBuffer,
      'float32x2',
      pointCount
    ),
    mask: importGraphBuffer(pyramidGraph, 'mask', maskBuffer, 'uint32', pointCount),
    values: importGraphBuffer(pyramidGraph, 'values', researchGradeBuffer, 'float32', pointCount),
    levels,
    levelCounts: importGraphBuffer(
      pyramidGraph,
      'level-counts',
      levelCountsBuffer,
      'uint32',
      LEVEL_COUNT
    )
  });
  pyramidGraph.add(pyramid);
  const compiledPyramid = resources.track(pyramidGraph.compile());

  const selectionGraph = new GPUCommandGraph<void>(device, {id: 'cell-level-selection'});
  const selectionDrawView = drawCommands.importToGraph(selectionGraph);
  selectionGraph.add(
    new GPUCellLevelSelection({
      id: 'nature-level',
      levelCounts: importGraphBuffer(
        selectionGraph,
        'level-counts',
        levelCountsBuffer,
        'uint32',
        LEVEL_COUNT
      ),
      activeLevel: activeLevel.importToGraph(selectionGraph),
      levelFirstRows: pyramid.levelFirstRows,
      output: {
        // The record's firstInstance stays 0: the shell does not request the
        // indirect-first-instance device feature, so the layer adds the first row itself.
        count: selectionGraph.createDataView(selectionDrawView.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: Uint32Array.BYTES_PER_ELEMENT
        }),
        firstRow: importGraphBuffer(selectionGraph, 'active-first-row', activeFirstRow, 'uint32', 1)
      }
    })
  );
  const compiledSelection = resources.track(selectionGraph.compile());

  // --- Roll-up variants (one per parent resolution, compiled on first use) ------------------------

  const rollupVariants = new Map<number, RollupVariant>();
  const ensureRollup = (parentResolution: number): RollupVariant => {
    const existing = rollupVariants.get(parentResolution);
    if (existing) return existing;
    const graph = new GPUCommandGraph<void>(device, {id: `cell-rollup-${parentResolution}`});
    const handles = new Map<Buffer, ReturnType<typeof graph.importBuffer>>();
    const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(
      buffer: Buffer,
      format: Format,
      rowByteLength: number,
      length: number,
      firstRow = 0
    ): GraphDataView<Format> => {
      let handle = handles.get(buffer);
      if (!handle) {
        handle = graph.importBuffer(
          {id: buffer.id, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        );
        handles.set(buffer, handle);
      }
      return graph.createDataView(handle, {format, length, byteOffset: firstRow * rowByteLength});
    };
    const drawView = rollupDrawCommands.importToGraph(graph);
    graph.add(
      new GPUCellRollup({
        id: `nature-rollup-${parentResolution}`,
        family: 'quadbin',
        sourceResolution: FINEST_RESOLUTION,
        resolution: parentResolution,
        source: {
          cells: view(cellsSlabs[0], 'uint32x2', 8, capacity),
          counts: view(countsSlabs[0], 'uint32', 4, capacity),
          sums: view(finestSumsBuffer, 'uint32x2', 8, capacity),
          count: view(tableCountBuffers[0], 'uint32', 4, 1),
          overflow: view(levelOverflowBuffers[0], 'uint32', 4, 1)
        },
        sumScale: GPU_CELL_DEFAULT_SUM_SCALE,
        output: {
          cells: view(rollupCellsBuffer, 'uint32x2', 8, capacity),
          counts: view(rollupCountsBuffer, 'uint32', 4, capacity),
          sums: view(rollupSumsBuffer, 'uint32x2', 8, capacity),
          sumValues: view(rollupSumValuesBuffer, 'float32', 4, capacity),
          // The record's instanceCount is the parent cell count (the clamped output count).
          count: graph.createDataView(drawView.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: Uint32Array.BYTES_PER_ELEMENT
          }),
          overflow: view(rollupOverflowBuffer, 'uint32', 4, 1)
        }
      })
    );
    const variant = {graph: resources.track(graph.compile())};
    rollupVariants.set(parentResolution, variant);
    return variant;
  };

  // --- Class variants (value column from the roll-up table) ---------------------------------------

  const classValuesBuffer = resources.createBuffer('class-values', capacity * 4);
  const classMaskBuffer = resources.createBuffer('class-mask', capacity * 4);
  const classBreaksBuffer = resources.createBuffer('class-breaks', (CLASS_CAPACITY + 1) * 4);
  const classCountBuffer = resources.createBuffer('class-count', 4);
  const classColorsBuffer = resources.createBuffer('class-colors', capacity * 4);
  const classMethods: readonly GPUClassBreaksMethod[] = [
    'equal-interval',
    'quantile',
    'natural-breaks',
    'standard-deviation',
    'head-tail',
    'box-plot'
  ];
  const classParameters = resources.createParameterBuffer(
    'class-breaks-parameters',
    'float32',
    getGPUClassBreaksParameterLength(CLASS_CAPACITY),
    getGPUClassBreaksParameterValues({method: 'quantile', classCount: 5}, CLASS_CAPACITY)
  );
  const classColorScale = resources.createParameterBuffer(
    'class-color-scale-parameters',
    'float32',
    9,
    getGPUColorScaleParameterValues({
      scale: 'threshold',
      domainCount: CLASS_CAPACITY + 1,
      paletteCount: 5,
      noDataColor: packGPUColor(0, 0, 0, 0)
    })
  );
  const classPalette = resources.createBuffer('class-palette', CLASS_CAPACITY * 4);
  const classVariants = new Map<CellPyramidOptions['classValue'], ClassVariant>();
  let classSnapshot: {breaks: Float32Array; classCount: number} | null = null;

  const ensureClasses = (value: CellPyramidOptions['classValue']): ClassVariant => {
    const existing = classVariants.get(value);
    if (existing) return existing;
    const graph = new GPUCommandGraph<void>(device, {id: `cell-classes-${value}`});
    const rollupCounts = importGraphBuffer(
      graph,
      'rollup-counts',
      rollupCountsBuffer,
      'uint32',
      capacity
    );
    const rollupSumValues = importGraphBuffer(
      graph,
      'rollup-sum-values',
      rollupSumValuesBuffer,
      'float32',
      capacity
    );
    const rollupDrawView = rollupDrawCommands.importToGraph(graph);
    const rowCount = graph.createDataView(rollupDrawView.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: Uint32Array.BYTES_PER_ELEMENT
    });
    const values = importGraphBuffer(graph, 'class-values', classValuesBuffer, 'float32', capacity);
    const mask = importGraphBuffer(graph, 'class-mask', classMaskBuffer, 'uint32', capacity);
    // Table to columns: observations per cell (counts) or research-grade observations per cell (the fixed-point sums as f32).
    addCellTableColumnsNode(
      graph,
      `classes-${value}`,
      value === 'confirmed'
        ? {counts: rollupCounts, sumValues: rollupSumValues}
        : {counts: rollupCounts},
      {values, mask},
      rowCount
    );
    const breaks = importGraphBuffer(
      graph,
      'class-breaks',
      classBreaksBuffer,
      'float32',
      CLASS_CAPACITY + 1
    );
    const classCount = importGraphBuffer(graph, 'class-count', classCountBuffer, 'uint32', 1);
    graph.add(
      new GPUClassBreaks({
        id: `class-breaks-${value}`,
        values,
        mask,
        parameters: classParameters.importToGraph(graph),
        maximumClassCount: CLASS_CAPACITY,
        methods: classMethods,
        output: {breaks, classCount}
      })
    );
    graph.add(
      new GPUColorScale({
        id: `class-colors-${value}`,
        values,
        mask,
        domain: breaks,
        domainCount: classCount,
        palette: importGraphBuffer(graph, 'class-palette', classPalette, 'uint32', CLASS_CAPACITY),
        parameters: classColorScale.importToGraph(graph),
        maximumDomainCount: CLASS_CAPACITY + 1,
        maximumPaletteCount: CLASS_CAPACITY,
        output: {
          colors: importGraphBuffer(graph, 'class-colors', classColorsBuffer, 'uint32', capacity)
        }
      })
    );
    const reader = new SummaryReader(
      resources,
      `classes-${value}`,
      [
        {buffer: classBreaksBuffer, size: (CLASS_CAPACITY + 1) * 4},
        {buffer: classCountBuffer, size: 4}
      ],
      bytes => {
        const breaksArray = new Float32Array(bytes, 0, CLASS_CAPACITY + 1);
        const classCountValue = new Uint32Array(bytes, (CLASS_CAPACITY + 1) * 4, 1)[0];
        classSnapshot = {breaks: breaksArray.slice(), classCount: classCountValue};
        const edges = Array.from(breaksArray.subarray(0, classCountValue + 1)).map(edge =>
          Number.isFinite(edge) ? formatCount(edge) : 'n/a'
        );
        ctx.setReadout('classEdges', `${classCountValue} classes: ${edges.join(' | ')}`);
      }
    );
    const variant = {graph: resources.track(graph.compile()), reader};
    classVariants.set(value, variant);
    return variant;
  };

  const writeClassParameters = () => {
    const options = ctx.options;
    const classCount = options.classMethod === 'box-plot' ? 6 : options.classCount;
    classParameters.write(
      getGPUClassBreaksParameterValues(
        {method: options.classMethod, classCount: Math.min(classCount, CLASS_CAPACITY)},
        CLASS_CAPACITY
      )
    );
    classColorScale.write(
      getGPUColorScaleParameterValues({
        scale: 'threshold',
        domainCount: CLASS_CAPACITY + 1,
        paletteCount: classCount,
        noDataColor: packGPUColor(0, 0, 0, 0)
      })
    );
    const palette = new Uint32Array(CLASS_CAPACITY);
    for (let index = 0; index < classCount; index++) {
      const [r, g, b] = sampleRamp(options.ramp, classCount === 1 ? 0.5 : index / (classCount - 1));
      palette[index] = packGPUColor(r, g, b, 235);
    }
    classPalette.write(palette);
  };

  // --- Compare variants (period comparison recipe) ------------------------------------------------

  const compareVariants = new Map<string, CompareVariant>();
  let compareSnapshot: {
    rows: number;
    cells: Uint32Array;
    before: Float32Array;
    after: Float32Array;
    delta: Float32Array;
    zScore: Float32Array;
    beforeCells: number;
    afterCells: number;
    overflow: boolean;
  } | null = null;
  let compareTileLookup: Map<number, number> | null = null;
  let maximumDelta = 20;

  const getCompareKey = (options: Readonly<CellPyramidOptions>) =>
    `${options.compareResolution}|${options.compareMeasure}|${options.zScoreKind}`;

  const writeCompareParameters = (variant: CompareVariant) => {
    const options = ctx.options;
    const band = options.stableBand;
    variant.classBreaks.write(
      options.changeClassing === 'band'
        ? getGPUClassBreaksParameterValues(
            {method: 'custom', customEdges: [-1e9, -4 * band, -band, band, 4 * band, 1e9]},
            5
          )
        : getGPUClassBreaksParameterValues({method: 'standard-deviation', classCount: 5}, 5)
    );
  };

  const ensureCompare = (options: Readonly<CellPyramidOptions>): CompareVariant => {
    const key = getCompareKey(options);
    const existing = compareVariants.get(key);
    if (existing) return existing;
    const resolution = options.compareResolution;
    const sum = options.compareMeasure === 'sum';
    const owned = new SpatialAnalysisResources(device, `cell-compare-${compareVariants.size}`);
    const graph = new GPUCommandGraph<void>(device, {id: `cell-compare-${key}`});
    const output = <Format extends Parameters<typeof importGraphBuffer>[3]>(
      name: string,
      rowBytes: number,
      length: number,
      format: Format
    ) => {
      const buffer = owned.createBuffer(name, Math.max(1, length) * rowBytes);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const union = COMPARE_UNION_CAPACITY;
    const cells = output('union-cells', 8, union, 'uint32x2');
    const delta = output('delta', 4, union, 'float32');
    const ratio = output('ratio', 4, union, 'float32');
    const percentChange = output('percent-change', 4, union, 'float32');
    const zScore = output('z-score', 4, union, 'float32');
    const beforeValue = output('before-value', 4, union, 'float32');
    const afterValue = output('after-value', 4, union, 'float32');
    const count = output('union-count', 4, 1, 'uint32');
    const overflow = output('union-overflow', 4, 1, 'uint32');
    const colors = output('colors', 4, union, 'uint32');
    const beforeCellCount = output('before-table-count', 4, 1, 'uint32');
    const afterCellCount = output('after-table-count', 4, 1, 'uint32');
    const classBreaks = owned.createParameterBuffer(
      'compare-class-breaks',
      'float32',
      getGPUClassBreaksParameterLength(5)
    );
    const colorScale = owned.createParameterBuffer(
      'compare-color-scale',
      'float32',
      9,
      getGPUColorScaleParameterValues({
        scale: 'threshold',
        domainCount: 6,
        paletteCount: 5,
        noDataColor: packGPUColor(0, 0, 0, 0)
      })
    );
    const paletteBuffer = owned.createBuffer(
      'compare-palette',
      Uint32Array.from(CHANGE_CLASS_COLORS, ([r, g, b]) => packGPUColor(r, g, b, 235))
    );
    const rows = (maskBuffer: Buffer, name: string) =>
      importGraphBuffer(graph, name, maskBuffer, 'uint32', pointCount);
    const positions = importGraphBuffer(
      graph,
      'positions',
      positionsBuffer,
      'float32x2',
      pointCount
    );
    const valueView = sum
      ? importGraphBuffer(graph, 'values', researchGradeBuffer, 'float32', pointCount)
      : undefined;
    addPeriodComparisonRecipe(graph, {
      id: 'period',
      family: 'quadbin',
      resolution,
      tableCapacity: COMPARE_TABLE_CAPACITY,
      unionCapacity: union,
      before: {
        positions,
        values: valueView,
        mask: rows(beforeMaskBuffer, 'before-mask')
      },
      after: {
        positions,
        values: valueView,
        mask: rows(afterMaskBuffer, 'after-mask')
      },
      scratch: {
        before: {count: beforeCellCount.view},
        after: {count: afterCellCount.view}
      },
      measure: options.compareMeasure,
      zScore: options.zScoreKind,
      outputs: {
        comparison: {
          cells: cells.view,
          before: beforeValue.view,
          after: afterValue.view,
          delta: delta.view,
          ratio: ratio.view,
          percentChange: percentChange.view,
          zScore: zScore.view,
          count: count.view,
          overflow: overflow.view
        },
        colors: colors.view
      },
      classify: 'delta',
      classBreaksParameters: classBreaks.importToGraph(graph),
      maximumClassCount: 5,
      methods: ['custom', 'standard-deviation'],
      palette: importGraphBuffer(graph, 'palette', paletteBuffer, 'uint32', 5),
      colorScaleParameters: colorScale.importToGraph(graph),
      maximumPaletteCount: 5
    });
    const compiled = owned.track(graph.compile());
    const reader = new SummaryReader(
      owned,
      `cell-compare-${key}`,
      [
        {buffer: count.buffer, size: 4},
        {buffer: overflow.buffer, size: 4},
        {buffer: beforeCellCount.buffer, size: 4},
        {buffer: afterCellCount.buffer, size: 4},
        {buffer: cells.buffer, size: union * 8},
        {buffer: beforeValue.buffer, size: union * 4},
        {buffer: afterValue.buffer, size: union * 4},
        {buffer: delta.buffer, size: union * 4},
        {buffer: zScore.buffer, size: union * 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        const unionRows = Math.min(words[0], union);
        const base = 4;
        compareSnapshot = {
          rows: unionRows,
          cells: words.slice(base, base + unionRows * 2),
          before: new Float32Array(bytes.slice((base + union * 2) * 4, (base + union * 3) * 4)),
          after: new Float32Array(bytes.slice((base + union * 3) * 4, (base + union * 4) * 4)),
          delta: new Float32Array(bytes.slice((base + union * 4) * 4, (base + union * 5) * 4)),
          zScore: new Float32Array(bytes.slice((base + union * 5) * 4, (base + union * 6) * 4)),
          beforeCells: words[2],
          afterCells: words[3],
          overflow: words[1] !== 0
        };
        compareTileLookup = null;
        updateCompareReadouts();
      }
    );
    const variant: CompareVariant = {
      key,
      resolution,
      resources: owned,
      graph: compiled,
      classBreaks,
      colorScale,
      cells: cells.buffer,
      count: count.buffer,
      colors: colors.buffer,
      columns: {
        delta: delta.buffer,
        ratio: ratio.buffer,
        percentChange: percentChange.buffer,
        zScore: zScore.buffer
      },
      reader
    };
    writeCompareParameters(variant);
    compareVariants.set(key, variant);
    return variant;
  };

  const formatPlace = (low: number, high: number): string => {
    const [longitude, latitude] = getQuadbinCenter(low, high);
    const area = findArea(areas, longitude, latitude);
    return `${area >= 0 ? areas.names[area] : 'outside the city'} (${latitude.toFixed(3)}, ${longitude.toFixed(3)})`;
  };

  const updateCompareReadouts = () => {
    const snapshot = compareSnapshot;
    if (!snapshot) return;
    const options = ctx.options;
    const unit = options.compareMeasure === 'sum' ? 'confirmed' : 'observations';
    const band = options.stableBand;
    let gained = 0;
    let lost = 0;
    let beforeTotal = 0;
    let afterTotal = 0;
    let largestGain = 0;
    let largestGainRow = -1;
    let largestLoss = 0;
    let largestLossRow = -1;
    let significant = 0;
    for (let row = 0; row < snapshot.rows; row++) {
      const change = snapshot.delta[row];
      beforeTotal += snapshot.before[row];
      afterTotal += snapshot.after[row];
      if (change > band) gained++;
      else if (change < -band) lost++;
      if (change > largestGain) {
        largestGain = change;
        largestGainRow = row;
      }
      if (change < largestLoss) {
        largestLoss = change;
        largestLossRow = row;
      }
      if (Math.abs(snapshot.zScore[row]) > 2) significant++;
    }
    maximumDelta = Math.max(Math.abs(largestGain), Math.abs(largestLoss), 5);
    ctx.setReadout(
      'compareCells',
      `${formatCount(snapshot.beforeCells)} before, ${formatCount(snapshot.afterCells)} after, ${formatCount(snapshot.rows)} in the union${snapshot.overflow ? ' (OVERFLOW)' : ''}`
    );
    ctx.setReadout(
      'compareTotals',
      `${formatCount(beforeTotal)} to ${formatCount(afterTotal)} ${unit} (${beforeTotal > 0 ? (((afterTotal - beforeTotal) / beforeTotal) * 100).toFixed(1) : 'n/a'}%)`
    );
    ctx.setReadout(
      'compareGainLoss',
      `${formatCount(gained)} cells up and ${formatCount(lost)} down by more than ${band} ${unit}`
    );
    ctx.setReadout(
      'compareSignificant',
      `${formatCount(significant)} cells with |z| > 2 (${options.zScoreKind})`
    );
    ctx.setReadout(
      'compareLargestGain',
      largestGainRow >= 0
        ? `+${formatCount(largestGain)} ${unit} in ${formatPlace(snapshot.cells[largestGainRow * 2], snapshot.cells[largestGainRow * 2 + 1])}`
        : 'none'
    );
    ctx.setReadout(
      'compareLargestLoss',
      largestLossRow >= 0
        ? `${formatCount(largestLoss)} ${unit} in ${formatPlace(snapshot.cells[largestLossRow * 2], snapshot.cells[largestLossRow * 2 + 1])}`
        : 'none'
    );
    updateChangeLegend();
    ctx.requestLayers();
  };

  const getColorLimit = (): number => {
    const options = ctx.options;
    return (getColumnScale(options.changeColumn, maximumDelta) * options.colorLimit) / 100;
  };

  const updateChangeLegend = () => {
    ctx.setLegendExtent('change', [-getColorLimit(), getColorLimit()]);
  };

  // --- Masks ------------------------------------------------------------------------------------

  let pyramidStale = true;
  let rollupStale = true;
  let classesStale = true;
  let compareStale = true;
  let areaObservationCounts = new Uint32Array(areas.names.length + 1);

  const updateMasks = () => {
    const options = ctx.options;
    const kept = fillObservationMask(columns, maskValues, {
      category: options.category,
      months: [1, 12],
      sampledPercent: options.sampledPercent
    });
    maskBuffer.write(maskValues);
    const categoryIndex = getCategoryIndex(columns, options.category);
    const [aFirst, aLast] = options.periodA;
    const [bFirst, bLast] = options.periodB;
    let beforeKept = 0;
    let afterKept = 0;
    areaObservationCounts = new Uint32Array(areas.names.length + 1);
    for (let index = 0; index < pointCount; index++) {
      const month = columns.month[index];
      let inside = categoryIndex < 0 || columns.category[index] === categoryIndex;
      if (inside && options.sampledPercent < 100) {
        inside = hashToUnit(index) * 100 < options.sampledPercent;
      }
      const a = inside && month >= aFirst && month <= aLast ? 1 : 0;
      const b = inside && month >= bFirst && month <= bLast ? 1 : 0;
      beforeMaskValues[index] = a;
      afterMaskValues[index] = b;
      beforeKept += a;
      afterKept += b;
      if (maskValues[index]) areaObservationCounts[columns.communityArea[index]]++;
    }
    beforeMaskBuffer.write(beforeMaskValues);
    afterMaskBuffer.write(afterMaskValues);
    pyramidStale = true;
    compareStale = true;
    ctx.setReadout('points', `${formatCount(kept)} of ${formatCount(pointCount)}`);
    ctx.setReadout(
      'periods',
      `${formatMonthWindow(options.periodA)}: ${formatCount(beforeKept)} observations, ${formatMonthWindow(options.periodB)}: ${formatCount(afterKept)} observations`
    );
  };

  // --- Summaries --------------------------------------------------------------------------------

  let lastLevelIndex = 0;
  const pyramidReader = new SummaryReader(
    resources,
    'pyramid',
    [
      {buffer: levelCountsBuffer, size: LEVEL_COUNT * 4},
      {buffer: levelOverflowBuffers[0], size: LEVEL_COUNT * 4},
      {buffer: levelOverflowBuffers[1], size: LEVEL_COUNT * 4},
      {buffer: levelTotalsBuffer, size: 4},
      {buffer: drawCommands.buffer, size: 8} // vertex count, then the GPU-written instance count
    ],
    bytes => {
      const words = new Uint32Array(bytes);
      const counts = words.subarray(0, LEVEL_COUNT);
      const overflows = [
        words.subarray(LEVEL_COUNT, LEVEL_COUNT * 2),
        words.subarray(LEVEL_COUNT * 2, LEVEL_COUNT * 3)
      ];
      const finestTotal = words[LEVEL_COUNT * 3];
      const describeLevels = (first: number, last: number) => {
        const parts: string[] = [];
        for (let level = first; level <= last; level++) {
          parts.push(`${getLevelResolution(level)}: ${formatCount(counts[level])}`);
        }
        return parts.join(', ');
      };
      ctx.setReadout('levelsFine', `res ${describeLevels(0, 4)}`);
      ctx.setReadout('levelsMiddle', `res ${describeLevels(5, 9)}`);
      ctx.setReadout('levelsCoarse', `res ${describeLevels(10, LEVEL_COUNT - 1)}`);
      const overflowed: number[] = [];
      for (let level = 0; level < LEVEL_COUNT; level++) {
        if (overflows[level & 1][level]) overflowed.push(getLevelResolution(level));
      }
      ctx.setReadout(
        'overflow',
        overflowed.length
          ? `res ${overflowed.join(', ')} (unclamped ${formatCount(finestTotal)})`
          : 'none'
      );
      if (ctx.options.view === 'pyramid') {
        ctx.setReadout(
          'activeCells',
          `${formatCount(words[LEVEL_COUNT * 3 + 2])} cells at res ${getLevelResolution(lastLevelIndex)}`
        );
      }
    }
  );

  const sumWords = (words: Uint32Array, rows: number): bigint => {
    let total = 0n;
    for (let row = 0; row < rows; row++) {
      total += BigInt.asIntN(64, (BigInt(words[row * 2 + 1]) << 32n) | BigInt(words[row * 2]));
    }
    return total;
  };
  const sumCounts = (words: Uint32Array, rows: number): number => {
    let total = 0;
    for (let row = 0; row < rows; row++) total += words[row];
    return total;
  };
  const TOTALS_HEADER_BYTES = 20;
  let totalsResolution = DEFAULT_ROLLUP_RESOLUTION;
  const totalsReader = new SummaryReader(
    resources,
    'rollup-totals',
    [
      {buffer: tableCountBuffers[0], size: 4},
      {buffer: levelOverflowBuffers[0], size: 4},
      {buffer: rollupDrawCommands.buffer, size: 8},
      {buffer: rollupOverflowBuffer, size: 4},
      {buffer: countsSlabs[0], size: capacity * 4},
      {buffer: finestSumsBuffer, size: capacity * 8},
      {buffer: rollupCountsBuffer, size: capacity * 4},
      {buffer: rollupSumsBuffer, size: capacity * 8}
    ],
    bytes => {
      const header = new Uint32Array(bytes, 0, 5);
      const fineRows = header[0];
      const parentRows = header[3];
      const wordsAt = (byteOffset: number, length: number) =>
        new Uint32Array(bytes, byteOffset, length);
      const fineCount = sumCounts(wordsAt(TOTALS_HEADER_BYTES, capacity), fineRows);
      const fineSum = sumWords(wordsAt(TOTALS_HEADER_BYTES + capacity * 4, capacity * 2), fineRows);
      const parentCount = sumCounts(
        wordsAt(TOTALS_HEADER_BYTES + capacity * 12, capacity),
        parentRows
      );
      const parentSum = sumWords(
        wordsAt(TOTALS_HEADER_BYTES + capacity * 16, capacity * 2),
        parentRows
      );
      const toConfirmed = (sum: bigint) => Number(sum) / GPU_CELL_DEFAULT_SUM_SCALE;
      ctx.setReadout(
        'rollupCells',
        `${formatCount(parentRows)} cells at res ${totalsResolution} (from ${formatCount(fineRows)} at res ${FINEST_RESOLUTION})${header[1] || header[4] ? ' (OVERFLOW)' : ''}`
      );
      ctx.setReadout(
        'rollupCounts',
        `${formatCount(fineCount)} to ${formatCount(parentCount)} observations ${fineCount === parentCount ? '(conserved exactly)' : '(DIFFERS)'}`
      );
      ctx.setReadout(
        'rollupSums',
        `${formatCount(toConfirmed(fineSum))} to ${formatCount(toConfirmed(parentSum))} research-grade observations ${fineSum === parentSum ? '(bit-exact)' : '(DIFFERS)'}`
      );
    }
  );

  // --- Build orchestration ----------------------------------------------------------------------

  let destroyed = false;
  let ensureTimer: ReturnType<typeof setTimeout> | undefined;
  const ensureVariants = () => {
    ensureTimer = undefined;
    if (destroyed) return;
    const options = ctx.options;
    if (options.view === 'rollup' || options.view === 'classes') {
      ensureRollup(options.rollupResolution);
      totalsResolution = options.rollupResolution;
      rollupStale = true;
    }
    if (options.view === 'classes') {
      ensureClasses(options.classValue);
      writeClassParameters();
      classesStale = true;
    }
    if (options.view === 'compare') {
      ensureCompare(options);
      compareStale = true;
    }
    ctx.requestLayers();
  };
  const scheduleEnsure = () => {
    clearTimeout(ensureTimer);
    ensureTimer = setTimeout(ensureVariants, REBUILD_DEBOUNCE_MILLISECONDS);
  };

  updateMasks();
  ctx.setReadout(
    'pyramid',
    `${LEVEL_COUNT} levels, res ${FINEST_RESOLUTION} to ${COARSEST_RESOLUTION}, ${formatCount(capacity)} rows each`
  );
  ensureVariants();

  const getActiveCompare = (): CompareVariant | null =>
    compareVariants.get(getCompareKey(ctx.options)) ?? null;

  let pyramidEncodes = 0;
  let rollupEncodes = 0;
  const getTileLookup = (): Map<number, number> | null => {
    if (!compareSnapshot) return null;
    if (!compareTileLookup) {
      compareTileLookup = new Map();
      for (let row = 0; row < compareSnapshot.rows; row++) {
        const {x, y} = decodeQuadbin(
          compareSnapshot.cells[row * 2],
          compareSnapshot.cells[row * 2 + 1]
        );
        compareTileLookup.set(y * 1048576 + x, row);
      }
    }
    return compareTileLookup;
  };

  const instance: SceneInstance<CellPyramidOptions> = {
    getCompiledGraphs: () => {
      const graphs: CompiledGPUCommandGraph<void>[] = [compiledPyramid, compiledSelection];
      const options = ctx.options;
      if (options.view === 'rollup' || options.view === 'classes') {
        const variant = rollupVariants.get(options.rollupResolution);
        if (variant) graphs.push(variant.graph);
      }
      if (options.view === 'classes') {
        const variant = classVariants.get(options.classValue);
        if (variant) graphs.push(variant.graph);
      }
      if (options.view === 'compare') {
        const variant = getActiveCompare();
        if (variant) graphs.push(variant.graph);
      }
      return graphs as CompiledGPUCommandGraph<never>[];
    },

    setOption(id) {
      switch (id) {
        case 'category':
        case 'sampledPercent':
        case 'periodA':
        case 'periodB':
          updateMasks();
          rollupStale = true;
          classesStale = true;
          break;
        case 'view':
        case 'rollupResolution':
        case 'classValue':
        case 'compareResolution':
        case 'compareMeasure':
        case 'zScoreKind':
          scheduleEnsure();
          ctx.requestLayers();
          break;
        case 'classMethod':
        case 'classCount':
          writeClassParameters();
          classesStale = true;
          break;
        case 'ramp':
          writeClassParameters();
          classesStale = true;
          ctx.requestLayers();
          break;
        case 'stableBand':
        case 'changeClassing': {
          const variant = getActiveCompare();
          if (variant) {
            writeCompareParameters(variant);
            compareStale = true;
          }
          updateCompareReadouts();
          break;
        }
        case 'changeColumn':
        case 'colorLimit':
          updateChangeLegend();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      const view = options.view;
      // The pyramid depends only on the points and the mask, never on the camera.
      if (view !== 'compare' && pyramidStale) {
        compiledPyramid.encode(commandEncoder, {parameters: undefined});
        pyramidStale = false;
        rollupStale = true;
        pyramidEncodes++;
        ctx.setReadout('encodes', `${pyramidEncodes} pyramid, ${rollupEncodes} roll-up`);
        pyramidReader.markStale();
      }
      if (view === 'pyramid') {
        const zoom = frame.viewport.zoom;
        const targetResolution = options.manualLevel
          ? options.manualResolution
          : Math.round(zoom) + options.resolutionOffset;
        const clamped = Math.min(
          FINEST_RESOLUTION,
          Math.max(COARSEST_RESOLUTION, targetResolution)
        );
        lastLevelIndex = FINEST_RESOLUTION - clamped;
        activeLevel.write(Uint32Array.of(lastLevelIndex));
        ctx.setReadout('zoom', `${zoom.toFixed(2)} (map) to res ${clamped} (cells)`);
        compiledSelection.encode(commandEncoder, {parameters: undefined});
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 1) pyramidReader.markStale();
      }
      if ((view === 'rollup' || view === 'classes') && rollupStale && !pyramidStale) {
        const variant = rollupVariants.get(options.rollupResolution);
        if (variant) {
          variant.graph.encode(commandEncoder, {parameters: undefined});
          rollupStale = false;
          classesStale = true;
          rollupEncodes++;
          ctx.setReadout('encodes', `${pyramidEncodes} pyramid, ${rollupEncodes} roll-up`);
          totalsReader.markStale();
        }
      }
      if (view === 'classes' && classesStale && !rollupStale) {
        const variant = classVariants.get(options.classValue);
        if (variant) {
          variant.graph.encode(commandEncoder, {parameters: undefined});
          classesStale = false;
          variant.reader.markStale();
        }
      }
      if (view === 'compare' && compareStale) {
        const variant = getActiveCompare();
        if (variant) {
          variant.graph.encode(commandEncoder, {parameters: undefined});
          compareStale = false;
          variant.reader.markStale();
        }
      }
      pyramidReader.flush(commandEncoder);
      if (view === 'rollup' || view === 'classes') totalsReader.flush(commandEncoder);
      if (view === 'classes') classVariants.get(options.classValue)?.reader.flush(commandEncoder);
      if (view === 'compare') getActiveCompare()?.reader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const layers: Layer[] = [];
      if (options.view === 'pyramid') {
        layers.push(
          new QuadbinTableLayer({
            id: 'cell-pyramid-cells',
            cellsEven: cellsSlabs[0],
            cellsOdd: cellsSlabs[1],
            countsEven: countsSlabs[0],
            countsOdd: countsSlabs[1],
            activeLevel: activeLevel.buffer,
            firstRow: activeFirstRow,
            drawCommands,
            mode: 'density',
            ramp: options.ramp,
            densityLogRange: [0, options.densityCeiling]
          })
        );
      } else if (options.view === 'rollup') {
        layers.push(
          new QuadbinTableLayer({
            id: 'cell-rollup-cells',
            cellsEven: rollupCellsBuffer,
            countsEven: rollupCountsBuffer,
            values: rollupSumValuesBuffer,
            activeLevel: zeroWordBuffer,
            firstRow: zeroWordBuffer,
            drawCommands: rollupDrawCommands,
            mode: options.rollupColor === 'researchShare' ? 'meanValue' : 'density',
            ramp: options.ramp,
            densityLogRange: [0, options.densityCeiling],
            meanRange: [0, options.rateCeiling / 100],
            inset: 0.04
          })
        );
      } else if (options.view === 'classes') {
        layers.push(
          new QuadbinTableLayer({
            id: 'cell-class-cells',
            cellsEven: rollupCellsBuffer,
            colors: classColorsBuffer,
            activeLevel: zeroWordBuffer,
            firstRow: zeroWordBuffer,
            drawCommands: rollupDrawCommands,
            mode: 'packed',
            inset: 0.04,
            opacity: 0.9
          })
        );
      } else {
        const variant = getActiveCompare();
        if (variant) {
          const classes = options.changeColumn === 'classes';
          layers.push(
            new QuadbinTableLayer({
              id: `cell-compare-cells-${variant.key}-${options.changeColumn}`,
              cellsEven: variant.cells,
              colors: variant.colors,
              values: classes ? null : variant.columns[options.changeColumn],
              rowCount: variant.count,
              instanceCount: COMPARE_UNION_CAPACITY,
              mode: classes ? 'packed' : 'signed',
              ramp: 'diverging',
              signedLimit: getColorLimit(),
              signedTransform: options.changeColumn === 'ratio' ? 'log2' : 'linear',
              inset: 0.04,
              opacity: 0.88
            })
          );
        }
      }
      if (options.showAreas) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'cell-pyramid-areas',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: areaOutline,
            instanceCount: areas.outlineSegments.length / 4,
            widthPixels: 1.2,
            color: [255, 255, 255, 140]
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [longitude, latitude] = event.coordinate;
      const area = findArea(areas, longitude, latitude);
      const name = area >= 0 ? areas.names[area] : null;
      const options = ctx.options;
      if (options.view === 'compare' && compareSnapshot) {
        const lookup = getTileLookup();
        const zoom = options.compareResolution;
        const tiles = 2 ** zoom;
        const tileX = Math.floor(((longitude + 180) / 360) * tiles);
        const sine = Math.sin((latitude * Math.PI) / 180);
        const tileY = Math.floor((0.5 - Math.log((1 + sine) / (1 - sine)) / (4 * Math.PI)) * tiles);
        const row = lookup?.get(tileY * 1048576 + tileX);
        if (row !== undefined) {
          const before = compareSnapshot.before[row];
          const after = compareSnapshot.after[row];
          const unit = options.compareMeasure === 'sum' ? 'confirmed' : 'observations';
          const change = compareSnapshot.delta[row];
          return `${name ?? 'Outside the city'}: ${formatCount(before)} to ${formatCount(after)} ${unit} (${change >= 0 ? '+' : ''}${formatCount(change)}${before > 0 ? `, ${(((after - before) / before) * 100).toFixed(0)}%` : ''}, z ${compareSnapshot.zScore[row].toFixed(1)})`;
        }
      }
      return name
        ? `${name}: ${formatCount(areaObservationCounts[area + 1] ?? 0)} observations in the current filter`
        : null;
    },

    destroy() {
      destroyed = true;
      clearTimeout(ensureTimer);
      pyramidReader.stop();
      totalsReader.stop();
      for (const variant of classVariants.values()) variant.reader.stop();
      for (const variant of compareVariants.values()) {
        variant.reader.stop();
        variant.resources.destroy();
      }
      compareVariants.clear();
      resources.destroy();
    }
  };
  void classSnapshot;
  return instance;
}
