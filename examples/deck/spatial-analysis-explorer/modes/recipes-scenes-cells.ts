// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Recipe scenes over Quadbin cells: hot spot analysis, period comparison and space-time hot spots.
 */

import type {Layer} from '@deck.gl/core';
import {
  getGPUCalendarBucketsParameterValues,
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  packGPUColor
} from '@luma.gl/experimental/gpu-dataframe';
import {
  addHotSpotAnalysisRecipe,
  addPeriodComparisonRecipe,
  addSpaceTimeHotSpotsRecipe,
  getGPUEmergingHotSpotParameterValues,
  getGPUNeighborSearchParameterValues,
  getGPUPermutationParameterValues,
  getGPUSpatialAutocorrelationParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {formatCount} from '../spatial-analysis-resources';
import {PackedColorRasterLayer, PackedQuadbinCellLayer} from './classification-layers';
import {
  addKernelPass,
  DirtyEncoder,
  getCoreBounds,
  metersToDegrees,
  RecipeKit,
  SummaryReader,
  sliceSummary,
  type RecipeParameter,
  type RecipeSceneBuilder
} from './recipes-kit';

const HOT_SPOT_RESOLUTION = 15;
const HOT_SPOT_TABLE_CAPACITY = 4096;
const HOT_SPOT_NEIGHBORS_PER_ROW = 40;
const HOT_SPOT_MAXIMUM_PERMUTATIONS = 199;
/** Width of a Quadbin cell in degrees of longitude at the resolution. */
const HOT_SPOT_CELL_DEGREES = 360 / 2 ** HOT_SPOT_RESOLUTION;

/**
 * `addHotSpotAnalysisRecipe`: points to cells to Gi* z-scores to permutation confirmation to class
 * colors. The radius, permutation count and significance level are parameter buffers.
 */
export const buildHotSpotScene: RecipeSceneBuilder = async host => {
  const {context} = host;
  const trips = await context.data.getNewYorkTrips();
  context.signal.throwIfAborted();
  const projection = new LocalMetricProjection(trips.origin);
  const degrees = metersToDegrees(trips.vertexPositions, (x, y) => projection.unproject(x, y));
  const pointCount = degrees.length / 2;
  let minimumLongitude = Infinity;
  let minimumLatitude = Infinity;
  let maximumLongitude = -Infinity;
  let maximumLatitude = -Infinity;
  for (let index = 0; index < pointCount; index++) {
    minimumLongitude = Math.min(minimumLongitude, degrees[index * 2]);
    maximumLongitude = Math.max(maximumLongitude, degrees[index * 2]);
    minimumLatitude = Math.min(minimumLatitude, degrees[index * 2 + 1]);
    maximumLatitude = Math.max(maximumLatitude, degrees[index * 2 + 1]);
  }
  const padding = HOT_SPOT_CELL_DEGREES * 4;
  const bounds = [
    minimumLongitude - padding,
    minimumLatitude - padding,
    maximumLongitude + padding,
    maximumLatitude + padding
  ] as const;

  let radiusCells = 1.5;
  let permutations = 99;
  let significanceLevel = 0.05;
  let gateBySignificance = true;

  const kit = new RecipeKit(context.device, 'recipe-hot-spot');
  const capacity = HOT_SPOT_TABLE_CAPACITY;
  const positions = kit.input('positions', degrees, 'float32x2', pointCount);
  const searchParameters = kit.parameter(
    'search-parameters',
    'float32',
    getGPUNeighborSearchParameterValues({
      bounds,
      radius: radiusCells * HOT_SPOT_CELL_DEGREES,
      weightKind: 'binary'
    })
  );
  const autocorrelation = kit.parameter(
    'autocorrelation-parameters',
    'float32',
    getGPUSpatialAutocorrelationParameterValues({significanceLevel})
  );
  const permutationParameters = kit.parameter(
    'permutation-parameters',
    'uint32',
    getGPUPermutationParameterValues({seed: 7, permutations, significanceLevel})
  );
  const classBreaksParameters = kit.parameter(
    'class-breaks-parameters',
    'float32',
    getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 5}, 5)
  );
  const colorScaleParameters = kit.parameter(
    'color-scale-parameters',
    'float32',
    getGPUColorScaleParameterValues({scale: 'quantile', domainCount: 6, paletteCount: 5})
  );
  const paletteColors = [
    [44, 123, 182],
    [171, 217, 233],
    [235, 235, 235],
    [253, 174, 97],
    [215, 25, 28]
  ] as const;
  const palette = kit.input(
    'palette',
    Uint32Array.from(paletteColors, ([r, g, b]) => packGPUColor(r, g, b)),
    'uint32',
    5
  );
  const gate = kit.parameter('gate', 'uint32', Uint32Array.of(1));

  const tableCells = kit.output('table-cells', 'uint32x2', capacity);
  const tableCount = kit.output('table-count', 'uint32', 1);
  const tableOverflow = kit.output('table-overflow', 'uint32', 1);
  const weightsOverflow = kit.output('weights-overflow', 'uint32', 1);
  const bins = kit.output('bins', 'sint32', capacity);
  const globalStatistics = kit.output('global-statistics', 'float32', 4);
  const significant = kit.output('significant', 'uint32', capacity);
  const colors = kit.output('colors', 'uint32', capacity);
  const gatedColors = kit.output('gated-colors', 'uint32', capacity);

  const recipe = addHotSpotAnalysisRecipe(kit.graph, {
    source: {
      kind: 'points',
      positions: positions.view,
      family: 'quadbin',
      resolution: HOT_SPOT_RESOLUTION,
      tableCapacity: capacity,
      neighborCapacity: capacity * HOT_SPOT_NEIGHBORS_PER_ROW,
      gridSize: [64, 64],
      neighborSearchParameters: searchParameters.view
    },
    parameters: autocorrelation.view,
    scratch: {
      table: {cells: tableCells.view, count: tableCount.view, overflow: tableOverflow.view}
    },
    outputs: {
      bins: bins.view,
      globalStatistics: globalStatistics.view,
      weightsOverflow: weightsOverflow.view,
      permutation: {significant: significant.view},
      color: {colors: colors.view}
    },
    permutation: {
      parameters: permutationParameters.view,
      maximumPermutations: HOT_SPOT_MAXIMUM_PERMUTATIONS
    },
    color: {
      classBreaksParameters: classBreaksParameters.view,
      maximumClassCount: 5,
      methods: ['equal-interval'],
      colorScaleParameters: colorScaleParameters.view,
      palette: palette.view,
      maximumPaletteCount: 5
    }
  });
  // Display adapter: dim the cells the permutation test does not confirm (a per-frame gate).
  addKernelPass(kit.graph, {
    id: 'hot-spot-gate',
    count: capacity,
    bindings: [
      {name: 'significant', view: significant.view, access: 'read', type: 'u32'},
      {name: 'colors', view: colors.view, access: 'read', type: 'u32'},
      {name: 'gate', view: gate.view, access: 'read', type: 'u32'},
      {name: 'gated', view: gatedColors.view, access: 'read_write', type: 'u32'}
    ],
    body: `let color = colors[colorsOffset + index];
  let keep = gate[gateOffset] == 0u || significant[significantOffset + index] != 0u;
  gated[gatedOffset + index] = select((color & 0x00ffffffu) | 0x30000000u, color, keep);`
  });
  const compiled = kit.compile();

  const summarySizes = [4, 4, 4, 16, capacity * 4, capacity * 4];
  const reader = new SummaryReader(
    kit.resources,
    'hot-spot',
    [
      {buffer: tableCount.buffer, size: 4},
      {buffer: tableOverflow.buffer, size: 4},
      {buffer: weightsOverflow.buffer, size: 4},
      {buffer: globalStatistics.buffer, size: 16},
      {buffer: bins.buffer, size: capacity * 4},
      {buffer: significant.buffer, size: capacity * 4}
    ],
    bytes => {
      const [count, tableFlag, weightsFlag, statistics, binRows, significantRows] = sliceSummary(
        bytes,
        summarySizes
      );
      const rows = Math.min(count.u32[0], capacity);
      let hot = 0;
      let cold = 0;
      let confirmed = 0;
      for (let row = 0; row < rows; row++) {
        if (binRows.i32[row] > 0) hot++;
        else if (binRows.i32[row] < 0) cold++;
        if (significantRows.u32[row] !== 0) confirmed++;
      }
      host.setOutputs([
        ['Occupied cells', `${formatCount(rows)} of ${formatCount(capacity)}`],
        ['Gi* hot / cold (90%+)', `${formatCount(hot)} / ${formatCount(cold)}`],
        ['Permutation-confirmed', `${formatCount(confirmed)} (${permutations} runs)`],
        ['Mean points per cell', statistics.f32[1].toFixed(1)],
        ['Table overflow', tableFlag.u32[0] ? 'YES' : 'no'],
        ['Neighbor overflow', weightsFlag.u32[0] ? 'YES' : 'no']
      ]);
    }
  );
  const encoder = new DirtyEncoder(compiled, reader);

  const writeNeighborParameters = () => {
    searchParameters.parameters.write(
      getGPUNeighborSearchParameterValues({
        bounds,
        radius: radiusCells * HOT_SPOT_CELL_DEGREES,
        weightKind: 'binary'
      })
    );
    encoder.markDirty();
  };
  const writeSignificance = () => {
    autocorrelation.parameters.write(
      getGPUSpatialAutocorrelationParameterValues({significanceLevel})
    );
    permutationParameters.parameters.write(
      getGPUPermutationParameterValues({seed: 7, permutations, significanceLevel})
    );
    encoder.markDirty();
  };
  const parameters: RecipeParameter[] = [
    {
      kind: 'slider',
      label: 'Neighborhood radius',
      minimum: 1,
      maximum: 3,
      step: 0.1,
      value: radiusCells,
      format: value => `${value.toFixed(1)} cells (${Math.round(value * 925)} m)`,
      onChange: value => {
        radiusCells = value;
        writeNeighborParameters();
      }
    },
    {
      kind: 'slider',
      label: 'Permutations',
      minimum: 19,
      maximum: HOT_SPOT_MAXIMUM_PERMUTATIONS,
      step: 10,
      value: permutations,
      format: value => `${value}`,
      onChange: value => {
        permutations = value;
        writeSignificance();
      }
    },
    {
      kind: 'slider',
      label: 'Significance level',
      minimum: 0.01,
      maximum: 0.2,
      step: 0.01,
      value: significanceLevel,
      format: value => `p <= ${value.toFixed(2)}`,
      onChange: value => {
        significanceLevel = value;
        writeSignificance();
      }
    },
    {
      kind: 'toggle',
      label: 'Dim cells the permutation test does not confirm',
      value: gateBySignificance,
      onChange: value => {
        gateBySignificance = value;
        gate.parameters.write(Uint32Array.of(value ? 1 : 0));
        encoder.markDirty();
      }
    }
  ];

  return {
    compiled,
    contributorCount: recipe.contributors.length,
    chain: [
      'GPUPointToCell (H3 only; Quadbin bins directly)',
      'GPUCellAggregation (points per cell)',
      'GPUCellGeometry (cell centers)',
      'GPUNeighborSearch (radius weights)',
      'GPUHotSpotAnalysis (Gi*)',
      'GPULocalPermutationTest (pseudo p)',
      'GPUClassBreaks + GPUColorScale'
    ],
    parameters,
    legend:
      'Cells colored by Gi* z-score class, cold blue to hot red; dimmed cells fail the permutation test.',
    dataNote: `${trips.attribution}, ${formatCount(pointCount)} vertices`,
    encode: commandEncoder => encoder.encode(commandEncoder),
    getLayers: (): Layer[] => [
      new PackedQuadbinCellLayer({
        id: 'recipe-hot-spot-cells',
        cells: tableCells.buffer,
        colors: gatedColors.buffer,
        count: tableCount.buffer,
        capacity,
        inset: 0.04,
        opacity: 0.8
      })
    ],
    destroy: () => {
      reader.stop();
      kit.resources.destroy();
    }
  };
};

const PERIOD_RESOLUTION = 15;
const PERIOD_TABLE_CAPACITY = 4096;
const PERIOD_UNION_CAPACITY = 8192;

/**
 * `addPeriodComparisonRecipe`: two windows of the same trip vertices become two cell tables, an
 * outer join with delta, and diverging class colors. The window length, the boundary time and the
 * "no change" band are per-frame writes (mask buffers and class-break edges).
 */
export const buildPeriodComparisonScene: RecipeSceneBuilder = async host => {
  const {context} = host;
  const trips = await context.data.getNewYorkTrips();
  context.signal.throwIfAborted();
  const projection = new LocalMetricProjection(trips.origin);
  const degrees = metersToDegrees(trips.vertexPositions, (x, y) => projection.unproject(x, y));
  const times = trips.vertexTimestamps;
  const pointCount = times.length;
  const [firstTime, lastTime] = trips.timeRange;

  let windowSeconds = 600;
  let boundaryTime = (firstTime + lastTime) / 2;
  let stableBand = 4;
  const beforeMask = new Uint32Array(pointCount);
  const afterMask = new Uint32Array(pointCount);

  const kit = new RecipeKit(context.device, 'recipe-period');
  const positions = kit.input('positions', degrees, 'float32x2', pointCount);
  const before = kit.input('before-mask', beforeMask, 'uint32', pointCount);
  const after = kit.input('after-mask', afterMask, 'uint32', pointCount);
  const getEdges = () =>
    [-1e6, -4 * stableBand, -stableBand, stableBand, 4 * stableBand, 1e6] as const;
  const classBreaks = kit.parameter(
    'class-breaks-parameters',
    'float32',
    getGPUClassBreaksParameterValues({method: 'custom', customEdges: getEdges()}, 5)
  );
  const colorScale = kit.parameter(
    'color-scale-parameters',
    'float32',
    getGPUColorScaleParameterValues({
      scale: 'threshold',
      domainCount: 6,
      paletteCount: 5,
      noDataColor: packGPUColor(0, 0, 0, 0)
    })
  );
  const palette = kit.input(
    'palette',
    Uint32Array.from(
      [
        [33, 102, 172],
        [146, 197, 222],
        [235, 235, 235],
        [244, 165, 130],
        [178, 24, 43]
      ],
      ([r, g, b]) => packGPUColor(r, g, b)
    ),
    'uint32',
    5
  );
  const unionCells = kit.output('union-cells', 'uint32x2', PERIOD_UNION_CAPACITY);
  const delta = kit.output('delta', 'float32', PERIOD_UNION_CAPACITY);
  const unionCount = kit.output('union-count', 'uint32', 1);
  const unionOverflow = kit.output('union-overflow', 'uint32', 1);
  const colors = kit.output('colors', 'uint32', PERIOD_UNION_CAPACITY);
  const beforeTableCount = kit.output('before-table-count', 'uint32', 1);
  const afterTableCount = kit.output('after-table-count', 'uint32', 1);

  const recipe = addPeriodComparisonRecipe(kit.graph, {
    family: 'quadbin',
    resolution: PERIOD_RESOLUTION,
    tableCapacity: PERIOD_TABLE_CAPACITY,
    unionCapacity: PERIOD_UNION_CAPACITY,
    before: {
      positions: positions.view,
      mask: before.view
    },
    after: {positions: positions.view, mask: after.view},
    scratch: {
      before: {count: beforeTableCount.view},
      after: {count: afterTableCount.view}
    },
    outputs: {
      comparison: {
        cells: unionCells.view,
        delta: delta.view,
        count: unionCount.view,
        overflow: unionOverflow.view
      },
      colors: colors.view
    },
    classify: 'delta',
    classBreaksParameters: classBreaks.view,
    maximumClassCount: 5,
    methods: ['custom'],
    palette: palette.view,
    colorScaleParameters: colorScale.view,
    maximumPaletteCount: 5
  });
  const compiled = kit.compile();

  const unionBytes = PERIOD_UNION_CAPACITY * 4;
  const summarySizes = [4, 4, 4, 4, unionBytes];
  const reader = new SummaryReader(
    kit.resources,
    'period',
    [
      {buffer: unionCount.buffer, size: 4},
      {buffer: unionOverflow.buffer, size: 4},
      {buffer: beforeTableCount.buffer, size: 4},
      {buffer: afterTableCount.buffer, size: 4},
      {buffer: delta.buffer, size: unionBytes}
    ],
    bytes => {
      const [count, overflow, beforeCells, afterCells, deltas] = sliceSummary(bytes, summarySizes);
      const rows = Math.min(count.u32[0], PERIOD_UNION_CAPACITY);
      let gained = 0;
      let lost = 0;
      let largestGain = 0;
      let largestLoss = 0;
      for (let row = 0; row < rows; row++) {
        const change = deltas.f32[row];
        if (change > stableBand) gained++;
        else if (change < -stableBand) lost++;
        largestGain = Math.max(largestGain, change);
        largestLoss = Math.min(largestLoss, change);
      }
      host.setOutputs([
        [
          'Cells before / after',
          `${formatCount(beforeCells.u32[0])} / ${formatCount(afterCells.u32[0])}`
        ],
        ['Cells in union', formatCount(rows)],
        ['Gained / lost beyond band', `${formatCount(gained)} / ${formatCount(lost)}`],
        ['Largest gain / loss', `+${largestGain.toFixed(0)} / ${largestLoss.toFixed(0)}`],
        ['Union overflow', overflow.u32[0] ? 'YES' : 'no']
      ]);
    }
  );
  const encoder = new DirtyEncoder(compiled, reader);

  const writeMasks = () => {
    const beforeStart = boundaryTime - windowSeconds;
    const afterEnd = boundaryTime + windowSeconds;
    let beforeCount = 0;
    let afterCount = 0;
    for (let row = 0; row < pointCount; row++) {
      const time = times[row];
      beforeMask[row] = time >= beforeStart && time < boundaryTime ? 1 : 0;
      afterMask[row] = time >= boundaryTime && time < afterEnd ? 1 : 0;
      beforeCount += beforeMask[row];
      afterCount += afterMask[row];
    }
    before.buffer.write(beforeMask);
    after.buffer.write(afterMask);
    void beforeCount;
    void afterCount;
    encoder.markDirty();
  };
  writeMasks();
  const parameters: RecipeParameter[] = [
    {
      kind: 'slider',
      label: 'Boundary between the periods',
      minimum: Math.round(firstTime + 300),
      maximum: Math.round(lastTime - 300),
      step: 30,
      value: Math.round(boundaryTime),
      format: value => `t = ${Math.round(value / 60)} min`,
      onChange: value => {
        boundaryTime = value;
        writeMasks();
      }
    },
    {
      kind: 'slider',
      label: 'Length of each period',
      minimum: 120,
      maximum: 1200,
      step: 60,
      value: windowSeconds,
      format: value => `${Math.round(value / 60)} min`,
      onChange: value => {
        windowSeconds = value;
        writeMasks();
      }
    },
    {
      kind: 'slider',
      label: 'No-change band (class break)',
      minimum: 1,
      maximum: 30,
      step: 1,
      value: stableBand,
      format: value => `+/- ${value} points`,
      onChange: value => {
        stableBand = value;
        classBreaks.parameters.write(
          getGPUClassBreaksParameterValues({method: 'custom', customEdges: getEdges()}, 5)
        );
        encoder.markDirty();
      }
    }
  ];

  return {
    compiled,
    contributorCount: recipe.contributors.length,
    chain: [
      'GPUCellAggregation (earlier period)',
      'GPUCellAggregation (later period)',
      'GPUCellTableCompare (outer join, delta)',
      'union mask adapter',
      'GPUClassBreaks (custom edges)',
      'GPUColorScale (threshold)'
    ].slice(0, 6),
    parameters,
    legend:
      'Red: more taxi-trip vertices in the later period; blue: fewer; gray: within the no-change band.',
    dataNote: `${trips.attribution}, ${formatCount(pointCount)} vertices split by time`,
    encode: commandEncoder => encoder.encode(commandEncoder),
    getLayers: (): Layer[] => [
      new PackedQuadbinCellLayer({
        id: 'recipe-period-cells',
        cells: unionCells.buffer,
        colors: colors.buffer,
        count: unionCount.buffer,
        capacity: PERIOD_UNION_CAPACITY,
        inset: 0.04,
        opacity: 0.8
      })
    ],
    destroy: () => {
      reader.stop();
      kit.resources.destroy();
    }
  };
};

const SPACE_TIME_WIDTH = 16;
const SPACE_TIME_HEIGHT = 20;
/** Minutes with trips: the real trips start in the first 30 minutes, then the last ones finish. */
const SPACE_TIME_SLICES = 30;
/** Palette of the 17 emerging hot spot categories: none, 8 hot (yellow to red), 8 cold (blue). */
const EMERGING_PALETTE: readonly (readonly [number, number, number, number])[] = [
  [0, 0, 0, 0],
  [255, 237, 160, 235],
  [254, 217, 118, 235],
  [254, 178, 76, 240],
  [253, 141, 60, 245],
  [252, 78, 42, 250],
  [227, 26, 28, 250],
  [189, 0, 38, 255],
  [128, 0, 38, 255],
  [198, 219, 239, 235],
  [158, 202, 225, 235],
  [107, 174, 214, 240],
  [66, 146, 198, 245],
  [33, 113, 181, 250],
  [8, 81, 156, 250],
  [8, 48, 107, 255],
  [8, 29, 88, 255]
];
const EMERGING_NAMES = [
  'no pattern',
  'new hot',
  'consecutive hot',
  'intensifying hot',
  'persistent hot',
  'diminishing hot',
  'sporadic hot',
  'oscillating hot',
  'historical hot',
  'new cold',
  'consecutive cold',
  'intensifying cold',
  'persistent cold',
  'diminishing cold',
  'sporadic cold',
  'oscillating cold',
  'historical cold'
];
const CONFIDENCE_LEVELS = [0.9, 0.95, 0.99] as const;

/**
 * `addSpaceTimeHotSpotsRecipe`: trip vertices are binned into a 16 x 20 lattice by calendar minute,
 * Gi* is run on every space-time bin and each cell gets an emerging hot spot category. Radius,
 * temporal window and confidence level are per-frame parameters.
 */
export const buildSpaceTimeScene: RecipeSceneBuilder = async host => {
  const {context} = host;
  const trips = await context.data.getNewYorkTrips();
  context.signal.throwIfAborted();
  const pointCount = trips.vertexTimestamps.length;
  const bounds = getCoreBounds(trips.vertexPositions);
  const cellCount = SPACE_TIME_WIDTH * SPACE_TIME_HEIGHT;

  // Calendar time for the trips: seconds since a fixed 12:00 UTC start, as epoch milliseconds.
  const epochStart = Date.UTC(2023, 5, 1, 12, 0, 0);
  const timestampWords = new Uint32Array(pointCount * 2);
  for (let row = 0; row < pointCount; row++) {
    const milliseconds = epochStart + Math.round(trips.vertexTimestamps[row] * 1000);
    timestampWords[row * 2] = milliseconds % 2 ** 32;
    timestampWords[row * 2 + 1] = Math.floor(milliseconds / 2 ** 32);
  }

  let radiusCells = 1.5;
  let temporalWindow = 1;
  let confidenceIndex = 0;
  const getParameters = () =>
    getGPUEmergingHotSpotParameterValues({
      radius: radiusCells,
      temporalWindow,
      confidenceLevel: CONFIDENCE_LEVELS[confidenceIndex]
    });

  const kit = new RecipeKit(context.device, 'recipe-space-time');
  const positions = kit.input('positions', trips.vertexPositions, 'float32x2', pointCount);
  const timestamps = kit.input('timestamps', timestampWords, 'uint32x2', pointCount);
  const calendar = kit.input(
    'calendar-parameters',
    getGPUCalendarBucketsParameterValues(0),
    'sint32',
    2
  );
  const emerging = kit.parameter('emerging-parameters', 'float32', getParameters());
  const colorScale = kit.parameter(
    'color-scale-parameters',
    'float32',
    getGPUColorScaleParameterValues({scale: 'ordinal', domainCount: 0, paletteCount: 17})
  );
  const palette = kit.input(
    'palette',
    Uint32Array.from(EMERGING_PALETTE, ([r, g, b, a]) => packGPUColor(r, g, b, a)),
    'uint32',
    17
  );
  const category = kit.output('category', 'uint32', cellCount);
  const hotSlices = kit.output('hot-slices', 'uint32', cellCount);
  const coldSlices = kit.output('cold-slices', 'uint32', cellCount);
  const colors = kit.output('colors', 'uint32', cellCount);
  const recipe = addSpaceTimeHotSpotsRecipe(kit.graph, {
    timestamps: timestamps.view,
    calendarParameters: calendar.view,
    slices: {field: 'minute', firstValue: 0, count: SPACE_TIME_SLICES},
    cells: {
      kind: 'lattice',
      positions: positions.view,
      width: SPACE_TIME_WIDTH,
      height: SPACE_TIME_HEIGHT,
      bounds
    },
    parameters: emerging.view,
    outputs: {
      category: category.view,
      hotSliceCount: hotSlices.view,
      coldSliceCount: coldSlices.view,
      colors: colors.view
    },
    color: {palette: palette.view, parameters: colorScale.view}
  });
  const compiled = kit.compile();

  const cellBytes = cellCount * 4;
  const summarySizes = [cellBytes, cellBytes];
  const reader = new SummaryReader(
    kit.resources,
    'space-time',
    [
      {buffer: category.buffer, size: cellBytes},
      {buffer: hotSlices.buffer, size: cellBytes}
    ],
    bytes => {
      const [categories, hot] = sliceSummary(bytes, summarySizes);
      const histogram = new Uint32Array(17);
      let mostHotSlices = 0;
      for (let cell = 0; cell < cellCount; cell++) {
        histogram[Math.min(16, categories.u32[cell])]++;
        mostHotSlices = Math.max(mostHotSlices, hot.u32[cell]);
      }
      const hotCells = histogram.slice(1, 9).reduce((total, value) => total + value, 0);
      const coldCells = histogram.slice(9).reduce((total, value) => total + value, 0);
      const dominant = Array.from(histogram.keys())
        .filter(code => code > 0 && histogram[code] > 0)
        .sort((a, b) => histogram[b] - histogram[a])[0];
      host.setOutputs([
        ['Hot / cold cells', `${hotCells} / ${coldCells} of ${cellCount}`],
        [
          'Most common pattern',
          dominant === undefined
            ? 'none'
            : `${EMERGING_NAMES[dominant]} (${histogram[dominant]} cells)`
        ],
        ['New or consecutive hot', `${histogram[1] + histogram[2]}`],
        ['Intensifying / persistent hot', `${histogram[3]} / ${histogram[4]}`],
        ['Most hot slices in a cell', `${mostHotSlices} of ${SPACE_TIME_SLICES}`]
      ]);
    }
  );
  const encoder = new DirtyEncoder(compiled, reader);
  const writeParameters = () => {
    emerging.parameters.write(getParameters());
    encoder.markDirty();
  };
  const parameters: RecipeParameter[] = [
    {
      kind: 'slider',
      label: 'Spatial neighborhood radius',
      minimum: 0,
      maximum: 4,
      step: 0.5,
      value: radiusCells,
      format: value => `${value.toFixed(1)} cells`,
      onChange: value => {
        radiusCells = value;
        writeParameters();
      }
    },
    {
      kind: 'slider',
      label: 'Temporal window (previous minutes)',
      minimum: 0,
      maximum: 4,
      step: 1,
      value: temporalWindow,
      format: value => `${value} min`,
      onChange: value => {
        temporalWindow = value;
        writeParameters();
      }
    },
    {
      kind: 'slider',
      label: 'Confidence level',
      minimum: 0,
      maximum: 2,
      step: 1,
      value: confidenceIndex,
      format: value => `${Math.round(CONFIDENCE_LEVELS[value] * 100)}%`,
      onChange: value => {
        confidenceIndex = value;
        writeParameters();
      }
    }
  ];

  return {
    compiled,
    contributorCount: recipe.contributors.length,
    chain: [
      'GPUCalendarBuckets (minute of each event)',
      'key adapter + GPUGroupStatistics (counts per cell and minute)',
      'dense cube scatter adapter',
      'GPUEmergingHotSpots (Gi* per bin, Mann-Kendall)',
      'GPUColorScale (ordinal categories)'
    ],
    parameters,
    legend:
      'Yellow to dark red: new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating, historical hot. Blues are the cold mirror.',
    dataNote: `${trips.attribution}; trip seconds mapped to calendar minutes of a fixed hour`,
    encode: commandEncoder => encoder.encode(commandEncoder),
    getLayers: (): Layer[] => [
      new PackedColorRasterLayer({
        id: 'recipe-space-time-cells',
        coordinateOrigin: [trips.origin[0], trips.origin[1], 0],
        gridSize: [SPACE_TIME_WIDTH, SPACE_TIME_HEIGHT],
        bounds,
        rowOrigin: 'south',
        values: colors.buffer,
        valueFormat: 'uint32',
        opacity: 0.8
      })
    ],
    destroy: () => {
      reader.stop();
      kit.resources.destroy();
    }
  };
};
