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
  getGPUNeighborSearchParameterValues,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_WGS84_SEMI_MAJOR_AXIS_METERS
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {formatCount} from '../spatial-analysis-resources';
import {PackedColorRasterLayer, PackedQuadbinCellLayer} from './classification-layers';
import {
  addKernelPass,
  DirtyEncoder,
  getCoreBounds,
  metersToDegrees,
  RecipeKit,
  SummaryReader,
  describeMultipleTesting,
  sliceSummary,
  type RecipeParameter,
  type RecipeSceneBuilder
} from './recipes-kit';

const HOT_SPOT_RESOLUTION = 15;
/** Neighbor slots per study cell: a 3-cell metric band holds about 28 cell centers. */
const HOT_SPOT_NEIGHBORS_PER_ROW = 40;
/** Quadbin cell width at the equator in meters (Web Mercator circumference over 2^resolution). */
const HOT_SPOT_CELL_EQUATOR_METERS = (2 * Math.PI * GPU_WGS84_SEMI_MAJOR_AXIS_METERS) / 2 ** 15;
const HOT_SPOT_MAXIMUM_PERMUTATIONS = 199;
/** Width of a Quadbin cell in degrees of longitude at the resolution. */
const HOT_SPOT_CELL_DEGREES = 360 / 2 ** HOT_SPOT_RESOLUTION;

/** Web Mercator tile row of a latitude at the hot spot resolution. */
function getHotSpotTileRow(latitude: number): number {
  const radians = (latitude * Math.PI) / 180;
  const fraction = (1 - Math.asinh(Math.tan(radians)) / Math.PI) / 2;
  return Math.floor(fraction * 2 ** HOT_SPOT_RESOLUTION);
}

/**
 * `addHotSpotAnalysisRecipe`: points to the cells of a study area (empty cells included) to Gi*
 * z-scores to permutation confirmation to class colors. The neighborhood is a great-circle
 * distance band in meters over the cell centers; the band radius, permutation count and
 * significance level are parameter buffers.
 */
export const buildHotSpotScene: RecipeSceneBuilder = async host => {
  const {context} = host;
  const trips = await context.data.getNewYorkTrips();
  context.signal.throwIfAborted();
  const projection = trips.frame;
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

  // Study area: the padded bounding rectangle of the trips, as planar longitude/latitude
  // polygon vertices (the ring closes implicitly). Every Quadbin cell whose center is inside it is
  // an analysis row, empty or not, so empty ground is the null Gi* compares against.
  const studyPolygon = Float32Array.of(
    bounds[0],
    bounds[1],
    bounds[2],
    bounds[1],
    bounds[2],
    bounds[3],
    bounds[0],
    bounds[3]
  );
  const tileColumns = Math.floor(((bounds[2] - bounds[0]) / 360) * 2 ** HOT_SPOT_RESOLUTION) + 2;
  const tileRows = getHotSpotTileRow(bounds[1]) - getHotSpotTileRow(bounds[3]) + 2;
  const capacity = tileColumns * tileRows;
  // Cell width in meters at the study area's mid latitude (Web Mercator is conformal, so the
  // cell is as tall as it is wide). The band radius is a great-circle distance in meters.
  const cellMeters =
    HOT_SPOT_CELL_EQUATOR_METERS * Math.cos((((bounds[1] + bounds[3]) / 2) * Math.PI) / 180);
  let radiusCells = 1.6;
  let permutations = 99;
  let significanceLevel = 0.05;
  let gateBySignificance = true;

  const kit = new RecipeKit(context.device, 'recipe-hot-spot');
  const positions = kit.input('positions', degrees, 'float32x2', pointCount);
  const studyPositions = kit.input('study-positions', studyPolygon, 'float32x2', 4);
  const studyFeatureOffsets = kit.input('study-feature-offsets', Uint32Array.of(0, 1), 'uint32', 2);
  const studyPolygonOffsets = kit.input('study-polygon-offsets', Uint32Array.of(0, 1), 'uint32', 2);
  const studyRingOffsets = kit.input('study-ring-offsets', Uint32Array.of(0, 4), 'uint32', 2);
  const searchParameters = kit.parameter(
    'search-parameters',
    'float32',
    getGPUNeighborSearchParameterValues({
      bounds,
      radius: radiusCells * cellMeters,
      weightKind: 'binary'
    })
  );
  const autocorrelation = kit.parameterBuffer(
    'autocorrelation-parameters',
    'float32',
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );
  const permutationParameters = kit.parameterBuffer(
    'permutation-parameters',
    'uint32',
    GPU_PERMUTATION_PARAMETER_LENGTH
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
  const invalidCount = kit.output('invalid-count', 'uint32', 1);
  const bins = kit.output('bins', 'sint32', capacity);
  const globalStatistics = kit.output('global-statistics', 'float32', 5);
  const significant = kit.output('significant', 'uint32', capacity);
  const colors = kit.output('colors', 'uint32', capacity);
  const gatedColors = kit.output('gated-colors', 'uint32', capacity);

  const recipe = addHotSpotAnalysisRecipe(kit.graph, {
    source: {
      kind: 'points',
      positions: positions.view,
      family: 'quadbin',
      resolution: HOT_SPOT_RESOLUTION,
      studyArea: {
        kind: 'polygons',
        positions: studyPositions.view,
        featureOffsets: studyFeatureOffsets.view,
        polygonOffsets: studyPolygonOffsets.view,
        ringOffsets: studyRingOffsets.view,
        candidateCapacity: capacity * 2
      },
      cellCapacity: capacity,
      distanceBand: {
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
        neighborSearchParameters: searchParameters.view,
        gridSize: [64, 64],
        neighborCapacity: capacity * HOT_SPOT_NEIGHBORS_PER_ROW
      }
    },
    settings: {significanceLevel, seed: 7, permutationCount: permutations},
    parameters: autocorrelation,
    scratch: {
      table: {
        cells: tableCells.view,
        status: {count: tableCount.view, overflow: tableOverflow.view}
      }
    },
    outputs: {
      bins: bins.view,
      globalStatistics: globalStatistics.view,
      status: {weightsOverflow: weightsOverflow.view, invalidCount: invalidCount.view},
      permutation: {significant: significant.view},
      color: {colors: colors.view}
    },
    permutation: {
      parameters: permutationParameters,
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
  const compiled = await kit.compileAsync();

  const summarySizes = [4, 4, 4, 20, 4, capacity * 4, capacity * 4];
  const reader = new SummaryReader(
    kit.resources,
    'hot-spot',
    [
      {buffer: tableCount.buffer, size: 4},
      {buffer: tableOverflow.buffer, size: 4},
      {buffer: weightsOverflow.buffer, size: 4},
      {buffer: globalStatistics.buffer, size: 20},
      {buffer: invalidCount.buffer, size: 4},
      {buffer: bins.buffer, size: capacity * 4},
      {buffer: significant.buffer, size: capacity * 4}
    ],
    bytes => {
      const [count, tableFlag, weightsFlag, statistics, invalid, binRows, significantRows] =
        sliceSummary(bytes, summarySizes);
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
        ['Study-area cells', `${formatCount(rows)} (capacity ${formatCount(capacity)})`],
        ['Not tested (under 8 neighbors)', formatCount(invalid.u32[0])],
        [
          `Gi* hot / cold (${describeMultipleTesting(recipe.inference.multipleTesting)}, 90%+)`,
          `${formatCount(hot)} / ${formatCount(cold)}`
        ],
        ['Permutation-confirmed', `${formatCount(confirmed)} (${permutations} runs)`],
        ['Mean trip vertices per km²', statistics.f32[1].toFixed(1)],
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
        radius: radiusCells * cellMeters,
        weightKind: 'binary'
      })
    );
    encoder.markDirty();
  };
  const writeSignificance = () => {
    recipe.setSettings({significanceLevel, seed: 7, permutationCount: permutations});
    encoder.markDirty();
  };
  const parameters: RecipeParameter[] = [
    {
      kind: 'slider',
      label: 'Neighborhood radius',
      // Below 1.5 cells a band holds fewer than the 8 neighbors a row needs to be tested.
      minimum: 1.5,
      maximum: 3,
      step: 0.1,
      value: radiusCells,
      format: value =>
        `${value.toFixed(1)} cells (${Math.round(value * cellMeters)} m great-circle)`,
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
      'GPUCellCover (study-area cells, empty ones included)',
      'GPUCellAggregation (points per cell, joined onto the study cells)',
      'GPUCellGeometry (cell centers) + per-km² density',
      'GPUNeighborSearch (great-circle distance band in meters)',
      'GPUHotSpotAnalysis (Gi*)',
      'GPULocalPermutationTest (pseudo p)',
      'GPUClassBreaks + GPUColorScale'
    ],
    parameters,
    legend:
      'Cells colored by Gi* z-score class of trip-vertex density per km², cold blue to hot red; empty cells count as zero. Rows with fewer than 8 neighbors in the band are not tested. Dimmed cells fail the (unadjusted) permutation test; the hot/cold counts use the Benjamini-Hochberg false discovery rate correction.',
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
  const projection = trips.frame;
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
      before: {status: {count: beforeTableCount.view}},
      after: {status: {count: afterTableCount.view}}
    },
    outputs: {
      comparison: {
        cells: unionCells.view,
        delta: delta.view,
        status: {count: unionCount.view, overflow: unionOverflow.view}
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
  const compiled = await kit.compileAsync();

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
      'GPUCellTableCompare (outer join, delta, union mask)',
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
/** Time steps with trips: the real trips start in the first 30 minutes, then the last ones finish. */
const SPACE_TIME_SLICES = 30;
/**
 * The trips are one 30 minute session, which is too short for a trend. Each trip minute is
 * stretched to one calendar hour (a linear `hour` step; the recipe rejects the cyclic `minute`
 * field), so the cube holds 30 consecutive steps and Mann-Kendall has the 10 slices it needs.
 */
const SPACE_TIME_SECONDS_PER_STEP = 60;
/** Palette of the 18 emerging hot spot categories: none, 8 hot (yellow to red), 8 cold (blue), undetermined. */
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
  [8, 29, 88, 255],
  [150, 150, 150, 200]
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
  'historical cold',
  'undetermined (final slice empty or too few slices)'
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

  // Calendar time for the trips: each trip minute becomes one hour after a fixed 12:00 UTC start,
  // as epoch milliseconds (see SPACE_TIME_SECONDS_PER_STEP).
  const epochStart = Date.UTC(2023, 5, 1, 12, 0, 0);
  const timestampWords = new Uint32Array(pointCount * 2);
  for (let row = 0; row < pointCount; row++) {
    const milliseconds =
      epochStart + Math.round((trips.vertexTimestamps[row] / SPACE_TIME_SECONDS_PER_STEP) * 3.6e6);
    timestampWords[row * 2] = milliseconds % 2 ** 32;
    timestampWords[row * 2 + 1] = Math.floor(milliseconds / 2 ** 32);
  }

  let radiusCells = 1.5;
  let temporalWindow = 1;
  let confidenceIndex = 0;
  const getSettings = () => ({
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
  const emerging = kit.parameterBuffer(
    'emerging-parameters',
    'float32',
    GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
  );
  const colorScale = kit.parameter(
    'color-scale-parameters',
    'float32',
    getGPUColorScaleParameterValues({scale: 'ordinal', domainCount: 0, paletteCount: 18})
  );
  const palette = kit.input(
    'palette',
    Uint32Array.from(EMERGING_PALETTE, ([r, g, b, a]) => packGPUColor(r, g, b, a)),
    'uint32',
    18
  );
  const category = kit.output('category', 'uint32', cellCount);
  const hotSlices = kit.output('hot-slices', 'uint32', cellCount);
  const coldSlices = kit.output('cold-slices', 'uint32', cellCount);
  const colors = kit.output('colors', 'uint32', cellCount);
  const recipe = addSpaceTimeHotSpotsRecipe(kit.graph, {
    timestamps: timestamps.view,
    calendarParameters: calendar.view,
    slices: {
      step: 'hour',
      origin: {year: 2023, month: 6, day: 1, hour: 12},
      count: SPACE_TIME_SLICES
    },
    cells: {
      kind: 'lattice',
      spatialContext: trips.frame.spatialContext,
      positions: positions.view,
      width: SPACE_TIME_WIDTH,
      height: SPACE_TIME_HEIGHT,
      bounds
    },
    settings: getSettings(),
    parameters: emerging,
    outputs: {
      category: category.view,
      hotSliceCount: hotSlices.view,
      coldSliceCount: coldSlices.view,
      colors: colors.view
    },
    color: {palette: palette.view, parameters: colorScale.view}
  });
  const compiled = await kit.compileAsync();

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
      const histogram = new Uint32Array(18);
      let mostHotSlices = 0;
      for (let cell = 0; cell < cellCount; cell++) {
        histogram[Math.min(17, categories.u32[cell])]++;
        mostHotSlices = Math.max(mostHotSlices, hot.u32[cell]);
      }
      const hotCells = histogram.slice(1, 9).reduce((total, value) => total + value, 0);
      const coldCells = histogram.slice(9, 17).reduce((total, value) => total + value, 0);
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
        ['Undetermined', `${histogram[17]}`],
        ['Most hot slices in a cell', `${mostHotSlices} of ${SPACE_TIME_SLICES}`]
      ]);
    }
  );
  const encoder = new DirtyEncoder(compiled, reader);
  const writeParameters = () => {
    recipe.setSettings(getSettings());
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
      label: 'Temporal window (previous steps)',
      minimum: 0,
      maximum: 4,
      step: 1,
      value: temporalWindow,
      format: value => `${value} step${value === 1 ? '' : 's'}`,
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
      'GPUCalendarBuckets (calendar fields of each event)',
      'atomic count adapter (counts per cell and hourly step into the dense cube)',
      'GPUEmergingHotSpots (Gi* per bin, Benjamini-Hochberg, Mann-Kendall)',
      'GPUColorScale (ordinal categories)'
    ],
    parameters,
    legend:
      'Yellow to dark red: new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating, historical hot. Blues are the cold mirror; gray cells are undetermined (final step empty or fewer than 10 usable steps). Confidence uses the Benjamini-Hochberg false discovery rate correction.',
    dataNote: `${trips.attribution}; each trip minute is stretched to one hourly time step so the 30 minute session forms a 30-slice cube`,
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
