// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Recipe scenes over polygon zones: rate cluster map, points-in-polygons choropleth and change of
 * support. The zones are a procedural mesh of districts over the area the New York trips cover
 * (the explorer has no polygon dataset there, so the zones are labelled synthetic); every column
 * the recipes analyse is counted from the real trips and points of interest.
 */

import type {Layer} from '@deck.gl/core';
import {
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  packGPUColor
} from '@luma.gl/experimental/gpu-dataframe';
import {
  addChangeOfSupportRecipe,
  addPointsInPolygonsChoroplethRecipe,
  addRateClusterMapRecipe,
  getGPUPermutationParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPU_EMPIRICAL_BAYES_SUMMARY,
  GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import type {Buffer} from '@luma.gl/core';
import {getGPUPolygonRasterizationExtentValues} from '@luma.gl/experimental/gpu-raster';
import type {SpatialAnalysisPointsOfInterest} from '../spatial-analysis-data';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import {formatCount} from '../spatial-analysis-resources';
import {formatCompact, getRampPalette, PackedColorRasterLayer} from './classification-layers';
import {
  countPointsPerDistrict,
  createDistrictPolygons,
  DirtyEncoder,
  getCoreBounds,
  getTripEndpoints,
  rasterizeFeatureRows,
  RecipeKit,
  SummaryReader,
  sliceSummary,
  type DistrictPolygons,
  type RecipeParameter,
  type RecipeSceneBuilder
} from './recipes-kit';

const RASTER_WIDTH = 200;
const RASTER_HEIGHT = 240;
const YELLOW_ORANGE_RED = [
  [255, 255, 178],
  [254, 204, 92],
  [253, 141, 60],
  [240, 59, 32],
  [189, 0, 38]
] as const;

/** District meshes over the area the points of interest cover, plus the raster that fills them for drawing. */
type DistrictScene = {
  polygons: DistrictPolygons;
  bounds: [number, number, number, number];
  /** Per raster cell, row 0 at the south: the district row, or `featureCount` outside every district. */
  cellRows: Uint32Array;
};

function createDistrictScene(
  pois: SpatialAnalysisPointsOfInterest,
  columns: number,
  rows: number,
  jitter: number,
  seed: number
): DistrictScene {
  const bounds = getCoreBounds(pois.positions, 0.03);
  const polygons = createDistrictPolygons(bounds, columns, rows, jitter, seed);
  return {
    polygons,
    bounds,
    cellRows: rasterizeFeatureRows(polygons, bounds, RASTER_WIDTH, RASTER_HEIGHT)
  };
}

/** Uploads a district polygon set as recipe inputs. */
function importPolygons(kit: RecipeKit, polygons: DistrictPolygons, name: string) {
  return {
    polygonPositions: kit.input(
      `${name}-positions`,
      polygons.polygonPositions,
      'float32x2',
      polygons.polygonPositions.length / 2
    ).view,
    featureOffsets: kit.input(
      `${name}-features`,
      polygons.featureOffsets,
      'uint32',
      polygons.featureOffsets.length
    ).view,
    polygonOffsets: kit.input(
      `${name}-polygons`,
      polygons.polygonOffsets,
      'uint32',
      polygons.polygonOffsets.length
    ).view,
    ringOffsets: kit.input(
      `${name}-rings`,
      polygons.ringOffsets,
      'uint32',
      polygons.ringOffsets.length
    ).view
  };
}

/** Outline layer of a district polygon set. */
function createOutlineLayer(
  id: string,
  origin: readonly [number, number],
  segments: Buffer,
  count: number,
  color: readonly [number, number, number, number]
): Layer {
  return new SpatialAnalysisSegmentLayer({
    id,
    coordinateOrigin: [origin[0], origin[1], 0],
    segments,
    instanceCount: count,
    color,
    widthPixels: 1.2
  });
}

const RATE_PALETTE = [
  [160, 160, 170, 90],
  [215, 25, 28, 235],
  [171, 217, 233, 235],
  [44, 123, 182, 235],
  [253, 174, 97, 235]
] as const;

/**
 * `addRateClusterMapRecipe`: taxi pickups and drop-offs per district over points of interest per
 * district, smoothed by empirical Bayes, clustered by local Moran and gated by permutation. The
 * analysed rate (standardized or smoothed) is compile-time, so each is its own select entry.
 */
export function createRateClusterScene(analyze: 'standardized' | 'smoothed'): RecipeSceneBuilder {
  return async host => {
    const {context} = host;
    const [trips, pois] = await Promise.all([
      context.data.getNewYorkTrips(),
      context.data.getNewYorkPointsOfInterest()
    ]);
    context.signal.throwIfAborted();
    const district = createDistrictScene(pois, 12, 14, 0.3, 11);
    const {polygons} = district;
    const featureCount = polygons.featureCount;
    const events = countPointsPerDistrict(
      polygons,
      getTripEndpoints(trips.vertexPositions, trips.tripOffsets)
    );
    const populations = countPointsPerDistrict(polygons, pois.positions);

    let permutations = 99;
    let significanceLevel = 0.05;
    const kit = new RecipeKit(context.device, `recipe-rate-${analyze}`);
    const eventsInput = kit.input('events', events, 'float32', featureCount);
    const populationsInput = kit.input('populations', populations, 'float32', featureCount);
    const geometry = importPolygons(kit, polygons, 'districts');
    const autocorrelation = kit.parameter(
      'autocorrelation-parameters',
      'float32',
      getGPUSpatialAutocorrelationParameterValues({significanceLevel})
    );
    const permutationParameters = kit.parameter(
      'permutation-parameters',
      'uint32',
      getGPUPermutationParameterValues({seed: 5, permutations, significanceLevel})
    );
    const palette = kit.input(
      'palette',
      Uint32Array.from(RATE_PALETTE, ([r, g, b, a]) => packGPUColor(r, g, b, a)),
      'uint32',
      GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH
    );
    const quadrants = kit.output('quadrants', 'uint32', featureCount);
    // One spare row past the districts holds a transparent color for raster cells outside them.
    const colors = kit.output('colors', 'uint32', featureCount, 1);
    const summary = kit.output('summary', 'float32', GPU_EMPIRICAL_BAYES_SUMMARY.length);
    const weightsOverflow = kit.output('weights-overflow', 'uint32', 1);
    const zScores = kit.output('z-scores', 'float32', featureCount);
    const rates = kit.output('analysed-rates', 'float32', featureCount);
    const significant = kit.output('significant', 'uint32', featureCount);
    const neighborCapacity = featureCount * 10;
    const recipe = addRateClusterMapRecipe(kit.graph, {
      events: eventsInput.view,
      populations: populationsInput.view,
      positions: geometry.polygonPositions,
      ringOffsets: geometry.ringOffsets,
      polygonOffsets: geometry.polygonOffsets,
      snapTolerance: 1,
      neighborCapacity,
      analyze,
      parameters: autocorrelation.view,
      permutation: {
        parameters: permutationParameters.view,
        maximumPermutations: 199
      },
      palette: palette.view,
      outputs: {
        weightsOverflow: weightsOverflow.view,
        zScores: zScores.view,
        standardizedRates: analyze === 'standardized' ? rates.view : undefined,
        smoothedRates: analyze === 'smoothed' ? rates.view : undefined,
        quadrants: quadrants.view,
        colors: colors.view,
        permutation: {significant: significant.view}
      },
      scratch: {summary: summary.view}
    });
    const compiled = kit.compile();

    const cellRows = kit.resources.createBuffer('cell-rows', district.cellRows);
    const outline = kit.resources.createBuffer('outline', polygons.outlineSegments);
    const summarySizes = [featureCount * 4, GPU_EMPIRICAL_BAYES_SUMMARY.length * 4, 4];
    const reader = new SummaryReader(
      kit.resources,
      'rate',
      [
        {buffer: quadrants.buffer, size: summarySizes[0]},
        {buffer: summary.buffer, size: summarySizes[1]},
        {buffer: weightsOverflow.buffer, size: 4}
      ],
      bytes => {
        const [quadrantRows, moments, overflow] = sliceSummary(bytes, summarySizes);
        const counts = [0, 0, 0, 0, 0];
        for (let row = 0; row < featureCount; row++) counts[quadrantRows.u32[row] % 5]++;
        host.setOutputs([
          ['Districts analysed', `${formatCount(moments.f32[GPU_EMPIRICAL_BAYES_SUMMARY.count])}`],
          [
            'Pooled rate (pings per POI)',
            moments.f32[GPU_EMPIRICAL_BAYES_SUMMARY.pooledRate].toFixed(3)
          ],
          [
            'Prior variance',
            moments.f32[GPU_EMPIRICAL_BAYES_SUMMARY.priorVariance].toExponential(2)
          ],
          ['High-high / low-low', `${counts[1]} / ${counts[3]}`],
          ['Low-high / high-low', `${counts[2]} / ${counts[4]}`],
          ['Weights overflow', overflow.u32[0] ? 'YES' : 'no']
        ]);
      }
    );
    const encoder = new DirtyEncoder(compiled, reader);
    const writeSignificance = () => {
      autocorrelation.parameters.write(
        getGPUSpatialAutocorrelationParameterValues({significanceLevel})
      );
      permutationParameters.parameters.write(
        getGPUPermutationParameterValues({seed: 5, permutations, significanceLevel})
      );
      encoder.markDirty();
    };
    const parameters: RecipeParameter[] = [
      {
        kind: 'slider',
        label: 'Permutations',
        minimum: 19,
        maximum: 199,
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
      }
    ];
    return {
      compiled,
      contributorCount: recipe.contributors.length,
      chain: [
        'GPUEmpiricalBayesRates (events per population)',
        'GPUContiguityWeights (queen, shared vertices)',
        'GPUSpatialWeightsTransform (row-standardize)',
        'GPULocalMoran (LISA quadrants, ungated)',
        'GPULocalPermutationTest (gate)',
        'quadrant color adapter'
      ],
      parameters,
      legend:
        'Red high-high, blue low-low, light blue low-high, orange high-low; gray is not significant. Rate: taxi pickups and drop-offs per point of interest.',
      dataNote:
        `${trips.attribution}; ${pois.attribution}. Districts: synthetic ${polygons.columns} x ${polygons.rows} mesh ` +
        `(no polygon data for New York), ${analyze} empirical-Bayes rate`,
      encode: commandEncoder => encoder.encode(commandEncoder),
      getLayers: (): Layer[] => [
        new PackedColorRasterLayer({
          id: 'recipe-rate-fill',
          coordinateOrigin: [trips.origin[0], trips.origin[1], 0],
          gridSize: [RASTER_WIDTH, RASTER_HEIGHT],
          bounds: district.bounds,
          rowOrigin: 'south',
          values: colors.buffer,
          valueFormat: 'uint32',
          valueIndices: cellRows,
          opacity: 0.72
        }),
        createOutlineLayer(
          'recipe-rate-outline',
          trips.origin,
          outline,
          polygons.outlineSegments.length / 4,
          [235, 240, 250, 150]
        )
      ],
      destroy: () => {
        reader.stop();
        kit.resources.destroy();
      }
    };
  };
}

const CHOROPLETH_MAXIMUM_CLASSES = 7;
const CHOROPLETH_METHODS = ['equal-interval', 'quantile', 'natural-breaks'] as const;

/**
 * `addPointsInPolygonsChoroplethRecipe`: points to a per-district statistic to class breaks to
 * colors. The `zonal` variant counts points of interest per district with `GPUZonalStatistics`;
 * the `group` variant joins taxi-trip vertices with `GPUPointInPolygonJoin` and averages their
 * trip time per district with `GPUGroupStatistics`. Class count and method are per-frame.
 */
export function createChoroplethScene(backend: 'zonal' | 'group'): RecipeSceneBuilder {
  return async host => {
    const {context} = host;
    const [trips, pois] = await Promise.all([
      context.data.getNewYorkTrips(),
      context.data.getNewYorkPointsOfInterest()
    ]);
    context.signal.throwIfAborted();
    const district = createDistrictScene(pois, 10, 12, 0.3, 23);
    const {polygons} = district;
    const featureCount = polygons.featureCount;
    const points = backend === 'zonal' ? pois.positions : trips.vertexPositions;
    const pointCount = points.length / 2;

    let classCount = 5;
    let methodIndex = 2;
    const kit = new RecipeKit(context.device, `recipe-choropleth-${backend}`);
    const pointsInput = kit.input('points', points, 'float32x2', pointCount);
    const valuesInput =
      backend === 'group'
        ? kit.input('values', trips.vertexTimestamps, 'float32', pointCount)
        : null;
    const geometry = importPolygons(kit, polygons, 'districts');
    const getBreakParameters = () =>
      getGPUClassBreaksParameterValues(
        {method: CHOROPLETH_METHODS[methodIndex], classCount},
        CHOROPLETH_MAXIMUM_CLASSES
      );
    const getScaleParameters = () =>
      getGPUColorScaleParameterValues({
        scale: 'quantile',
        domainCount: classCount + 1,
        paletteCount: classCount,
        noDataColor: packGPUColor(0, 0, 0, 0)
      });
    const classBreaks = kit.parameter('class-breaks-parameters', 'float32', getBreakParameters());
    const colorScale = kit.parameter('color-scale-parameters', 'float32', getScaleParameters());
    const palette = kit.input(
      'palette',
      getRampPalette(YELLOW_ORANGE_RED, CHOROPLETH_MAXIMUM_CLASSES),
      'uint32',
      CHOROPLETH_MAXIMUM_CLASSES
    );
    const counts = kit.output('counts', 'uint32', featureCount);
    const overflow = kit.output('overflow', 'uint32', 1);
    const breaks = kit.output('breaks', 'float32', CHOROPLETH_MAXIMUM_CLASSES + 1);
    const classes = kit.output('class-count', 'uint32', 1);
    const colors = kit.output('colors', 'uint32', featureCount, 1);
    const meanValues =
      backend === 'group' ? kit.output('feature-values', 'float32', featureCount) : null;
    const recipe = addPointsInPolygonsChoroplethRecipe(kit.graph, {
      backend,
      points: pointsInput.view,
      values: valuesInput?.view,
      statistic: backend === 'group' ? 'mean' : 'count',
      polygons: {
        ...geometry,
        candidateCapacity: Math.max(4096, pointCount * 4)
      },
      outputs: {
        counts: counts.view,
        featureValues: meanValues?.view,
        overflow: overflow.view,
        color: {breaks: breaks.view, classCount: classes.view, colors: colors.view}
      },
      color: {
        classBreaksParameters: classBreaks.view,
        maximumClassCount: CHOROPLETH_MAXIMUM_CLASSES,
        methods: [...CHOROPLETH_METHODS],
        colorScaleParameters: colorScale.view,
        palette: palette.view,
        maximumPaletteCount: CHOROPLETH_MAXIMUM_CLASSES
      }
    });
    const compiled = kit.compile();
    const cellRows = kit.resources.createBuffer('cell-rows', district.cellRows);
    const outline = kit.resources.createBuffer('outline', polygons.outlineSegments);

    const summarySizes = [featureCount * 4, (CHOROPLETH_MAXIMUM_CLASSES + 1) * 4, 4, 4];
    const reader = new SummaryReader(
      kit.resources,
      'choropleth',
      [
        {buffer: counts.buffer, size: summarySizes[0]},
        {buffer: breaks.buffer, size: summarySizes[1]},
        {buffer: classes.buffer, size: 4},
        {buffer: overflow.buffer, size: 4}
      ],
      bytes => {
        const [countRows, breakRows, classRows, overflowFlag] = sliceSummary(bytes, summarySizes);
        let joined = 0;
        let occupied = 0;
        for (let row = 0; row < featureCount; row++) {
          joined += countRows.u32[row];
          if (countRows.u32[row] > 0) occupied++;
        }
        const usedClasses = classRows.u32[0];
        const edges = Array.from(breakRows.f32.subarray(0, usedClasses + 1), formatCompact);
        host.setOutputs([
          ['Points joined', `${formatCount(joined)} of ${formatCount(pointCount)}`],
          ['Districts with points', `${occupied} of ${featureCount}`],
          ['Class breaks', edges.join(' | ')],
          ['Method / classes', `${CHOROPLETH_METHODS[methodIndex]} / ${usedClasses}`],
          ['Join overflow', overflowFlag.u32[0] ? 'YES' : 'no']
        ]);
      }
    );
    const encoder = new DirtyEncoder(compiled, reader);
    const writeClassification = () => {
      classBreaks.parameters.write(getBreakParameters());
      colorScale.parameters.write(getScaleParameters());
      palette.buffer.write(getRampPalette(YELLOW_ORANGE_RED, classCount));
      encoder.markDirty();
    };
    const parameters: RecipeParameter[] = [
      {
        kind: 'slider',
        label: 'Classes',
        minimum: 2,
        maximum: CHOROPLETH_MAXIMUM_CLASSES,
        step: 1,
        value: classCount,
        format: value => `${value} classes`,
        onChange: value => {
          classCount = value;
          writeClassification();
        }
      },
      {
        kind: 'slider',
        label: 'Classification method',
        minimum: 0,
        maximum: CHOROPLETH_METHODS.length - 1,
        step: 1,
        value: methodIndex,
        format: value => CHOROPLETH_METHODS[value],
        onChange: value => {
          methodIndex = value;
          writeClassification();
        }
      }
    ];
    return {
      compiled,
      contributorCount: recipe.contributors.length,
      chain:
        backend === 'zonal'
          ? [
              'GPUZonalStatistics (points per district)',
              'GPUClassBreaks (equal, quantile, natural)',
              'GPUColorScale (class colors)'
            ]
          : [
              'GPUPointInPolygonJoin (vertex to district)',
              'GPUGroupStatistics (dense, mean per district)',
              'GPUClassBreaks (equal, quantile, natural)',
              'GPUColorScale (class colors)'
            ],
      parameters,
      legend:
        backend === 'zonal'
          ? 'Districts colored by points of interest per district, light to dark.'
          : 'Districts colored by the mean trip time (seconds) of the taxi-trip vertices inside, light to dark.',
      dataNote:
        `${backend === 'zonal' ? pois.attribution : trips.attribution}. Districts: synthetic ` +
        `${polygons.columns} x ${polygons.rows} mesh (no polygon data for New York)`,
      encode: commandEncoder => encoder.encode(commandEncoder),
      getLayers: (): Layer[] => [
        new PackedColorRasterLayer({
          id: `recipe-choropleth-${backend}-fill`,
          coordinateOrigin: [trips.origin[0], trips.origin[1], 0],
          gridSize: [RASTER_WIDTH, RASTER_HEIGHT],
          bounds: district.bounds,
          rowOrigin: 'south',
          values: colors.buffer,
          valueFormat: 'uint32',
          valueIndices: cellRows,
          opacity: 0.72
        }),
        createOutlineLayer(
          `recipe-choropleth-${backend}-outline`,
          trips.origin,
          outline,
          polygons.outlineSegments.length / 4,
          [235, 240, 250, 150]
        )
      ],
      destroy: () => {
        reader.stop();
        kit.resources.destroy();
      }
    };
  };
}

const SUPPORT_COLUMNS = 8;
const SUPPORT_ROWS = 10;
const TARGET_COLUMNS = 14;
const TARGET_ROWS = 17;
const SUPPORT_RASTER_WIDTH = 160;
const SUPPORT_RASTER_HEIGHT = 190;

/**
 * `addChangeOfSupportRecipe`: points of interest counted per irregular source district are moved
 * onto a regular target grid by overlap area. The dasymetric strength blends the area weights with
 * a point-of-interest density raster; both the weights raster and the choice of extensive or
 * intensive transfer are per-frame.
 */
export const buildChangeOfSupportScene: RecipeSceneBuilder = async host => {
  const {context} = host;
  const [trips, pois] = await Promise.all([
    context.data.getNewYorkTrips(),
    context.data.getNewYorkPointsOfInterest()
  ]);
  context.signal.throwIfAborted();
  const bounds = getCoreBounds(pois.positions, 0.03);
  const source = createDistrictPolygons(bounds, SUPPORT_COLUMNS, SUPPORT_ROWS, 0.45, 41);
  const target = createDistrictPolygons(bounds, TARGET_COLUMNS, TARGET_ROWS, 0, 0);
  const sourceCount = source.featureCount;
  const targetCount = target.featureCount;
  const sourceValues = countPointsPerDistrict(source, pois.positions);
  let sourceTotal = 0;
  for (const value of sourceValues) sourceTotal += value;

  // POI density raster for the dasymetric weights, normalized to a mean of 1.
  const cellCount = SUPPORT_RASTER_WIDTH * SUPPORT_RASTER_HEIGHT;
  const density = new Float32Array(cellCount);
  const cellWidth = (bounds[2] - bounds[0]) / SUPPORT_RASTER_WIDTH;
  const cellHeight = (bounds[3] - bounds[1]) / SUPPORT_RASTER_HEIGHT;
  for (let index = 0; index < pois.positions.length; index += 2) {
    const column = Math.floor((pois.positions[index] - bounds[0]) / cellWidth);
    const row = Math.floor((pois.positions[index + 1] - bounds[1]) / cellHeight);
    if (column >= 0 && column < SUPPORT_RASTER_WIDTH && row >= 0 && row < SUPPORT_RASTER_HEIGHT) {
      density[row * SUPPORT_RASTER_WIDTH + column]++;
    }
  }
  let densityTotal = 0;
  for (const value of density) densityTotal += value;
  const densityMean = densityTotal / cellCount || 1;
  const weights = new Float32Array(cellCount);
  let strength = 0;
  let intensive = false;
  const writeWeights = () => {
    for (let index = 0; index < cellCount; index++) {
      weights[index] = 1 - strength + strength * (0.05 + density[index] / densityMean);
    }
    cellWeights.buffer.write(weights);
  };

  const kit = new RecipeKit(context.device, 'recipe-change-of-support');
  const cellWeights = kit.input('cell-weights', weights, 'float32', cellCount);
  const extent = kit.input(
    'extent',
    getGPUPolygonRasterizationExtentValues(bounds[0], bounds[1], cellWidth, cellHeight),
    'float32',
    4
  );
  const sourceInput = kit.input('source-values', sourceValues, 'float32', sourceCount);
  const pairCapacity = 8192;
  const extensive = kit.output('extensive', 'float32', targetCount);
  const intensiveValues = kit.output('intensive', 'float32', targetCount);
  const overflow = kit.output('overflow', 'uint32', 1);
  const totalPairs = kit.output('total-pairs', 'uint32', 1);
  const recipe = addChangeOfSupportRecipe(kit.graph, {
    width: SUPPORT_RASTER_WIDTH,
    height: SUPPORT_RASTER_HEIGHT,
    extent: extent.view,
    source: {...importPolygons(kit, source, 'source'), crossingCapacity: 65536},
    target: {...importPolygons(kit, target, 'target'), crossingCapacity: 65536},
    cellWeights: cellWeights.view,
    pairCapacity,
    sourceValues: sourceInput.view,
    outputs: {
      extensiveValues: extensive.view,
      intensiveValues: intensiveValues.view,
      overflow: overflow.view,
      requiredCount: totalPairs.view
    }
  });
  writeWeights();
  const compiled = kit.compile();
  const targetOutline = kit.resources.createBuffer('target-outline', target.outlineSegments);
  const sourceOutline = kit.resources.createBuffer('source-outline', source.outlineSegments);

  let displayMaximum = Math.max(1, sourceTotal / targetCount) * 3;
  const summarySizes = [targetCount * 4, targetCount * 4, 4, 4];
  const reader = new SummaryReader(
    kit.resources,
    'support',
    [
      {buffer: extensive.buffer, size: summarySizes[0]},
      {buffer: intensiveValues.buffer, size: summarySizes[1]},
      {buffer: totalPairs.buffer, size: 4},
      {buffer: overflow.buffer, size: 4}
    ],
    bytes => {
      const [extensiveRows, intensiveRows, pairs, overflowFlag] = sliceSummary(bytes, summarySizes);
      let transferred = 0;
      let largest = 0;
      let largestIntensive = 0;
      for (let row = 0; row < targetCount; row++) {
        transferred += extensiveRows.f32[row];
        largest = Math.max(largest, extensiveRows.f32[row]);
        largestIntensive = Math.max(largestIntensive, intensiveRows.f32[row]);
      }
      const nextMaximum = Math.max(1, intensive ? largestIntensive : largest);
      if (Math.abs(nextMaximum - displayMaximum) > displayMaximum * 0.05) {
        displayMaximum = nextMaximum;
        host.updateLayers();
      }
      host.setOutputs([
        ['Source / target zones', `${sourceCount} / ${targetCount}`],
        ['Overlap pairs', `${formatCount(pairs.u32[0])} of ${formatCount(pairCapacity)}`],
        ['Source total', formatCount(sourceTotal)],
        [
          'Transferred (extensive)',
          `${transferred.toFixed(0)} (${((100 * transferred) / sourceTotal).toFixed(1)}%)`
        ],
        [
          'Largest target cell',
          `${largest.toFixed(1)} counts, ${largestIntensive.toFixed(1)} mean`
        ],
        ['Overflow', overflowFlag.u32[0] ? 'YES' : 'no']
      ]);
    }
  );
  const encoder = new DirtyEncoder(compiled, reader);
  const parameters: RecipeParameter[] = [
    {
      kind: 'slider',
      label: 'Dasymetric strength (area to POI density)',
      minimum: 0,
      maximum: 1,
      step: 0.05,
      value: strength,
      format: value => (value === 0 ? 'area only' : `${Math.round(value * 100)}% density`),
      onChange: value => {
        strength = value;
        writeWeights();
        encoder.markDirty();
      }
    },
    {
      kind: 'toggle',
      label: 'Show intensive transfer (area-weighted mean)',
      value: intensive,
      onChange: value => {
        intensive = value;
        host.updateLayers();
        encoder.markDirty();
      }
    }
  ];
  const origin = trips.origin;
  return {
    compiled,
    contributorCount: recipe.contributors.length,
    chain: [
      'GPUPolygonRasterization (source zones)',
      'GPUPolygonRasterization (target zones)',
      'GPUArealInterpolation (overlap areas, weights)',
      'GPUSpatialLag (extensive transfer)',
      'GPUSpatialLag (intensive transfer)'
    ],
    parameters,
    legend:
      'Target grid colored by points of interest moved from the orange source districts by overlap area (extensive: counts, intensive: area-weighted mean).',
    dataNote:
      `${pois.attribution}. Source: synthetic ${SUPPORT_COLUMNS} x ${SUPPORT_ROWS} irregular districts; ` +
      `target: synthetic ${TARGET_COLUMNS} x ${TARGET_ROWS} grid`,
    encode: commandEncoder => encoder.encode(commandEncoder),
    getLayers: (): Layer[] => [
      new SpatialAnalysisRasterLayer({
        id: 'recipe-support-target',
        coordinateOrigin: [origin[0], origin[1], 0],
        gridSize: [TARGET_COLUMNS, TARGET_ROWS],
        bounds,
        rowOrigin: 'south',
        values: intensive ? intensiveValues.buffer : extensive.buffer,
        valueFormat: 'float32',
        colormap: 'viridis',
        valueRange: [0, displayMaximum],
        discardAtOrBelow: 0,
        noDataColor: [0, 0, 0, 0],
        opacity: 0.8
      }),
      createOutlineLayer(
        'recipe-support-target-outline',
        origin,
        targetOutline,
        target.outlineSegments.length / 4,
        [210, 220, 240, 70]
      ),
      createOutlineLayer(
        'recipe-support-source-outline',
        origin,
        sourceOutline,
        source.outlineSegments.length / 4,
        [255, 150, 60, 230]
      )
    ],
    destroy: () => {
      reader.stop();
      kit.resources.destroy();
    }
  };
};
