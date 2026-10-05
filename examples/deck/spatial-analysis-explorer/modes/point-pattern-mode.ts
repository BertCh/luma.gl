// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Point-pattern analysis of New York points of interest by category. Two compiled graphs share the
 * POI positions:
 *
 * - A distribution graph: `GPUGeographicDistribution` per category (mean and median centres,
 *   standard distance circles, standard deviational ellipses), optionally weighted by nearby taxi
 *   activity. The ellipse and circle rings it writes are turned into segments on the GPU and drawn
 *   directly.
 * - A pattern graph over one selected category (or all): `GPURipley` (L(r) - r with edge
 *   correction), `GPUVariogram` (directional semivariogram of local taxi activity, CPU model fit on
 *   the readback), `GPUSpatialCorrelogram` (Moran's I by distance band) and `GPUPointPatternIndices`
 *   (Clark-Evans and quadrat variance-to-mean ratio).
 *
 * The analysis window, the maximum distance, the edge correction, the azimuth offset, the category
 * and the weighting are all buffer writes (the window can follow the map view): the rebuild counter
 * stays 0. Both graphs are encoded only when an input changed; the readbacks are one ring-buffered
 * summary of a few kilobytes that feeds the readouts and three small canvas charts.
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, GPUReadbackRing} from '@luma.gl/gpgpu/gpu-core';
import {
  fitVariogramModel,
  evaluateVariogramModel,
  getGPUPointPatternIndicesParameterValues,
  getGPURipleyParameterValues,
  getGPUSpatialCorrelogramParameterValues,
  getGPUVariogramParameterValues,
  GPU_CLARK_EVANS_LENGTH,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  GPU_QUADRAT_STATISTICS_LENGTH,
  GPU_RIPLEY_PARAMETER_LENGTH,
  GPU_SPATIAL_CORRELOGRAM_NO_BAND,
  GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
  GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPU_VARIOGRAM_STATISTICS_LENGTH,
  GPUPointPatternIndices,
  GPURipley,
  GPUSpatialCorrelogram,
  GPUVariogram,
  type GPURipleyEdgeCorrection,
  type GPUSpatialCorrelogramVarianceAssumption,
  type VariogramModel,
  type VariogramModelType
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUGeographicDistributionParameterValues,
  GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  GPUGeographicDistribution
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor
} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {
  formatCount,
  getViewportMetricBounds,
  SpatialAnalysisResources
} from '../spatial-analysis-resources';
import {addKernelPass} from './mode-kernels';
import {MiniChart} from './point-pattern-chart';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

/** Number of categories analyzed (the most frequent ones). */
const GROUP_COUNT = 6;
/** Neighbor-search lattice of the pair statistics (compile-time; speed only). */
const GRID_SIZE: readonly [number, number] = [64, 64];
const RIPLEY_RADII = 30;
const LAG_COUNT = 20;
const DIRECTION_COUNT = 4;
const BAND_COUNT = 20;
const QUADRAT_GRID: readonly [number, number] = [12, 12];
const RING_VERTICES = 64;
/** Cell size in meters used to turn nearby trip vertices into a POI activity value. */
const ACTIVITY_CELL_METERS = 150;

const HIDDEN: SpatialAnalysisColor = [0, 0, 0, 0];
const GROUP_COLORS: readonly (readonly [number, number, number])[] = [
  [78, 201, 255],
  [255, 148, 72],
  [189, 122, 255],
  [87, 235, 168],
  [255, 105, 168],
  [245, 220, 87]
];

// Float32 word offsets of the readback summary.
type SummaryLayoutKey =
  | 'rippleL'
  | 'semivariances'
  | 'variogramPairs'
  | 'variogramDistances'
  | 'variogramStatistics'
  | 'moransI'
  | 'zScores'
  | 'bandPairs'
  | 'peakBands'
  | 'correlogramStatistics'
  | 'clarkEvans'
  | 'quadrat'
  | 'counts'
  | 'meanCenters'
  | 'medianCenters'
  | 'standardDistances'
  | 'ellipses';

const SUMMARY_LAYOUT = (() => {
  const layout = {} as Record<SummaryLayoutKey | 'words', number>;
  let offset = 0;
  const add = (name: SummaryLayoutKey, words: number) => {
    layout[name] = offset;
    offset += words;
  };
  add('rippleL', RIPLEY_RADII);
  add('semivariances', LAG_COUNT * DIRECTION_COUNT);
  add('variogramPairs', LAG_COUNT * DIRECTION_COUNT);
  add('variogramDistances', LAG_COUNT * DIRECTION_COUNT);
  add('variogramStatistics', GPU_VARIOGRAM_STATISTICS_LENGTH);
  add('moransI', BAND_COUNT);
  add('zScores', BAND_COUNT);
  add('bandPairs', BAND_COUNT);
  add('peakBands', 2);
  add('correlogramStatistics', GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH);
  add('clarkEvans', GPU_CLARK_EVANS_LENGTH);
  add('quadrat', GPU_QUADRAT_STATISTICS_LENGTH);
  add('counts', GROUP_COUNT);
  add('meanCenters', GROUP_COUNT * 2);
  add('medianCenters', GROUP_COUNT * 2);
  add('standardDistances', GROUP_COUNT);
  add('ellipses', GROUP_COUNT * 3);
  layout.words = offset;
  return layout;
})();

type WindowChoice = 'extent' | 'view';
type CategoryChoice = 'all' | `${number}`;
type Bounds = [number, number, number, number];

/** Upper tail probability of a chi-square statistic by the Wilson-Hilferty approximation. */
function getChiSquareUpperTail(statistic: number, degreesOfFreedom: number): number {
  if (!(degreesOfFreedom > 0) || !Number.isFinite(statistic)) return NaN;
  const scale = 2 / (9 * degreesOfFreedom);
  const z = ((statistic / degreesOfFreedom) ** (1 / 3) - (1 - scale)) / Math.sqrt(scale);
  return 0.5 * erfc(z / Math.SQRT2);
}

function erfc(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const polynomial =
    t *
    (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const value = polynomial * Math.exp(-x * x);
  return x >= 0 ? value : 2 - value;
}

export const pointPatternMode: SpatialAnalysisModeDefinition = {
  id: 'point-pattern',
  title: 'Pattern',
  contributors: [
    'GPUVariogram',
    'GPUSpatialCorrelogram',
    'GPURipley',
    'GPUPointPatternIndices',
    'GPUGeographicDistribution'
  ],
  description:
    'Where do the points of interest of each category sit, and how clustered are they? Mean ' +
    'centres, standard distances and deviational ellipses per category; Ripley L(r) - r, ' +
    'Clark-Evans, quadrats, a semivariogram and a correlogram for the selected category, over a ' +
    'window that can follow the map.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const [pois, trips] = await Promise.all([
      context.data.getNewYorkPointsOfInterest(),
      context.data.getNewYorkTrips()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'point-pattern');
    const pointCount = pois.positions.length / 2;
    const projection = new LocalMetricProjection(pois.origin);

    // Top categories by count; every other category maps to a group id that is excluded.
    const categoryTotals = new Map<number, number>();
    for (const category of pois.categories) {
      categoryTotals.set(category, (categoryTotals.get(category) ?? 0) + 1);
    }
    const topCategories = [...categoryTotals.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, GROUP_COUNT)
      .map(entry => entry[0]);
    const groupOfCategory = new Map(topCategories.map((category, group) => [category, group]));
    const groupIds = new Uint32Array(pointCount);
    const groupTotals = new Array<number>(GROUP_COUNT).fill(0);
    for (let index = 0; index < pointCount; index++) {
      const group = groupOfCategory.get(pois.categories[index]) ?? GROUP_COUNT;
      groupIds[index] = group;
      if (group < GROUP_COUNT) groupTotals[group]++;
    }
    const groupNames = topCategories.map(category => pois.categoryNames[category]);

    // POI activity: trip vertices in the POI's cell (sqrt keeps the variogram well scaled).
    const activityCells = new Map<number, number>();
    const getCellKey = (x: number, y: number) =>
      (Math.floor(x / ACTIVITY_CELL_METERS) + 32768) * 65536 +
      (Math.floor(y / ACTIVITY_CELL_METERS) + 32768);
    for (let index = 0; index < trips.vertexPositions.length; index += 2) {
      const key = getCellKey(trips.vertexPositions[index], trips.vertexPositions[index + 1]);
      activityCells.set(key, (activityCells.get(key) ?? 0) + 1);
    }
    const activity = new Float32Array(pointCount);
    for (let index = 0; index < pointCount; index++) {
      activity[index] = Math.sqrt(
        activityCells.get(getCellKey(pois.positions[index * 2], pois.positions[index * 2 + 1])) ?? 0
      );
    }
    const unitWeights = new Float32Array(pointCount).fill(1);
    const activityWeights = Float32Array.from(activity, value => 1 + value);

    let minimumX = Infinity;
    let minimumY = Infinity;
    let maximumX = -Infinity;
    let maximumY = -Infinity;
    for (let index = 0; index < pointCount; index++) {
      minimumX = Math.min(minimumX, pois.positions[index * 2]);
      maximumX = Math.max(maximumX, pois.positions[index * 2]);
      minimumY = Math.min(minimumY, pois.positions[index * 2 + 1]);
      maximumY = Math.max(maximumY, pois.positions[index * 2 + 1]);
    }
    const extentBounds: Bounds = [minimumX - 5, minimumY - 5, maximumX + 5, maximumY + 5];

    // Controls state.
    let windowChoice: WindowChoice = 'extent';
    let maximumDistance = 800;
    let edgeCorrection: GPURipleyEdgeCorrection = 'isotropic';
    let varianceAssumption: GPUSpatialCorrelogramVarianceAssumption = 'randomization';
    let azimuthDegrees = 60;
    let directionChoice = 'all';
    let modelType: VariogramModelType = 'spherical';
    let categoryChoice: CategoryChoice = 'all';
    let standardDeviations = 1;
    let ellipseConvention: 'arcgis' | 'standard' = 'arcgis';
    let weighted = false;
    let inspectedGroup = 0;
    const visibleGroups = new Array<boolean>(GROUP_COUNT).fill(true);
    let showEllipses = true;
    let showCircles = false;
    let showMedian = false;
    let showPoints = true;
    let currentBounds: Bounds = extentBounds;
    let patternDirty = true;
    let distributionDirty = true;
    let needsReadback = true;
    let readbackPending = false;
    let destroyed = false;
    let lastSummary: Float32Array | null = null;
    let lastSummaryWords: Uint32Array | null = null;

    // Shared inputs.
    const positionsBuffer = resources.createBuffer('positions', pois.positions);
    const valuesBuffer = resources.createBuffer('activity', activity);
    const groupIdsBuffer = resources.createBuffer('group-ids', groupIds);
    const weightsBuffer = resources.createBuffer('weights', unitWeights);
    const maskBuffer = resources.createBuffer('mask', new Uint32Array(pointCount).fill(1));
    const groupIndexBuffer = resources.createBuffer(
      'group-index',
      Uint32Array.from({length: GROUP_COUNT}, (_, group) => group)
    );
    const groupVisibleBuffer = resources.createBuffer(
      'group-visible',
      new Uint32Array(GROUP_COUNT).fill(1)
    );

    const parameterBuffers = {
      ripley: resources.createParameterBuffer('ripley', 'float32', GPU_RIPLEY_PARAMETER_LENGTH),
      variogram: resources.createParameterBuffer(
        'variogram',
        'float32',
        GPU_VARIOGRAM_PARAMETER_LENGTH
      ),
      correlogram: resources.createParameterBuffer(
        'correlogram',
        'float32',
        GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH
      ),
      indices: resources.createParameterBuffer(
        'indices',
        'float32',
        GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH
      ),
      distribution: resources.createParameterBuffer(
        'distribution',
        'float32',
        GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH
      )
    };

    // Pattern outputs.
    const outputs = {
      lMinusR: resources.createBuffer('l-minus-r', RIPLEY_RADII * 4),
      semivariances: resources.createBuffer('semivariances', LAG_COUNT * DIRECTION_COUNT * 4),
      variogramPairs: resources.createBuffer('variogram-pairs', LAG_COUNT * DIRECTION_COUNT * 4),
      variogramDistances: resources.createBuffer(
        'variogram-distances',
        LAG_COUNT * DIRECTION_COUNT * 4
      ),
      variogramStatistics: resources.createBuffer(
        'variogram-statistics',
        GPU_VARIOGRAM_STATISTICS_LENGTH * 4
      ),
      moransI: resources.createBuffer('morans-i', BAND_COUNT * 4),
      zScores: resources.createBuffer('z-scores', BAND_COUNT * 4),
      bandPairs: resources.createBuffer('band-pairs', BAND_COUNT * 4),
      peakBands: resources.createBuffer('peak-bands', 2 * 4),
      correlogramStatistics: resources.createBuffer(
        'correlogram-statistics',
        GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH * 4
      ),
      clarkEvans: resources.createBuffer('clark-evans', GPU_CLARK_EVANS_LENGTH * 4),
      quadrat: resources.createBuffer('quadrat', GPU_QUADRAT_STATISTICS_LENGTH * 4),
      counts: resources.createBuffer('counts', GROUP_COUNT * 4),
      meanCenters: resources.createBuffer('mean-centers', GROUP_COUNT * 8),
      medianCenters: resources.createBuffer('median-centers', GROUP_COUNT * 8),
      standardDistances: resources.createBuffer('standard-distances', GROUP_COUNT * 4),
      ellipses: resources.createBuffer('ellipses', GROUP_COUNT * 12),
      ellipseVertices: resources.createBuffer('ellipse-vertices', GROUP_COUNT * RING_VERTICES * 8),
      circleVertices: resources.createBuffer('circle-vertices', GROUP_COUNT * RING_VERTICES * 8),
      ellipseSegments: resources.createBuffer('ellipse-segments', GROUP_COUNT * RING_VERTICES * 16),
      circleSegments: resources.createBuffer('circle-segments', GROUP_COUNT * RING_VERTICES * 16)
    };
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {
        id: 'point-pattern-summary',
        byteLength: SUMMARY_LAYOUT['words'] * 4
      })
    );

    // The pattern graph: four contributors over the same points, mask and bounds.
    const patternGraph = new GPUCommandGraph<void>(device, {id: 'pattern'});
    {
      const positions = importGraphBuffer(
        patternGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        pointCount
      );
      const values = importGraphBuffer(patternGraph, 'values', valuesBuffer, 'float32', pointCount);
      const mask = importGraphBuffer(patternGraph, 'mask', maskBuffer, 'uint32', pointCount);
      const view = <Format extends 'float32' | 'uint32'>(
        name: keyof typeof outputs,
        format: Format,
        length: number
      ) => importGraphBuffer(patternGraph, name, outputs[name], format, length);
      patternGraph.add(
        new GPURipley({
          id: 'ripley',
          positions,
          mask,
          parameters: parameterBuffers.ripley.importToGraph(patternGraph),
          gridSize: GRID_SIZE,
          radiusCount: RIPLEY_RADII,
          k: importGraphBuffer(
            patternGraph,
            'ripley-k',
            resources.createBuffer('ripley-k', RIPLEY_RADII * 4),
            'float32',
            RIPLEY_RADII
          ),
          lMinusR: view('lMinusR', 'float32', RIPLEY_RADII)
        })
      );
      patternGraph.add(
        new GPUVariogram({
          id: 'variogram',
          positions,
          values,
          mask,
          parameters: parameterBuffers.variogram.importToGraph(patternGraph),
          gridSize: GRID_SIZE,
          lagCount: LAG_COUNT,
          directionCount: DIRECTION_COUNT,
          semivariances: view('semivariances', 'float32', LAG_COUNT * DIRECTION_COUNT),
          pairCounts: view('variogramPairs', 'uint32', LAG_COUNT * DIRECTION_COUNT),
          meanDistances: view('variogramDistances', 'float32', LAG_COUNT * DIRECTION_COUNT),
          statistics: view('variogramStatistics', 'float32', GPU_VARIOGRAM_STATISTICS_LENGTH)
        })
      );
      patternGraph.add(
        new GPUSpatialCorrelogram({
          id: 'correlogram',
          positions,
          values,
          mask,
          parameters: parameterBuffers.correlogram.importToGraph(patternGraph),
          gridSize: GRID_SIZE,
          bandCount: BAND_COUNT,
          bandMode: 'cumulative',
          moransI: view('moransI', 'float32', BAND_COUNT),
          zScores: view('zScores', 'float32', BAND_COUNT),
          pairCounts: view('bandPairs', 'uint32', BAND_COUNT),
          peakBands: view('peakBands', 'uint32', 2),
          statistics: view(
            'correlogramStatistics',
            'float32',
            GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH
          )
        })
      );
      patternGraph.add(
        new GPUPointPatternIndices({
          id: 'indices',
          positions,
          mask,
          parameters: parameterBuffers.indices.importToGraph(patternGraph),
          gridSize: GRID_SIZE,
          quadratGrid: QUADRAT_GRID,
          clarkEvans: view('clarkEvans', 'float32', GPU_CLARK_EVANS_LENGTH),
          quadratStatistics: view('quadrat', 'float32', GPU_QUADRAT_STATISTICS_LENGTH),
          quadratCounts: importGraphBuffer(
            patternGraph,
            'quadrat-counts',
            resources.createBuffer('quadrat-counts', QUADRAT_GRID[0] * QUADRAT_GRID[1] * 4),
            'uint32',
            QUADRAT_GRID[0] * QUADRAT_GRID[1]
          )
        })
      );
    }
    const patternCompiled = resources.track(patternGraph.compile());

    // The distribution graph plus the passes that turn its rings into drawable segments.
    const distributionGraph = new GPUCommandGraph<void>(device, {id: 'distribution'});
    {
      const positions = importGraphBuffer(
        distributionGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        pointCount
      );
      const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
        name: keyof typeof outputs,
        format: Format,
        length: number
      ) => importGraphBuffer(distributionGraph, name, outputs[name], format, length);
      const ellipseVertices = view('ellipseVertices', 'float32x2', GROUP_COUNT * RING_VERTICES);
      const circleVertices = view('circleVertices', 'float32x2', GROUP_COUNT * RING_VERTICES);
      distributionGraph.add(
        new GPUGeographicDistribution({
          id: 'distribution',
          positions,
          weights: importGraphBuffer(
            distributionGraph,
            'weights',
            weightsBuffer,
            'float32',
            pointCount
          ),
          groupIds: importGraphBuffer(
            distributionGraph,
            'group-ids',
            groupIdsBuffer,
            'uint32',
            pointCount
          ),
          groupCount: GROUP_COUNT,
          polygonVertexCount: RING_VERTICES,
          parameters: parameterBuffers.distribution.importToGraph(distributionGraph),
          output: {
            counts: view('counts', 'uint32', GROUP_COUNT),
            meanCenters: view('meanCenters', 'float32x2', GROUP_COUNT),
            medianCenters: view('medianCenters', 'float32x2', GROUP_COUNT),
            standardDistances: view('standardDistances', 'float32', GROUP_COUNT),
            ellipses: view('ellipses', 'float32', GROUP_COUNT * 3),
            ellipseVertices,
            circleVertices
          }
        })
      );
      for (const [name, vertices, segments] of [
        ['ellipse', ellipseVertices, outputs.ellipseSegments],
        ['circle', circleVertices, outputs.circleSegments]
      ] as const) {
        addKernelPass(distributionGraph, {
          id: `${name}-segments`,
          invocationCount: GROUP_COUNT * RING_VERTICES,
          bindings: [
            {name: 'vertices', view: vertices, type: 'f32', access: 'read'},
            {
              name: 'segments',
              view: importGraphBuffer(
                distributionGraph,
                `${name}-segments`,
                segments,
                'float32',
                GROUP_COUNT * RING_VERTICES * 4
              ),
              type: 'f32',
              access: 'read_write'
            }
          ],
          declarations: `const RING: u32 = ${RING_VERTICES}u;`,
          body: /* wgsl */ `
  let ring = index / RING;
  let vertex = index % RING;
  let next = ring * RING + (vertex + 1u) % RING;
  let current = index;
  segments[segmentsOffset + index * 4u] = vertices[verticesOffset + current * 2u];
  segments[segmentsOffset + index * 4u + 1u] = vertices[verticesOffset + current * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = vertices[verticesOffset + next * 2u];
  segments[segmentsOffset + index * 4u + 3u] = vertices[verticesOffset + next * 2u + 1u];`
        });
      }
    }
    const distributionCompiled = resources.track(distributionGraph.compile());

    const writePatternParameters = () => {
      const bounds = currentBounds;
      patternBoundsGuard(bounds);
      parameterBuffers.ripley.write(
        getGPURipleyParameterValues({bounds, maximumDistance, edgeCorrection})
      );
      parameterBuffers.variogram.write(
        getGPUVariogramParameterValues({
          bounds,
          maximumDistance,
          azimuthOffset: (azimuthDegrees * Math.PI) / 180
        })
      );
      parameterBuffers.correlogram.write(
        getGPUSpatialCorrelogramParameterValues({bounds, maximumDistance, varianceAssumption})
      );
      parameterBuffers.indices.write(
        getGPUPointPatternIndicesParameterValues({bounds, maximumDistance})
      );
      patternDirty = true;
      needsReadback = true;
    };
    const patternBoundsGuard = (bounds: Bounds) => {
      if (!(bounds[2] > bounds[0] && bounds[3] > bounds[1])) {
        throw new Error('Pattern window has no area');
      }
    };
    const writeDistributionParameters = () => {
      parameterBuffers.distribution.write(
        getGPUGeographicDistributionParameterValues({standardDeviations, ellipseConvention})
      );
      distributionDirty = true;
      needsReadback = true;
    };
    const writeMask = () => {
      const mask = new Uint32Array(pointCount);
      const selected = categoryChoice === 'all' ? -1 : Number(categoryChoice);
      for (let index = 0; index < pointCount; index++) {
        mask[index] = selected < 0 || groupIds[index] === selected ? 1 : 0;
      }
      maskBuffer.write(mask);
      patternDirty = true;
      needsReadback = true;
    };

    // ---- Controls ----
    context.controls.addSelect<CategoryChoice>({
      label: 'Pattern analysis: category (mask buffer write)',
      options: [
        {value: 'all', label: `All points of interest (${formatCount(pointCount)})`},
        ...groupNames.map((name, group) => ({
          value: `${group}` as CategoryChoice,
          label: `${name} (${formatCount(groupTotals[group])})`
        }))
      ],
      value: categoryChoice,
      onChange: value => {
        categoryChoice = value;
        writeMask();
      }
    });
    context.controls.addSelect<WindowChoice>({
      label: 'Analysis window (per-frame bounds)',
      options: [
        {value: 'extent', label: 'Data extent'},
        {value: 'view', label: 'Current map view (follows pan and zoom)'}
      ],
      value: windowChoice,
      onChange: value => {
        windowChoice = value;
        if (value === 'extent') {
          currentBounds = extentBounds;
          writePatternParameters();
        }
      }
    });
    context.controls.addSlider({
      label: 'Maximum distance (per-frame parameter)',
      min: 100,
      max: 3000,
      step: 100,
      value: maximumDistance,
      format: value => `${value} m`,
      onChange: value => {
        maximumDistance = value;
        writePatternParameters();
      }
    });
    context.controls.addSelect<GPURipleyEdgeCorrection>({
      label: 'Ripley edge correction (per-frame parameter)',
      options: [
        {value: 'isotropic', label: 'Isotropic (Ripley 1977)'},
        {value: 'border', label: 'Border (reduced sample)'},
        {value: 'none', label: 'None'}
      ],
      value: edgeCorrection,
      onChange: value => {
        edgeCorrection = value;
        writePatternParameters();
      }
    });
    context.controls.addSelect<GPUSpatialCorrelogramVarianceAssumption>({
      label: 'Correlogram variance (per-frame parameter)',
      options: [
        {value: 'randomization', label: 'Randomization'},
        {value: 'normality', label: 'Normality'}
      ],
      value: varianceAssumption,
      onChange: value => {
        varianceAssumption = value;
        writePatternParameters();
      }
    });
    context.controls.addSlider({
      label: 'Variogram azimuth offset (per-frame parameter)',
      min: 0,
      max: 165,
      step: 15,
      value: azimuthDegrees,
      format: value => `${value}°`,
      onChange: value => {
        azimuthDegrees = value;
        writePatternParameters();
      }
    });
    context.controls.addSelect<string>({
      label: 'Variogram direction (sector of 45°, read back)',
      options: [
        {value: 'all', label: 'All directions'},
        ...Array.from({length: DIRECTION_COUNT}, (_, sector) => ({
          value: `${sector}`,
          label: `Sector ${sector + 1}`
        }))
      ],
      value: directionChoice,
      onChange: value => {
        directionChoice = value;
        redrawCharts();
      }
    });
    context.controls.addSelect<VariogramModelType>({
      label: 'Variogram model (CPU weighted fit of the readback)',
      options: [
        {value: 'spherical', label: 'Spherical'},
        {value: 'exponential', label: 'Exponential'},
        {value: 'gaussian', label: 'Gaussian'}
      ],
      value: modelType,
      onChange: value => {
        modelType = value;
        redrawCharts();
      }
    });
    context.controls.addSlider({
      label: 'Distribution: standard deviations (per-frame parameter)',
      min: 1,
      max: 3,
      step: 1,
      value: standardDeviations,
      format: value => `${value}`,
      onChange: value => {
        standardDeviations = value;
        writeDistributionParameters();
      }
    });
    context.controls.addSelect<'arcgis' | 'standard'>({
      label: 'Ellipse convention (per-frame parameter)',
      options: [
        {value: 'arcgis', label: 'ArcGIS (axes scaled by sqrt 2)'},
        {value: 'standard', label: 'Standard deviation'}
      ],
      value: ellipseConvention,
      onChange: value => {
        ellipseConvention = value;
        writeDistributionParameters();
      }
    });
    context.controls.addToggle({
      label: 'Weight by nearby taxi activity (weights buffer write)',
      value: weighted,
      onChange: value => {
        weighted = value;
        weightsBuffer.write(value ? activityWeights : unitWeights);
        distributionDirty = true;
        needsReadback = true;
      }
    });
    groupNames.forEach((name, group) => {
      context.controls.addToggle({
        label: `Show ${name}`,
        value: true,
        onChange: value => {
          visibleGroups[group] = value;
          groupVisibleBuffer.write(Uint32Array.from(visibleGroups, flag => (flag ? 1 : 0)));
          context.updateLayers();
        }
      });
    });
    context.controls.addToggle({
      label: 'Ellipses',
      value: showEllipses,
      onChange: value => {
        showEllipses = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Standard distance circles',
      value: showCircles,
      onChange: value => {
        showCircles = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Median centres (Weiszfeld)',
      value: showMedian,
      onChange: value => {
        showMedian = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Points of interest',
      value: showPoints,
      onChange: value => {
        showPoints = value;
        context.updateLayers();
      }
    });
    context.controls.addSelect<string>({
      label: 'Inspect distribution of',
      options: groupNames.map((name, group) => ({value: `${group}`, label: name})),
      value: '0',
      onChange: value => {
        inspectedGroup = Number(value);
        updateReadouts();
      }
    });
    context.controls.addLegend({
      title: 'Category',
      entries: groupNames.map((name, group) => ({
        color: [...GROUP_COLORS[group], 255],
        label: `${name} (${formatCount(groupTotals[group])})`
      }))
    });
    context.controls.addNote(
      'Discs are mean centres; circles are standard distances, ellipses the standard deviational ' +
        'ellipse. Charts: Ripley L(r) - r (above 0 is clustered), the empirical semivariogram ' +
        '(points sized by pair count) with its fitted model and the variance as sill, and ' +
        "Moran's I per distance band (filled where |z| > 1.96)."
    );

    const rippleChart = new MiniChart('Ripley L(r) - r', 'm');
    const variogramChart = new MiniChart('Semivariogram of POI taxi activity', 'm');
    const correlogramChart = new MiniChart("Correlogram: Moran's I by band", 'm');
    const readoutAnchor = document.querySelector('[data-mode-readouts]');
    for (const chart of [rippleChart, variogramChart, correlogramChart]) {
      chart.insertBefore(readoutAnchor);
    }

    context.controls.addReadout('Points of interest', formatCount(pointCount));
    const windowReadout = context.controls.addReadout('Window (area)');
    const selectedReadout = context.controls.addReadout('Included in the pattern');
    const clarkEvansReadout = context.controls.addReadout('Clark-Evans R (z)');
    const nearestReadout = context.controls.addReadout('Mean NN distance (expected)');
    const quadratReadout = context.controls.addReadout('Quadrats VMR (12 x 12)');
    const quadratTestReadout = context.controls.addReadout('Quadrat chi-square (df), p');
    const rippleReadout = context.controls.addReadout('Ripley max L - r (at r)');
    const modelReadout = context.controls.addReadout('Variogram nugget / sill / range');
    const peakReadout = context.controls.addReadout('Correlogram first peak / max z band');
    const countReadout = context.controls.addReadout('Distribution: count');
    const meanReadout = context.controls.addReadout('Mean centre (x, y from origin)');
    const medianReadout = context.controls.addReadout('Median centre offset');
    const distanceReadout = context.controls.addReadout('Standard distance');
    const ellipseReadout = context.controls.addReadout('Ellipse sigma long / short');
    const bearingReadout = context.controls.addReadout('Ellipse long axis bearing');
    context.controls.addReadout('Data', `${pois.attribution}; ${trips.attribution}`);
    const patternTimingReadout = context.controls.addReadout('Pattern graph (GPU)', '...');
    const distributionTimingReadout = context.controls.addReadout(
      'Distribution graph (GPU)',
      '...'
    );
    context.controls.addButton({
      label: 'Measure the graphs',
      onClick: () => {
        void (async () => {
          const options = {
            parameters: undefined,
            completionBuffer: outputs.peakBands,
            signal: context.signal
          };
          try {
            const pattern = await measureCompiledGraph(device, patternCompiled, options);
            if (destroyed) return;
            patternTimingReadout.setValue(
              `${patternCompiled.stats.nodeOrder.length} nodes, ${formatCompiledGraphTiming(pattern)}`
            );
            const distribution = await measureCompiledGraph(device, distributionCompiled, {
              ...options,
              runs: 1,
              warmUpRuns: 0
            });
            if (destroyed) return;
            distributionTimingReadout.setValue(
              `${distributionCompiled.stats.nodeOrder.length} nodes, ${formatCompiledGraphTiming(distribution)} (1 run)`
            );
          } catch (error) {
            if (!destroyed) patternTimingReadout.setValue(`failed: ${String(error)}`);
          }
        })();
      }
    });

    writePatternParameters();
    writeDistributionParameters();
    writeMask();

    // ---- Readout and chart updates from the last summary ----
    const formatNumber = (value: number, digits = 3) =>
      Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
    const formatP = (value: number) =>
      !Number.isFinite(value) ? 'n/a' : value < 0.001 ? '< 0.001' : value.toFixed(3);

    function getSlice(name: SummaryLayoutKey, length: number): Float32Array {
      return lastSummary!.subarray(SUMMARY_LAYOUT[name], SUMMARY_LAYOUT[name] + length);
    }

    function redrawCharts() {
      if (!lastSummary || !lastSummaryWords) return;
      const bounds = currentBounds;
      void bounds;
      // Ripley.
      const rippleValues = getSlice('rippleL', RIPLEY_RADII);
      const radii = Array.from(
        {length: RIPLEY_RADII},
        (_, index) => (maximumDistance * (index + 1)) / RIPLEY_RADII
      );
      rippleChart.update({
        series: [{kind: 'line', x: radii, y: rippleValues, color: '#4ec9ff'}],
        referenceLines: [{y: 0, color: '#7f90ad', dashed: true}],
        xRange: [0, maximumDistance]
      });

      // Variogram: aggregate sectors for 'all', otherwise one sector.
      const semivariances = getSlice('semivariances', LAG_COUNT * DIRECTION_COUNT);
      const pairs = lastSummaryWords.subarray(
        SUMMARY_LAYOUT['variogramPairs'],
        SUMMARY_LAYOUT['variogramPairs'] + LAG_COUNT * DIRECTION_COUNT
      );
      const distances = getSlice('variogramDistances', LAG_COUNT * DIRECTION_COUNT);
      const lagDistances: number[] = [];
      const lagSemivariances: number[] = [];
      const lagPairs: number[] = [];
      for (let lag = 0; lag < LAG_COUNT; lag++) {
        let pairTotal = 0;
        let distanceSum = 0;
        let semivarianceSum = 0;
        for (let sector = 0; sector < DIRECTION_COUNT; sector++) {
          if (directionChoice !== 'all' && Number(directionChoice) !== sector) continue;
          const bin = sector * LAG_COUNT + lag;
          const count = pairs[bin];
          if (count === 0 || !Number.isFinite(semivariances[bin])) continue;
          pairTotal += count;
          distanceSum += count * distances[bin];
          semivarianceSum += count * semivariances[bin];
        }
        if (pairTotal > 0) {
          lagDistances.push(distanceSum / pairTotal);
          lagSemivariances.push(semivarianceSum / pairTotal);
          lagPairs.push(pairTotal);
        }
      }
      const variance = lastSummary[SUMMARY_LAYOUT['variogramStatistics'] + 2];
      let model: VariogramModel | null = null;
      try {
        model = fitVariogramModel(
          {distances: lagDistances, semivariances: lagSemivariances, pairCounts: lagPairs},
          {model: modelType}
        );
      } catch {
        model = null;
      }
      const largestPairCount = Math.max(1, ...lagPairs);
      const curveX = Array.from({length: 40}, (_, index) => (maximumDistance * index) / 39);
      variogramChart.update({
        series: [
          {
            kind: 'points',
            x: lagDistances,
            y: lagSemivariances,
            color: '#ff9448',
            radius: lagPairs.map(count => 1.5 + 3 * Math.sqrt(count / largestPairCount))
          },
          ...(model
            ? [
                {
                  kind: 'line' as const,
                  x: curveX,
                  y: curveX.map(distance => evaluateVariogramModel(model, distance)),
                  color: '#57eba8'
                }
              ]
            : [])
        ],
        referenceLines: [{y: variance, color: '#7f90ad', dashed: true}],
        xRange: [0, maximumDistance]
      });
      modelReadout.setValue(
        model
          ? `${formatNumber(model.nugget, 2)} / ${formatNumber(model.nugget + model.sill, 2)} / ${formatNumber(model.range, 0)} m`
          : 'n/a (too few bins)'
      );

      // Correlogram.
      const moransI = getSlice('moransI', BAND_COUNT);
      const zScores = getSlice('zScores', BAND_COUNT);
      const bandUpper = Array.from(
        {length: BAND_COUNT},
        (_, band) => (maximumDistance * (band + 1)) / BAND_COUNT
      );
      const statistics = getSlice(
        'correlogramStatistics',
        GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH
      );
      const expected = -1 / (statistics[0] - 1);
      correlogramChart.update({
        series: [
          {kind: 'line', x: bandUpper, y: moransI, color: '#bd7aff'},
          {
            kind: 'points',
            x: bandUpper,
            y: moransI,
            color: '#bd7aff',
            radius: 2.8,
            filled: index => Math.abs(zScores[index]) > 1.96
          }
        ],
        referenceLines: [{y: expected, color: '#7f90ad', dashed: true}],
        xRange: [0, maximumDistance]
      });
    }

    function updateReadouts() {
      if (!lastSummary || !lastSummaryWords) return;
      const area = (currentBounds[2] - currentBounds[0]) * (currentBounds[3] - currentBounds[1]);
      windowReadout.setValue(
        `${((currentBounds[2] - currentBounds[0]) / 1000).toFixed(1)} x ${(
          (currentBounds[3] - currentBounds[1]) / 1000
        ).toFixed(1)} km (${(area / 1e6).toFixed(1)} km²)`
      );
      const clark = getSlice('clarkEvans', GPU_CLARK_EVANS_LENGTH);
      selectedReadout.setValue(formatCount(clark[0]));
      clarkEvansReadout.setValue(`${formatNumber(clark[3])} (z ${formatNumber(clark[5], 1)})`);
      nearestReadout.setValue(`${formatNumber(clark[1], 1)} m (${formatNumber(clark[2], 1)} m)`);
      const quadrat = getSlice('quadrat', GPU_QUADRAT_STATISTICS_LENGTH);
      quadratReadout.setValue(
        `${formatNumber(quadrat[3], 2)} (mean ${formatNumber(quadrat[1], 1)}, var ${formatNumber(quadrat[2], 1)})`
      );
      quadratTestReadout.setValue(
        `${formatNumber(quadrat[4], 0)} (${formatNumber(quadrat[5], 0)}), p ${formatP(
          getChiSquareUpperTail(quadrat[4], quadrat[5])
        )}`
      );
      const rippleValues = getSlice('rippleL', RIPLEY_RADII);
      let bestIndex = -1;
      for (let index = 0; index < rippleValues.length; index++) {
        if (
          Number.isFinite(rippleValues[index]) &&
          (bestIndex < 0 || rippleValues[index] > rippleValues[bestIndex])
        ) {
          bestIndex = index;
        }
      }
      rippleReadout.setValue(
        bestIndex < 0
          ? 'n/a'
          : `${formatNumber(rippleValues[bestIndex], 1)} m at ${formatNumber(
              (maximumDistance * (bestIndex + 1)) / RIPLEY_RADII,
              0
            )} m`
      );
      const peakWords = lastSummaryWords.subarray(
        SUMMARY_LAYOUT['peakBands'],
        SUMMARY_LAYOUT['peakBands'] + 2
      );
      const describeBand = (band: number) =>
        band === GPU_SPATIAL_CORRELOGRAM_NO_BAND
          ? 'none'
          : `${formatNumber((maximumDistance * (band + 1)) / BAND_COUNT, 0)} m`;
      peakReadout.setValue(`${describeBand(peakWords[0])} / ${describeBand(peakWords[1])}`);

      const group = inspectedGroup;
      const counts = lastSummaryWords.subarray(
        SUMMARY_LAYOUT['counts'],
        SUMMARY_LAYOUT['counts'] + GROUP_COUNT
      );
      const meanCenters = getSlice('meanCenters', GROUP_COUNT * 2);
      const medianCenters = getSlice('medianCenters', GROUP_COUNT * 2);
      const standardDistances = getSlice('standardDistances', GROUP_COUNT);
      const ellipses = getSlice('ellipses', GROUP_COUNT * 3);
      countReadout.setValue(
        `${formatCount(counts[group])}${weighted ? ' (activity-weighted)' : ''}`
      );
      meanReadout.setValue(
        `${formatNumber(meanCenters[group * 2], 0)}, ${formatNumber(meanCenters[group * 2 + 1], 0)} m`
      );
      medianReadout.setValue(
        `${formatNumber(
          Math.hypot(
            medianCenters[group * 2] - meanCenters[group * 2],
            medianCenters[group * 2 + 1] - meanCenters[group * 2 + 1]
          ),
          0
        )} m`
      );
      distanceReadout.setValue(`${formatNumber(standardDistances[group], 0)} m`);
      ellipseReadout.setValue(
        `${formatNumber(ellipses[group * 3 + 2], 0)} / ${formatNumber(ellipses[group * 3 + 1], 0)} m`
      );
      // Short axis angle counter-clockwise from +x; the long axis is 90 degrees further, which
      // is a clockwise-from-north bearing of -angle (modulo 180).
      const bearing = ((((-ellipses[group * 3] * 180) / Math.PI) % 180) + 180) % 180;
      bearingReadout.setValue(`${formatNumber(bearing, 0)}° from north`);
    }

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const copy = (name: keyof typeof outputs, key: SummaryLayoutKey, words: number) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: outputs[name],
          destinationBuffer: ticket.buffer,
          destinationOffset: SUMMARY_LAYOUT[key] * 4,
          size: words * 4
        });
      copy('lMinusR', 'rippleL', RIPLEY_RADII);
      copy('semivariances', 'semivariances', LAG_COUNT * DIRECTION_COUNT);
      copy('variogramPairs', 'variogramPairs', LAG_COUNT * DIRECTION_COUNT);
      copy('variogramDistances', 'variogramDistances', LAG_COUNT * DIRECTION_COUNT);
      copy('variogramStatistics', 'variogramStatistics', GPU_VARIOGRAM_STATISTICS_LENGTH);
      copy('moransI', 'moransI', BAND_COUNT);
      copy('zScores', 'zScores', BAND_COUNT);
      copy('bandPairs', 'bandPairs', BAND_COUNT);
      copy('peakBands', 'peakBands', 2);
      copy(
        'correlogramStatistics',
        'correlogramStatistics',
        GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH
      );
      copy('clarkEvans', 'clarkEvans', GPU_CLARK_EVANS_LENGTH);
      copy('quadrat', 'quadrat', GPU_QUADRAT_STATISTICS_LENGTH);
      copy('counts', 'counts', GROUP_COUNT);
      copy('meanCenters', 'meanCenters', GROUP_COUNT * 2);
      copy('medianCenters', 'medianCenters', GROUP_COUNT * 2);
      copy('standardDistances', 'standardDistances', GROUP_COUNT);
      copy('ellipses', 'ellipses', GROUP_COUNT * 3);
      ticket.markEncoded({byteOffset: 0, byteLength: SUMMARY_LAYOUT['words'] * 4});
      readbackPending = true;
      needsReadback = false;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        // Copy: the ring recycles its mapped memory.
        const copyBuffer = bytes.slice().buffer;
        lastSummary = new Float32Array(copyBuffer, 0, SUMMARY_LAYOUT['words']);
        lastSummaryWords = new Uint32Array(copyBuffer, 0, SUMMARY_LAYOUT['words']);
        updateReadouts();
        redrawCharts();
      } catch {
        needsReadback = true;
      } finally {
        readbackPending = false;
      }
    };

    // The view window follows the map: re-derive it when the view moved noticeably.
    let lastViewSignature = '';
    const followView = (
      viewport: Parameters<SpatialAnalysisModeInstance['encode']>[1]['viewport']
    ) => {
      const bounds = getViewportMetricBounds(viewport, projection);
      const width = bounds[2] - bounds[0];
      const height = bounds[3] - bounds[1];
      const quantum = Math.max(10, Math.min(width, height) / 100);
      const signature = bounds.map(value => Math.round(value / quantum)).join(',');
      if (signature === lastViewSignature) return;
      lastViewSignature = signature;
      currentBounds = bounds;
      writePatternParameters();
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [patternCompiled, distributionCompiled],
      encode(commandEncoder, frame) {
        if (windowChoice === 'view') followView(frame.viewport);
        // Static data: re-run a graph only when one of its inputs or parameters changed.
        if (patternDirty || frame.frameIndex < 2) {
          patternCompiled.encode(commandEncoder, {parameters: undefined});
          patternDirty = false;
        }
        if (distributionDirty || frame.frameIndex < 2) {
          distributionCompiled.encode(commandEncoder, {parameters: undefined});
          distributionDirty = false;
        }
        if (needsReadback && !readbackPending && frame.frameIndex >= 1) {
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [pois.origin[0], pois.origin[1], 0];
        const palette = (alpha: number): SpatialAnalysisColor[] =>
          Array.from({length: 8}, (_, index) =>
            index < GROUP_COUNT && visibleGroups[index] ? [...GROUP_COLORS[index], alpha] : HIDDEN
          ) as SpatialAnalysisColor[];
        const layers: Layer[] = [];
        if (showPoints) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'pattern-points',
              coordinateOrigin,
              positions: positionsBuffer,
              instanceCount: pointCount,
              values: groupIdsBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: palette(150),
              radiusPixels: 1.8
            })
          );
        }
        const ringCommon = {
          coordinateOrigin,
          instanceCount: GROUP_COUNT * RING_VERTICES,
          values: groupIndexBuffer,
          valueFormat: 'uint32' as const,
          valueDivisor: RING_VERTICES,
          colormap: 'category' as const,
          palette: palette(255)
        };
        if (showCircles) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              ...ringCommon,
              id: 'pattern-circles',
              segments: outputs.circleSegments,
              widthPixels: 1.6,
              palette: palette(190)
            })
          );
        }
        if (showEllipses) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              ...ringCommon,
              id: 'pattern-ellipses',
              segments: outputs.ellipseSegments,
              widthPixels: 2.6
            })
          );
        }
        const centerCommon = {
          coordinateOrigin,
          instanceCount: GROUP_COUNT,
          values: groupIndexBuffer,
          valueFormat: 'uint32' as const,
          colormap: 'category' as const
        };
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'pattern-center-halos',
            coordinateOrigin,
            positions: outputs.meanCenters,
            instanceCount: GROUP_COUNT,
            values: groupVisibleBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [255, 255, 255, 255],
            noDataColor: HIDDEN,
            radiusPixels: 9
          }),
          new SpatialAnalysisPointLayer({
            ...centerCommon,
            id: 'pattern-mean-centers',
            positions: outputs.meanCenters,
            palette: palette(255),
            radiusPixels: 7
          })
        );
        if (showMedian) {
          layers.push(
            new SpatialAnalysisPointLayer({
              ...centerCommon,
              id: 'pattern-median-centers',
              positions: outputs.medianCenters,
              palette: palette(255),
              radiusPixels: 4
            })
          );
        }
        return layers;
      },
      destroy() {
        destroyed = true;
        for (const chart of [rippleChart, variogramChart, correlogramChart]) chart.destroy();
        resources.destroy();
      }
    };
    return instance;
  }
};
