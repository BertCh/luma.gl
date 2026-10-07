// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  GPUClassBreaks,
  type GPUClassBreaksMethod
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPU_LISA_MARKOV_STATE_COUNT,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPUClassAssignment,
  GPULISAMarkov,
  GPUSpatialMarkov,
  GPUTransitionMatrix
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {addKernelPass} from '../../engine/mode-kernels';
import {
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor
} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  getGeoJsonOutlineSegments,
  getPolygonSource,
  lookupFeature,
  rasterizePolygons,
  type B13PolygonRaster
} from './b13-polygon-raster';
import {createViewImporter} from './b13-views';

/** Option state of the election-dynamics scene. */
export type ElectionDynamicsOptions = {
  through: '2020' | '2024';
  periodLag: '1' | '2';
  variable: 'demTwoParty' | 'demShare' | 'turnout';
  classMethod: 'equal-interval' | 'quantile' | 'natural-breaks';
  classCount: number;
  lagMethod: 'equal-interval' | 'quantile';
  lagClassCount: number;
  lisaSignificance: number;
  lisaMoments: 'per-year' | 'pooled';
  matrix: 'pooled' | 'lag0' | 'lag1' | 'lag2' | 'lisa';
  mapView: 'class' | 'lag' | 'lisa' | 'move';
  year: number;
  play: boolean;
  showStates: boolean;
  opacity: number;
};

export const ELECTION_YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024] as const;
const MAXIMUM_CLASS_COUNT = 5;
const LAG_CLASS_COUNT = 3;
const STATE_COUNT = GPU_LISA_MARKOV_STATE_COUNT;
const NO_CLASS = 0xffffffff;
const PLAY_SECONDS_PER_PERIOD = 1.4;
const METHODS: GPUClassBreaksMethod[] = ['quantile', 'equal-interval', 'natural-breaks'];
const LAG_METHODS: GPUClassBreaksMethod[] = ['equal-interval', 'quantile'];

/** Colors of the five activity classes for the share variables (low to high: red to blue). */
export const SHARE_CLASS_COLORS: readonly SpatialAnalysisColor[] = [
  [178, 24, 43, 245],
  [239, 138, 98, 245],
  [205, 205, 215, 245],
  [103, 169, 207, 245],
  [33, 102, 172, 245]
];
/** Sequential colors of the five classes for turnout. */
export const TURNOUT_CLASS_COLORS: readonly SpatialAnalysisColor[] = [
  [68, 1, 84, 245],
  [59, 82, 139, 245],
  [33, 145, 140, 245],
  [94, 201, 98, 245],
  [253, 231, 37, 245]
];
export const LISA_COLORS: readonly SpatialAnalysisColor[] = [
  [120, 130, 145, 80],
  [215, 48, 39, 250],
  [145, 191, 219, 250],
  [49, 54, 149, 250],
  [253, 174, 97, 250]
];
export const LISA_LABELS = ['Not significant', 'HH', 'LH', 'LL', 'HL'] as const;
/** Move colors: down two or more classes, down one, same, up one, up two or more. */
export const MOVE_COLORS: readonly SpatialAnalysisColor[] = [
  [84, 39, 136, 250],
  [153, 142, 195, 245],
  [225, 225, 225, 120],
  [241, 163, 64, 245],
  [179, 88, 6, 250]
];

/** Colors of the three spatial-lag classes. */
export function getLagColors(
  variable: ElectionDynamicsOptions['variable']
): SpatialAnalysisColor[] {
  return variable === 'turnout'
    ? [TURNOUT_CLASS_COLORS[0], TURNOUT_CLASS_COLORS[2], TURNOUT_CLASS_COLORS[4]]
    : [SHARE_CLASS_COLORS[0], SHARE_CLASS_COLORS[2], SHARE_CLASS_COLORS[4]];
}

/** Queen contiguity (any shared vertex) among the `valid` features, row-standardised. */
function buildQueenWeights(
  source: ReturnType<typeof getPolygonSource>,
  valid: Uint8Array
): {offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array; links: number} {
  const {vertices, ringOffsets, polygonRingOffsets, partFeature, featureCount} = source;
  const owners = new Map<number, number[]>();
  for (let part = 0; part + 1 < polygonRingOffsets.length; part++) {
    const feature = partFeature[part];
    if (!valid[feature]) continue;
    for (let ring = polygonRingOffsets[part]; ring < polygonRingOffsets[part + 1]; ring++) {
      for (let vertex = ringOffsets[ring]; vertex < ringOffsets[ring + 1]; vertex++) {
        const key =
          (Math.round(vertices[vertex * 2] * 1e5) + 12_600_000) * 6_000_000 +
          Math.round(vertices[vertex * 2 + 1] * 1e5);
        const list = owners.get(key);
        if (!list) owners.set(key, [feature]);
        else if (list[list.length - 1] !== feature && !list.includes(feature)) list.push(feature);
      }
    }
  }
  const adjacency: Set<number>[] = Array.from({length: featureCount}, () => new Set<number>());
  for (const list of owners.values()) {
    if (list.length < 2) continue;
    for (const a of list) for (const b of list) if (a !== b) adjacency[a].add(b);
  }
  const offsets = new Uint32Array(featureCount + 1);
  for (let row = 0; row < featureCount; row++)
    offsets[row + 1] = offsets[row] + adjacency[row].size;
  const neighbors = new Uint32Array(Math.max(1, offsets[featureCount]));
  const weights = new Float32Array(Math.max(1, offsets[featureCount]));
  for (let row = 0; row < featureCount; row++) {
    const sorted = [...adjacency[row]].sort((a, b) => a - b);
    sorted.forEach((neighbor, index) => {
      neighbors[offsets[row] + index] = neighbor;
      weights[offsets[row] + index] = 1 / sorted.length;
    });
  }
  return {offsets, neighbors, weights, links: offsets[featureCount]};
}

type Variant = {
  key: string;
  periods: number;
  lag: number;
  rows: number;
  analysis: CompiledGPUCommandGraph<void>;
  display: CompiledGPUCommandGraph<void>;
  values: Buffer;
  valid: Uint8Array;
  breaksParameters: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
  lagParameters: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
  reader: SummaryReader;
  displayReader: SummaryReader;
  displayParameters: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
  displayBuffer: Buffer;
  links: number;
  validCount: number;
};

type MatrixSummary = {
  counts: Uint32Array;
  probabilities: Float32Array;
  classEdges: Float32Array;
  classCount: number;
  lagEdges: Float32Array;
  lagClassCount: number;
  spatialCounts: Uint32Array;
  spatialProbabilities: Float32Array;
  lisaCounts: Uint32Array;
  lisaProbabilities: Float32Array;
  ignored: number;
};

/**
 * Distribution dynamics of US presidential voting, 2000 to 2024. One compiled graph per
 * (last year, period lag) runs the giddy-style chain on the GPU: pooled `GPUClassBreaks` and
 * `GPUClassAssignment` classify every county-year with one legend, `GPUTransitionMatrix` counts
 * class moves, `GPUSpatialMarkov` conditions them on the class of the neighbors, and
 * `GPULISAMarkov` counts moves between local Moran states. Variable, class method and count, lag
 * method and count, significance and moments are parameter writes.
 */
export async function createElectionDynamics(
  ctx: SceneContext<ElectionDynamicsOptions>
): Promise<SceneInstance<ElectionDynamicsOptions>> {
  const {device} = ctx;
  const counties = ctx.datasets.get('us-counties');
  const elections = ctx.datasets.get('us-elections');
  const statesDataset = ctx.datasets.get('us-states');
  const origin = counties.defaultOrigin;
  const projection = counties.getProjection(origin);
  const source = getPolygonSource(counties);
  const rows = source.featureCount;
  const raster: B13PolygonRaster = rasterizePolygons(
    source,
    projection,
    counties.manifest.bbox,
    4096
  );
  const names = (counties.geojson?.features ?? []).map(feature => {
    const properties = feature.properties as {name?: string; state?: string};
    return `${properties?.name ?? '?'}${properties?.state ? `, ${properties.state}` : ''}`;
  });
  const stateSegments = statesDataset.geojson
    ? getGeoJsonOutlineSegments(
        statesDataset.geojson as unknown as Parameters<typeof getGeoJsonOutlineSegments>[0],
        projection
      )
    : new Float32Array(4);

  const yearColumn = (variable: ElectionDynamicsOptions['variable'], year: number) =>
    elections.column<Float32Array>(`${variable === 'turnout' ? 'turnoutProxy' : variable}${year}`);

  const resources = new SpatialAnalysisResources(device, 'elections');
  const cellFeatureBuffer = resources.createBuffer('cell-feature', raster.cellFeature);
  const stateBuffer = resources.createBuffer('states', stateSegments);
  const lisaParameters = resources.createParameterBuffer(
    'lisa-parameters',
    'float32',
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );

  let destroyed = false;
  let dirty = true;
  let current: Variant | null = null;
  let summary: MatrixSummary | null = null;
  let cpuDisplay: Uint32Array | null = null;
  let shownPeriod = 0;
  let lastAdvance = 0;
  const variants = new Map<string, Variant>();

  const validityOf = (periods: number): Uint8Array => {
    const valid = new Uint8Array(rows);
    for (let row = 0; row < rows; row++) {
      let ok = 1;
      for (let period = 0; period < periods && ok; period++) {
        const year = ELECTION_YEARS[period];
        const share = yearColumn('demTwoParty', year)[row];
        const turnout = yearColumn('turnout', year)[row];
        if (!Number.isFinite(share) || !Number.isFinite(turnout)) ok = 0;
      }
      valid[row] = ok;
    }
    return valid;
  };

  function buildVariant(through: '2020' | '2024', lag: number): Variant {
    const periods = through === '2020' ? 6 : 7;
    const key = `${through}-${lag}`;
    const valid = validityOf(periods);
    let validCount = 0;
    for (const flag of valid) validCount += flag;
    const queen = buildQueenWeights(source, valid);
    const pooledCount = rows * periods;
    const pooledMask = new Uint32Array(pooledCount);
    for (let period = 0; period < periods; period++) {
      for (let row = 0; row < rows; row++) pooledMask[period * rows + row] = valid[row];
    }
    const buffer = (name: string, data: Float32Array | Uint32Array | number) =>
      resources.createBuffer(`${key}-${name}`, data);
    const rowMask = buffer('row-mask', Uint32Array.from(valid));
    const pooledMaskBuffer = buffer('pooled-mask', pooledMask);
    const values = buffer('values', pooledCount * 4);
    const offsets = buffer('offsets', queen.offsets);
    const neighbors = buffer('neighbors', queen.neighbors);
    const weightsBuffer = buffer('weights', queen.weights);
    const breaks = buffer('breaks', (MAXIMUM_CLASS_COUNT + 1) * 4);
    const classCount = buffer('class-count', 4);
    const classes = buffer('classes', pooledCount * 4);
    const counts = buffer('counts', MAXIMUM_CLASS_COUNT ** 2 * 4);
    const probabilities = buffer('probabilities', MAXIMUM_CLASS_COUNT ** 2 * 4);
    const totals = buffer('totals', MAXIMUM_CLASS_COUNT * 4);
    const ignored = buffer('ignored', 4);
    const spatialCells = LAG_CLASS_COUNT * MAXIMUM_CLASS_COUNT ** 2;
    const spatialCounts = buffer('spatial-counts', spatialCells * 4);
    const spatialProbabilities = buffer('spatial-probabilities', spatialCells * 4);
    const spatialTotals = buffer('spatial-totals', LAG_CLASS_COUNT * MAXIMUM_CLASS_COUNT * 4);
    const lagBreaks = buffer('lag-breaks', (LAG_CLASS_COUNT + 1) * 4);
    const lagClassCount = buffer('lag-class-count', 4);
    const lagClasses = buffer('lag-classes', pooledCount * 4);
    const lisaCounts = buffer('lisa-counts', STATE_COUNT ** 2 * 4);
    const lisaProbabilities = buffer('lisa-probabilities', STATE_COUNT ** 2 * 4);
    const lisaTotals = buffer('lisa-totals', STATE_COUNT * 4);
    const quadrants = buffer('quadrants', pooledCount * 4);
    const displayBuffer = buffer('display', (rows + 1) * 4);
    const breaksParameters = resources.createParameterBuffer(
      `${key}-breaks-parameters`,
      'float32',
      getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT)
    );
    const lagParameters = resources.createParameterBuffer(
      `${key}-lag-parameters`,
      'float32',
      getGPUClassBreaksParameterLength(LAG_CLASS_COUNT)
    );
    const displayParameters = resources.createParameterBuffer(
      `${key}-display-parameters`,
      'uint32',
      4
    );

    const analysisGraph = new GPUCommandGraph<void>(device, {id: `elections-analysis-${key}`});
    const v = createViewImporter(analysisGraph, key);
    const valuesView = v('values', values, 'float32', pooledCount);
    const rowMaskView = v('row-mask', rowMask, 'uint32', rows);
    const weights = {
      offsets: v('offsets', offsets, 'uint32', rows + 1),
      neighbors: v('neighbors', neighbors, 'uint32', queen.neighbors.length),
      weights: v('weights', weightsBuffer, 'float32', queen.weights.length)
    };
    const breaksView = v('breaks', breaks, 'float32', MAXIMUM_CLASS_COUNT + 1);
    const classCountView = v('class-count', classCount, 'uint32', 1);
    const classesView = v('classes', classes, 'uint32', pooledCount);
    const pooledMaskView = v('pooled-mask', pooledMaskBuffer, 'uint32', pooledCount);
    analysisGraph.add(
      new GPUClassBreaks({
        id: `${key}-breaks`,
        values: valuesView,
        mask: pooledMaskView,
        parameters: breaksParameters.importToGraph(analysisGraph),
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        methods: METHODS,
        output: {breaks: breaksView, classCount: classCountView}
      })
    );
    analysisGraph.add(
      new GPUClassAssignment({
        id: `${key}-assign`,
        values: valuesView,
        breaks: breaksView,
        classCount: classCountView,
        mask: pooledMaskView,
        output: classesView
      })
    );
    analysisGraph.add(
      new GPUTransitionMatrix({
        id: `${key}-matrix`,
        classes: classesView,
        rows,
        periods,
        periodLag: lag,
        classCount: MAXIMUM_CLASS_COUNT,
        mask: rowMaskView,
        output: {
          counts: v('counts', counts, 'uint32', MAXIMUM_CLASS_COUNT ** 2),
          probabilities: v('probabilities', probabilities, 'float32', MAXIMUM_CLASS_COUNT ** 2),
          rowTotals: v('totals', totals, 'uint32', MAXIMUM_CLASS_COUNT),
          ignored: v('ignored', ignored, 'uint32', 1)
        }
      })
    );
    analysisGraph.add(
      new GPUSpatialMarkov({
        id: `${key}-spatial`,
        values: valuesView,
        classes: classesView,
        classCount: MAXIMUM_CLASS_COUNT,
        weights,
        periods,
        periodLag: lag,
        lagMaximumClassCount: LAG_CLASS_COUNT,
        lagParameters: lagParameters.importToGraph(analysisGraph),
        lagMethods: LAG_METHODS,
        mask: rowMaskView,
        output: {
          counts: v('spatial-counts', spatialCounts, 'uint32', spatialCells),
          probabilities: v('spatial-probabilities', spatialProbabilities, 'float32', spatialCells),
          rowTotals: v(
            'spatial-totals',
            spatialTotals,
            'uint32',
            LAG_CLASS_COUNT * MAXIMUM_CLASS_COUNT
          ),
          lagBreaks: v('lag-breaks', lagBreaks, 'float32', LAG_CLASS_COUNT + 1),
          lagClassCount: v('lag-class-count', lagClassCount, 'uint32', 1),
          lagClasses: v('lag-classes', lagClasses, 'uint32', pooledCount)
        }
      })
    );
    analysisGraph.add(
      new GPULISAMarkov({
        id: `${key}-lisa`,
        values: valuesView,
        weights,
        periods,
        periodLag: lag,
        parameters: lisaParameters.importToGraph(analysisGraph),
        mask: rowMaskView,
        output: {
          counts: v('lisa-counts', lisaCounts, 'uint32', STATE_COUNT ** 2),
          probabilities: v('lisa-probabilities', lisaProbabilities, 'float32', STATE_COUNT ** 2),
          rowTotals: v('lisa-totals', lisaTotals, 'uint32', STATE_COUNT),
          quadrants: v('quadrants', quadrants, 'uint32', pooledCount)
        }
      })
    );
    const analysis = resources.track(analysisGraph.compile());

    // Display kernel: one county's class, lag class, LISA state or class move for one period.
    const displayGraph = new GPUCommandGraph<void>(device, {id: `elections-display-${key}`});
    const d = createViewImporter(displayGraph, `${key}-d`);
    addKernelPass(displayGraph, {
      id: `${key}-display`,
      invocationCount: rows + 1,
      declarations: `const ROWS: u32 = ${rows}u; const PERIODS: u32 = ${periods}u; const LAG: u32 = ${lag}u;`,
      bindings: [
        {
          name: 'parameters',
          view: displayParameters.importToGraph(displayGraph),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'rowMask',
          view: d('row-mask', rowMask, 'uint32', rows),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'classes',
          view: d('classes', classes, 'uint32', pooledCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'lagClasses',
          view: d('lag-classes', lagClasses, 'uint32', pooledCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'quadrants',
          view: d('quadrants', quadrants, 'uint32', pooledCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'shown',
          view: d('display', displayBuffer, 'uint32', rows + 1),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  if (index >= ROWS || rowMask[rowMaskOffset + index] == 0u) {
    shown[shownOffset + index] = 0xffffffffu;
    return;
  }
  let period = min(parameters[parametersOffset], PERIODS - 1u);
  let mapView = parameters[parametersOffset + 1u];
  let source = period * ROWS + index;
  var value = classes[classesOffset + source];
  if (mapView == 1u) {
    value = lagClasses[lagClassesOffset + source];
  } else if (mapView == 2u) {
    value = quadrants[quadrantsOffset + source];
  } else if (mapView == 3u) {
    // Class move over the next LAG periods (the last LAG periods have no successor).
    if (period + LAG >= PERIODS) {
      value = 0xffffffffu;
    } else {
      let before = classes[classesOffset + source];
      let after = classes[classesOffset + (period + LAG) * ROWS + index];
      if (before == 0xffffffffu || after == 0xffffffffu) {
        value = 0xffffffffu;
      } else {
        value = u32(clamp(i32(after) - i32(before) + 2, 0, 4));
      }
    }
  }
  shown[shownOffset + index] = value;`
    });
    const display = resources.track(displayGraph.compile());

    const reader = new SummaryReader(
      resources,
      `${key}-matrices`,
      [
        {buffer: counts, size: MAXIMUM_CLASS_COUNT ** 2 * 4},
        {buffer: probabilities, size: MAXIMUM_CLASS_COUNT ** 2 * 4},
        {buffer: breaks, size: (MAXIMUM_CLASS_COUNT + 1) * 4},
        {buffer: classCount, size: 4},
        {buffer: lagBreaks, size: (LAG_CLASS_COUNT + 1) * 4},
        {buffer: lagClassCount, size: 4},
        {buffer: spatialCounts, size: spatialCells * 4},
        {buffer: spatialProbabilities, size: spatialCells * 4},
        {buffer: lisaCounts, size: STATE_COUNT ** 2 * 4},
        {buffer: lisaProbabilities, size: STATE_COUNT ** 2 * 4},
        {buffer: ignored, size: 4}
      ],
      bytes => {
        if (destroyed || current?.key !== key) return;
        let offset = 0;
        const take = (length: number) => {
          const slice = bytes.slice(offset, offset + length * 4);
          offset += length * 4;
          return slice;
        };
        const unsigned = (length: number) => new Uint32Array(take(length));
        const floating = (length: number) => new Float32Array(take(length));
        summary = {
          counts: unsigned(MAXIMUM_CLASS_COUNT ** 2),
          probabilities: floating(MAXIMUM_CLASS_COUNT ** 2),
          classEdges: floating(MAXIMUM_CLASS_COUNT + 1),
          classCount: unsigned(1)[0],
          lagEdges: floating(LAG_CLASS_COUNT + 1),
          lagClassCount: unsigned(1)[0],
          spatialCounts: unsigned(spatialCells),
          spatialProbabilities: floating(spatialCells),
          lisaCounts: unsigned(STATE_COUNT ** 2),
          lisaProbabilities: floating(STATE_COUNT ** 2),
          ignored: unsigned(1)[0]
        };
        showSummary();
      }
    );
    const displayReader = new SummaryReader(
      resources,
      `${key}-display`,
      [{buffer: displayBuffer, size: (rows + 1) * 4}],
      bytes => {
        if (destroyed || current?.key !== key) return;
        cpuDisplay = new Uint32Array(bytes);
      }
    );
    return {
      key,
      periods,
      lag,
      rows,
      analysis,
      display,
      values,
      valid,
      breaksParameters,
      lagParameters,
      reader,
      displayReader,
      displayParameters,
      displayBuffer,
      links: queen.links,
      validCount
    };
  }

  function getVariant(): Variant {
    const {through, periodLag} = ctx.options;
    const key = `${through}-${periodLag}`;
    let variant = variants.get(key);
    if (!variant) {
      variant = buildVariant(through, Number(periodLag));
      variants.set(key, variant);
    }
    return variant;
  }

  function writeValues(): void {
    if (!current) return;
    const {variable} = ctx.options;
    const data = new Float32Array(current.rows * current.periods);
    for (let period = 0; period < current.periods; period++) {
      const column = yearColumn(variable, ELECTION_YEARS[period]);
      for (let row = 0; row < current.rows; row++) {
        const value = column[row];
        data[period * current.rows + row] =
          current.valid[row] && Number.isFinite(value) ? value : 0;
      }
    }
    current.values.write(data);
    writeLisaParameters(data);
    dirty = true;
  }

  function writeLisaParameters(data?: Float32Array): void {
    if (!current) return;
    const options = ctx.options;
    let fixedMoments: {count: number; mean: number; variance: number} | undefined;
    if (options.lisaMoments === 'pooled') {
      const pooled = data ?? new Float32Array(0);
      let count = 0;
      let sum = 0;
      for (let period = 0; period < current.periods; period++) {
        for (let row = 0; row < current.rows; row++) {
          if (!current.valid[row]) continue;
          count++;
          sum += pooled[period * current.rows + row] ?? 0;
        }
      }
      if (pooled.length > 0 && count > 1) {
        const mean = sum / count;
        let squares = 0;
        for (let period = 0; period < current.periods; period++) {
          for (let row = 0; row < current.rows; row++) {
            if (!current.valid[row]) continue;
            squares += ((pooled[period * current.rows + row] ?? 0) - mean) ** 2;
          }
        }
        fixedMoments = {count, mean, variance: Math.max(squares / count, 1e-12)};
      }
    }
    lisaParameters.write(
      getGPUSpatialAutocorrelationParameterValues({
        significanceLevel: options.lisaSignificance,
        fixedMoments
      })
    );
    dirty = true;
  }

  function writeClassParameters(): void {
    if (!current) return;
    const options = ctx.options;
    current.breaksParameters.write(
      getGPUClassBreaksParameterValues(
        {method: options.classMethod, classCount: options.classCount},
        MAXIMUM_CLASS_COUNT
      )
    );
    current.lagParameters.write(
      getGPUClassBreaksParameterValues(
        {method: options.lagMethod, classCount: options.lagClassCount},
        LAG_CLASS_COUNT
      )
    );
    dirty = true;
  }

  const formatEdge = (value: number) =>
    ctx.options.variable === 'turnout'
      ? `${(value * 100).toFixed(0)}%`
      : `${(value * 100).toFixed(0)}%`;

  function showSummary(): void {
    if (!summary || !current) return;
    const data = summary;
    const options = ctx.options;
    const classes = Math.min(MAXIMUM_CLASS_COUNT, Math.max(1, data.classCount));
    ctx.setReadout(
      'classEdges',
      Array.from(data.classEdges.subarray(0, classes + 1), formatEdge).join('  |  ')
    );
    ctx.setReadout(
      'lagEdges',
      Array.from(data.lagEdges.subarray(0, Math.max(1, data.lagClassCount) + 1), formatEdge).join(
        '  |  '
      )
    );
    const total = (counts: Uint32Array) => counts.reduce((sum, value) => sum + value, 0);
    const stay = (counts: Uint32Array, size: number, offset = 0) => {
      let diagonal = 0;
      let all = 0;
      for (let from = 0; from < size; from++) {
        for (let to = 0; to < size; to++) {
          const value = counts[offset + from * size + to];
          all += value;
          if (from === to) diagonal += value;
        }
      }
      return all > 0 ? diagonal / all : NaN;
    };
    const percent = (value: number) =>
      Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'n/a';
    ctx.setReadout(
      'transitions',
      `${formatCount(total(data.counts))} county moves (${formatCount(data.ignored)} skipped)`
    );
    ctx.setReadout('stay', percent(stay(data.counts, MAXIMUM_CLASS_COUNT)));
    ctx.setReadout(
      'stayLag',
      [0, 1, 2]
        .map(lagClass =>
          percent(
            stay(data.spatialCounts, MAXIMUM_CLASS_COUNT, lagClass * MAXIMUM_CLASS_COUNT ** 2)
          )
        )
        .join(' / ')
    );
    ctx.setReadout('stayLisa', percent(stay(data.lisaCounts, STATE_COUNT)));

    // Transition matrix rows of the chosen matrix.
    let shownCounts = data.counts;
    let shownProbabilities = data.probabilities;
    let offset = 0;
    let labels: readonly string[] = Array.from(
      {length: MAXIMUM_CLASS_COUNT},
      (_, index) => `class ${index + 1}`
    );
    let limit = classes;
    if (options.matrix === 'lisa') {
      shownCounts = data.lisaCounts;
      shownProbabilities = data.lisaProbabilities;
      labels = LISA_LABELS;
      limit = STATE_COUNT;
    } else if (options.matrix !== 'pooled') {
      const lagClass = Number(options.matrix.slice(3));
      offset = lagClass * MAXIMUM_CLASS_COUNT ** 2;
      shownCounts = data.spatialCounts;
      shownProbabilities = data.spatialProbabilities;
    }
    const stride = MAXIMUM_CLASS_COUNT;
    for (let from = 0; from < STATE_COUNT; from++) {
      if (from >= limit) {
        ctx.setReadout(`row${from}`, '-');
        continue;
      }
      let rowTotal = 0;
      const cells: string[] = [];
      for (let to = 0; to < limit; to++) {
        rowTotal += shownCounts[offset + from * stride + to];
        cells.push(shownProbabilities[offset + from * stride + to].toFixed(2));
      }
      ctx.setReadout(
        `row${from}`,
        `${labels[from]}: ${cells.join('  ')}   (n = ${formatCount(rowTotal)})`
      );
    }
  }

  function setPeriod(period: number): void {
    if (!current) return;
    shownPeriod = Math.max(0, Math.min(period, current.periods - 1));
    const lag = current.lag;
    const next = shownPeriod + lag < current.periods ? ELECTION_YEARS[shownPeriod + lag] : null;
    ctx.setReadout(
      'yearShown',
      ctx.options.mapView === 'move' && next
        ? `${ELECTION_YEARS[shownPeriod]} to ${next}`
        : String(ELECTION_YEARS[shownPeriod])
    );
    displayDirty = true;
  }

  let displayDirty = true;

  function adopt(): void {
    current = getVariant();
    summary = null;
    cpuDisplay = null;
    ctx.setReadout('counties', `${formatCount(current.validCount)} of ${formatCount(rows)}`);
    ctx.setReadout(
      'links',
      `${formatCount(current.links)} queen links (${(current.links / Math.max(1, current.validCount)).toFixed(1)} per county)`
    );
    writeValues();
    writeClassParameters();
    writeLisaParameters(readValues());
    setPeriod(ctx.options.year);
    ctx.requestLayers();
  }

  function readValues(): Float32Array | undefined {
    if (!current) return undefined;
    const data = new Float32Array(current.rows * current.periods);
    for (let period = 0; period < current.periods; period++) {
      const column = yearColumn(ctx.options.variable, ELECTION_YEARS[period]);
      for (let row = 0; row < current.rows; row++) {
        data[period * current.rows + row] =
          current.valid[row] && Number.isFinite(column[row]) ? column[row] : 0;
      }
    }
    return data;
  }

  adopt();

  return {
    getCompiledGraphs: () =>
      current ? ([current.analysis, current.display] as CompiledGPUCommandGraph<never>[]) : [],

    setOption(id, _value, state) {
      if (id === 'through' || id === 'periodLag') {
        adopt();
      } else if (id === 'variable') {
        writeValues();
        ctx.requestLayers();
      } else if (
        id === 'classMethod' ||
        id === 'classCount' ||
        id === 'lagMethod' ||
        id === 'lagClassCount'
      ) {
        writeClassParameters();
      } else if (id === 'lisaSignificance' || id === 'lisaMoments') {
        writeLisaParameters(readValues());
      } else if (id === 'year' || id === 'mapView') {
        setPeriod(state.year);
        ctx.requestLayers();
      } else if (id === 'matrix') {
        showSummary();
      } else {
        ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip({coordinate}) {
      if (!current || !coordinate) return null;
      const feature = lookupFeature(raster, coordinate[0], coordinate[1]);
      if (feature < 0) return null;
      const lines = [names[feature] ?? `County ${feature}`];
      if (!current.valid[feature]) {
        lines.push('Missing returns in this period range');
        return lines.join('\n');
      }
      const series = ELECTION_YEARS.slice(0, current.periods).map(year => {
        const value = yearColumn(ctx.options.variable, year)[feature];
        return `${year}: ${Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'n/a'}`;
      });
      lines.push(
        ctx.options.variable === 'turnout'
          ? 'Votes per resident'
          : ctx.options.variable === 'demShare'
            ? 'Democratic share of all votes'
            : 'Democratic two-party share'
      );
      lines.push(series.join('   '));
      const shown = cpuDisplay?.[feature];
      if (shown !== undefined && shown !== NO_CLASS) {
        const options = ctx.options;
        lines.push(
          options.mapView === 'class'
            ? `Class ${shown + 1} in ${ELECTION_YEARS[shownPeriod]}`
            : options.mapView === 'lag'
              ? `Neighbors in lag class ${shown + 1}`
              : options.mapView === 'lisa'
                ? `LISA: ${LISA_LABELS[shown] ?? shown}`
                : (['Down 2+ classes', 'Down 1 class', 'Same class', 'Up 1 class', 'Up 2+ classes'][
                    shown
                  ] ?? '')
        );
      }
      return lines.join('\n');
    },

    encode(commandEncoder, frame) {
      if (!current) return;
      const options = ctx.options;
      if (options.play) {
        if (frame.timeSeconds - lastAdvance > PLAY_SECONDS_PER_PERIOD) {
          lastAdvance = frame.timeSeconds;
          setPeriod((shownPeriod + 1) % current.periods);
        }
      } else if (shownPeriod !== Math.min(options.year, current.periods - 1)) {
        setPeriod(options.year);
      }
      if (dirty || frame.frameIndex < 3) {
        current.analysis.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        current.reader.request(commandEncoder);
        displayDirty = true;
      }
      if (displayDirty || frame.frameIndex < 3) {
        const view =
          options.mapView === 'class'
            ? 0
            : options.mapView === 'lag'
              ? 1
              : options.mapView === 'lisa'
                ? 2
                : 3;
        current.displayParameters.write(Uint32Array.of(shownPeriod, view, 0, 0));
        current.display.encode(commandEncoder, {parameters: undefined});
        current.displayReader.request(commandEncoder);
        displayDirty = false;
      }
      current.reader.flush(commandEncoder);
      current.displayReader.flush(commandEncoder);
    },

    getLayers() {
      if (!current) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const palette: readonly SpatialAnalysisColor[] =
        options.mapView === 'class'
          ? options.variable === 'turnout'
            ? TURNOUT_CLASS_COLORS
            : SHARE_CLASS_COLORS
          : options.mapView === 'lag'
            ? getLagColors(options.variable)
            : options.mapView === 'lisa'
              ? LISA_COLORS
              : MOVE_COLORS;
      const layers: Layer[] = [
        new SpatialAnalysisRasterLayer({
          id: `elections-${options.mapView}`,
          coordinateOrigin,
          gridSize: [raster.width, raster.height],
          bounds: raster.bounds,
          valueIndices: cellFeatureBuffer,
          values: current.displayBuffer,
          valueFormat: 'uint32',
          colormap: 'category',
          palette,
          noDataColor: [0, 0, 0, 0],
          opacity: options.opacity,
          color: [255, 255, 255, 255]
        })
      ];
      if (options.showStates) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'elections-states',
            coordinateOrigin,
            segments: stateBuffer,
            instanceCount: stateSegments.length / 4,
            widthPixels: 0.9,
            colormap: 'uniform',
            color: dark ? [235, 240, 250, 150] : [30, 40, 60, 140]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const variant of variants.values()) {
        variant.reader.stop();
        variant.displayReader.stop();
      }
      resources.destroy();
    }
  };
}
