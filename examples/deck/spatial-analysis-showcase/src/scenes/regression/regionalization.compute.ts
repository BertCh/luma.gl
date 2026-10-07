// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {COORDINATE_SYSTEM} from '@deck.gl/core';
import {
  GPUContiguityWeights,
  GPUKMeans,
  GPURegionPartitionEvaluation,
  GPUSkaterRegions,
  GPUSpatialWeightsMinimumSpanningTree,
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPU_SKATER_NO_CUT,
  GPU_SKATER_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {REGION_PALETTE} from './b6-colors';
import {createNamedReader} from './b6-reader';
import {PolygonFillLayer} from './b6-fill-layer';
import {
  buildOutlineSegments,
  computeCentroids,
  computeQueenAdjacency,
  createFeatureLocator,
  formatInteger,
  formatNumber,
  getLargestComponent,
  getMoments,
  getQuantile,
  NO_ROW,
  readPolygonGeometry,
  subsetPolygonGeometry,
  triangulatePolygonGeometry
} from './b6-polygons';

/** Attribute columns of the regionalization (the id is the option key). */
export const REGION_ATTRIBUTES = [
  {id: 'poverty', label: 'Poverty rate'},
  {id: 'income', label: 'ln income per capita'},
  {id: 'uninsured', label: 'Uninsured rate'},
  {id: 'age65', label: 'Age 65+ share'},
  {id: 'noVehicle', label: 'No-vehicle households'},
  {id: 'diabetes', label: 'Diabetes prevalence'},
  {id: 'black', label: 'Black share'},
  {id: 'hispanic', label: 'Hispanic share'}
] as const;

/** Option state of the regionalization scene. */
export type RegionalizationOptions = {
  poverty: boolean;
  income: boolean;
  uninsured: boolean;
  age65: boolean;
  noVehicle: boolean;
  diabetes: boolean;
  black: boolean;
  hispanic: boolean;
  standardize: boolean;
  criterion: 'queen' | 'rook';
  regions: number;
  minimumSize: number;
  kmeansClusters: '3' | '4' | '6' | '8' | '10' | '12' | '16';
  map: 'skater' | 'kmeans';
  showTree: boolean;
  treeByCost: boolean;
  showCuts: boolean;
};

const MAXIMUM_REGIONS = 20;
const SLOTS_PER_ROW = 24;
const MAXIMUM_KMEANS_CLUSTERS = 16;
const KMEANS_ITERATIONS = 24;
const COLUMN_COUNT = REGION_ATTRIBUTES.length;

/**
 * Greedy graph coloring of the labels so that adjacent regions get different palette entries.
 * Largest regions first; a region takes the lowest-use color none of its neighbors holds.
 */
function assignRegionColors(labels: Uint32Array, adjacency: readonly number[][]): Uint32Array {
  const sizes = new Map<number, number>();
  for (const label of labels) sizes.set(label, (sizes.get(label) ?? 0) + 1);
  const neighborLabels = new Map<number, Set<number>>();
  for (let row = 0; row < labels.length; row++) {
    for (const neighbor of adjacency[row]) {
      if (labels[neighbor] !== labels[row]) {
        const set = neighborLabels.get(labels[row]) ?? new Set<number>();
        set.add(labels[neighbor]);
        neighborLabels.set(labels[row], set);
      }
    }
  }
  const order = [...sizes.keys()].sort((a, b) => sizes.get(b)! - sizes.get(a)! || a - b);
  const colors = new Map<number, number>();
  const paletteSize = REGION_PALETTE.length;
  for (const label of order) {
    const used = new Array<number>(paletteSize).fill(0);
    for (const other of neighborLabels.get(label) ?? []) {
      const color = colors.get(other);
      if (color !== undefined) used[color]++;
    }
    let best = 0;
    for (let color = 1; color < paletteSize; color++) if (used[color] < used[best]) best = color;
    colors.set(label, best);
  }
  return Uint32Array.from(labels, label => colors.get(label)!);
}

/** Number of connected pieces of every label, counting through `adjacency`. */
function countPieces(
  labels: Uint32Array,
  adjacency: readonly number[][]
): {total: number; labelCount: number} {
  const seen = new Uint8Array(labels.length);
  let total = 0;
  const distinct = new Set<number>();
  for (let start = 0; start < labels.length; start++) {
    distinct.add(labels[start]);
    if (seen[start]) continue;
    total++;
    const stack = [start];
    seen[start] = 1;
    while (stack.length) {
      const row = stack.pop()!;
      for (const neighbor of adjacency[row]) {
        if (!seen[neighbor] && labels[neighbor] === labels[row]) {
          seen[neighbor] = 1;
          stack.push(neighbor);
        }
      }
    }
  }
  return {total, labelCount: distinct.size};
}

/** Top two principal components of the rows of `table` restricted to `columns`, as `x, y` pairs. */
function computePrincipalPlane(
  table: Float32Array,
  rowCount: number,
  columns: number[]
): Float32Array {
  const dimension = columns.length;
  const plane = new Float32Array(rowCount * 2);
  if (dimension === 0) return plane;
  const covariance = Array.from({length: dimension}, () => new Float64Array(dimension));
  for (let row = 0; row < rowCount; row++) {
    for (let a = 0; a < dimension; a++) {
      for (let b = 0; b < dimension; b++) {
        covariance[a][b] +=
          (table[row * COLUMN_COUNT + columns[a]] * table[row * COLUMN_COUNT + columns[b]]) /
          rowCount;
      }
    }
  }
  // Power iteration with deflation for two components.
  const components: Float64Array[] = [];
  for (let component = 0; component < Math.min(2, dimension); component++) {
    let vector = Float64Array.from({length: dimension}, (_, index) => 1 + index * 0.37 + component);
    for (let iteration = 0; iteration < 200; iteration++) {
      const next = new Float64Array(dimension);
      for (let a = 0; a < dimension; a++)
        for (let b = 0; b < dimension; b++) next[a] += covariance[a][b] * vector[b];
      for (const previous of components) {
        let dot = 0;
        for (let a = 0; a < dimension; a++) dot += next[a] * previous[a];
        for (let a = 0; a < dimension; a++) next[a] -= dot * previous[a];
      }
      const length = Math.hypot(...next) || 1;
      vector = next.map(value => value / length);
    }
    components.push(vector);
  }
  for (let row = 0; row < rowCount; row++) {
    for (let component = 0; component < components.length; component++) {
      let score = 0;
      for (let a = 0; a < dimension; a++)
        score += table[row * COLUMN_COUNT + columns[a]] * components[component][a];
      plane[row * 2 + component] = score;
    }
  }
  return plane;
}

/**
 * Regionalization of Chicago tracts. One graph per (contiguity criterion, standardization) holds
 * `GPUContiguityWeights`, `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions` and a
 * `GPURegionPartitionEvaluation`; a second graph per k-means size holds an aspatial `GPUKMeans`
 * on the first two principal components of the same attributes and a second evaluation over the
 * same attribute table, so the two partitions are scored by one metric. Attribute switches, the
 * region count and the minimum region size are buffer writes. Variants compile the first time
 * they are chosen; small result columns are read back once per change.
 */
export async function createRegionalization(
  ctx: SceneContext<RegionalizationOptions>
): Promise<SceneInstance<RegionalizationOptions>> {
  const tracts = ctx.datasets.get('chicago-tracts');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'regionalization');

  // ---- Study set. ----
  const fullGeometry = readPolygonGeometry(tracts);
  const featureCount = fullGeometry.featureCount;
  const column = (name: string) => tracts.column<Float32Array>(name);
  const population = column('population');
  const income = column('perCapitaIncome');
  const share = (name: string) => {
    const values = column(name);
    return Float32Array.from(values, (value, index) =>
      population[index] > 0 ? (100 * value) / population[index] : Number.NaN
    );
  };
  const raw: Record<(typeof REGION_ATTRIBUTES)[number]['id'], Float32Array> = {
    poverty: column('poverty150Pct'),
    income: Float32Array.from(income, value => (value > 0 ? Math.log(value) : Number.NaN)),
    uninsured: column('uninsuredPct'),
    age65: column('age65Pct'),
    noVehicle: column('noVehiclePct'),
    diabetes: column('diabetes'),
    black: share('nhBlack'),
    hispanic: share('hispanic')
  };
  const candidates: number[] = [];
  for (let feature = 0; feature < featureCount; feature++) {
    let finite = population[feature] >= 300;
    for (const values of Object.values(raw)) finite &&= Number.isFinite(values[feature]);
    if (finite) candidates.push(feature);
  }
  const features = getLargestComponent(computeQueenAdjacency(fullGeometry), candidates);
  const rowCount = features.length;
  const rowOfFeature = new Uint32Array(featureCount).fill(NO_ROW);
  features.forEach((feature, row) => {
    rowOfFeature[feature] = row;
  });
  const geometry = subsetPolygonGeometry(fullGeometry, features);
  const centroids = computeCentroids(geometry);
  const geoids = (tracts.manifest as unknown as {geoid: string[]}).geoid;

  // Attribute table (row-major, raw units) and its z-scored copy for the principal plane.
  const table = new Float32Array(rowCount * COLUMN_COUNT);
  const zTable = new Float32Array(rowCount * COLUMN_COUNT);
  REGION_ATTRIBUTES.forEach(({id}, columnIndex) => {
    const values = Float32Array.from(features, feature => raw[id][feature]);
    const {mean, deviation} = getMoments(values);
    for (let row = 0; row < rowCount; row++) {
      table[row * COLUMN_COUNT + columnIndex] = values[row];
      zTable[row * COLUMN_COUNT + columnIndex] = (values[row] - mean) / deviation;
    }
  });

  // ---- Geometry buffers. ----
  const triangles = triangulatePolygonGeometry(geometry);
  const excludedRows = new Uint32Array(featureCount).fill(NO_ROW);
  for (let feature = 0; feature < featureCount; feature++)
    if (rowOfFeature[feature] === NO_ROW) excludedRows[feature] = 0;
  const excludedTriangles = triangulatePolygonGeometry(fullGeometry, excludedRows);
  const trianglesBuffer = resources.createBuffer('triangles', triangles.corners);
  const ownersBuffer = resources.createBuffer('owners', triangles.owners);
  const excludedTrianglesBuffer = resources.createBuffer(
    'excluded-triangles',
    excludedTriangles.corners
  );
  const excludedOwnersBuffer = resources.createBuffer('excluded-owners', excludedTriangles.owners);
  const outlineSegments = buildOutlineSegments(fullGeometry);
  const outlineBuffer = resources.createBuffer('outline', outlineSegments);
  const locate = createFeatureLocator(geometry);
  const centroidBuffer = resources.createBuffer('centroids', centroids);

  // ---- Contributor buffers. ----
  const capacity = rowCount * SLOTS_PER_ROW;
  const stepCount = MAXIMUM_REGIONS - 1;
  const verticesBuffer = resources.createBuffer('vertices', geometry.vertices);
  const ringOffsetsBuffer = resources.createBuffer('ring-offsets', geometry.ringOffsets);
  const polygonOffsetsBuffer = resources.createBuffer(
    'polygon-offsets',
    geometry.featureRingOffsets
  );
  const valuesBuffer = resources.createBuffer('values', rowCount * COLUMN_COUNT * 4);
  const standardizedBuffer = resources.createBuffer('standardized', rowCount * COLUMN_COUNT * 4);
  const offsetsBuffer = resources.createBuffer('offsets', (rowCount + 1) * 4);
  const neighborsBuffer = resources.createBuffer('neighbors', capacity * 4);
  const weightValuesBuffer = resources.createBuffer('weight-values', capacity * 4);
  const overflowBuffer = resources.createBuffer('overflow', 4);
  const treeFlagsBuffer = resources.createBuffer('tree-flags', capacity * 4);
  const componentLabelsBuffer = resources.createBuffer('component-labels', rowCount * 4);
  const edgeIdsBuffer = resources.createBuffer('edge-ids', rowCount * 4);
  const edgeCountBuffer = resources.createBuffer('edge-count', 4);
  const edgeOverflowBuffer = resources.createBuffer('edge-overflow', 4);
  const edgeCostsBuffer = resources.createBuffer('edge-costs', rowCount * 4);
  const labelsBuffer = resources.createBuffer('labels', rowCount * 4);
  const regionCountBuffer = resources.createBuffer('region-count', 4);
  const cutEdgesBuffer = resources.createBuffer('cut-edges', stepCount * 4);
  const cutGainsBuffer = resources.createBuffer('cut-gains', stepCount * 4);
  const summaryBuffer = resources.createBuffer(
    'summary',
    GPU_REGION_PARTITION_EVALUATION_LAYOUT.length * 4
  );
  const regionSizesBuffer = resources.createBuffer('region-sizes', rowCount * 4);
  const regionWithinBuffer = resources.createBuffer('region-within-ssd', rowCount * 4);
  const skaterParameters = resources.createParameterBuffer(
    'skater-parameters',
    'uint32',
    GPU_SKATER_PARAMETER_LENGTH
  );
  const planeBuffer = resources.createBuffer('principal-plane', rowCount * 2 * 4);
  const kLabelsBuffer = resources.createBuffer('kmeans-labels', rowCount * 4);
  const kCentersBuffer = resources.createBuffer('kmeans-centers', MAXIMUM_KMEANS_CLUSTERS * 2 * 4);
  const kSizesBuffer = resources.createBuffer('kmeans-sizes', MAXIMUM_KMEANS_CLUSTERS * 4);
  const kConvergenceBuffer = resources.createBuffer('kmeans-convergence', 8);
  const kSummaryBuffer = resources.createBuffer(
    'kmeans-summary',
    GPU_REGION_PARTITION_EVALUATION_LAYOUT.length * 4
  );
  const kRegionSizesBuffer = resources.createBuffer('kmeans-region-sizes', rowCount * 4);
  const kRegionWithinBuffer = resources.createBuffer('kmeans-region-within', rowCount * 4);
  const regionColorsBuffer = resources.createBuffer('region-colors', rowCount * 4);
  const clusterColorsBuffer = resources.createBuffer('cluster-colors', rowCount * 4);
  const slotSegmentsBuffer = resources.createBuffer('slot-segments', capacity * 16);
  const slotCostBuffer = resources.createBuffer('slot-cost', capacity * 4);
  const cutMaskBuffer = resources.createBuffer('cut-mask', capacity * 4);

  const importWeights = (graph: GPUCommandGraph<void>) => ({
    offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', rowCount + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', neighborsBuffer, 'uint32', capacity),
    weights: importGraphBuffer(graph, 'weight-values', weightValuesBuffer, 'float32', capacity)
  });

  // ---- Variant graphs. ----
  const skaterGraphs = new Map<string, CompiledGPUCommandGraph<void>>();
  const getSkaterGraph = (criterion: 'queen' | 'rook', standardize: boolean) => {
    const key = `${criterion}:${standardize}`;
    let compiled = skaterGraphs.get(key);
    if (compiled) return compiled;
    const graph = new GPUCommandGraph<void>(device, {
      id: `regions-${criterion}-${standardize ? 'z' : 'raw'}`
    });
    const weights = importWeights(graph);
    const treeFlags = importGraphBuffer(graph, 'tree-flags', treeFlagsBuffer, 'uint32', capacity);
    const componentLabels = importGraphBuffer(
      graph,
      'component-labels',
      componentLabelsBuffer,
      'uint32',
      rowCount
    );
    const values = importGraphBuffer(
      graph,
      'values',
      valuesBuffer,
      'float32',
      rowCount * COLUMN_COUNT
    );
    const standardized = importGraphBuffer(
      graph,
      'standardized',
      standardizedBuffer,
      'float32',
      rowCount * COLUMN_COUNT
    );
    const labels = importGraphBuffer(graph, 'labels', labelsBuffer, 'uint32', rowCount);
    graph.add(
      new GPUContiguityWeights({
        id: 'contiguity',
        criterion,
        positions: importGraphBuffer(
          graph,
          'vertices',
          verticesBuffer,
          'float32x2',
          geometry.vertices.length / 2
        ),
        ringOffsets: importGraphBuffer(
          graph,
          'ring-offsets',
          ringOffsetsBuffer,
          'uint32',
          geometry.ringOffsets.length
        ),
        polygonOffsets: importGraphBuffer(
          graph,
          'polygon-offsets',
          polygonOffsetsBuffer,
          'uint32',
          rowCount + 1
        ),
        weights,
        overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
      })
    );
    graph.add(
      new GPUSpatialWeightsMinimumSpanningTree({
        id: 'tree',
        weights,
        values,
        columnCount: COLUMN_COUNT,
        standardize,
        treeEdgeFlags: treeFlags,
        componentLabels,
        standardizedValues: standardize ? standardized : undefined,
        edges: {
          ids: importGraphBuffer(graph, 'edge-ids', edgeIdsBuffer, 'uint32', rowCount),
          count: importGraphBuffer(graph, 'edge-count', edgeCountBuffer, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'edge-overflow', edgeOverflowBuffer, 'uint32', 1)
        },
        edgeCosts: importGraphBuffer(graph, 'edge-costs', edgeCostsBuffer, 'float32', rowCount)
      })
    );
    // Without standardization the cuts and the evaluation use the raw table, as the tree does.
    const metric = standardize ? standardized : values;
    graph.add(
      new GPUSkaterRegions({
        id: 'skater',
        weights,
        treeEdgeFlags: treeFlags,
        componentLabels,
        values: metric,
        columnCount: COLUMN_COUNT,
        maximumRegionCount: MAXIMUM_REGIONS,
        parameters: skaterParameters.importToGraph(graph),
        labels,
        regionCount: importGraphBuffer(graph, 'region-count', regionCountBuffer, 'uint32', 1),
        cutEdges: importGraphBuffer(graph, 'cut-edges', cutEdgesBuffer, 'uint32', stepCount),
        cutGains: importGraphBuffer(graph, 'cut-gains', cutGainsBuffer, 'float32', stepCount)
      })
    );
    graph.add(
      new GPURegionPartitionEvaluation({
        id: 'skater-evaluation',
        values: metric,
        columnCount: COLUMN_COUNT,
        labels,
        weights,
        summary: importGraphBuffer(
          graph,
          'summary',
          summaryBuffer,
          'float32',
          GPU_REGION_PARTITION_EVALUATION_LAYOUT.length
        ),
        regionSizes: importGraphBuffer(
          graph,
          'region-sizes',
          regionSizesBuffer,
          'uint32',
          rowCount
        ),
        regionWithinSsd: importGraphBuffer(
          graph,
          'region-within-ssd',
          regionWithinBuffer,
          'float32',
          rowCount
        )
      })
    );
    compiled = resources.track(graph.compile());
    skaterGraphs.set(key, compiled);
    return compiled;
  };

  // The k-means evaluation scores with the standardized table; it is written by the MST graph, so
  // a raw-table scene scores both partitions on the raw metric by uploading the raw table once.
  const kmeansGraphs = new Map<string, CompiledGPUCommandGraph<void>>();
  const getKmeansGraph = (clusters: number, standardize: boolean) => {
    const key = `${clusters}:${standardize}`;
    let compiled = kmeansGraphs.get(key);
    if (compiled) return compiled;
    const graph = new GPUCommandGraph<void>(device, {
      id: `kmeans-${clusters}-${standardize ? 'z' : 'raw'}`
    });
    const labels = importGraphBuffer(graph, 'kmeans-labels', kLabelsBuffer, 'uint32', rowCount);
    graph.add(
      new GPUKMeans({
        id: 'kmeans',
        positions: importGraphBuffer(graph, 'principal-plane', planeBuffer, 'float32x2', rowCount),
        k: clusters,
        iterations: KMEANS_ITERATIONS,
        initialization: 'kmeans++',
        seed: 7,
        labels,
        centers: importGraphBuffer(graph, 'kmeans-centers', kCentersBuffer, 'float32x2', clusters),
        sizes: importGraphBuffer(graph, 'kmeans-sizes', kSizesBuffer, 'uint32', clusters),
        convergence: importGraphBuffer(graph, 'kmeans-convergence', kConvergenceBuffer, 'uint32', 2)
      })
    );
    graph.add(
      new GPURegionPartitionEvaluation({
        id: 'kmeans-evaluation',
        values: importGraphBuffer(
          graph,
          'metric',
          standardize ? standardizedBuffer : valuesBuffer,
          'float32',
          rowCount * COLUMN_COUNT
        ),
        columnCount: COLUMN_COUNT,
        labels,
        labelCapacity: MAXIMUM_KMEANS_CLUSTERS,
        weights: importWeights(graph),
        summary: importGraphBuffer(
          graph,
          'kmeans-summary',
          kSummaryBuffer,
          'float32',
          GPU_REGION_PARTITION_EVALUATION_LAYOUT.length
        ),
        regionSizes: importGraphBuffer(
          graph,
          'kmeans-region-sizes',
          kRegionSizesBuffer,
          'uint32',
          MAXIMUM_KMEANS_CLUSTERS
        ),
        regionWithinSsd: importGraphBuffer(
          graph,
          'kmeans-region-within',
          kRegionWithinBuffer,
          'float32',
          MAXIMUM_KMEANS_CLUSTERS
        )
      })
    );
    compiled = resources.track(graph.compile());
    kmeansGraphs.set(key, compiled);
    return compiled;
  };

  // ---- State. ----
  let activeCriterion = ctx.options.criterion;
  let activeStandardize = ctx.options.standardize;
  let activeClusters = ctx.options.kmeansClusters;
  let dirty = true;
  let weightsChanged = true;
  let adjacency: number[][] = Array.from({length: rowCount}, () => []);
  let csrReady = false;
  let latestLabels: Uint32Array = new Uint32Array(rowCount);
  let latestKLabels: Uint32Array = new Uint32Array(rowCount);
  let costRange: [number, number] = [0, 1];
  let treeEdges = 0;

  const writeInputs = () => {
    const enabled = REGION_ATTRIBUTES.map(({id}) => ctx.options[id]);
    if (!enabled.some(Boolean)) {
      // A constant table would make every cut worthless: fall back to the first attribute.
      enabled[0] = true;
    }
    const values = new Float32Array(rowCount * COLUMN_COUNT);
    for (let row = 0; row < rowCount; row++) {
      for (let columnIndex = 0; columnIndex < COLUMN_COUNT; columnIndex++) {
        values[row * COLUMN_COUNT + columnIndex] = enabled[columnIndex]
          ? table[row * COLUMN_COUNT + columnIndex]
          : 0;
      }
    }
    valuesBuffer.write(values);
    if (!ctx.options.standardize)
      standardizedBuffer.write(new Float32Array(rowCount * COLUMN_COUNT));
    const columns = enabled.flatMap((on, index) => (on ? [index] : []));
    planeBuffer.write(computePrincipalPlane(zTable, rowCount, columns));
    skaterParameters.write(new Uint32Array([ctx.options.regions, ctx.options.minimumSize]));
  };
  ctx.setReadout('rows', `${formatInteger(rowCount)} of ${formatInteger(featureCount)} tracts`);
  ctx.setStatus(`${formatInteger(rowCount)} Chicago tracts`);
  getSkaterGraph(activeCriterion, activeStandardize);
  getKmeansGraph(Number(activeClusters), activeStandardize);
  writeInputs();

  const csrReader = createNamedReader(
    resources,
    'regions-csr',
    [
      {name: 'offsets', buffer: offsetsBuffer, bytes: (rowCount + 1) * 4},
      {name: 'neighbors', buffer: neighborsBuffer, bytes: capacity * 4}
    ],
    get => {
      const offsets = get('offsets').u32;
      const neighbors = get('neighbors').u32;
      const segments = new Float32Array(capacity * 4);
      const lists: number[][] = Array.from({length: rowCount}, () => []);
      for (let row = 0; row < rowCount; row++) {
        for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
          const neighbor = neighbors[slot];
          lists[row].push(neighbor);
          segments[slot * 4] = centroids[row * 2];
          segments[slot * 4 + 1] = centroids[row * 2 + 1];
          segments[slot * 4 + 2] = centroids[neighbor * 2];
          segments[slot * 4 + 3] = centroids[neighbor * 2 + 1];
        }
      }
      slotSegmentsBuffer.write(segments);
      adjacency = lists;
      csrReady = true;
      reader.markStale();
      ctx.requestLayers();
    }
  );

  const reader = createNamedReader(
    resources,
    'regions-results',
    [
      {
        name: 'summary',
        buffer: summaryBuffer,
        bytes: GPU_REGION_PARTITION_EVALUATION_LAYOUT.length * 4
      },
      {name: 'labels', buffer: labelsBuffer, bytes: rowCount * 4},
      {name: 'regionCount', buffer: regionCountBuffer, bytes: 4},
      {name: 'cutEdges', buffer: cutEdgesBuffer, bytes: stepCount * 4},
      {name: 'cutGains', buffer: cutGainsBuffer, bytes: stepCount * 4},
      {name: 'edgeCount', buffer: edgeCountBuffer, bytes: 4},
      {name: 'edgeOverflow', buffer: edgeOverflowBuffer, bytes: 4},
      {name: 'edgeIds', buffer: edgeIdsBuffer, bytes: rowCount * 4},
      {name: 'edgeCosts', buffer: edgeCostsBuffer, bytes: rowCount * 4},
      {name: 'componentLabels', buffer: componentLabelsBuffer, bytes: rowCount * 4},
      {
        name: 'kSummary',
        buffer: kSummaryBuffer,
        bytes: GPU_REGION_PARTITION_EVALUATION_LAYOUT.length * 4
      },
      {name: 'kLabels', buffer: kLabelsBuffer, bytes: rowCount * 4},
      {name: 'kConvergence', buffer: kConvergenceBuffer, bytes: 8},
      {name: 'overflow', buffer: overflowBuffer, bytes: 4}
    ],
    get => {
      const layout = GPU_REGION_PARTITION_EVALUATION_LAYOUT;
      const summary = get('summary').f32;
      const kSummary = get('kSummary').f32;
      const labels = get('labels').u32;
      const regions = get('regionCount').u32[0];
      const cuts = get('cutEdges').u32;
      const gains = get('cutGains').f32;
      treeEdges = get('edgeCount').u32[0];
      const edgeIds = get('edgeIds').u32;
      const edgeCosts = get('edgeCosts').f32;
      const components = new Set(get('componentLabels').u32).size;
      latestLabels = labels;
      latestKLabels = get('kLabels').u32;
      const convergence = get('kConvergence').u32;

      const explained = (words: Float32Array) =>
        words[layout.totalSsd] > 0
          ? (100 * words[layout.betweenSsd]) / words[layout.totalSsd]
          : Number.NaN;
      const target = ctx.options.regions;
      ctx.setReadout(
        'regions',
        `${regions} (target ${target}${regions < target ? ', size floor or islands bind' : ''})`
      );
      ctx.setReadout(
        'explained',
        `SKATER ${formatNumber(explained(summary), 1)}% vs k-means ${formatNumber(explained(kSummary), 1)}% of variance`
      );
      ctx.setReadout(
        'within',
        `SKATER ${formatNumber(summary[layout.withinSsd], 0)} vs k-means ${formatNumber(kSummary[layout.withinSsd], 0)} (of ${formatNumber(summary[layout.totalSsd], 0)})`
      );
      ctx.setReadout(
        'sizes',
        `SKATER ${formatInteger(summary[layout.minimumSize])} to ${formatInteger(summary[layout.maximumSize])} tracts; k-means ${formatInteger(kSummary[layout.minimumSize])} to ${formatInteger(kSummary[layout.maximumSize])}`
      );
      ctx.setReadout(
        'boundary',
        `SKATER ${(100 * summary[layout.crossLinkFraction]).toFixed(1)}% vs k-means ${(100 * kSummary[layout.crossLinkFraction]).toFixed(1)}% of neighbour links`
      );
      const pieces = countPieces(latestKLabels, adjacency);
      const skaterPieces = countPieces(labels, adjacency);
      ctx.setReadout(
        'pieces',
        csrReady
          ? `SKATER ${skaterPieces.total} piece${skaterPieces.total === 1 ? '' : 's'} for ${skaterPieces.labelCount} regions; k-means ${pieces.total} pieces for ${pieces.labelCount} clusters`
          : null
      );
      ctx.setReadout(
        'tree',
        `${formatInteger(treeEdges)} edges in ${components} tree${components === 1 ? '' : 's'}${get('edgeOverflow').u32[0] ? ' (overflow)' : ''}`
      );
      ctx.setReadout(
        'kmeans',
        `${convergence[0]} iterations, ${convergence[1] ? 'converged' : 'not converged'}`
      );
      let applied = 0;
      const mask = new Uint32Array(capacity);
      for (let step = 0; step < stepCount; step++) {
        if (cuts[step] !== GPU_SKATER_NO_CUT) {
          mask[cuts[step]] = 1;
          applied = step + 1;
        }
      }
      cutMaskBuffer.write(mask);
      ctx.setReadout(
        'gains',
        applied > 0
          ? Array.from(gains.slice(0, Math.min(applied, 8)))
              .map(value => formatNumber(value, 0))
              .join(' / ') + (applied > 8 ? ' / …' : '')
          : 'no cuts'
      );
      // Tree edge costs per slot for the line layer.
      const slotCosts = new Float32Array(capacity).fill(Number.NaN);
      const finiteCosts: number[] = [];
      for (let index = 0; index < Math.min(treeEdges, rowCount); index++) {
        slotCosts[edgeIds[index]] = edgeCosts[index];
        finiteCosts.push(edgeCosts[index]);
      }
      slotCostBuffer.write(slotCosts);
      costRange = [0, Math.max(1e-6, getQuantile(finiteCosts, 0.95))];
      ctx.setLegendExtent('tree', costRange);
      if (csrReady) {
        regionColorsBuffer.write(assignRegionColors(labels, adjacency));
        clusterColorsBuffer.write(assignRegionColors(latestKLabels, adjacency));
      }
      ctx.setReadout('capacity', get('overflow').u32[0] ? 'overflow: raise slots per tract' : 'ok');
      ctx.requestLayers();
    }
  );

  const dark = () => ctx.theme() === 'dark';

  return {
    getCompiledGraphs: () => [
      getSkaterGraph(activeCriterion, activeStandardize),
      getKmeansGraph(Number(activeClusters), activeStandardize)
    ],

    setOption(id) {
      if (id === 'map' || id === 'showTree' || id === 'showCuts' || id === 'treeByCost') {
        ctx.requestLayers();
        return;
      }
      if (id === 'criterion') {
        activeCriterion = ctx.options.criterion;
        getSkaterGraph(activeCriterion, activeStandardize);
        weightsChanged = true;
      }
      if (id === 'standardize') {
        activeStandardize = ctx.options.standardize;
        getSkaterGraph(activeCriterion, activeStandardize);
        getKmeansGraph(Number(activeClusters), activeStandardize);
      }
      if (id === 'kmeansClusters') {
        activeClusters = ctx.options.kmeansClusters;
        getKmeansGraph(Number(activeClusters), activeStandardize);
      }
      writeInputs();
      dirty = true;
    },

    encode(commandEncoder, frame) {
      if (frame.frameIndex < 2) {
        dirty = true;
        weightsChanged = true;
      }
      if (dirty) {
        getSkaterGraph(activeCriterion, activeStandardize).encode(commandEncoder, {
          parameters: undefined
        });
        getKmeansGraph(Number(activeClusters), activeStandardize).encode(commandEncoder, {
          parameters: undefined
        });
        reader.request(commandEncoder);
        if (weightsChanged) {
          csrReader.request(commandEncoder);
          weightsChanged = false;
        }
        dirty = false;
      }
      csrReader.flush(commandEncoder);
      reader.flush(commandEncoder);
    },

    getLayers() {
      const isDark = dark();
      const {map, showTree, showCuts, treeByCost} = ctx.options;
      const layers: Layer[] = [
        new PolygonFillLayer({
          id: 'regions-excluded',
          triangles: excludedTrianglesBuffer,
          owners: excludedOwnersBuffer,
          triangleCount: excludedTriangles.triangleCount,
          colormap: 'uniform',
          color: isDark ? [90, 94, 104, 150] : [170, 172, 178, 150]
        }),
        new PolygonFillLayer({
          id: `regions-fill-${map}`,
          triangles: trianglesBuffer,
          owners: ownersBuffer,
          triangleCount: triangles.triangleCount,
          values: map === 'skater' ? regionColorsBuffer : clusterColorsBuffer,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: REGION_PALETTE,
          opacity: isDark ? 0.78 : 0.82
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'regions-outline',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          segments: outlineBuffer,
          instanceCount: outlineSegments.length / 4,
          widthPixels: 0.7,
          color: isDark ? [235, 238, 248, 60] : [30, 34, 46, 70]
        })
      ];
      if (csrReady && showTree) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'regions-tree',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: slotSegmentsBuffer,
            instanceCount: capacity,
            values: treeByCost ? slotCostBuffer : treeFlagsBuffer,
            valueFormat: treeByCost ? 'float32' : 'uint32',
            colormap: treeByCost ? 'inferno' : 'mask',
            valueRange: costRange,
            sqrtScale: treeByCost,
            color: [255, 255, 255, 240],
            noDataColor: [0, 0, 0, 0],
            widthPixels: 2
          })
        );
      }
      if (csrReady && showCuts) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'regions-cuts',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: slotSegmentsBuffer,
            instanceCount: capacity,
            values: cutMaskBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [230, 40, 40, 255],
            noDataColor: [0, 0, 0, 0],
            widthPixels: 5
          })
        );
      }
      return layers;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const row = locate(event.coordinate[0], event.coordinate[1]);
      if (row < 0) return null;
      const feature = features[row];
      const parts = REGION_ATTRIBUTES.map(
        ({id, label}) => `${label}: ${raw[id][feature].toFixed(id === 'income' ? 2 : 1)}`
      );
      return [
        `Tract ${geoids[feature]}`,
        `SKATER region ${latestLabels[row]}, k-means cluster ${latestKLabels[row]}`,
        ...parts.slice(0, 6)
      ].join('\n');
    },

    destroy() {
      csrReader.stop();
      reader.stop();
      void centroidBuffer;
      resources.destroy();
    }
  };
}
