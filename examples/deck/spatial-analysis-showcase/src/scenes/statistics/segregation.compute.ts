// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUNeighborSearchParameterValues,
  getGPUSegregationLayout,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPUNeighborSearch,
  GPUSegregation,
  GPUSpatialWeightsTransform,
  type GPUSpatialWeights,
  type GPUSpatialWeightsKernel
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createByteReader,
  createChoroplethGeometry,
  formatNumber,
  getOutlineColor,
  getQuantile,
  getSortedFinite
} from './b5-common';
import {clearLegendData} from './b5-legend-bus';

/** Population groups, in the column order of `groupCounts`. */
export const SEGREGATION_GROUPS = [
  {id: 'nhBlack', label: 'Black', color: [213, 94, 0, 235]},
  {id: 'nhWhite', label: 'White', color: [86, 180, 233, 235]},
  {id: 'hispanic', label: 'Hispanic', color: [0, 158, 115, 235]},
  {id: 'nhAsian', label: 'Asian', color: [240, 228, 66, 235]},
  {id: 'nhOther', label: 'Other or multiracial', color: [150, 150, 150, 235]}
] as const;

/** Option state of the segregation scene. */
export type SegregationOptions = {
  localIndex: 'dominant' | 'composition' | 'entropy' | 'dissimilarity' | 'theil';
  focalGroup: string;
  scale: number;
  bandwidth: number;
  weights: 'band' | GPUSpatialWeightsKernel;
  rowStandardize: boolean;
  spatialForm: 'environment' | 'smoothed-population';
  selfWeight: string;
  atkinsonB: string;
  outlines: boolean;
};

/** Ladder of bandwidths as multiples of the bandwidth slider. */
const BANDWIDTH_LADDER = [1, 2, 4, 8, 16] as const;
const SCALE_COUNT = BANDWIDTH_LADDER.length + 1;
const GROUP_COUNT = SEGREGATION_GROUPS.length;
const LOCAL_MODE = {composition: 0, entropy: 1, dissimilarity: 2, theil: 3, dominant: 4} as const;
const NO_GROUP = 0xffffffff;

type Variant = {key: string; compiled: CompiledGPUCommandGraph<void>};

/** Display range data shared with legends. */
export type SegregationRange = {range: [number, number]};

/**
 * Residential segregation of Chicago census tracts by race and ethnicity. A `GPUNeighborSearch`
 * distance band per scale writes a weights CSR, an optional `GPUSpatialWeightsTransform` turns it
 * into a kernel or row-standardises it, and one `GPUSegregation` reads every scale and writes the
 * global indices and the per-tract local terms. Slider changes are parameter writes; the weights
 * variant, self weight, spatial form and Atkinson parameter are compile-time and compile on demand.
 */
export async function createSegregation(
  ctx: SceneContext<SegregationOptions>
): Promise<SceneInstance<SegregationOptions>> {
  const tracts = ctx.datasets.get('chicago-tracts');
  const areas = ctx.datasets.get('chicago-community-areas');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'segregation');
  const geometry = createChoroplethGeometry(resources, tracts);
  const n = geometry.featureCount;
  const projection = tracts.getProjection();
  const centers = new Float32Array(n * 2);
  {
    // Population-free centroid: mean of the exterior vertices in local meters.
    const {vertices, ringOffsets, featureRingOffsets} = geometry.layout;
    for (let feature = 0; feature < n; feature++) {
      let sumX = 0;
      let sumY = 0;
      let count = 0;
      const first = ringOffsets[featureRingOffsets[feature]];
      const last = ringOffsets[featureRingOffsets[feature + 1]];
      for (let vertex = first; vertex < last; vertex++) {
        const [x, y] = projection.project(vertices[vertex * 2], vertices[vertex * 2 + 1]);
        sumX += x;
        sumY += y;
        count++;
      }
      centers[feature * 2] = sumX / Math.max(count, 1);
      centers[feature * 2 + 1] = sumY / Math.max(count, 1);
    }
  }
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let row = 0; row < n; row++) {
    minimumX = Math.min(minimumX, centers[row * 2]);
    maximumX = Math.max(maximumX, centers[row * 2]);
    minimumY = Math.min(minimumY, centers[row * 2 + 1]);
    maximumY = Math.max(maximumY, centers[row * 2 + 1]);
  }
  const margin = 2000;
  const searchBounds: [number, number, number, number] = [
    minimumX - margin,
    minimumY - margin,
    maximumX + margin,
    maximumY + margin
  ];

  const groupCountsHost = new Float32Array(n * GROUP_COUNT);
  const groupColumns = SEGREGATION_GROUPS.map(group => tracts.column<Float32Array>(group.id));
  const totals = new Float32Array(n);
  for (let row = 0; row < n; row++) {
    for (let group = 0; group < GROUP_COUNT; group++) {
      const value = groupColumns[group][row];
      const count = Number.isFinite(value) && value > 0 ? value : 0;
      groupCountsHost[row * GROUP_COUNT + group] = count;
      totals[row] += count;
    }
  }
  const tractFeatures = tracts.geojson?.features ?? [];
  const areaFeatures = areas.geojson?.features ?? [];
  const communityArea = tracts.column<Uint8Array>('communityArea');
  const describeTract = (row: number): string => {
    const properties = tractFeatures[row]?.properties;
    const area = areaFeatures[communityArea[row] - 1]?.properties?.name;
    const shares = SEGREGATION_GROUPS.map(
      (group, index) =>
        `${group.label} ${totals[row] > 0 ? ((100 * groupCountsHost[row * GROUP_COUNT + index]) / totals[row]).toFixed(0) : 0}%`
    ).join(', ');
    return [
      `Tract ${properties?.GEOID ?? row}${area ? `, ${area}` : ''}`,
      `${formatNumber(totals[row])} residents`,
      shares,
      latestDisplay && Number.isFinite(latestDisplay[row])
        ? `Map value: ${latestDisplay[row].toFixed(3)}`
        : ''
    ]
      .filter(Boolean)
      .join('\n');
  };

  const layout = getGPUSegregationLayout(GROUP_COUNT);
  const positionsBuffer = resources.createBuffer('positions', centers);
  const groupCounts = resources.createBuffer('group-counts', groupCountsHost);
  const indices = resources.createBuffer('indices', SCALE_COUNT * layout.stride * 4);
  const localEnvironment = resources.createBuffer(
    'local-environment',
    SCALE_COUNT * n * GROUP_COUNT * 4
  );
  const localEntropy = resources.createBuffer('local-entropy', SCALE_COUNT * n * 4);
  const localDissimilarity = resources.createBuffer(
    'local-dissimilarity',
    SCALE_COUNT * n * GROUP_COUNT * 4
  );
  const localTheil = resources.createBuffer('local-theil', SCALE_COUNT * n * 4);
  const display = resources.createBuffer('display', n * 4);
  const dominant = resources.createBuffer('dominant', n * 4);
  const displayParameter = resources.createParameterBuffer('display-parameter', 'uint32', 4);
  const scaleBuffers = BANDWIDTH_LADDER.map((_, scale) => {
    const capacity = n * n;
    return {
      capacity,
      searchParameter: resources.createParameterBuffer(
        `search-parameter-${scale}`,
        'float32',
        GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
      ),
      offsets: resources.createBuffer(`offsets-${scale}`, (n + 1) * 4),
      neighbors: resources.createBuffer(`neighbors-${scale}`, capacity * 4),
      weights: resources.createBuffer(`weights-${scale}`, capacity * 4),
      distances: resources.createBuffer(`distances-${scale}`, capacity * 4),
      overflow: resources.createBuffer(`overflow-${scale}`, 4)
    };
  });

  const importWeights = (graph: GPUCommandGraph<void>, scale: number): GPUSpatialWeights => {
    const buffers = scaleBuffers[scale];
    return {
      offsets: importGraphBuffer(graph, `offsets-${scale}`, buffers.offsets, 'uint32', n + 1),
      neighbors: importGraphBuffer(
        graph,
        `neighbors-${scale}`,
        buffers.neighbors,
        'uint32',
        buffers.capacity
      ),
      weights: importGraphBuffer(
        graph,
        `weights-${scale}`,
        buffers.weights,
        'float32',
        buffers.capacity
      ),
      distances: importGraphBuffer(
        graph,
        `distances-${scale}`,
        buffers.distances,
        'float32',
        buffers.capacity
      )
    };
  };

  // Search graph: one distance band per scale; the radius is a parameter write.
  const searchGraph = new GPUCommandGraph<void>(device, {id: 'segregation-search'});
  const positionsView = importGraphBuffer(
    searchGraph,
    'positions',
    positionsBuffer,
    'float32x2',
    n
  );
  BANDWIDTH_LADDER.forEach((_, scale) => {
    const buffers = scaleBuffers[scale];
    searchGraph.add(
      new GPUNeighborSearch({
        id: `search-${scale}`,
        mode: 'radius',
        gridSize: [32, 32],
        positions: positionsView,
        parameters: buffers.searchParameter.importToGraph(searchGraph),
        weights: importWeights(searchGraph, scale),
        overflow: importGraphBuffer(searchGraph, `overflow-${scale}`, buffers.overflow, 'uint32', 1)
      })
    );
  });
  const searchCompiled = resources.track(searchGraph.compile());

  const getKey = (options: SegregationOptions) =>
    [
      options.weights,
      options.rowStandardize ? 'row' : '-',
      options.spatialForm,
      options.selfWeight,
      options.atkinsonB
    ].join('|');

  const compileVariant = (options: SegregationOptions): Variant => {
    const key = getKey(options);
    const graph = new GPUCommandGraph<void>(device, {id: `segregation-${key}`});
    const scaleWeights: (GPUSpatialWeights | null)[] = [null];
    BANDWIDTH_LADDER.forEach((_, scale) => {
      const weights = importWeights(graph, scale);
      if (options.weights !== 'band') {
        graph.add(
          new GPUSpatialWeightsTransform({
            id: `kernel-${scale}`,
            operation: 'kernel',
            weights,
            kernel: options.weights,
            bandwidth: 'adaptive'
          })
        );
      }
      if (options.rowStandardize) {
        graph.add(new GPUSpatialWeightsTransform({id: `row-${scale}`, operation: 'row', weights}));
      }
      scaleWeights.push(weights);
    });
    const local = (
      name: string,
      buffer: typeof localEntropy,
      length: number
    ): GraphDataView<'float32'> => importGraphBuffer(graph, name, buffer, 'float32', length);
    graph.add(
      new GPUSegregation({
        id: 'segregation',
        unitCount: n,
        groupCount: GROUP_COUNT,
        groupCounts: importGraphBuffer(
          graph,
          'group-counts',
          groupCounts,
          'float32',
          n * GROUP_COUNT
        ),
        scales: scaleWeights,
        selfWeight: Number(options.selfWeight),
        spatialForm: options.spatialForm,
        atkinsonB: Number(options.atkinsonB),
        indices: local('indices', indices, SCALE_COUNT * layout.stride),
        local: {
          environment: local('local-environment', localEnvironment, SCALE_COUNT * n * GROUP_COUNT),
          entropy: local('local-entropy', localEntropy, SCALE_COUNT * n),
          dissimilarity: local(
            'local-dissimilarity',
            localDissimilarity,
            SCALE_COUNT * n * GROUP_COUNT
          ),
          theil: local('local-theil', localTheil, SCALE_COUNT * n)
        }
      })
    );
    return {key, compiled: resources.track(graph.compile())};
  };

  const variants = new Map<string, Variant>();
  const getVariant = (options: SegregationOptions): Variant => {
    const key = getKey(options);
    let variant = variants.get(key);
    if (!variant) {
      variant = compileVariant(options);
      variants.set(key, variant);
    }
    return variant;
  };

  // Display graph: the chosen local index of the chosen scale and group, one value per tract.
  const displayGraph = new GPUCommandGraph<void>(device, {id: 'segregation-display'});
  const importDisplay = (name: string, buffer: typeof display, length: number) =>
    importGraphBuffer(displayGraph, name, buffer, 'float32', length);
  addKernelPass(displayGraph, {
    id: 'segregation-display',
    invocationCount: n,
    bindings: [
      {
        name: 'parameters',
        view: displayParameter.importToGraph(displayGraph),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'counts',
        view: importDisplay('group-counts', groupCounts, n * GROUP_COUNT),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'environment',
        view: importDisplay('local-environment', localEnvironment, SCALE_COUNT * n * GROUP_COUNT),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'entropy',
        view: importDisplay('local-entropy', localEntropy, SCALE_COUNT * n),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'dissimilarity',
        view: importDisplay(
          'local-dissimilarity',
          localDissimilarity,
          SCALE_COUNT * n * GROUP_COUNT
        ),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'theil',
        view: importDisplay('local-theil', localTheil, SCALE_COUNT * n),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'display',
        view: importDisplay('display', display, n),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'dominant',
        view: importGraphBuffer(displayGraph, 'dominant', dominant, 'uint32', n),
        type: 'u32',
        access: 'read_write'
      }
    ],
    declarations: `const UNITS: u32 = ${n}u;\nconst GROUPS: u32 = ${GROUP_COUNT}u;`,
    body: /* wgsl */ `
  let mode = parameters[parametersOffset];
  let scale = parameters[parametersOffset + 1u];
  let group = parameters[parametersOffset + 2u];
  var total = 0.0;
  var best = 0u;
  var bestCount = -1.0;
  for (var member = 0u; member < GROUPS; member++) {
    let count = counts[countsOffset + index * GROUPS + member];
    total += count;
    if (count > bestCount) {
      bestCount = count;
      best = member;
    }
  }
  // Keeps the NaN pattern a runtime value because WGSL rejects a constant NaN.
  var value = bitcast<f32>(0x7fc00000u | (parameters[parametersOffset + 3u] >> 31u));
  var winner = ${NO_GROUP}u;
  if (total > 0.0) {
    winner = best;
    let row = scale * UNITS + index;
    if (mode == 0u) {
      value = environment[environmentOffset + row * GROUPS + group];
    } else if (mode == 1u) {
      value = entropy[entropyOffset + row];
    } else if (mode == 2u) {
      value = dissimilarity[dissimilarityOffset + row * GROUPS + group];
    } else if (mode == 3u) {
      value = theil[theilOffset + row];
    } else {
      value = f32(best);
    }
  }
  display[displayOffset + index] = value;
  dominant[dominantOffset + index] = winner;`
  });
  const displayCompiled = resources.track(displayGraph.compile());

  let active = getVariant(ctx.options);
  let analysisDirty = true;
  let displayDirty = true;
  let latestDisplay: Float32Array | null = null;
  let selectedRow = -1;
  let mapRange: [number, number] = [0, 1];

  const getFocalIndex = () =>
    Math.max(
      0,
      SEGREGATION_GROUPS.findIndex(group => group.id === ctx.options.focalGroup)
    );
  const getBandwidthMeters = (ladderIndex: number) =>
    BANDWIDTH_LADDER[ladderIndex] * ctx.options.bandwidth * 1000;

  const writeSearchParameters = () => {
    scaleBuffers.forEach((buffers, index) => {
      buffers.searchParameter.write(
        getGPUNeighborSearchParameterValues({
          bounds: searchBounds,
          radius: getBandwidthMeters(index),
          weightKind: 'binary'
        })
      );
    });
    analysisDirty = true;
    reader.markStale();
  };
  const writeDisplayParameter = () => {
    const {localIndex, scale} = ctx.options;
    displayParameter.write(Uint32Array.of(LOCAL_MODE[localIndex], scale, getFocalIndex(), 0));
    displayDirty = true;
    reader.markStale();
  };
  const decimal = (value: number) => value.toFixed(3);
  const formatScale = (scale: number) =>
    scale === 0 ? 'aspatial' : `${(getBandwidthMeters(scale - 1) / 1000).toFixed(1)} km`;

  const reader = new SummaryReader(
    resources,
    'segregation',
    [
      {buffer: indices, size: SCALE_COUNT * layout.stride * 4},
      ...scaleBuffers.map(buffers => ({buffer: buffers.overflow, size: 4})),
      {buffer: display, size: n * 4},
      {buffer: localTheil, size: SCALE_COUNT * n * 4},
      {buffer: localDissimilarity, size: SCALE_COUNT * n * GROUP_COUNT * 4}
    ],
    bytes => {
      const read = createByteReader(bytes);
      const indexValues = read.floats(SCALE_COUNT * layout.stride);
      const flags = read.words(BANDWIDTH_LADDER.length);
      const displayValues = read.floats(n);
      const theil = read.floats(SCALE_COUNT * n);
      const dissimilarity = read.floats(SCALE_COUNT * n * GROUP_COUNT);
      const {scale, localIndex} = ctx.options;
      const g = getFocalIndex();
      const focal = SEGREGATION_GROUPS[g];
      const column = (scaleIndex: number, field: number) =>
        indexValues[scaleIndex * layout.stride + field];
      ctx.setReadout(
        'scaleLabel',
        scale === 0
          ? 'aspatial: each tract on its own'
          : `environment within ${formatScale(scale)} of each tract`
      );
      ctx.setReadout(
        'entropy',
        `H ${decimal(column(scale, layout.entropy))} · D (multigroup) ${decimal(column(scale, layout.multiGroupDissimilarity))} · diversity ${decimal(column(scale, layout.diversity))} nats`
      );
      ctx.setReadout(
        'group',
        `${focal.label}: D ${decimal(column(scale, layout.dissimilarity + g))} · isolation ${decimal(column(scale, layout.isolation + g))} · Atkinson ${decimal(column(scale, layout.atkinson + g))}`
      );
      ctx.setReadout(
        'interaction',
        SEGREGATION_GROUPS.map((group, h) =>
          h === g
            ? null
            : `${group.label} ${decimal(column(scale, layout.interaction + g * GROUP_COUNT + h))}`
        )
          .filter(Boolean)
          .join(' · ')
      );
      ctx.setReadout(
        'profile',
        Array.from(
          {length: SCALE_COUNT},
          (_, index) =>
            `${formatScale(index)} D=${column(index, layout.multiGroupDissimilarity).toFixed(2)}`
        ).join(' → ')
      );
      ctx.setReadout(
        'profileGroup',
        Array.from(
          {length: SCALE_COUNT},
          (_, index) =>
            `${formatScale(index)} ${column(index, layout.dissimilarity + g).toFixed(2)}`
        ).join(' → ')
      );
      let localTheilSum = 0;
      let localDissimilaritySum = 0;
      for (let unit = 0; unit < n; unit++) {
        localTheilSum += theil[scale * n + unit];
        localDissimilaritySum += dissimilarity[(scale * n + unit) * GROUP_COUNT + g];
      }
      ctx.setReadout(
        'localSum',
        `local H sums to ${decimal(localTheilSum)} (global ${decimal(column(scale, layout.entropy))}); local D(${focal.label}) sums to ${decimal(localDissimilaritySum)} (global ${decimal(column(scale, layout.dissimilarity + g))})`
      );
      ctx.setReadout(
        'overflow',
        flags.some(flag => flag) ? `YES in scales ${Array.from(flags).join(', ')}` : 'no'
      );
      latestDisplay = displayValues.slice();
      if (localIndex === 'composition') mapRange = [0, 1];
      else if (localIndex !== 'dominant') {
        const sorted = getSortedFinite(displayValues);
        if (localIndex === 'theil') {
          const limit = Math.max(
            Math.abs(getQuantile(sorted, 0.02)),
            Math.abs(getQuantile(sorted, 0.98)),
            1e-6
          );
          mapRange = [-limit, limit];
        } else {
          mapRange = [getQuantile(sorted, 0.02), getQuantile(sorted, 0.98)];
          if (!(mapRange[1] > mapRange[0])) mapRange = [mapRange[0], mapRange[0] + 1e-6];
        }
        ctx.setLegendExtent('local', mapRange);
      }
      ctx.requestLayers();
    }
  );

  ctx.setReadout('tracts', n);
  writeSearchParameters();
  writeDisplayParameter();

  return {
    getCompiledGraphs: () => [
      searchCompiled,
      ...[...variants.values()].map(v => v.compiled),
      displayCompiled
    ],

    setOption(id) {
      if (['weights', 'rowStandardize', 'spatialForm', 'selfWeight', 'atkinsonB'].includes(id)) {
        active = getVariant(ctx.options);
        analysisDirty = true;
        reader.markStale();
      } else if (id === 'bandwidth') {
        writeSearchParameters();
      } else if (['localIndex', 'focalGroup', 'scale'].includes(id)) {
        writeDisplayParameter();
      }
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (analysisDirty || frame.frameIndex < 2) {
        searchCompiled.encode(commandEncoder, {parameters: undefined});
        active.compiled.encode(commandEncoder, {parameters: undefined});
        analysisDirty = false;
        displayDirty = true;
      }
      if (displayDirty || frame.frameIndex < 2) {
        displayCompiled.encode(commandEncoder, {parameters: undefined});
        displayDirty = false;
        reader.markStale();
      }
      if (frame.frameIndex >= 2) reader.flush(commandEncoder);
    },

    getLayers() {
      const {localIndex, outlines} = ctx.options;
      const layers: Layer[] = [];
      if (localIndex === 'dominant') {
        layers.push(
          geometry.createFillLayer('segregation-dominant', {
            values: dominant,
            mode: 'category',
            palette: SEGREGATION_GROUPS.map(group => group.color),
            noDataColor: [128, 128, 128, 60],
            selectedRow,
            fillOpacity: 0.88
          })
        );
      } else {
        layers.push(
          geometry.createFillLayer('segregation-local', {
            values: display,
            mode: 'ramp',
            ramp:
              localIndex === 'theil' ? 'diverging' : localIndex === 'entropy' ? 'viridis' : 'magma',
            valueRange: mapRange,
            noDataColor: [128, 128, 128, 60],
            selectedRow,
            fillOpacity: 0.9
          })
        );
      }
      layers.push(
        geometry.createOutlineLayer(
          'segregation-outline',
          getOutlineColor(ctx.theme(), outlines ? 110 : 40),
          outlines ? 0.9 : 0.4
        )
      );
      return layers;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const row = geometry.locator.locate(event.coordinate[0], event.coordinate[1]);
      return row >= 0 ? describeTract(row) : null;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const row = geometry.locator.locate(event.coordinate[0], event.coordinate[1]);
      selectedRow = row === selectedRow ? -1 : row;
      ctx.setReadout(
        'selected',
        selectedRow >= 0 ? describeTract(selectedRow).replace(/\n/g, ' | ') : null
      );
      ctx.requestLayers();
      return true;
    },

    destroy() {
      reader.stop();
      clearLegendData('segregation');
      resources.destroy();
    }
  };
}
