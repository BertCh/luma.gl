// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Segregation: residential segregation indices of three population groups over a 30-column grid of
 * San Francisco units, aspatial and at five spatial scales (a multiscale profile).
 *
 * The group counts are SYNTHETIC: seeded smooth composition fields over a population that follows
 * the ZIP-code land area. A "segregation strength" slider rewrites the counts buffer (a per-frame
 * write); at 0 the groups are spatially uniform apart from noise.
 *
 * Per scale, `GPUNeighborSearch` writes a distance-band weights CSR (radius is a per-frame
 * parameter) and `GPUSpatialWeightsTransform` optionally turns it into an Epanechnikov kernel or
 * row-standardizes it. `GPUSegregation` reads all scales in one graph and writes the global
 * indices (entropy H, multigroup D, per-group D, isolation, interaction, Atkinson) and per-unit
 * local indices. The weights variant is a compile-time choice and rebuilds the analysis graph; every
 * slider is a buffer write. Readouts check that the local terms add up to the global indices.
 */

import type {Layer} from '@deck.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUNeighborSearchParameterValues,
  getGPUSegregationLayout,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPUNeighborSearch,
  GPUSegregation,
  GPUSpatialWeightsTransform,
  type GPUSpatialWeights
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {createSeededRandom} from '../spatial-analysis-data';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance,
  SpatialAnalysisPointerEvent
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {createZipLocator, getPolygonBounds} from './areal-interpolation-layers';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';

const GRID_COLUMNS = 30;
const GROUP_COUNT = 3;
const GROUP_NAMES = ['Group A', 'Group B', 'Group C'] as const;
const GROUP_COLORS = [
  [78, 201, 255, 215],
  [255, 148, 72, 215],
  [189, 122, 255, 215]
] as const;
/** Bandwidth ladder in units of the bandwidth slider (cell widths). */
const BANDWIDTH_LADDER = [1, 2, 3, 5, 8] as const;
/** Largest value of the bandwidth slider, which sizes the slot capacities. */
const MAXIMUM_BANDWIDTH_UNIT = 1.5;
/** Scale 0 is aspatial; scales 1..5 follow the ladder. */
const SCALE_COUNT = BANDWIDTH_LADDER.length + 1;
const HIDDEN = -1e9;
const NO_GROUP = 0xffffffff;

type WeightsVariant = 'band' | 'kernel' | 'row';
type LocalIndex = 'composition' | 'dominant' | 'entropy' | 'dissimilarity' | 'theil';

const LOCAL_MODE: Record<LocalIndex, number> = {
  composition: 0,
  entropy: 1,
  dissimilarity: 2,
  theil: 3,
  dominant: 4
};

/** Segregation demo: aspatial and multiscale segregation indices of synthetic group counts. */
export const segregationMode: SpatialAnalysisModeDefinition = {
  id: 'segregation',
  title: 'Segregation',
  contributors: ['GPUSegregation', 'GPUNeighborSearch', 'GPUSpatialWeightsTransform'],
  description:
    'Segregation of three synthetic population groups on a San Francisco grid. Pick the aspatial ' +
    'or a spatial scale, change the segregation strength or the bandwidth, and compare the local ' +
    'index map with the global indices and the multiscale profile.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.6},

  async create(context) {
    const zips = await context.data.getSanFranciscoZipCodes();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'segregation');
    const projection = new LocalMetricProjection(zips.origin);
    const locate = createZipLocator(zips);

    // --- Units: a square grid over the ZIP bounds (CPU setup) ----------------------------------------
    const bounds = getPolygonBounds(zips);
    const cellSize = Math.ceil((bounds[2] - bounds[0]) / GRID_COLUMNS);
    const columns = GRID_COLUMNS;
    const rows = Math.ceil((bounds[3] - bounds[1]) / cellSize);
    const unitCount = columns * rows;
    const gridBounds: [number, number, number, number] = [
      bounds[0],
      bounds[1],
      bounds[0] + columns * cellSize,
      bounds[1] + rows * cellSize
    ];
    const centers = new Float32Array(unitCount * 2);
    const landFraction = new Float32Array(unitCount);
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const unit = row * columns + column;
        centers[unit * 2] = gridBounds[0] + (column + 0.5) * cellSize;
        centers[unit * 2 + 1] = gridBounds[1] + (row + 0.5) * cellSize;
        let inside = 0;
        for (let sampleY = 0; sampleY < 3; sampleY++) {
          for (let sampleX = 0; sampleX < 3; sampleX++) {
            const x = gridBounds[0] + (column + (sampleX + 0.5) / 3) * cellSize;
            const y = gridBounds[1] + (row + (sampleY + 0.5) / 3) * cellSize;
            if (locate(x, y) >= 0) inside++;
          }
        }
        landFraction[unit] = inside / 9;
      }
    }

    // --- Synthetic groups: smooth seeded composition fields ----------------------------------------------
    const random = createSeededRandom(20261004);
    const width = gridBounds[2] - gridBounds[0];
    const height = gridBounds[3] - gridBounds[1];
    const blobs = Array.from({length: GROUP_COUNT}, () =>
      Array.from({length: 3}, () => ({
        x: gridBounds[0] + random() * width,
        y: gridBounds[1] + random() * height,
        radius: (0.18 + random() * 0.22) * Math.max(width, height),
        amplitude: random() > 0.4 ? 1 : -0.6
      }))
    );
    const field = (group: number, x: number, y: number): number =>
      blobs[group].reduce((sum, blob) => {
        const distance2 = (x - blob.x) ** 2 + (y - blob.y) ** 2;
        return sum + blob.amplitude * Math.exp(-distance2 / (2 * blob.radius * blob.radius));
      }, 0);
    const populationBlobs = Array.from({length: 4}, () => ({
      x: gridBounds[0] + random() * width,
      y: gridBounds[1] + random() * height,
      radius: (0.12 + random() * 0.15) * Math.max(width, height)
    }));
    const noise = Float32Array.from({length: unitCount * GROUP_COUNT}, () => 0.8 + random() * 0.4);
    const basePopulation = new Float32Array(unitCount);
    const fields = new Float32Array(unitCount * GROUP_COUNT);
    for (let unit = 0; unit < unitCount; unit++) {
      const x = centers[unit * 2];
      const y = centers[unit * 2 + 1];
      const density = populationBlobs.reduce(
        (sum, blob) =>
          sum + Math.exp(-((x - blob.x) ** 2 + (y - blob.y) ** 2) / (2 * blob.radius ** 2)),
        0.35
      );
      basePopulation[unit] = Math.round(2600 * landFraction[unit] * density);
      for (let group = 0; group < GROUP_COUNT; group++) {
        fields[unit * GROUP_COUNT + group] = field(group, x, y);
      }
    }
    const groupBias = [0.9, 0.3, 0];
    const groupCountsHost = new Float32Array(unitCount * GROUP_COUNT);
    let landUnits = 0;
    for (let unit = 0; unit < unitCount; unit++) if (basePopulation[unit] > 0) landUnits++;
    const writeGroupCounts = (strength: number): void => {
      for (let unit = 0; unit < unitCount; unit++) {
        let normalizer = 0;
        const weights = [0, 0, 0];
        for (let group = 0; group < GROUP_COUNT; group++) {
          weights[group] = Math.exp(
            groupBias[group] + strength * 2.4 * fields[unit * GROUP_COUNT + group]
          );
          normalizer += weights[group];
        }
        for (let group = 0; group < GROUP_COUNT; group++) {
          groupCountsHost[unit * GROUP_COUNT + group] = Math.round(
            (basePopulation[unit] * weights[group] * noise[unit * GROUP_COUNT + group]) / normalizer
          );
        }
      }
    };

    // --- Buffers -------------------------------------------------------------------------------------
    const layout = getGPUSegregationLayout(GROUP_COUNT);
    const positions = resources.createBuffer('positions', centers);
    const outlineSegmentsBuffer = resources.createBuffer('outline-segments', zips.outlineSegments);
    const groupCounts = resources.createBuffer('group-counts', unitCount * GROUP_COUNT * 4);
    const indices = resources.createBuffer('indices', SCALE_COUNT * layout.stride * 4);
    const localEnvironment = resources.createBuffer(
      'local-environment',
      SCALE_COUNT * unitCount * GROUP_COUNT * 4
    );
    const localEntropy = resources.createBuffer('local-entropy', SCALE_COUNT * unitCount * 4);
    const localDissimilarity = resources.createBuffer(
      'local-dissimilarity',
      SCALE_COUNT * unitCount * GROUP_COUNT * 4
    );
    const localTheil = resources.createBuffer('local-theil', SCALE_COUNT * unitCount * 4);
    const display = resources.createBuffer('display', unitCount * 4);
    const dominant = resources.createBuffer('dominant', unitCount * 4);
    const displayParameter = resources.createParameterBuffer('display-parameter', 'uint32', 4);
    const scaleBuffers = BANDWIDTH_LADDER.map((multiple, scale) => {
      const capacityPerRow = Math.min(
        unitCount,
        Math.ceil(Math.PI * (multiple * MAXIMUM_BANDWIDTH_UNIT + 1) ** 2)
      );
      const capacity = unitCount * capacityPerRow;
      return {
        capacity,
        searchParameter: resources.createParameterBuffer(
          `search-parameter-${scale}`,
          'float32',
          GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
        ),
        offsets: resources.createBuffer(`offsets-${scale}`, (unitCount + 1) * 4),
        neighbors: resources.createBuffer(`neighbors-${scale}`, capacity * 4),
        weights: resources.createBuffer(`weights-${scale}`, capacity * 4),
        distances: resources.createBuffer(`distances-${scale}`, capacity * 4),
        overflow: resources.createBuffer(`overflow-${scale}`, 4)
      };
    });

    // --- Graphs ------------------------------------------------------------------------------------------
    const buildAnalysisGraph = (variant: WeightsVariant): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {id: `segregation-${variant}`});
      const scaleWeights: (GPUSpatialWeights | null)[] = [null];
      const positionsView = importGraphBuffer(
        graph,
        'positions',
        positions,
        'float32x2',
        unitCount
      );
      BANDWIDTH_LADDER.forEach((_, scale) => {
        const buffers = scaleBuffers[scale];
        const weights: GPUSpatialWeights = {
          offsets: importGraphBuffer(
            graph,
            `offsets-${scale}`,
            buffers.offsets,
            'uint32',
            unitCount + 1
          ),
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
        graph.add(
          new GPUNeighborSearch({
            id: `search-${scale}`,
            mode: 'radius',
            gridSize: [32, 32],
            positions: positionsView,
            parameters: buffers.searchParameter.importToGraph(graph),
            weights,
            overflow: importGraphBuffer(graph, `overflow-${scale}`, buffers.overflow, 'uint32', 1)
          })
        );
        if (variant === 'kernel') {
          graph.add(
            new GPUSpatialWeightsTransform({
              id: `kernel-${scale}`,
              operation: 'kernel',
              weights,
              kernel: 'epanechnikov',
              bandwidth: 'adaptive'
            })
          );
        } else if (variant === 'row') {
          graph.add(
            new GPUSpatialWeightsTransform({id: `row-${scale}`, operation: 'row', weights})
          );
        }
        scaleWeights.push(weights);
      });
      const importLocal = (
        name: string,
        buffer: typeof localEntropy,
        length: number
      ): GraphDataView<'float32'> => importGraphBuffer(graph, name, buffer, 'float32', length);
      graph.add(
        new GPUSegregation({
          id: 'segregation',
          unitCount,
          groupCount: GROUP_COUNT,
          groupCounts: importGraphBuffer(
            graph,
            'group-counts',
            groupCounts,
            'float32',
            unitCount * GROUP_COUNT
          ),
          scales: scaleWeights,
          indices: importLocal('indices', indices, SCALE_COUNT * layout.stride),
          local: {
            environment: importLocal(
              'local-environment',
              localEnvironment,
              SCALE_COUNT * unitCount * GROUP_COUNT
            ),
            entropy: importLocal('local-entropy', localEntropy, SCALE_COUNT * unitCount),
            dissimilarity: importLocal(
              'local-dissimilarity',
              localDissimilarity,
              SCALE_COUNT * unitCount * GROUP_COUNT
            ),
            theil: importLocal('local-theil', localTheil, SCALE_COUNT * unitCount)
          }
        })
      );
      return resources.track(graph.compile());
    };

    // Display: the selected local index of the selected scale and group, one value per unit.
    const displayGraph = new GPUCommandGraph<void>(device, {id: 'segregation-display'});
    const importDisplay = (name: string, buffer: typeof display, length: number) =>
      importGraphBuffer(displayGraph, name, buffer, 'float32', length);
    addKernelPass(displayGraph, {
      id: 'segregation-display',
      invocationCount: unitCount,
      bindings: [
        {
          name: 'parameters',
          view: displayParameter.importToGraph(displayGraph),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'counts',
          view: importDisplay('group-counts', groupCounts, unitCount * GROUP_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'environment',
          view: importDisplay(
            'local-environment',
            localEnvironment,
            SCALE_COUNT * unitCount * GROUP_COUNT
          ),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'entropy',
          view: importDisplay('local-entropy', localEntropy, SCALE_COUNT * unitCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'dissimilarity',
          view: importDisplay(
            'local-dissimilarity',
            localDissimilarity,
            SCALE_COUNT * unitCount * GROUP_COUNT
          ),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'theil',
          view: importDisplay('local-theil', localTheil, SCALE_COUNT * unitCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'display',
          view: importDisplay('display', display, unitCount),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'dominant',
          view: importGraphBuffer(displayGraph, 'dominant', dominant, 'uint32', unitCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      declarations: `const UNITS: u32 = ${unitCount}u;\nconst GROUPS: u32 = ${GROUP_COUNT}u;`,
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
  var value = ${HIDDEN.toFixed(1)};
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

    // --- State -----------------------------------------------------------------------------------------------
    let variant: WeightsVariant = 'band';
    let analysis = buildAnalysisGraph(variant);
    let strength = 0.6;
    let bandwidthUnit = 1;
    let localIndex: LocalIndex = 'dissimilarity';
    let scale = 3;
    let focalGroup = 0;
    let analysisDirty = true;
    let displayDirty = true;
    let displayMinimum = 0;
    let displayMaximum = 1;
    let destroyed = false;

    const writeSearchParameters = (): void => {
      scaleBuffers.forEach((buffers, index) => {
        buffers.searchParameter.write(
          getGPUNeighborSearchParameterValues({
            bounds: [
              gridBounds[0] - cellSize,
              gridBounds[1] - cellSize,
              gridBounds[2] + cellSize,
              gridBounds[3] + cellSize
            ],
            radius: BANDWIDTH_LADDER[index] * bandwidthUnit * cellSize,
            weightKind: 'binary'
          })
        );
      });
    };
    const writeDisplayParameter = (): void => {
      displayParameter.write(Uint32Array.of(LOCAL_MODE[localIndex], scale, focalGroup, 0));
    };
    const formatBandwidth = (index: number): string =>
      index === 0
        ? 'aspatial'
        : `${((BANDWIDTH_LADDER[index - 1] * bandwidthUnit * cellSize) / 1000).toFixed(1)} km`;
    const decimal = (value: number): string => value.toFixed(3);

    // Readback layout: indices, overflow flags, display, local Theil, local dissimilarity.
    const summaryReader = new SummaryReader(
      resources,
      'segregation',
      [
        {buffer: indices, size: SCALE_COUNT * layout.stride * 4},
        ...scaleBuffers.map(buffers => ({buffer: buffers.overflow, size: 4})),
        {buffer: display, size: unitCount * 4},
        {buffer: localTheil, size: SCALE_COUNT * unitCount * 4},
        {buffer: localDissimilarity, size: SCALE_COUNT * unitCount * GROUP_COUNT * 4}
      ],
      bytes => {
        let offset = 0;
        const indexValues = new Float32Array(bytes, offset, SCALE_COUNT * layout.stride);
        offset += SCALE_COUNT * layout.stride * 4;
        const flags = new Uint32Array(bytes, offset, BANDWIDTH_LADDER.length);
        offset += BANDWIDTH_LADDER.length * 4;
        const displayValues = new Float32Array(bytes, offset, unitCount);
        offset += unitCount * 4;
        const theil = new Float32Array(bytes, offset, SCALE_COUNT * unitCount);
        offset += SCALE_COUNT * unitCount * 4;
        const dissimilarity = new Float32Array(
          bytes,
          offset,
          SCALE_COUNT * unitCount * GROUP_COUNT
        );
        const column = (scaleIndex: number, field: number): number =>
          indexValues[scaleIndex * layout.stride + field];
        const g = focalGroup;
        entropyReadout.setValue(
          `H ${decimal(column(scale, layout.entropy))}   D (multigroup) ${decimal(column(scale, layout.multiGroupDissimilarity))}   diversity ${decimal(column(scale, layout.diversity))} nats`
        );
        groupReadout.setValue(
          `D ${decimal(column(scale, layout.dissimilarity + g))}   isolation ${decimal(column(scale, layout.isolation + g))}   Atkinson(b=0.5) ${decimal(column(scale, layout.atkinson + g))}`
        );
        interactionReadout.setValue(
          GROUP_NAMES.map((name, h) =>
            h === g
              ? null
              : `${name} ${decimal(column(scale, layout.interaction + g * GROUP_COUNT + h))}`
          )
            .filter(Boolean)
            .join('   ')
        );
        profileReadout.setValue(
          Array.from(
            {length: SCALE_COUNT},
            (_, index) =>
              `${formatBandwidth(index)}: D ${decimal(column(index, layout.multiGroupDissimilarity))}, H ${decimal(column(index, layout.entropy))}`
          ).join('\n')
        );
        let localTheilSum = 0;
        let localDissimilaritySum = 0;
        for (let unit = 0; unit < unitCount; unit++) {
          localTheilSum += theil[scale * unitCount + unit];
          localDissimilaritySum += dissimilarity[(scale * unitCount + unit) * GROUP_COUNT + g];
        }
        localSumReadout.setValue(
          `sum of local H ${decimal(localTheilSum)} vs H ${decimal(column(scale, layout.entropy))}; ` +
            `sum of local D(${GROUP_NAMES[g]}) ${decimal(localDissimilaritySum)} vs ${decimal(column(scale, layout.dissimilarity + g))}`
        );
        overflowReadout.setValue(
          flags.some(flag => flag) ? `YES (scales ${Array.from(flags).join(', ')})` : 'no'
        );
        let minimum = Infinity;
        let maximum = -Infinity;
        for (const value of displayValues) {
          if (value > HIDDEN / 2) {
            minimum = Math.min(minimum, value);
            maximum = Math.max(maximum, value);
          }
        }
        if (Number.isFinite(minimum)) {
          displayMinimum = minimum;
          displayMaximum = maximum > minimum ? maximum : minimum + 1e-6;
          legendReadout.setValue(`${decimal(displayMinimum)} to ${decimal(displayMaximum)}`);
        }
        context.updateLayers();
      }
    );

    // --- Controls --------------------------------------------------------------------------------------------
    context.controls.addSelect<LocalIndex>({
      label: 'Map: local index',
      options: [
        {value: 'dissimilarity', label: 'Contribution to D (focal group)'},
        {value: 'theil', label: 'Contribution to H (entropy index)'},
        {value: 'entropy', label: 'Local entropy of the environment'},
        {value: 'composition', label: 'Environment share of focal group'},
        {value: 'dominant', label: 'Dominant group (unit counts)'}
      ],
      value: localIndex,
      onChange: value => {
        localIndex = value;
        writeDisplayParameter();
        displayDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addSelect<string>({
      label: 'Scale (aspatial or bandwidth)',
      options: Array.from({length: SCALE_COUNT}, (_, index) => ({
        value: String(index),
        label:
          index === 0
            ? 'Aspatial (unit compositions)'
            : `Spatial: ${BANDWIDTH_LADDER[index - 1]} x bandwidth unit`
      })),
      value: String(scale),
      onChange: value => {
        scale = Number(value);
        writeDisplayParameter();
        displayDirty = true;
      }
    });
    context.controls.addSelect<string>({
      label: 'Focal group',
      options: GROUP_NAMES.map((name, index) => ({value: String(index), label: name})),
      value: String(focalGroup),
      onChange: value => {
        focalGroup = Number(value);
        writeDisplayParameter();
        displayDirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Segregation strength (rewrites group counts)',
      min: 0,
      max: 1,
      step: 0.05,
      value: strength,
      format: value => value.toFixed(2),
      onChange: value => {
        strength = value;
        writeGroupCounts(strength);
        groupCounts.write(groupCountsHost);
        analysisDirty = true;
        displayDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Bandwidth unit (per-frame search radius)',
      min: 0.5,
      max: MAXIMUM_BANDWIDTH_UNIT,
      step: 0.25,
      value: bandwidthUnit,
      format: value => `${((value * cellSize) / 1000).toFixed(2)} km`,
      onChange: value => {
        bandwidthUnit = value;
        writeSearchParameters();
        analysisDirty = true;
        displayDirty = true;
      }
    });
    context.controls.addSelect<WeightsVariant>({
      label: 'Weights (compile-time: rebuilds the graph)',
      options: [
        {value: 'band', label: 'Distance band, binary'},
        {value: 'kernel', label: 'Epanechnikov kernel (Transform)'},
        {value: 'row', label: 'Row-standardized band (Transform)'}
      ],
      value: variant,
      onChange: value => {
        variant = value;
        const previous = analysis;
        analysis = buildAnalysisGraph(variant);
        // Deck may still be encoding the old graph for one more frame; free it afterwards.
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            if (!destroyed) resources.release(previous);
          })
        );
        analysisDirty = true;
        displayDirty = true;
      }
    });
    context.controls.addLegend({
      title: 'Map value (range read back from the GPU)',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: 'min',
        maximumLabel: 'max'
      }
    });
    context.controls.addLegend({
      title: 'Dominant group (when selected)',
      entries: GROUP_NAMES.map((name, index) => ({color: GROUP_COLORS[index], label: name}))
    });
    const legendReadout = context.controls.addReadout('Map range', '...');
    const entropyReadout = context.controls.addReadout('Global (scale)', '...');
    const groupReadout = context.controls.addReadout('Focal group', '...');
    const interactionReadout = context.controls.addReadout('Interaction with', '...');
    const localSumReadout = context.controls.addReadout('Local terms add up', '...');
    const profileReadout = context.controls.addReadout('Multiscale profile', '...');
    const overflowReadout = context.controls.addReadout('Weights overflow', '...');
    const shareReadout = context.controls.addReadout('Group shares');
    context.controls.addReadout(
      'Units',
      `${columns} x ${rows} grid, ${formatCount(landUnits)} populated, ${(cellSize / 1000).toFixed(2)} km cells`
    );
    context.controls.addNote(
      'Group counts are synthetic and seeded (smooth composition fields over a land-weighted population), not census data. ' +
        "Spatial scales use the Reardon and O'Sullivan environment: compositions come from the weighted neighborhood, unit populations stay the weights. " +
        'Segregation that persists as the bandwidth grows is large-scale; segregation that vanishes is local.'
    );
    context.controls.addReadout('Data', `synthetic group counts on ${zips.attribution}`);

    writeGroupCounts(strength);
    groupCounts.write(groupCountsHost);
    writeSearchParameters();
    writeDisplayParameter();
    const updateShares = (): void => {
      const totals = [0, 0, 0];
      for (let unit = 0; unit < unitCount; unit++) {
        for (let group = 0; group < GROUP_COUNT; group++) {
          totals[group] += groupCountsHost[unit * GROUP_COUNT + group];
        }
      }
      const sum = totals.reduce((a, b) => a + b, 0) || 1;
      shareReadout.setValue(
        totals
          .map((total, group) => `${GROUP_NAMES[group]} ${((100 * total) / sum).toFixed(0)}%`)
          .join(', ') + ` of ${formatCount(sum)}`
      );
    };
    updateShares();

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [analysis, displayCompiled],
      encode(commandEncoder) {
        if (analysisDirty) {
          analysis.encode(commandEncoder, {parameters: undefined});
          analysisDirty = false;
          updateShares();
          summaryReader.markStale();
        }
        if (displayDirty) {
          displayCompiled.encode(commandEncoder, {parameters: undefined});
          displayDirty = false;
          summaryReader.markStale();
        }
        summaryReader.flush(commandEncoder);
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [zips.origin[0], zips.origin[1], 0];
        const grid = {
          coordinateOrigin,
          gridSize: [columns, rows] as const,
          bounds: gridBounds,
          rowOrigin: 'south' as const
        };
        return [
          localIndex === 'dominant'
            ? new SpatialAnalysisRasterLayer({
                ...grid,
                id: 'segregation-dominant',
                values: dominant,
                valueFormat: 'uint32',
                colormap: 'category',
                palette: GROUP_COLORS,
                noDataValue: NO_GROUP,
                noDataColor: [0, 0, 0, 0]
              })
            : new SpatialAnalysisRasterLayer({
                ...grid,
                id: 'segregation-local',
                values: display,
                valueFormat: 'float32',
                colormap: 'viridis',
                valueRange: [displayMinimum, displayMaximum],
                discardAtOrBelow: HIDDEN / 2,
                color: [255, 255, 255, 220],
                noDataColor: [0, 0, 0, 0]
              }),
          new SpatialAnalysisSegmentLayer({
            id: 'segregation-outline',
            coordinateOrigin,
            segments: outlineSegmentsBuffer,
            instanceCount: zips.outlineSegments.length / 4,
            widthPixels: 1.2,
            color: [255, 255, 255, 140]
          })
        ] as Layer[];
      },
      getTooltip(event: SpatialAnalysisPointerEvent) {
        if (!event.coordinate) return null;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const column = Math.floor((x - gridBounds[0]) / cellSize);
        const row = Math.floor((y - gridBounds[1]) / cellSize);
        if (column < 0 || row < 0 || column >= columns || row >= rows) return null;
        const unit = row * columns + column;
        if (basePopulation[unit] <= 0) return null;
        return GROUP_NAMES.map(
          (name, group) => `${name} ${groupCountsHost[unit * GROUP_COUNT + group]}`
        ).join(', ');
      },
      destroy() {
        destroyed = true;
        summaryReader.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
