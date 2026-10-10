// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPURasterArithmeticParameterValues,
  getGPURasterConditionalParameterValues,
  getGPURasterReclassifyParameterValues,
  getGPURasterSieveParameterValues,
  getGPURasterStretchParameterValues,
  GPURasterArithmetic,
  GPURasterConditional,
  GPURasterConnectedComponents,
  GPURasterDenseComponents,
  GPURasterPatchMetrics,
  GPURasterReclassify,
  GPURasterSieve,
  GPURasterStatistics,
  GPURasterStretch,
  GPU_RASTER_SIEVE_PARAMETER_LENGTH,
  type GPURasterBufferBand,
  type GPURasterSieveMode
} from '@luma.gl/experimental/gpu-raster';
import {SpatialAnalysisResources, formatCount} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {RampName} from '../../engine/ramps';
import type {SceneContext, SceneInstance} from '../scene';
import {
  formatFixed,
  formatHectares,
  createGraphViewer,
  formatPercent,
  loadDixieRasters,
  readDixieReference,
  UtmRasterLayer
} from './b16-common';
import {PATCH_COLORS, SEVERITY_COLORS, SEVERITY_NAMES} from './b16-colors';

/** Option state of the burn-severity scene. */
export type BurnSeverityOptions = {
  layer:
    | 'dnbr'
    | 'severity'
    | 'burned'
    | 'patches'
    | 'nbr-before'
    | 'nbr-after'
    | 'ndvi-before'
    | 'ndvi-after'
    | 'dndvi';
  ramp: RampName;
  opacity: number;
  indexFormula: 'normalizedDifference' | 'divide' | 'subtract';
  changeOperation: 'subtract' | 'absoluteDifference';
  maskClouds: boolean;
  cloudClass: number;
  comparison: '>' | '>=' | 'between' | '<';
  threshold: number;
  upperThreshold: number;
  classScheme: 'usgs' | 'custom' | 'binary';
  closed: 'left' | 'right';
  lowBreak: number;
  moderateBreak: number;
  highBreak: number;
  stretchMode: 'linear' | 'percentile' | 'equalize';
  percentiles: readonly [number, number];
  domain: 'auto' | 'fixed';
  gamma: number;
  sigmoidContrast: number;
  sigmoidMidpoint: number;
  stretchToView: boolean;
  patchTarget: 'burned' | 'unburned';
  connectivity: '4' | '8';
  minimumPatchHa: number;
  sieveMode: GPURasterSieveMode;
};

type StretchLayer = 'dnbr' | 'nbr-before' | 'nbr-after' | 'ndvi-before' | 'ndvi-after' | 'dndvi';

const STRETCH_LAYERS: readonly StretchLayer[] = [
  'dnbr',
  'nbr-before',
  'nbr-after',
  'ndvi-before',
  'ndvi-after',
  'dndvi'
];
/** Largest dense patch label measured and sieved (compile-time). */
const PATCH_CAPACITY = 8192;
const MAXIMUM_BREAKS = 6;
const CLASS_COUNT = MAXIMUM_BREAKS + 1;
const HISTOGRAM_BINS = 1024;
/** Hectares in one 20 m pixel. */
const PIXEL_HECTARES = 0.04;
const SETTLE_MILLISECONDS = 350;
const SPARK = ['.', ':', '-', '=', '+', '*', '#', '@'];
const COLUMN_NAMES = [
  'pixelCounts',
  'areas',
  'perimeters',
  'minColumns',
  'minRows',
  'maxColumns',
  'maxRows'
] as const;

/** Statistics products, in the order they are read back. */
const STATISTIC_NAMES = [
  'dnbr',
  'nbrBefore',
  'nbrAfter',
  'ndviBefore',
  'ndviAfter',
  'dndvi',
  'burned'
] as const;

type PatchBuffers = {
  sparse: Buffer;
  sparseValidity: Buffer;
  sparseConverged: Buffer;
  iterations: Buffer;
  labels: Buffer;
  labelValidity: Buffer;
  componentCount: Buffer;
  overflow: Buffer;
  columns: Record<(typeof COLUMN_NAMES)[number], Buffer>;
  sieves: Record<GPURasterSieveMode, {labels: Buffer; targets: Buffer; count: Buffer}>;
};

/** Everything the scene reads back after the camera and controls settle. */
type Summary = {
  stretch: Float32Array;
  histogram: Uint32Array;
  classCounts: Uint32Array;
  statistics: {count: number; mean: number; minimum: number; maximum: number}[];
  componentCount: number;
  converged: number;
  overflow: number;
  sievedRemove: number;
  sievedMerge: number;
  columns: Record<(typeof COLUMN_NAMES)[number], Uint32Array | Float32Array>;
};

/**
 * Burn severity of the 2021 Dixie Fire. One compiled graph holds the band math, the cloud screen,
 * the burned condition and the reclassification; a statistics graph reduces the products; one
 * stretch graph per displayable raster makes the picture; two patch graphs (4 and 8 connectivity)
 * label, measure and sieve the burned mask. Every control is a parameter-buffer write; only the
 * graphs that depend on a change are re-encoded, and results are read back once the controls and
 * camera settle.
 */
export async function createBurnSeverity(
  ctx: SceneContext<BurnSeverityOptions>
): Promise<SceneInstance<BurnSeverityOptions>> {
  const dataset = ctx.datasets.get('dixie-fire');
  const {device} = ctx;
  ctx.setStatus('Loading Sentinel-2 bands');
  const rasters = await loadDixieRasters(dataset, ctx.signal);
  ctx.signal.throwIfAborted();
  const reference = readDixieReference(dataset);
  const {width, height, cellCount, frame} = rasters;
  const resources = new SpatialAnalysisResources(device, 'burn');

  ctx.setReadout('grid', `${width} × ${height} px, 20 m`);

  // --- Buffers ---------------------------------------------------------------------------------
  const input = {
    redBefore: resources.createBuffer('red-before', rasters.redBefore),
    nirBefore: resources.createBuffer('nir-before', rasters.nirBefore),
    swirBefore: resources.createBuffer('swir-before', rasters.swirBefore),
    redAfter: resources.createBuffer('red-after', rasters.redAfter),
    nirAfter: resources.createBuffer('nir-after', rasters.nirAfter),
    swirAfter: resources.createBuffer('swir-after', rasters.swirAfter),
    scl: resources.createBuffer('scl', rasters.sclAfter)
  };
  const makeFloat = (name: string) => resources.createBuffer(name, cellCount * 4);
  const makeWord = (name: string, length = cellCount) => resources.createBuffer(name, length * 4);
  const product = {
    nbrBefore: makeFloat('nbr-before'),
    nbrAfter: makeFloat('nbr-after'),
    ndviBefore: makeFloat('ndvi-before'),
    ndviAfter: makeFloat('ndvi-after'),
    dnbr: makeFloat('dnbr'),
    dndvi: makeFloat('dndvi'),
    cloudFlag: makeFloat('cloud-flag'),
    cloudMask: makeWord('cloud-mask'),
    clean: makeFloat('clean-dnbr'),
    burnedFlag: makeFloat('burned-flag'),
    burnedMask: makeWord('burned-mask'),
    patchMask: makeWord('patch-mask'),
    classes: makeWord('classes'),
    classCounts: makeWord('class-counts', CLASS_COUNT),
    stretched: makeFloat('stretched'),
    stretchStatistics: resources.createBuffer('stretch-statistics', 32),
    histogram: makeWord('histogram', HISTOGRAM_BINS)
  };
  const patch: PatchBuffers = {
    sparse: makeWord('sparse'),
    sparseValidity: makeWord('sparse-validity'),
    sparseConverged: makeWord('sparse-converged', 1),
    iterations: makeWord('iterations', 1),
    labels: makeWord('labels'),
    labelValidity: makeWord('label-validity'),
    componentCount: makeWord('component-count', 1),
    overflow: makeWord('overflow', 1),
    columns: {
      pixelCounts: makeWord('pixel-counts', PATCH_CAPACITY),
      areas: makeWord('areas', PATCH_CAPACITY),
      perimeters: makeWord('perimeters', PATCH_CAPACITY),
      minColumns: makeWord('min-columns', PATCH_CAPACITY),
      minRows: makeWord('min-rows', PATCH_CAPACITY),
      maxColumns: makeWord('max-columns', PATCH_CAPACITY),
      maxRows: makeWord('max-rows', PATCH_CAPACITY)
    },
    sieves: {
      remove: {
        labels: makeWord('remove-labels'),
        targets: makeWord('remove-targets', PATCH_CAPACITY),
        count: makeWord('remove-count', 1)
      },
      merge: {
        labels: makeWord('merge-labels'),
        targets: makeWord('merge-targets', PATCH_CAPACITY),
        count: makeWord('merge-count', 1)
      }
    }
  };
  const statisticBuffers = STATISTIC_NAMES.map(name => ({
    count: resources.createBuffer(`${name}-count`, 4),
    sum: resources.createBuffer(`${name}-sum`, 4),
    mean: resources.createBuffer(`${name}-mean`, 4),
    extent: resources.createBuffer(`${name}-extent`, 8)
  }));

  // Parameter buffers: every one is rewritten between encodings, never recompiled.
  const indexParameters = resources.createParameterBuffer('index-parameters', 'float32', 8);
  const changeParameters = resources.createParameterBuffer('change-parameters', 'float32', 8);
  const cloudParameters = resources.createParameterBuffer('cloud-parameters', 'float32', 8);
  const cleanParameters = resources.createParameterBuffer('clean-parameters', 'float32', 8);
  const burnedParameters = resources.createParameterBuffer('burned-parameters', 'float32', 8);
  const patchMaskParameters = resources.createParameterBuffer(
    'patch-mask-parameters',
    'float32',
    8
  );
  const reclassifyParameters = resources.createParameterBuffer(
    'reclassify-parameters',
    'float32',
    4
  );
  const breaks = resources.createParameterBuffer('breaks', 'float32', MAXIMUM_BREAKS);
  const stretchParameters = resources.createParameterBuffer('stretch-parameters', 'float32', 16);
  const sieveParameters = resources.createParameterBuffer(
    'sieve-parameters',
    'uint32',
    GPU_RASTER_SIEVE_PARAMETER_LENGTH
  );

  // --- Graphs ----------------------------------------------------------------------------------
  const floatBand = (
    id: string,
    values: GraphDataView<'float32'>
  ): GPURasterBufferBand<'float32'> => ({id, format: 'float32', storage: {kind: 'buffer', values}});

  // Main graph: band math, cloud screen, burned condition, severity classes.
  const main = new GPUCommandGraph<void>(device, {id: 'burn-main'});
  {
    const view = createGraphViewer(main);
    const f = (name: string, buffer: Buffer) => view(name, buffer, 'float32', cellCount);
    const w = (name: string, buffer: Buffer, length = cellCount) =>
      view(name, buffer, 'uint32', length);
    const indexView = indexParameters.importToGraph(main);
    const changeView = changeParameters.importToGraph(main);
    // NBR = (NIR - SWIR) / (NIR + SWIR) and NDVI = (NIR - red) / (NIR + red), before and after.
    const indices = [
      ['nbr-before', input.nirBefore, input.swirBefore, product.nbrBefore],
      ['nbr-after', input.nirAfter, input.swirAfter, product.nbrAfter],
      ['ndvi-before', input.nirBefore, input.redBefore, product.ndviBefore],
      ['ndvi-after', input.nirAfter, input.redAfter, product.ndviAfter]
    ] as const;
    for (const [id, a, b, output] of indices) {
      main.add(
        new GPURasterArithmetic({
          id,
          cellCount,
          a: f(`${id}-a`, a),
          b: f(`${id}-b`, b),
          noDataValue: 0,
          parameters: indexView,
          output: {values: f(`${id}-out`, output)}
        })
      );
    }
    for (const [id, before, after, output] of [
      ['dnbr', product.nbrBefore, product.nbrAfter, product.dnbr],
      ['dndvi', product.ndviBefore, product.ndviAfter, product.dndvi]
    ] as const) {
      main.add(
        new GPURasterArithmetic({
          id,
          cellCount,
          a: f(`${id}-before`, before),
          b: f(`${id}-after`, after),
          parameters: changeView,
          output: {values: f(`${id}-out`, output)}
        })
      );
    }
    main.add(
      new GPURasterConditional({
        id: 'cloud',
        cellCount,
        conditionValues: f('scl', input.scl),
        parameters: cloudParameters.importToGraph(main),
        output: {
          values: f('cloud-flag', product.cloudFlag),
          mask: w('cloud-mask', product.cloudMask)
        }
      })
    );
    main.add(
      new GPURasterConditional({
        id: 'clean',
        cellCount,
        mask: w('cloud-mask-in', product.cloudMask),
        b: f('dnbr-in', product.dnbr),
        parameters: cleanParameters.importToGraph(main),
        output: {values: f('clean-out', product.clean)}
      })
    );
    main.add(
      new GPURasterConditional({
        id: 'burned',
        cellCount,
        conditionValues: f('clean-in', product.clean),
        parameters: burnedParameters.importToGraph(main),
        output: {
          values: f('burned-flag', product.burnedFlag),
          mask: w('burned-mask', product.burnedMask)
        }
      })
    );
    main.add(
      new GPURasterConditional({
        id: 'patch-mask',
        cellCount,
        conditionValues: f('burned-flag-in', product.burnedFlag),
        parameters: patchMaskParameters.importToGraph(main),
        output: {
          values: f('patch-flag', makeFloat('patch-flag')),
          mask: w('patch-mask', product.patchMask)
        }
      })
    );
    main.add(
      new GPURasterReclassify({
        id: 'severity',
        values: f('clean-classify', product.clean),
        breaks: breaks.importToGraph(main),
        parameters: reclassifyParameters.importToGraph(main),
        output: {
          classes: w('classes', product.classes),
          classCounts: w('class-counts', product.classCounts, CLASS_COUNT)
        }
      })
    );
  }
  const compiledMain = resources.track(main.compile());

  // One stretch graph per displayable continuous raster; only the displayed one is encoded.
  const stretchSources: Record<StretchLayer, Buffer> = {
    dnbr: product.clean,
    'nbr-before': product.nbrBefore,
    'nbr-after': product.nbrAfter,
    'ndvi-before': product.ndviBefore,
    'ndvi-after': product.ndviAfter,
    dndvi: product.dndvi
  };
  const compiledStretch = {} as Record<StretchLayer, CompiledGPUCommandGraph<void>>;
  for (const layer of STRETCH_LAYERS) {
    const graph = new GPUCommandGraph<void>(device, {id: `burn-stretch-${layer}`});
    const view = createGraphViewer(graph);
    graph.add(
      new GPURasterStretch({
        id: `stretch-${layer}`,
        values: view('values', stretchSources[layer], 'float32', cellCount),
        width,
        height,
        binCount: HISTOGRAM_BINS,
        parameters: stretchParameters.importToGraph(graph),
        output: {
          stretched: view('stretched', product.stretched, 'float32', cellCount),
          statistics: view('statistics', product.stretchStatistics, 'float32', 8),
          histogram: view('histogram', product.histogram, 'uint32', HISTOGRAM_BINS)
        }
      })
    );
    compiledStretch[layer] = resources.track(graph.compile());
  }

  // Statistics graph: GPU means and extents of every product, compared with the CPU reference.
  const statisticsGraph = new GPUCommandGraph<void>(device, {id: 'burn-statistics'});
  {
    const view = createGraphViewer(statisticsGraph);
    const sources = [
      product.clean,
      product.nbrBefore,
      product.nbrAfter,
      product.ndviBefore,
      product.ndviAfter,
      product.dndvi,
      product.burnedFlag
    ];
    sources.forEach((source, index) => {
      const name = STATISTIC_NAMES[index];
      const outputs = statisticBuffers[index];
      new GPURasterStatistics({
        id: `stats-${name}`,
        width,
        height,
        input: floatBand(name, view(`${name}-values`, source, 'float32', cellCount)),
        count: view(`${name}-count`, outputs.count, 'uint32', 1),
        sum: view(`${name}-sum`, outputs.sum, 'float32', 1),
        mean: view(`${name}-mean`, outputs.mean, 'float32', 1),
        extent: view(`${name}-extent`, outputs.extent, 'float32', 2)
      }).addToGraph(statisticsGraph);
    });
  }
  const compiledStatistics = resources.track(statisticsGraph.compile());

  // Patch graphs: label, measure and sieve the patch mask; one per connectivity.
  const buildPatchGraph = (connectivity: 4 | 8): CompiledGPUCommandGraph<void> => {
    const graph = new GPUCommandGraph<void>(device, {id: `burn-patches-${connectivity}`});
    const view = createGraphViewer(graph);
    const w = (name: string, buffer: Buffer, length = cellCount) =>
      view(name, buffer, 'uint32', length);
    const sparse = w('sparse', patch.sparse);
    const sparseValidity = w('sparse-validity', patch.sparseValidity);
    const sparseConverged = w('sparse-converged', patch.sparseConverged, 1);
    new GPURasterConnectedComponents({
      id: 'components',
      width,
      height,
      input: {
        id: 'patch-mask',
        format: 'uint32',
        storage: {kind: 'buffer', values: w('patch-mask', product.patchMask)}
      },
      output: sparse,
      outputValidity: sparseValidity,
      converged: sparseConverged,
      iterationCount: w('iterations', patch.iterations, 1),
      connectivity
    }).addToGraph(graph);
    const labels = w('labels', patch.labels);
    const labelValidity = w('label-validity', patch.labelValidity);
    const componentCount = w('component-count', patch.componentCount, 1);
    const overflow = w('overflow', patch.overflow, 1);
    new GPURasterDenseComponents({
      id: 'dense',
      width,
      height,
      input: sparse,
      inputValidity: sparseValidity,
      converged: sparseConverged,
      output: labels,
      outputValidity: labelValidity,
      componentCount,
      overflow,
      capacity: PATCH_CAPACITY
    }).addToGraph(graph);
    const patchInput = {
      width,
      height,
      labels,
      labelValidity,
      converged: sparseConverged,
      componentCount,
      overflow
    };
    const column = <F extends 'uint32' | 'float32'>(
      name: (typeof COLUMN_NAMES)[number],
      format: F
    ) => view(name, patch.columns[name], format, PATCH_CAPACITY);
    graph.add(
      new GPURasterPatchMetrics({
        ...patchInput,
        id: 'metrics',
        // Pixel to local UTM meters: x = 20 * column - half width, y = half height - 20 * row.
        affine: [frame.cellSize, 0, -frame.halfWidth, 0, -frame.cellSize, frame.halfHeight],
        output: {
          pixelCounts: column('pixelCounts', 'uint32'),
          areas: column('areas', 'float32'),
          perimeters: column('perimeters', 'float32'),
          minColumns: column('minColumns', 'uint32'),
          minRows: column('minRows', 'uint32'),
          maxColumns: column('maxColumns', 'uint32'),
          maxRows: column('maxRows', 'uint32')
        }
      })
    );
    const sieveView = sieveParameters.importToGraph(graph);
    for (const mode of ['remove', 'merge'] as GPURasterSieveMode[]) {
      const sieve = patch.sieves[mode];
      graph.add(
        new GPURasterSieve({
          ...patchInput,
          id: `sieve-${mode}`,
          patchCapacity: PATCH_CAPACITY,
          parameters: sieveView,
          mode,
          connectivity,
          output: {
            labels: w(`${mode}-labels`, sieve.labels),
            patchTargets: w(`${mode}-targets`, sieve.targets, PATCH_CAPACITY),
            sievedCount: w(`${mode}-count`, sieve.count, 1)
          }
        })
      );
    }
    return resources.track(graph.compile());
  };
  const compiledPatches = {4: buildPatchGraph(4), 8: buildPatchGraph(8)};

  // --- Parameters ------------------------------------------------------------------------------
  let destroyed = false;
  let dirty = true;
  let summaryStale = true;
  let inspectStale = true;
  let labelsStale = true;
  let lastChange = performance.now();
  let lastWindowKey = '';
  let selectedLabel = 0;
  let summary: Summary | null = null;
  let inspectClean: Float32Array | null = null;
  let inspectClasses: Uint32Array | null = null;
  const labelArrays: Partial<Record<GPURasterSieveMode, Uint32Array>> = {};

  const markDirty = () => {
    dirty = true;
    lastChange = performance.now();
  };

  const getBreaks = (state: BurnSeverityOptions): {values: number[]; count: number} => {
    if (state.classScheme === 'usgs')
      return {values: [-0.25, -0.1, 0.1, 0.27, 0.44, 0.66], count: 6};
    if (state.classScheme === 'binary') return {values: [state.moderateBreak], count: 1};
    const sorted = [state.lowBreak, state.moderateBreak, state.highBreak].sort((a, b) => a - b);
    return {values: sorted, count: 3};
  };

  const getFixedDomain = (layer: StretchLayer): [number, number] =>
    layer === 'dnbr' || layer === 'dndvi' ? [-0.5, 1.5] : [-1, 1];

  const getWindow = (viewport: {
    width: number;
    height: number;
    unproject: (xy: number[]) => number[];
  }): [number, number, number, number] => {
    let minColumn = Infinity;
    let minRow = Infinity;
    let maxColumn = -Infinity;
    let maxRow = -Infinity;
    for (const [x, y] of [
      [0, 0],
      [viewport.width, 0],
      [viewport.width, viewport.height],
      [0, viewport.height]
    ]) {
      const [longitude, latitude] = viewport.unproject([x, y]);
      const [column, row] = frame.lngLatToCell(longitude, latitude);
      minColumn = Math.min(minColumn, column);
      maxColumn = Math.max(maxColumn, column);
      minRow = Math.min(minRow, row);
      maxRow = Math.max(maxRow, row);
    }
    const clamp = (value: number, high: number) => Math.min(Math.max(Math.floor(value), 0), high);
    return [
      clamp(minColumn, width - 1),
      clamp(minRow, height - 1),
      clamp(maxColumn, width - 1) + 1,
      clamp(maxRow, height - 1) + 1
    ];
  };

  const writeParameters = (stretchWindow?: [number, number, number, number]) => {
    const o = ctx.options;
    indexParameters.write(
      getGPURasterArithmeticParameterValues({
        operation: o.indexFormula,
        // Digital numbers are reflectance times 10000.
        scaleA: 1e-4,
        scaleB: 1e-4
      })
    );
    changeParameters.write(getGPURasterArithmeticParameterValues({operation: o.changeOperation}));
    cloudParameters.write(
      getGPURasterConditionalParameterValues({
        comparison: '>=',
        threshold: o.maskClouds ? o.cloudClass : 99,
        constantA: 1,
        constantB: 0
      })
    );
    cleanParameters.write(getGPURasterConditionalParameterValues({constantA: Number.NaN}));
    burnedParameters.write(
      getGPURasterConditionalParameterValues({
        comparison: o.comparison,
        threshold: o.threshold,
        upperThreshold: o.upperThreshold,
        constantA: 1,
        constantB: 0
      })
    );
    patchMaskParameters.write(
      getGPURasterConditionalParameterValues({
        comparison: '==',
        threshold: o.patchTarget === 'burned' ? 1 : 0
      })
    );
    const {values, count} = getBreaks(o);
    breaks.write(Float32Array.from({length: MAXIMUM_BREAKS}, (_, index) => values[index] ?? 1e9));
    reclassifyParameters.write(
      getGPURasterReclassifyParameterValues({breakCount: count, closed: o.closed})
    );
    const stretchLayer = getStretchLayer(o.layer);
    stretchParameters.write(
      getGPURasterStretchParameterValues({
        window: stretchWindow,
        domain: o.domain === 'fixed' ? getFixedDomain(stretchLayer) : 'auto',
        mode: o.stretchMode,
        percentiles: [Math.min(o.percentiles[0], o.percentiles[1]), Math.max(...o.percentiles)],
        gamma: o.gamma,
        sigmoidContrast: o.sigmoidContrast,
        sigmoidMidpoint: o.sigmoidMidpoint
      })
    );
    sieveParameters.write(
      getGPURasterSieveParameterValues({
        minimumPixels: Math.max(1, Math.round(o.minimumPatchHa / PIXEL_HECTARES))
      })
    );
  };

  // --- Readback --------------------------------------------------------------------------------
  const summaryReader = new SummaryReader(
    resources,
    'burn-summary',
    [
      {buffer: product.stretchStatistics, size: 32},
      {buffer: product.histogram, size: HISTOGRAM_BINS * 4},
      {buffer: product.classCounts, size: CLASS_COUNT * 4},
      ...statisticBuffers.flatMap(outputs => [
        {buffer: outputs.count, size: 4},
        {buffer: outputs.sum, size: 4},
        {buffer: outputs.mean, size: 4},
        {buffer: outputs.extent, size: 8}
      ]),
      {buffer: patch.componentCount, size: 4},
      {buffer: patch.sparseConverged, size: 4},
      {buffer: patch.overflow, size: 4},
      {buffer: patch.sieves.remove.count, size: 4},
      {buffer: patch.sieves.merge.count, size: 4},
      ...COLUMN_NAMES.map(name => ({buffer: patch.columns[name], size: PATCH_CAPACITY * 4}))
    ],
    bytes => {
      if (destroyed) return;
      summary = parseSummary(bytes);
      updateReadouts();
    }
  );
  const inspectReader = new SummaryReader(
    resources,
    'burn-inspect',
    [
      {buffer: product.clean, size: cellCount * 4},
      {buffer: product.classes, size: cellCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      inspectClean = new Float32Array(bytes, 0, cellCount);
      inspectClasses = new Uint32Array(bytes, cellCount * 4, cellCount);
    }
  );
  const labelReaders = {} as Record<GPURasterSieveMode, SummaryReader>;
  for (const mode of ['remove', 'merge'] as GPURasterSieveMode[]) {
    labelReaders[mode] = new SummaryReader(
      resources,
      `burn-labels-${mode}`,
      [{buffer: patch.sieves[mode].labels, size: cellCount * 4}],
      bytes => {
        if (destroyed) return;
        labelArrays[mode] = new Uint32Array(bytes);
        if (selectedLabel === 0) return;
        updateReadouts();
      }
    );
  }

  function parseSummary(bytes: ArrayBuffer): Summary {
    let offset = 0;
    const take = <T extends Float32Array | Uint32Array>(
      Type: {new (buffer: ArrayBuffer, offset: number, length: number): T},
      length: number
    ): T => {
      const result = new Type(bytes, offset, length);
      offset += length * 4;
      return result;
    };
    const stretch = take(Float32Array, 8);
    const histogram = take(Uint32Array, HISTOGRAM_BINS);
    const classCounts = take(Uint32Array, CLASS_COUNT);
    const statistics = STATISTIC_NAMES.map(() => {
      const count = take(Uint32Array, 1)[0];
      take(Float32Array, 1);
      const mean = take(Float32Array, 1)[0];
      const extent = take(Float32Array, 2);
      return {count, mean, minimum: extent[0], maximum: extent[1]};
    });
    const scalars = take(Uint32Array, 5);
    const columns = {
      pixelCounts: take(Uint32Array, PATCH_CAPACITY),
      areas: take(Float32Array, PATCH_CAPACITY),
      perimeters: take(Float32Array, PATCH_CAPACITY),
      minColumns: take(Uint32Array, PATCH_CAPACITY),
      minRows: take(Uint32Array, PATCH_CAPACITY),
      maxColumns: take(Uint32Array, PATCH_CAPACITY),
      maxRows: take(Uint32Array, PATCH_CAPACITY)
    };
    return {
      stretch,
      histogram,
      classCounts,
      statistics,
      componentCount: scalars[0],
      converged: scalars[1],
      overflow: scalars[2],
      sievedRemove: scalars[3],
      sievedMerge: scalars[4],
      columns
    };
  }

  function getClassNames(state: BurnSeverityOptions): string[] {
    if (state.classScheme === 'binary') return ['Below break', 'Burned'];
    if (state.classScheme === 'custom') return ['Unburned', 'Low', 'Moderate', 'High'];
    return [...SEVERITY_NAMES];
  }

  function formatReference(gpu: number, expected: number, digits: number, unit = ''): string {
    if (!Number.isFinite(gpu)) return 'n/a';
    const difference = Math.abs(gpu - expected);
    const text =
      difference < 5 * 10 ** -(digits + 1) ? 'match' : `Δ ${difference.toExponential(1)}`;
    return `${formatFixed(gpu, digits)}${unit} · CPU ${formatFixed(expected, digits)}${unit} (${text})`;
  }

  function getPatchStatistics(state: BurnSeverityOptions) {
    if (!summary) return null;
    const {columns, componentCount} = summary;
    const rows = Math.min(componentCount, PATCH_CAPACITY);
    const minimumPixels = Math.max(1, Math.round(state.minimumPatchHa / PIXEL_HECTARES));
    let kept = 0;
    let keptPixels = 0;
    let perimeter = 0;
    let largest = 0;
    let largestRow = -1;
    let totalPixels = 0;
    for (let row = 0; row < rows; row++) {
      const pixels = columns.pixelCounts[row] as number;
      totalPixels += pixels;
      if (pixels < minimumPixels) continue;
      kept++;
      keptPixels += pixels;
      perimeter += columns.perimeters[row] as number;
      if (pixels > largest) {
        largest = pixels;
        largestRow = row;
      }
    }
    return {rows, kept, keptPixels, perimeter, largest, largestRow, totalPixels, minimumPixels};
  }

  function describePatch(label: number): string {
    if (!summary || label < 1 || label > PATCH_CAPACITY) return 'none';
    const row = label - 1;
    const {columns} = summary;
    const pixels = columns.pixelCounts[row] as number;
    if (pixels === 0) return 'removed by the sieve';
    const hectares = pixels * PIXEL_HECTARES;
    const perimeter = columns.perimeters[row] as number;
    const boxWidth =
      ((columns.maxColumns[row] as number) - (columns.minColumns[row] as number) + 1) * 0.02;
    const boxHeight =
      ((columns.maxRows[row] as number) - (columns.minRows[row] as number) + 1) * 0.02;
    return `#${label}: ${formatHectares(hectares)}, perimeter ${(perimeter / 1000).toFixed(2)} km, box ${boxWidth.toFixed(1)} × ${boxHeight.toFixed(1)} km`;
  }

  function updateReadouts(): void {
    if (!summary) return;
    const state = ctx.options;
    const defaults =
      state.indexFormula === 'normalizedDifference' && state.changeOperation === 'subtract';
    const [dnbr, nbrBefore, nbrAfter, ndviBefore, ndviAfter, dndvi, burned] = summary.statistics;
    ctx.setReadout('validCells', `${formatCount(dnbr.count)} of ${formatCount(cellCount)}`);
    ctx.setReadout(
      'meanDnbr',
      defaults
        ? formatReference(dnbr.mean, reference.meanDNBR, 4)
        : `${formatFixed(dnbr.mean, 4)} (the CPU reference needs the defaults)`
    );
    const pair = (before: number, after: number, expectedBefore: number, expectedAfter: number) =>
      defaults
        ? `${formatFixed(before, 3)} / ${formatFixed(after, 3)} (CPU ${formatFixed(expectedBefore, 3)} / ${formatFixed(expectedAfter, 3)})`
        : `${formatFixed(before, 3)} / ${formatFixed(after, 3)}`;
    ctx.setReadout(
      'meanNbr',
      pair(nbrBefore.mean, nbrAfter.mean, reference.meanNBRBefore, reference.meanNBRAfter)
    );
    ctx.setReadout(
      'meanNdvi',
      pair(ndviBefore.mean, ndviAfter.mean, reference.meanNDVIBefore, reference.meanNDVIAfter)
    );
    ctx.setReadout(
      'meanDndvi',
      defaults ? formatReference(dndvi.mean, reference.meanDNDVI, 4) : formatFixed(dndvi.mean, 4)
    );
    const referenceBurned = state.comparison === '>' && Math.abs(state.threshold - 0.27) < 1e-6;
    ctx.setReadout(
      'burnedShare',
      defaults && referenceBurned && state.maskClouds
        ? `${formatPercent(burned.mean, 2)} · CPU ${formatPercent(reference.moderateFraction, 2)}`
        : `${formatPercent(burned.mean, 2)} (reference at dNBR > 0.27)`
    );
    const counts = summary.classCounts;
    const classNames = getClassNames(state);
    const used = classNames.length;
    let total = 0;
    for (let index = 0; index < used; index++) total += counts[index];
    const top = total > 0 ? counts[used - 1] / total : Number.NaN;
    ctx.setReadout(
      'highShare',
      defaults && state.classScheme === 'usgs' && state.closed === 'left'
        ? `${formatPercent(top, 2)} · CPU ${formatPercent(reference.highFraction, 2)}`
        : `${formatPercent(top, 2)} (top class of this scheme)`
    );
    ctx.setReadout(
      'classShares',
      classNames
        .map(
          (name, index) =>
            `${name.split(',')[0].replace(' severity', '')} ${formatPercent(total > 0 ? counts[index] / total : 0, 1)}`
        )
        .join(' · ')
    );
    const [domainMin, domainMax, lo, hi, valid] = summary.stretch;
    ctx.setReadout(
      'stretchRange',
      `${formatFixed(lo, 3)} to ${formatFixed(hi, 3)} (data ${formatFixed(domainMin, 2)} to ${formatFixed(domainMax, 2)}, ${formatCount(valid)} cells)`
    );
    ctx.setLegendExtent('index', [lo, hi]);
    ctx.setReadout('histogram', sparkline(summary.histogram));

    const patches = getPatchStatistics(state);
    if (patches) {
      const note = summary.converged ? '' : ' (labelling did not converge)';
      const overflow = summary.overflow
        ? ` · overflow: more than ${formatCount(PATCH_CAPACITY)} patches`
        : '';
      ctx.setReadout(
        'patchCount',
        `${formatCount(patches.kept)} of ${formatCount(patches.rows)}${overflow}${note}`
      );
      ctx.setReadout(
        'patchArea',
        `${formatHectares(patches.keptPixels * PIXEL_HECTARES)} of ${formatHectares(patches.totalPixels * PIXEL_HECTARES)}`
      );
      ctx.setReadout(
        'largestPatch',
        patches.largestRow >= 0 ? formatHectares(patches.largest * PIXEL_HECTARES) : 'none'
      );
      ctx.setReadout(
        'edgeDensity',
        patches.keptPixels > 0
          ? `${(patches.perimeter / (patches.keptPixels * PIXEL_HECTARES)).toFixed(1)} m / ha`
          : 'n/a'
      );
      const sieved = state.sieveMode === 'remove' ? summary.sievedRemove : summary.sievedMerge;
      ctx.setReadout(
        'sieved',
        `${formatCount(sieved)} patches under ${formatHectares(patches.minimumPixels * PIXEL_HECTARES)}`
      );
    }
    ctx.setReadout('selectedPatch', selectedLabel ? describePatch(selectedLabel) : 'click a patch');
  }

  function sparkline(histogram: Uint32Array): string {
    const bins = 40;
    const sums = new Array<number>(bins).fill(0);
    const perBin = histogram.length / bins;
    for (let index = 0; index < histogram.length; index++) {
      sums[Math.min(bins - 1, Math.floor(index / perBin))] += histogram[index];
    }
    const maximum = Math.log1p(Math.max(...sums));
    if (maximum <= 0) return 'no data';
    return sums
      .map(
        value =>
          SPARK[
            Math.min(
              SPARK.length - 1,
              Math.floor((Math.log1p(value) / maximum) * (SPARK.length - 1))
            )
          ]
      )
      .join('');
  }

  // --- Cell lookup for tooltips and clicks ------------------------------------------------------
  const getCell = (coordinate: readonly [number, number] | null): number => {
    if (!coordinate) return -1;
    const [column, row] = frame.lngLatToCell(coordinate[0], coordinate[1]);
    const c = Math.floor(column);
    const r = Math.floor(row);
    if (c < 0 || r < 0 || c >= width || r >= height) return -1;
    return r * width + c;
  };

  const getDisplayedLabels = () => labelArrays[ctx.options.sieveMode];

  ctx.setReadout('selectedPatch', 'click a patch');
  ctx.setStatus('');

  // --- Instance --------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [
      compiledMain,
      ...STRETCH_LAYERS.map(layer => compiledStretch[layer]),
      compiledStatistics,
      compiledPatches[4],
      compiledPatches[8]
    ],

    setOption(id) {
      markDirty();
      summaryStale = true;
      inspectStale = true;
      labelsStale = true;
      if (id === 'layer' || id === 'ramp' || id === 'opacity' || id === 'sieveMode') {
        ctx.requestLayers();
      } else {
        ctx.requestLayers();
      }
    },

    encode(commandEncoder, frameInfo) {
      const state = ctx.options;
      let window: [number, number, number, number] | undefined;
      if (state.stretchToView) {
        window = getWindow(frameInfo.viewport);
        const key = window.join(',');
        if (key !== lastWindowKey) {
          lastWindowKey = key;
          markDirty();
        }
      } else if (lastWindowKey !== '') {
        lastWindowKey = '';
        markDirty();
      }
      if (dirty || frameInfo.frameIndex < 2) {
        writeParameters(window);
        compiledMain.encode(commandEncoder, {parameters: undefined});
        compiledStretch[getStretchLayer(state.layer)].encode(commandEncoder, {
          parameters: undefined
        });
        compiledStatistics.encode(commandEncoder, {parameters: undefined});
        compiledPatches[state.connectivity === '8' ? 8 : 4].encode(commandEncoder, {
          parameters: undefined
        });
        dirty = false;
        summaryStale = true;
        inspectStale = true;
        labelsStale = true;
      }
      const settled = performance.now() - lastChange > SETTLE_MILLISECONDS;
      if (settled) {
        if (summaryStale && !summaryReader.isPending) {
          summaryReader.request(commandEncoder);
          summaryStale = false;
        }
        if (inspectStale && !inspectReader.isPending) {
          inspectReader.request(commandEncoder);
          inspectStale = false;
        }
        if (labelsStale && !labelReaders[state.sieveMode].isPending) {
          labelReaders[state.sieveMode].request(commandEncoder);
          labelsStale = false;
        }
      }
      summaryReader.flush(commandEncoder);
      inspectReader.flush(commandEncoder);
      labelReaders[state.sieveMode].flush(commandEncoder);
    },

    getLayers() {
      const state = ctx.options;
      const base = {
        id: `burn-${state.layer}`,
        frame,
        opacity: state.opacity,
        noDataColor: [0, 0, 0, 0] as const
      };
      let layer: Layer;
      if (state.layer === 'severity') {
        const palette =
          state.classScheme === 'usgs'
            ? SEVERITY_COLORS
            : state.classScheme === 'custom'
              ? [2, 3, 5, 6].map(index => SEVERITY_COLORS[index])
              : [2, 6].map(index => SEVERITY_COLORS[index]);
        layer = new UtmRasterLayer({
          ...base,
          values: product.classes,
          valueFormat: 'uint32',
          colormap: 'category',
          palette
        });
      } else if (state.layer === 'burned') {
        layer = new UtmRasterLayer({
          ...base,
          values: product.burnedMask,
          valueFormat: 'uint32',
          colormap: 'mask',
          color: [255, 90, 40, 215]
        });
      } else if (state.layer === 'patches') {
        layer = new UtmRasterLayer({
          ...base,
          values: patch.sieves[state.sieveMode].labels,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: PATCH_COLORS,
          noDataValue: 0
        });
      } else {
        layer = new UtmRasterLayer({
          ...base,
          values: product.stretched,
          valueFormat: 'float32',
          colormap: state.ramp,
          valueRange: [0, 1]
        });
      }
      return [layer];
    },

    getTooltip(event) {
      const cell = getCell(event.coordinate);
      if (cell < 0) return null;
      const state = ctx.options;
      const nbr = (nir: Float32Array, swir: Float32Array) =>
        nir[cell] > 0 && swir[cell] > 0
          ? (nir[cell] - swir[cell]) / (nir[cell] + swir[cell])
          : Number.NaN;
      const nbrBefore = nbr(rasters.nirBefore, rasters.swirBefore);
      const nbrAfter = nbr(rasters.nirAfter, rasters.swirAfter);
      const lines: string[] = [];
      const dnbr = inspectClean ? inspectClean[cell] : nbrBefore - nbrAfter;
      lines.push(
        `dNBR ${formatFixed(dnbr, 3)}  (NBR ${formatFixed(nbrBefore, 2)} → ${formatFixed(nbrAfter, 2)})`
      );
      if (inspectClasses && state.layer === 'severity') {
        const index = inspectClasses[cell];
        const names = getClassNames(state);
        lines.push(index < names.length ? names[index] : 'No data');
      }
      if (state.layer === 'patches') {
        const label = getDisplayedLabels()?.[cell] ?? 0;
        lines.push(label ? `Patch ${describePatch(label)}` : 'Not in a kept patch');
      }
      lines.push(`${Math.round(rasters.elevation[cell]).toLocaleString('en-US')} m elevation`);
      return lines.join('\n');
    },

    onClick(event) {
      if (ctx.options.layer !== 'patches') return false;
      const cell = getCell(event.coordinate);
      if (cell < 0) return false;
      const label = getDisplayedLabels()?.[cell] ?? 0;
      selectedLabel = label;
      updateReadouts();
      return true;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    destroy() {
      destroyed = true;
      summaryReader.stop();
      inspectReader.stop();
      for (const reader of Object.values(labelReaders)) reader.stop();
      resources.destroy();
    }
  };
}

function getStretchLayer(layer: BurnSeverityOptions['layer']): StretchLayer {
  return (STRETCH_LAYERS as readonly string[]).includes(layer) ? (layer as StretchLayer) : 'dnbr';
}
