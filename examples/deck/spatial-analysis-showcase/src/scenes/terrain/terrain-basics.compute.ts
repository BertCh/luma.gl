// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainVectorRuggednessParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH,
  GPUTerrainDerivatives,
  GPUTerrainRGBDecode,
  GPUTerrainRuggedness,
  GPUTerrainSpikeRepair,
  GPUTerrainVectorRuggedness
} from '@luma.gl/experimental/gpu-terrain';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {createSeededRandom} from '../../engine/projection';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {loadAlpsGrid, type AlpsGrid} from './b14a-grid';
import {TerrainSession, type ProductBuild, type ValueStats} from './b14a-session';
import {
  getBasicsPaint,
  VALID_RANGES,
  type BasicsOptions,
  type BasicsProduct
} from './terrain-basics.style';

export type {BasicsOptions, BasicsProduct};

/** Pixel rectangle `[column, row, width, height]` blanked by the "missing tile" option (Gorner glacier). */
const MISSING_PATCH = [1150, 780, 380, 380] as const;
const REBUILD_DELAY_MILLISECONDS = 220;

/** Builds the encoded input: the Terrarium words, optionally with +/-256 m spikes and a nodata patch. */
function buildInput(grid: AlpsGrid, spikeDensity: number, missingTile: boolean): Uint32Array {
  const words = grid.packedWords.slice();
  if (spikeDensity > 0) {
    const random = createSeededRandom(2024);
    const probability = spikeDensity / 100;
    for (let index = 0; index < words.length; index++) {
      if (random() < probability) {
        const word = words[index];
        const red = word & 0xff;
        // One count in the red byte is exactly 256 m in Terrarium.
        const nextRed = random() < 0.5 ? red + 1 : red - 1;
        const safeRed = nextRed < 0 || nextRed > 255 ? red + (red < 128 ? 1 : -1) : nextRed;
        words[index] = (word & 0xffffff00) | safeRed;
      }
    }
  }
  if (missingTile) {
    const [x, y, width, height] = MISSING_PATCH;
    for (let row = y; row < y + height; row++) {
      for (let column = x; column < x + width; column++) {
        // Alpha 0 is how PNG tiles mark missing data.
        words[row * grid.width + column] &= 0x00ffffff;
      }
    }
  }
  return words;
}

/** Hover text of the scene. */
function describeCell(state: BasicsOptions, value: number, elevation: number): string | null {
  if (!Number.isFinite(elevation)) return 'No data (invalid decoded height)';
  const elevationText = `${elevation.toFixed(1)} m`;
  if (!Number.isFinite(value))
    return `Elevation ${elevationText}\nNo ${state.product} (edge or nodata)`;
  switch (state.product) {
    case 'elevation':
      return `Elevation ${elevationText}`;
    case 'slope':
      return `Slope ${value.toFixed(1)}${state.slopeUnits === 'degrees' ? '°' : '%'}\nElevation ${elevationText}`;
    case 'aspect':
      return value < 0
        ? `Flat (no aspect)\nElevation ${elevationText}`
        : `Faces ${compass(value)} (${value.toFixed(0)}° from north)\nElevation ${elevationText}`;
    case 'hillshade':
      return `Hillshade ${value.toFixed(2)}\nElevation ${elevationText}`;
    case 'tpi':
      return `TPI ${value.toFixed(1)} m (${value >= 0 ? 'above' : 'below'} its 8 neighbours)\nElevation ${elevationText}`;
    case 'tri':
      return `TRI ${value.toFixed(1)} m\nElevation ${elevationText}`;
    case 'roughness':
      return `Roughness ${value.toFixed(1)} m\nElevation ${elevationText}`;
    case 'vrm':
      return `VRM ${value.toFixed(3)}\nElevation ${elevationText}`;
  }
}

function compass(degrees: number): string {
  const names = [
    'north',
    'north-east',
    'east',
    'south-east',
    'south',
    'south-west',
    'west',
    'north-west'
  ];
  return names[Math.round(degrees / 45) % 8];
}

/**
 * Terrain basics on the Matterhorn tile. The whole pipeline is GPU work on one 2048 x 2048 raster:
 *
 * 1. `GPUTerrainRGBDecode` turns the Terrarium PNG words into float32 heights and a validity mask.
 * 2. `GPUTerrainSpikeRepair` repairs +/-256 m red-byte errors; a select pass picks raw or repaired.
 * 3. One graph per product (`GPUTerrainDerivatives`, `GPUTerrainRuggedness`,
 *    `GPUTerrainVectorRuggedness`) reads the elevation buffer with `cellSizeMode: 'web-mercator'`.
 *
 * Graphs are compiled the first time a product is shown, and again only when a compile-time option
 * of that product changes. Sliders write parameter buffers; the elevation pipeline re-runs only
 * when its input changes.
 */
export async function createTerrainBasics(
  ctx: SceneContext<BasicsOptions>
): Promise<SceneInstance<BasicsOptions>> {
  const {device} = ctx;
  const dataset = ctx.datasets.get('alps-dem');
  const grid = await loadAlpsGrid(dataset, ctx.signal);
  ctx.signal.throwIfAborted();
  const {width, height, pixelCount} = grid;
  const resources = new SpatialAnalysisResources(device, 'terrain-basics');
  const session = new TerrainSession(ctx, resources, grid);
  session.enableUnderlay();

  // --- The elevation pipeline: decode, repair, select ------------------------------------------
  const encodedBuffer = resources.createBuffer(
    'encoded',
    buildInput(grid, ctx.options.spikeDensity, ctx.options.missingTile)
  );
  const decodedBuffer = resources.createBuffer('decoded', pixelCount * 4);
  const decodedValidity = resources.createBuffer('decoded-validity', pixelCount * 4);
  const repairedBuffer = resources.createBuffer('repaired', pixelCount * 4);
  const repairedValidity = resources.createBuffer('repaired-validity', pixelCount * 4);
  const repairStatistics = resources.createBuffer('repair-statistics', 32);
  const selectFlag = resources.createParameterBuffer('select-flag', 'float32', 4);

  let decodeGraph: CompiledGPUCommandGraph<void> | null = null;
  let decodeKey = '';
  function buildDecode(state: BasicsOptions): void {
    const key = `${state.encoding}|${state.validRange}|${state.alphaNoData}|${state.clampBathymetry}`;
    if (key === decodeKey && decodeGraph) return;
    const graph = new GPUCommandGraph<void>(device, {id: 'terrain-basics-decode'});
    graph.add(
      new GPUTerrainRGBDecode({
        id: 'decode',
        width,
        height,
        encoding: state.encoding,
        input: {buffer: importGraphBuffer(graph, 'encoded', encodedBuffer, 'uint32', pixelCount)},
        alphaNoData: state.alphaNoData,
        validRange: VALID_RANGES[state.validRange],
        clampBathymetry: state.clampBathymetry,
        values: importGraphBuffer(graph, 'decoded', decodedBuffer, 'float32', pixelCount),
        validity: importGraphBuffer(
          graph,
          'decoded-validity',
          decodedValidity,
          'uint32',
          pixelCount
        )
      })
    );
    const next = resources.track(graph.compile());
    if (decodeGraph) resources.release(decodeGraph);
    decodeGraph = next;
    decodeKey = key;
  }

  const repairGraphSource = new GPUCommandGraph<void>(device, {id: 'terrain-basics-repair'});
  repairGraphSource.add(
    new GPUTerrainSpikeRepair({
      id: 'repair',
      width,
      height,
      elevation: {
        id: 'decoded',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(
            repairGraphSource,
            'decoded',
            decodedBuffer,
            'float32',
            pixelCount
          )
        },
        validity: importGraphBuffer(
          repairGraphSource,
          'decoded-validity',
          decodedValidity,
          'uint32',
          pixelCount
        )
      },
      values: importGraphBuffer(
        repairGraphSource,
        'repaired',
        repairedBuffer,
        'float32',
        pixelCount
      ),
      validity: importGraphBuffer(
        repairGraphSource,
        'repaired-validity',
        repairedValidity,
        'uint32',
        pixelCount
      ),
      statistics: importGraphBuffer(repairGraphSource, 'statistics', repairStatistics, 'uint32', 5)
    })
  );
  const repairGraph = resources.track(repairGraphSource.compile());

  const selectSource = new GPUCommandGraph<void>(device, {id: 'terrain-basics-select'});
  addKernelPass(selectSource, {
    id: 'select-elevation',
    invocationCount: pixelCount,
    bindings: [
      {
        name: 'flag',
        view: selectFlag.importToGraph(selectSource),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'rawHeights',
        view: importGraphBuffer(selectSource, 'decoded', decodedBuffer, 'float32', pixelCount),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'rawValid',
        view: importGraphBuffer(
          selectSource,
          'decoded-validity',
          decodedValidity,
          'uint32',
          pixelCount
        ),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'fixedHeights',
        view: importGraphBuffer(selectSource, 'repaired', repairedBuffer, 'float32', pixelCount),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'fixedValid',
        view: importGraphBuffer(
          selectSource,
          'repaired-validity',
          repairedValidity,
          'uint32',
          pixelCount
        ),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'outHeights',
        view: importGraphBuffer(
          selectSource,
          'elevation',
          session.elevationBuffer,
          'float32',
          pixelCount
        ),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'outValid',
        view: importGraphBuffer(
          selectSource,
          'validity',
          session.validityBuffer,
          'uint32',
          pixelCount
        ),
        type: 'u32',
        access: 'read_write'
      }
    ],
    body: /* wgsl */ `
  if (flag[flagOffset] > 0.5) {
    outHeights[outHeightsOffset + index] = fixedHeights[fixedHeightsOffset + index];
    outValid[outValidOffset + index] = fixedValid[fixedValidOffset + index];
  } else {
    outHeights[outHeightsOffset + index] = rawHeights[rawHeightsOffset + index];
    outValid[outValidOffset + index] = rawValid[rawValidOffset + index];
  }`
  });
  const selectGraph = resources.track(selectSource.compile());
  buildDecode(ctx.options);

  // --- Products ---------------------------------------------------------------------------------
  const cellSettings = grid.cellSettings;
  const elevationBuild = session.addBuild(
    session.createStaticBuild('elevation', session.elevationBuffer, 'float32'),
    ''
  );

  function getConfigKey(state: BasicsOptions): string {
    switch (state.product) {
      case 'slope':
      case 'aspect':
      case 'hillshade':
        return `${state.slopeUnits}|${state.borderMode}`;
      case 'tri':
      case 'tpi':
      case 'roughness':
        return `${state.triAlgorithm}|${state.edgeMode}`;
      case 'vrm':
        return `${state.vrmRadius}|${state.borderMode}`;
      default:
        return '';
    }
  }

  function buildProduct(state: BasicsOptions): ProductBuild {
    const {product} = state;
    const builder = session.builder(product);
    const elevation = builder.elevation();
    if (product === 'slope' || product === 'aspect' || product === 'hillshade') {
      const settings = builder.settings(GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH);
      const output = builder.floats(product);
      builder.graph.add(
        new GPUTerrainDerivatives({
          id: 'derivatives',
          width,
          height,
          elevation,
          settings: settings.view,
          [product]: output,
          cellSizeMode: 'web-mercator',
          rowDirection: 'south',
          slopeUnits: state.slopeUnits,
          borderMode: state.borderMode
        })
      );
      return builder.finish({
        value: product,
        format: 'float32',
        write: () =>
          settings.parameters.write(
            getGPUTerrainDerivativesParameterValues({
              ...cellSettings,
              zFactor: ctx.options.zFactor,
              azimuthDegrees: ctx.options.sunAzimuth,
              altitudeDegrees: ctx.options.sunAltitude
            })
          )
      });
    }
    if (product === 'tpi' || product === 'tri' || product === 'roughness') {
      const output = builder.floats(product);
      builder.graph.add(
        new GPUTerrainRuggedness({
          id: 'ruggedness',
          width,
          height,
          elevation,
          ...(product === 'tpi'
            ? {topographicPositionIndex: output}
            : product === 'tri'
              ? {terrainRuggednessIndex: output}
              : {roughness: output}),
          terrainRuggednessAlgorithm: state.triAlgorithm,
          edgeMode: state.edgeMode
        })
      );
      return builder.finish({value: product, format: 'float32', write: () => {}});
    }
    // vrm
    const settings = builder.settings(GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH);
    const output = builder.floats('vrm');
    builder.graph.add(
      new GPUTerrainVectorRuggedness({
        id: 'vector-ruggedness',
        width,
        height,
        elevation,
        settings: settings.view,
        radius: state.vrmRadius,
        vectorRuggedness: output,
        cellSizeMode: 'web-mercator',
        rowDirection: 'south',
        borderMode: state.borderMode
      })
    );
    return builder.finish({
      value: 'vrm',
      format: 'float32',
      minIntervalMs: 120,
      write: () =>
        settings.parameters.write(
          getGPUTerrainVectorRuggednessParameterValues({
            ...cellSettings,
            zFactor: ctx.options.zFactor
          })
        )
    });
  }

  function showProduct(): void {
    const state = ctx.options;
    let build: ProductBuild;
    if (state.product === 'elevation') {
      build = elevationBuild;
    } else {
      const key = getConfigKey(state);
      build = session.getBuild(state.product, key) ?? session.addBuild(buildProduct(state), key);
    }
    session.activate(build, getBasicsPaint(state));
    session.setPaint(getBasicsPaint(state));
    ctx.requestLayers();
    updateStatsReadouts(null);
  }

  // --- Readouts ----------------------------------------------------------------------------------
  const repairReader = new SummaryReader(
    resources,
    'repair-statistics',
    [{buffer: repairStatistics, size: 20}],
    bytes => {
      const [jumps, repaired, shifted, remaining, converged] = new Uint32Array(bytes);
      ctx.setReadout('repairJumps', `${formatCount(jumps)} jumps found`);
      ctx.setReadout(
        'repairResult',
        `${formatCount(repaired)} px repaired in ${formatCount(shifted)} components, ${formatCount(remaining)} jumps left${converged ? '' : ' (did not converge)'}`
      );
    }
  );
  let preparedVersion = 0;
  let verifyVersion = -1;
  let destroyed = false;
  let timers: ReturnType<typeof setTimeout>[] = [];

  ctx.setReadout(
    'grid',
    `${width} x ${height} px, ${grid.groundCellSize.toFixed(2)} m ground (${grid.mercatorCellSize.toFixed(2)} m Web Mercator)`
  );

  session.describeHover = ({value, elevation}) => describeCell(ctx.options, value, elevation);
  session.onStats = (_id, stats) => updateStatsReadouts(stats);

  function updateStatsReadouts(stats: ValueStats | null): void {
    const state = ctx.options;
    if (!stats || stats.kind !== 'float') {
      ctx.setReadout('median', null);
      ctx.setReadout('p98', null);
      ctx.setReadout('maximum', null);
      return;
    }
    const unit =
      state.product === 'slope'
        ? state.slopeUnits === 'degrees'
          ? '°'
          : '%'
        : state.product === 'elevation' ||
            state.product === 'tpi' ||
            state.product === 'tri' ||
            state.product === 'roughness'
          ? ' m'
          : '';
    const digits = state.product === 'vrm' || state.product === 'hillshade' ? 3 : 1;
    const format = (value: number) => `${value.toFixed(digits)}${unit}`;
    ctx.setReadout('median', format(stats.quantile(0.5)));
    ctx.setReadout('p98', format(stats.quantile(0.98)));
    ctx.setReadout('maximum', format(stats.max));
  }

  function verifyElevation(bytes: Uint8Array, version: number): void {
    if (version !== preparedVersion) return;
    const heights = new Float32Array(bytes.buffer, bytes.byteOffset, pixelCount);
    const state = ctx.options;
    let valid = 0;
    let minimum = Infinity;
    let maximum = -Infinity;
    let maximumDifference = 0;
    const comparable =
      state.encoding === 'terrarium' &&
      state.spikeDensity === 0 &&
      !state.missingTile &&
      !state.clampBathymetry;
    for (let index = 0; index < pixelCount; index++) {
      const value = heights[index];
      if (Number.isFinite(value)) {
        valid++;
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
        if (comparable) {
          maximumDifference = Math.max(
            maximumDifference,
            Math.abs(value - grid.cpuElevation[index])
          );
        }
      }
    }
    ctx.setReadout(
      'validPixels',
      `${((valid / pixelCount) * 100).toFixed(2)}% (${formatCount(valid)} px)`
    );
    ctx.setReadout(
      'elevationRange',
      valid > 0 ? `${minimum.toFixed(1)} to ${maximum.toFixed(1)} m` : 'no valid pixels'
    );
    ctx.setReadout(
      'decodeDifference',
      comparable
        ? maximumDifference === 0
          ? 'exactly 0 m (bit-identical)'
          : `${maximumDifference.toExponential(2)} m`
        : 'n/a (input modified)'
    );
  }

  // --- Option handling ---------------------------------------------------------------------------
  let inputDirty = false;
  let decodeDirty = true;
  let selectDirty = true;
  function scheduleProductRebuild(): void {
    timers.push(setTimeout(() => !destroyed && showProduct(), REBUILD_DELAY_MILLISECONDS));
  }

  const writeSelect = () =>
    selectFlag.write(Float32Array.of(ctx.options.repairSpikes ? 1 : 0, 0, 0, 0));
  writeSelect();
  showProduct();

  return {
    getCompiledGraphs: () =>
      [
        ...(decodeGraph ? [decodeGraph] : []),
        repairGraph,
        selectGraph,
        ...session.getCompiledGraphs()
      ] as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id, _value, state) {
      switch (id) {
        case 'product':
          showProduct();
          break;
        case 'encoding':
        case 'validRange':
        case 'alphaNoData':
        case 'clampBathymetry':
          buildDecode(state);
          decodeDirty = true;
          break;
        case 'missingTile':
        case 'spikeDensity':
          inputDirty = true;
          decodeDirty = true;
          break;
        case 'repairSpikes':
          writeSelect();
          selectDirty = true;
          break;
        case 'slopeUnits':
        case 'borderMode':
        case 'triAlgorithm':
        case 'edgeMode':
        case 'vrmRadius':
          scheduleProductRebuild();
          break;
        case 'zFactor':
        case 'sunAzimuth':
        case 'sunAltitude':
          session.markAllDirty();
          break;
        case 'ramp':
        case 'rangeScale':
          session.setPaint(getBasicsPaint(state));
          break;
        case 'opacity':
        case 'underlay':
          ctx.requestLayers();
          break;
      }
      if (id === 'slopeUnits' || id === 'rangeScale') {
        session.setPaint(getBasicsPaint(state));
      }
    },

    onAction(id) {
      if (id !== 'measure') return;
      ctx.setReadout('timing', 'measuring...');
      void session.measure().then(results => {
        if (destroyed) return;
        const total = results.reduce((sum, result) => sum + result.milliseconds, 0);
        ctx.setReadout(
          'timing',
          results.length === 0
            ? 'static raster (no graph)'
            : `${total.toFixed(2)} ms for ${(grid.pixelCount / 1e6) | 0}M px (${results[0].method === 'gpu-timestamps' ? 'GPU timestamps' : 'wall clock'})`
        );
      });
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip: event => session.getTooltip(event),

    encode(commandEncoder) {
      if (inputDirty) {
        encodedBuffer.write(buildInput(grid, ctx.options.spikeDensity, ctx.options.missingTile));
        inputDirty = false;
      }
      if (decodeDirty && decodeGraph) {
        decodeGraph.encode(commandEncoder, {parameters: undefined});
        repairGraph.encode(commandEncoder, {parameters: undefined});
        selectGraph.encode(commandEncoder, {parameters: undefined});
        decodeDirty = false;
        selectDirty = false;
        preparedVersion++;
        session.elevationChanged();
        repairReader.request(commandEncoder);
      } else if (selectDirty) {
        selectGraph.encode(commandEncoder, {parameters: undefined});
        selectDirty = false;
        preparedVersion++;
        session.elevationChanged();
      }
      if (verifyVersion !== preparedVersion) {
        const version = preparedVersion;
        if (
          session.readBulk(commandEncoder, session.elevationBuffer, bytes =>
            verifyElevation(bytes, version)
          )
        ) {
          verifyVersion = version;
        }
      }
      repairReader.flush(commandEncoder);
      session.encode(commandEncoder);
    },

    getLayers(): Layer[] {
      const state = ctx.options;
      return session.getLayers({
        underlay: state.underlay && state.product !== 'hillshade',
        underlayAlpha: 1,
        alpha: state.product === 'hillshade' ? Math.max(state.opacity, 0.9) : state.opacity
      });
    },

    destroy() {
      destroyed = true;
      for (const timer of timers) clearTimeout(timer);
      timers = [];
      repairReader.stop();
      session.destroy();
      resources.destroy();
    }
  };
}
