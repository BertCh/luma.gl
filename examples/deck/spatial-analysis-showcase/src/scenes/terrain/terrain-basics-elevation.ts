// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The elevation pipeline of terrain-basics, every step on the GPU over one 2048 x 2048 raster:
 *
 * 1. `GPUTerrainRGBDecode` turns the Terrarium PNG words into float32 heights and a validity mask.
 * 2. `GPUTerrainSpikeRepair` repairs +/-256 m red-byte errors (an expert sub-block of the decode
 *    step); a select pass picks the raw or the repaired heights.
 * 3. The chosen heights land in the session's elevation buffer, which every product graph reads.
 */

import {GPUTerrainRGBDecode, GPUTerrainSpikeRepair} from '@luma.gl/experimental/gpu-terrain';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createSeededRandom} from '../../engine/projection';
import {formatCount, type SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext} from '../scene';
import type {AlpsGrid} from './b14a-grid';
import type {TerrainSession} from './b14a-session';
import {VALID_RANGES, type BasicsOptions} from './terrain-basics.style';

/**
 * Pixel rectangle `[column, row, width, height]` blanked by the "missing tile" option: the
 * Riffelberg and Rotenboden shoulder above the Gornergrat railway.
 */
const MISSING_PATCH = [1150, 780, 380, 380] as const;

/** `[west, south, east, north]` degrees of the blanked patch, for the "alpha-0 patch" outline. */
export function getMissingPatchBounds(grid: AlpsGrid): [number, number, number, number] {
  const [column, row, patchWidth, patchHeight] = MISSING_PATCH;
  // getLongitudeLatitude returns pixel centres: step half a pixel out to the outer cell edges.
  const [west, north] = grid.getLongitudeLatitude(column - 0.5, row - 0.5);
  const [east, south] = grid.getLongitudeLatitude(
    column + patchWidth - 0.5,
    row + patchHeight - 0.5
  );
  return [west, south, east, north];
}

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
    const [x, y, patchWidth, patchHeight] = MISSING_PATCH;
    for (let row = y; row < y + patchHeight; row++) {
      for (let column = x; column < x + patchWidth; column++) {
        // Alpha 0 is how PNG tiles mark missing data.
        words[row * grid.width + column] &= 0x00ffffff;
      }
    }
  }
  return words;
}

/** The running elevation pipeline. */
export type ElevationPipeline = {
  /** The compiled graphs (decode, repair, select) for "Under the hood". */
  getCompiledGraphs(): CompiledGPUCommandGraph<never>[];
  /** A compile-time decode option changed: rebuild the decode graph and re-run everything. */
  decodeOptionChanged(): void;
  /** The encoded input changed (spikes, the missing patch): re-upload and re-run everything. */
  inputChanged(): void;
  /** Only the raw / repaired choice changed. */
  selectionChanged(): void;
  /**
   * Encodes the pending work. Returns true when the session's elevation changed (the caller then
   * bumps its elevation version and verifies the heights).
   */
  encode(commandEncoder: Parameters<TerrainSession['encode']>[0]): boolean;
  stop(): void;
};

/** Creates the pipeline on the session's elevation and validity buffers. */
export function createElevationPipeline(
  ctx: SceneContext<BasicsOptions>,
  resources: SpatialAnalysisResources,
  session: TerrainSession,
  grid: AlpsGrid
): ElevationPipeline {
  const {device} = ctx;
  const {width, height, pixelCount} = grid;
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
  const buildDecode = (state: BasicsOptions): void => {
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
  };

  const repairSource = new GPUCommandGraph<void>(device, {id: 'terrain-basics-repair'});
  repairSource.add(
    new GPUTerrainSpikeRepair({
      id: 'repair',
      width,
      height,
      elevation: {
        id: 'decoded',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(repairSource, 'decoded', decodedBuffer, 'float32', pixelCount)
        },
        validity: importGraphBuffer(
          repairSource,
          'decoded-validity',
          decodedValidity,
          'uint32',
          pixelCount
        )
      },
      values: importGraphBuffer(repairSource, 'repaired', repairedBuffer, 'float32', pixelCount),
      validity: importGraphBuffer(
        repairSource,
        'repaired-validity',
        repairedValidity,
        'uint32',
        pixelCount
      ),
      statistics: importGraphBuffer(repairSource, 'statistics', repairStatistics, 'uint32', 5)
    })
  );
  const repairGraph = resources.track(repairSource.compile());

  const selectSource = new GPUCommandGraph<void>(device, {id: 'terrain-basics-select'});
  addKernelPass(selectSource, {
    id: 'select-elevation',
    invocationCount: pixelCount,
    bindings: [
      {name: 'flag', view: selectFlag.importToGraph(selectSource), type: 'f32', access: 'read'},
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

  const writeSelection = () =>
    selectFlag.write(Float32Array.of(ctx.options.repairSpikes ? 1 : 0, 0, 0, 0));
  writeSelection();
  let inputDirty = false;
  let decodeDirty = true;
  let selectDirty = true;

  return {
    getCompiledGraphs: () =>
      [
        ...(decodeGraph ? [decodeGraph] : []),
        repairGraph,
        selectGraph
      ] as unknown as CompiledGPUCommandGraph<never>[],

    decodeOptionChanged() {
      buildDecode(ctx.options);
      decodeDirty = true;
    },

    inputChanged() {
      inputDirty = true;
      decodeDirty = true;
    },

    selectionChanged() {
      writeSelection();
      selectDirty = true;
    },

    encode(commandEncoder) {
      if (inputDirty) {
        encodedBuffer.write(buildInput(grid, ctx.options.spikeDensity, ctx.options.missingTile));
        inputDirty = false;
      }
      let changed = false;
      if (decodeDirty && decodeGraph) {
        decodeGraph.encode(commandEncoder, {parameters: undefined});
        repairGraph.encode(commandEncoder, {parameters: undefined});
        selectGraph.encode(commandEncoder, {parameters: undefined});
        decodeDirty = false;
        selectDirty = false;
        repairReader.request(commandEncoder);
        changed = true;
      } else if (selectDirty) {
        selectGraph.encode(commandEncoder, {parameters: undefined});
        selectDirty = false;
        changed = true;
      }
      repairReader.flush(commandEncoder);
      return changed;
    },

    stop() {
      repairReader.stop();
    }
  };
}
