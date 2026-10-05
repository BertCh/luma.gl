// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GPUCommandGraphEncoding
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPURasterExtremaPyramidLayout,
  GPURasterExtremaPyramid,
  type GPURasterExtremaPyramidOutput
} from '../../../src/gpu-raster/raster-pyramid/index';
import {
  getGPUTerrainSightLineParameterValues,
  getGPUTerrainViewshedParameterValues,
  GPUTerrainCumulativeViewshed,
  GPUTerrainLineOfSight,
  GPUTerrainViewshed
} from '../../../src/gpu-terrain/terrain-analysis/index';
import {
  getGPUPointHorizonParameterValues,
  getGPUPointHorizonVisibilityParameterValues,
  GPUPointHorizonProfile,
  GPUPointHorizonVisibility
} from '../../../src/gpu-terrain/point-horizon/index';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createFractalTerrain} from './terrain-analysis-oracle';

/**
 * How the five terrain contributors obtain their traversal data:
 * - `march`: exhaustive march, no pyramid.
 * - `self`: each contributor builds its own pyramid (`traversal: 'pyramid'`, no `pyramid` prop).
 * - `shared`: one `GPURasterExtremaPyramid` in the same graph feeds every contributor.
 * - `prebuilt`: the pyramid was built by an earlier graph and is only read.
 */
export type SharedPyramidMode = 'march' | 'self' | 'shared' | 'prebuilt';

export type SharedPyramidSceneOptions = {
  width: number;
  height: number;
  mode: SharedPyramidMode;
  /** Number of line-of-sight pairs, point-horizon targets and point-horizon observers. */
  pairCount?: number;
  observerCount?: number;
  azimuthCount?: number;
  /** Cumulative viewshed observers. */
  cumulativeObserverCount?: number;
  /** Fractal terrain amplitude in metres. Defaults to 400. */
  terrainAmplitude?: number;
  /** Eye height above ground in metres for every contributor. Defaults to 1.7. */
  observerHeight?: number;
  /** Earth-curvature drop coefficient in 1/metres. Defaults to 3e-5 (exaggerated). */
  curvatureCoefficient?: number;
  /** Restrict the graph to some consumers; defaults to all five. */
  only?: readonly SharedPyramidConsumer[];
  /** Replace the pyramid prop given to the consumers (used by rejection tests). */
  pyramidOverride?: (output: GPURasterExtremaPyramidOutput) => GPURasterExtremaPyramidOutput;
};

export type SharedPyramidConsumer =
  | 'viewshed'
  | 'lineOfSight'
  | 'cumulative'
  | 'profile'
  | 'horizonVisibility';

export type SharedPyramidOutputs = {
  viewshed: number[];
  lineOfSightVisibility: number[];
  lineOfSightClearance: number[];
  cumulative: number[];
  profileTangent: number[];
  profileDistance: number[];
  horizonVisibility: number[];
};

export type SharedPyramidScene = {
  /** Compiled consumer graph. */
  compiled: CompiledGPUCommandGraph<undefined>;
  /** Submits the consumer graph (and, for `shared`, its pyramid nodes) once. */
  run: () => GPUCommandGraphEncoding;
  /** Reads every output. */
  read: () => Promise<SharedPyramidOutputs>;
  destroy: () => void;
};

const CELL_SIZE = 30;

/** Deterministic pseudo-random sequence in [0, 1). */
function createSequence(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * Builds one graph holding a viewshed, line of sight, cumulative viewshed, point-horizon profile
 * and point-horizon visibility over a fractal terrain. Used by the shared-pyramid spec and
 * benchmark.
 */
export function createSharedPyramidScene(
  device: Device,
  options: SharedPyramidSceneOptions
): SharedPyramidScene {
  const {width, height, mode, only} = options;
  const pixelCount = width * height;
  const pairCount = options.pairCount ?? 256;
  const observerCount = options.observerCount ?? 4;
  const azimuthCount = options.azimuthCount ?? 90;
  const cumulativeObserverCount = options.cumulativeObserverCount ?? 2;
  const observerHeight = options.observerHeight ?? 1.7;
  const curvatureCoefficient = options.curvatureCoefficient ?? 3e-5;
  const random = createSequence(1234);
  const buffers: Buffer[] = [];
  const parameterBuffers: GPUParameterBuffer<'float32'>[] = [];
  const track = (buffer: Buffer): Buffer => {
    buffers.push(buffer);
    return buffer;
  };
  const terrain = createFractalTerrain(width, height, 7, options.terrainAmplitude ?? 400);
  const elevationBuffer = track(createInputBuffer(device, terrain));
  const column = () => 1 + random() * (width - 3);
  const row = () => 1 + random() * (height - 3);

  const observers = Float32Array.from(
    {length: observerCount * 4},
    (_, index) => [column(), row(), observerHeight, 0][index % 4]
  );
  const pairs = Float32Array.from({length: pairCount * 4}, (_, index) =>
    index % 2 === 0 ? column() : row()
  );
  const targets = Float32Array.from(
    {length: pairCount * 4},
    (_, index) => [column(), row(), 5, Math.floor(random() * observerCount)][index % 4]
  );
  const cumulativeObservers = Float32Array.from(
    {length: cumulativeObserverCount * 2},
    (_, index) => (index % 2 === 0 ? column() : row())
  );
  const observerBuffer = track(createInputBuffer(device, observers));
  const pairBuffer = track(createInputBuffer(device, pairs));
  const targetBuffer = track(createInputBuffer(device, targets));
  const cumulativeObserverBuffer = track(createInputBuffer(device, cumulativeObservers));
  const viewshedOutput = track(createOutputBuffer(device, pixelCount));
  const lineOfSightVisibility = track(createOutputBuffer(device, pairCount));
  const lineOfSightClearance = track(createOutputBuffer(device, pairCount));
  const cumulativeOutput = track(createOutputBuffer(device, pixelCount));
  const rayCount = observerCount * azimuthCount;
  const profileTangent = track(createOutputBuffer(device, rayCount));
  const profileDistance = track(createOutputBuffer(device, rayCount));
  const horizonVisibility = track(createOutputBuffer(device, pairCount));

  const makeParameters = (id: string, length: number, values: Float32Array) => {
    const buffer = new GPUParameterBuffer<'float32'>(device, {
      id,
      format: 'float32',
      length,
      values
    });
    parameterBuffers.push(buffer);
    return buffer;
  };
  const viewshedSettings = makeParameters(
    'viewshed-settings',
    8,
    getGPUTerrainViewshedParameterValues({
      observer: [width / 2, height / 2],
      observerHeight,
      cellSize: [CELL_SIZE, CELL_SIZE],
      curvatureCoefficient
    })
  );
  const sightLineValues = getGPUTerrainSightLineParameterValues({
    observerHeight,
    cellSize: [CELL_SIZE, CELL_SIZE],
    curvatureCoefficient,
    toleranceMeters: 2
  });
  const lineOfSightSettings = makeParameters('line-of-sight-settings', 12, sightLineValues);
  const cumulativeSettings = makeParameters('cumulative-settings', 12, sightLineValues);
  const horizonSettings = makeParameters(
    'horizon-settings',
    8,
    getGPUPointHorizonParameterValues({
      cellSize: [CELL_SIZE, CELL_SIZE],
      curvatureCoefficient
    })
  );
  const horizonVisibilitySettings = makeParameters(
    'horizon-visibility-settings',
    12,
    getGPUPointHorizonVisibilityParameterValues({
      cellSize: [CELL_SIZE, CELL_SIZE],
      curvatureCoefficient,
      toleranceDegrees: 0.05
    })
  );

  const traversal = mode === 'march' ? 'march' : 'pyramid';
  const layout = getGPURasterExtremaPyramidLayout(width, height, {
    firstBlockSize: 4,
    footprint: 'bilinear'
  });
  const importElevation = (graph: GPUCommandGraph, id: string) => ({
    id,
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: importGraphBuffer(graph, id, elevationBuffer, 'float32', pixelCount)
    }
  });

  let producerCompiled: CompiledGPUCommandGraph<undefined> | undefined;
  let pyramidBuffer: Buffer | undefined;
  if (mode === 'prebuilt') {
    // Earlier graph: build the pyramid once into a buffer that the consumer graph imports.
    pyramidBuffer = track(createOutputBuffer(device, 2 * layout.length));
    const earlier = new GPUCommandGraph(device, {id: 'shared-pyramid-earlier'});
    earlier.add(
      new GPURasterExtremaPyramid({
        id: 'earlier-pyramid',
        width,
        height,
        firstBlockSize: 4,
        footprint: 'bilinear',
        input: importElevation(earlier, 'earlier-elevation'),
        combined: importGraphBuffer(
          earlier,
          'earlier-combined',
          pyramidBuffer,
          'float32',
          2 * layout.length
        )
      })
    );
    producerCompiled = earlier.compile();
    submitGraph(device, producerCompiled, undefined);
  }

  const graph = new GPUCommandGraph(device, {id: `shared-pyramid-${mode}`});
  let pyramid: GPURasterExtremaPyramidOutput | undefined;
  if (mode === 'shared') {
    const producer = new GPURasterExtremaPyramid({
      id: 'shared-pyramid',
      width,
      height,
      firstBlockSize: 4,
      footprint: 'bilinear',
      input: importElevation(graph, 'shared-elevation'),
      combined: createTransientView(graph, 'shared-pyramid-combined', 'float32', 2 * layout.length)
    });
    graph.add(producer);
    pyramid = producer.output;
  } else if (mode === 'prebuilt') {
    pyramid = {
      layout,
      combined: importGraphBuffer(
        graph,
        'prebuilt-combined',
        pyramidBuffer!,
        'float32',
        2 * layout.length
      )
    };
  }
  if (pyramid && options.pyramidOverride) {
    pyramid = options.pyramidOverride(pyramid);
  }

  const settingsView = (buffer: GPUParameterBuffer<'float32'>) => buffer.importToGraph(graph);
  if (!only || only.includes('viewshed')) {
    graph.add(
      new GPUTerrainViewshed({
        id: 'viewshed',
        width,
        height,
        elevation: importElevation(graph, 'viewshed-elevation'),
        settings: settingsView(viewshedSettings),
        traversal,
        pyramid,
        visibility: importGraphBuffer(graph, 'viewshed-out', viewshedOutput, 'uint32', pixelCount)
      })
    );
  }
  if (!only || only.includes('lineOfSight')) {
    graph.add(
      new GPUTerrainLineOfSight({
        id: 'line-of-sight',
        width,
        height,
        elevation: importElevation(graph, 'los-elevation'),
        pairs: importGraphBuffer(graph, 'pairs', pairBuffer, 'float32x4', pairCount),
        settings: settingsView(lineOfSightSettings),
        traversal,
        pyramid,
        visibility: importGraphBuffer(graph, 'los-out', lineOfSightVisibility, 'uint32', pairCount),
        clearance: importGraphBuffer(graph, 'los-clear', lineOfSightClearance, 'float32', pairCount)
      })
    );
  }
  if (!only || only.includes('cumulative')) {
    graph.add(
      new GPUTerrainCumulativeViewshed({
        id: 'cumulative',
        width,
        height,
        elevation: importElevation(graph, 'cumulative-elevation'),
        observers: importGraphBuffer(
          graph,
          'cumulative-observers',
          cumulativeObserverBuffer,
          'float32x2',
          cumulativeObserverCount
        ),
        settings: settingsView(cumulativeSettings),
        traversal,
        pyramid,
        visibleCount: importGraphBuffer(
          graph,
          'cumulative-out',
          cumulativeOutput,
          'uint32',
          pixelCount
        )
      })
    );
  }
  if (!only || only.includes('profile')) {
    graph.add(
      new GPUPointHorizonProfile({
        id: 'profile',
        width,
        height,
        elevation: importElevation(graph, 'profile-elevation'),
        traversal,
        pyramid,
        azimuthCount,
        maximumDistance: Math.max(width, height) * CELL_SIZE * 0.5,
        cellSize: CELL_SIZE,
        observers: importGraphBuffer(
          graph,
          'profile-observers',
          observerBuffer,
          'float32x4',
          observerCount
        ),
        settings: settingsView(horizonSettings),
        tangent: importGraphBuffer(graph, 'profile-tangent', profileTangent, 'float32', rayCount),
        distance: importGraphBuffer(graph, 'profile-distance', profileDistance, 'float32', rayCount)
      })
    );
  }
  if (!only || only.includes('horizonVisibility')) {
    graph.add(
      new GPUPointHorizonVisibility({
        id: 'horizon-visibility',
        width,
        height,
        elevation: importElevation(graph, 'horizon-elevation'),
        traversal,
        pyramid,
        maximumDistance: Math.max(width, height) * CELL_SIZE * 0.5,
        cellSize: CELL_SIZE,
        observers: importGraphBuffer(
          graph,
          'horizon-observers',
          observerBuffer,
          'float32x4',
          observerCount
        ),
        targets: importGraphBuffer(graph, 'horizon-targets', targetBuffer, 'float32x4', pairCount),
        settings: settingsView(horizonVisibilitySettings),
        visibility: importGraphBuffer(graph, 'horizon-out', horizonVisibility, 'uint32', pairCount)
      })
    );
  }
  const compiled = graph.compile();

  return {
    compiled,
    run: () => submitGraph(device, compiled, undefined),
    read: async () => ({
      viewshed: await readUint32(viewshedOutput, pixelCount),
      lineOfSightVisibility: await readUint32(lineOfSightVisibility, pairCount),
      lineOfSightClearance: await readFloat32(lineOfSightClearance, pairCount),
      cumulative: await readUint32(cumulativeOutput, pixelCount),
      profileTangent: await readFloat32(profileTangent, rayCount),
      profileDistance: await readFloat32(profileDistance, rayCount),
      horizonVisibility: await readUint32(horizonVisibility, pairCount)
    }),
    destroy: () => {
      compiled.destroy();
      producerCompiled?.destroy();
      for (const buffer of parameterBuffers) {
        buffer.destroy();
      }
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}
