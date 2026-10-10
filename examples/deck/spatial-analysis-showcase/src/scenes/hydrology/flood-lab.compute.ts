// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT,
  GPUColumnProfile
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  FrontierFloodSimulation,
  FRONTIER_FLOOD_PARAMETER_LENGTH,
  getFrontierFloodParameterValues
} from '../frontier/frontier-flood';
import {createCanyonGrid, flipRows} from './b15-common';

/** Live controls of Flood Lab. */
export type FloodLabOptions = {
  play: boolean;
  rainfall: number;
  infiltration: number;
  sourceRate: number;
  sourceRadius: number;
  roughness: number;
  timeStep: number;
  opacity: number;
};

const GRID_STRIDE = 8;
const STEPS_PER_FRAME = 3;

/** Persistent shallow-water simulation over the real Grand Canyon DEM. */
export async function createFloodLab(
  ctx: SceneContext<FloodLabOptions>
): Promise<SceneInstance<FloodLabOptions>> {
  const grid = createCanyonGrid(ctx.datasets.get('grand-canyon-dem'), GRID_STRIDE);
  const {device} = ctx;
  const {width, height, cellCount} = grid;
  const resources = new SpatialAnalysisResources(device, 'flood-lab');
  const terrain = resources.createBuffer('terrain', flipRows(grid.elevation, width, height));
  const depth = resources.createBuffer('depth', new Float32Array(cellCount));
  const xFlux = resources.createBuffer('x-flux', (width + 1) * height * 4);
  const yFlux = resources.createBuffer('y-flux', width * (height + 1) * 4);
  const display = resources.createBuffer('display', cellCount * 4);
  const statistics = resources.createBuffer(
    'depth-statistics',
    GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4
  );
  const parameters = resources.createParameterBuffer(
    'parameters',
    'float32',
    FRONTIER_FLOOD_PARAMETER_LENGTH
  );
  const source: [number, number] = [Math.floor(width * 0.47), Math.floor(height * 0.46)];
  const graph = new GPUCommandGraph<void>(device, {id: 'flood-lab'});
  graph.add(
    new FrontierFloodSimulation({
      width,
      height,
      terrain: importGraphBuffer(graph, 'terrain', terrain, 'float32', cellCount),
      parameters: parameters.importToGraph(graph),
      state: {
        depth: importGraphBuffer(graph, 'depth', depth, 'float32', cellCount),
        xFlux: importGraphBuffer(graph, 'x-flux', xFlux, 'float32', (width + 1) * height),
        yFlux: importGraphBuffer(graph, 'y-flux', yFlux, 'float32', width * (height + 1))
      },
      display: importGraphBuffer(graph, 'display', display, 'float32', cellCount),
      stepsPerEncoding: STEPS_PER_FRAME
    })
  );
  const compiled = resources.track(graph.compile());
  const profileGraph = new GPUCommandGraph<void>(device, {id: 'flood-lab-profile'});
  profileGraph.add(
    new GPUColumnProfile({
      id: 'depth-profile',
      columns: [
        {
          values: importGraphBuffer(profileGraph, 'depth', depth, 'float32', cellCount)
        }
      ],
      histogramBinCount: 1,
      hyperLogLogPrecision: 4,
      topCategoryCount: 1,
      output: {
        statistics: importGraphBuffer(
          profileGraph,
          'statistics',
          statistics,
          'float32',
          GPU_COLUMN_PROFILE_STATISTIC_COUNT
        )
      }
    })
  );
  const compiledProfile = resources.track(profileGraph.compile());
  let simulatedSeconds = 0;
  let simulatedFrames = 0;
  let profileStale = true;
  let displayMaximum = 2.5;

  const cellArea = grid.groundCellSize[0] * grid.groundCellSize[1];
  const studyArea = cellArea * cellCount;
  const getSourceWeightSum = (radius: number): number => {
    let sum = 0;
    const bound = Math.ceil(radius);
    for (let row = -bound; row <= bound; row++) {
      for (let column = -bound; column <= bound; column++) {
        sum += Math.max(1 - Math.hypot(column, row) / Math.max(radius, 0.0001), 0);
      }
    }
    return sum;
  };

  const updateScenarioReadouts = () => {
    const options = ctx.options;
    const rainfallCubicMetersPerSecond = (options.rainfall / 1000 / 3600) * studyArea;
    const releaseCubicMetersPerSecond =
      (options.sourceRate / 1000) * getSourceWeightSum(options.sourceRadius) * cellArea;
    const infiltrationCapacityCubicMetersPerSecond =
      (options.infiltration / 1000 / 3600) * studyArea;
    ctx.setReadout(
      'forcing',
      `${(rainfallCubicMetersPerSecond + releaseCubicMetersPerSecond).toFixed(1)} m³/s input · ≤${infiltrationCapacityCubicMetersPerSecond.toFixed(1)} m³/s infiltration capacity`
    );
  };

  const writeParameters = () => {
    const options = ctx.options;
    parameters.write(
      getFrontierFloodParameterValues({
        cellSize: grid.groundCellSize,
        timeStep: options.timeStep,
        rainfallRate: options.rainfall / 1000 / 3600,
        infiltrationRate: options.infiltration / 1000 / 3600,
        manningCoefficient: options.roughness,
        maximumDepth: 20,
        source,
        sourceRadius: options.sourceRadius,
        sourceRate: options.sourceRate / 1000
      })
    );
  };

  const reset = () => {
    depth.write(new Float32Array(cellCount));
    xFlux.write(new Float32Array((width + 1) * height));
    yFlux.write(new Float32Array(width * (height + 1)));
    display.write(new Float32Array(cellCount));
    simulatedSeconds = 0;
    simulatedFrames = 0;
    profileStale = true;
    ctx.setReadout('simulatedTime', '0 min');
  };
  reset();
  ctx.setReadout(
    'grid',
    `${width.toLocaleString()} × ${height.toLocaleString()} cells · ${Math.round(grid.groundCellSize[0])} m`
  );
  updateScenarioReadouts();

  let destroyed = false;
  const summary = new SummaryReader(
    resources,
    'flood-lab-depth',
    [{buffer: statistics, size: GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4}],
    bytes => {
      if (destroyed) return;
      const values = new Float32Array(bytes);
      const maximum = values[GPU_COLUMN_PROFILE_STATISTIC.maximum];
      const sum = values[GPU_COLUMN_PROFILE_STATISTIC.sum];
      if (!Number.isFinite(maximum) || !Number.isFinite(sum)) return;
      ctx.setReadout('maximumDepth', maximum);
      ctx.setReadout('storedWater', `${Math.round(sum * cellArea).toLocaleString('en-US')} m³`);
      const nextDisplayMaximum = Math.max(maximum, 0.01);
      ctx.setLegendExtent('depth', [0, nextDisplayMaximum]);
      if (Math.abs(nextDisplayMaximum - displayMaximum) > displayMaximum * 0.01) {
        displayMaximum = nextDisplayMaximum;
        ctx.requestLayers();
      }
    }
  );

  return {
    getCompiledGraphs: () => [compiled, compiledProfile] as CompiledGPUCommandGraph<never>[],
    encode(commandEncoder, frame) {
      writeParameters();
      if (ctx.options.play) {
        compiled.encode(commandEncoder, {parameters: undefined});
        simulatedSeconds += ctx.options.timeStep * STEPS_PER_FRAME;
        simulatedFrames++;
        if (simulatedFrames % 20 === 0) profileStale = true;
        if (frame.frameIndex % 20 === 0) {
          ctx.setReadout('simulatedTime', `${Math.round(simulatedSeconds / 60)} min`);
        }
      }
      if (profileStale) {
        compiledProfile.encode(commandEncoder, {parameters: undefined});
        summary.request(commandEncoder);
        profileStale = false;
      }
      summary.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      return [
        new SpatialAnalysisRasterLayer({
          id: 'flood-lab-depth',
          coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
          gridSize: [width, height],
          bounds: grid.bounds,
          rowOrigin: 'south',
          values: display,
          valueFormat: 'float32',
          colormap: 'cividis',
          valueRange: [0, displayMaximum],
          discardAtOrBelow: 0.002,
          color: [255, 255, 255, Math.round(ctx.options.opacity * 255)]
        })
      ];
    },
    setOption(id) {
      if (id === 'opacity') ctx.requestLayers();
      if (
        id === 'rainfall' ||
        id === 'infiltration' ||
        id === 'sourceRate' ||
        id === 'sourceRadius'
      ) {
        updateScenarioReadouts();
      }
    },
    onAction(id) {
      if (id === 'reset') reset();
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      destroyed = true;
      summary.stop();
      resources.destroy();
    }
  };
}
