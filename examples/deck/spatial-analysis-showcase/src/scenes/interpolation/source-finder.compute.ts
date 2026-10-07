// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisRasterLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {
  FRONTIER_SOURCE_PARAMETER_LENGTH,
  FrontierSourceInference,
  getFrontierSourceParameterValues
} from '../frontier/frontier-source';

/** Live inverse-model assumptions. */
export type SourceFinderOptions = {
  windBearing: number;
  emissionRate: number;
  dispersion: number;
  decayLength: number;
  noiseSigma: number;
  showSensors: boolean;
  revealSource: boolean;
  opacity: number;
};

const GRID_WIDTH = 180;
const GRID_HEIGHT = 200;
const SENSOR_COUNT = 32;
const TRUE_SOURCE_LNG_LAT = [-87.655, 41.885] as const;
const TRUE_WIND_BEARING = 65;

const windVector = (bearing: number): [number, number] => {
  const radians = (bearing * Math.PI) / 180;
  return [Math.sin(radians), Math.cos(radians)];
};

/** Sparse real facility sites around a staged, synthetic release in Chicago. */
export async function createSourceFinder(
  ctx: SceneContext<SourceFinderOptions>
): Promise<SceneInstance<SourceFinderOptions>> {
  const dataset = ctx.datasets.get('chicago-facilities');
  const projection = dataset.getProjection();
  const allPositions = dataset.projectColumn('position');
  const categories = dataset.column<Uint8Array>('category');
  const sensorRows: number[] = [];
  for (let row = 0; row < dataset.count && sensorRows.length < SENSOR_COUNT; row++) {
    // Hospitals and fire stations provide a plausible distributed emergency sensor network.
    if (categories[row] === 0 || categories[row] === 3) sensorRows.push(row);
  }
  const sensorPositions = new Float32Array(SENSOR_COUNT * 2);
  sensorRows.forEach((row, sensor) => {
    sensorPositions[sensor * 2] = allPositions[row * 2];
    sensorPositions[sensor * 2 + 1] = allPositions[row * 2 + 1];
  });
  const [trueX, trueY] = projection.project(...TRUE_SOURCE_LNG_LAT);
  const trueWind = windVector(TRUE_WIND_BEARING);
  const sensorValues = new Float32Array(SENSOR_COUNT);
  for (let sensor = 0; sensor < SENSOR_COUNT; sensor++) {
    const dx = sensorPositions[sensor * 2] - trueX;
    const dy = sensorPositions[sensor * 2 + 1] - trueY;
    const downwind = dx * trueWind[0] + dy * trueWind[1];
    let value = 0.025;
    if (downwind > 0) {
      const sigma = 450 + 9 * Math.sqrt(downwind);
      const crosswind = dx * -trueWind[1] + dy * trueWind[0];
      value +=
        (1 / (1 + downwind / 9000)) * Math.exp((-0.5 * crosswind * crosswind) / (sigma * sigma));
    }
    // Deterministic small perturbation: the incident is reproducible and honest about being staged.
    value += 0.012 * Math.sin(sensor * 12.9898);
    sensorValues[sensor] = Math.max(0, value);
  }
  const [west, south, east, north] = dataset.manifest.bbox;
  const [minimumX, minimumY] = projection.project(west, south);
  const [maximumX, maximumY] = projection.project(east, north);
  const bounds = [minimumX, minimumY, maximumX, maximumY] as const;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'source-finder');
  const positionsBuffer = resources.createBuffer('sensor-positions', sensorPositions);
  const valuesBuffer = resources.createBuffer('sensor-values', sensorValues);
  const likelihoodBuffer = resources.createBuffer('likelihood', GRID_WIDTH * GRID_HEIGHT * 4);
  const summaryBuffer = resources.createBuffer('summary', 8);
  const bestPositionBuffer = resources.createBuffer('best-position', 8);
  const truePositionBuffer = resources.createBuffer('true-position', Float32Array.of(trueX, trueY));
  const parameters = resources.createParameterBuffer(
    'parameters',
    'float32',
    FRONTIER_SOURCE_PARAMETER_LENGTH
  );
  const graph = new GPUCommandGraph<void>(device, {id: 'source-finder'});
  graph.add(
    new FrontierSourceInference({
      width: GRID_WIDTH,
      height: GRID_HEIGHT,
      sensorCount: SENSOR_COUNT,
      sensorPositions: importGraphBuffer(
        graph,
        'sensor-positions',
        positionsBuffer,
        'float32x2',
        SENSOR_COUNT
      ),
      sensorValues: importGraphBuffer(
        graph,
        'sensor-values',
        valuesBuffer,
        'float32',
        SENSOR_COUNT
      ),
      parameters: parameters.importToGraph(graph),
      likelihoods: importGraphBuffer(
        graph,
        'likelihood',
        likelihoodBuffer,
        'float32',
        GRID_WIDTH * GRID_HEIGHT
      ),
      summary: importGraphBuffer(graph, 'summary', summaryBuffer, 'uint32', 2),
      bestPosition: importGraphBuffer(graph, 'best-position', bestPositionBuffer, 'float32x2', 1)
    })
  );
  const compiled = resources.track(graph.compile());
  const writeParameters = () => {
    const options = ctx.options;
    parameters.write(
      getFrontierSourceParameterValues({
        bounds,
        wind: windVector(options.windBearing),
        emissionRate: options.emissionRate,
        dispersionBase: options.dispersion,
        dispersionGrowth: 9,
        decayLength: options.decayLength,
        noiseSigma: options.noiseSigma,
        background: 0.025
      })
    );
  };
  ctx.setReadout('sensors', SENSOR_COUNT);
  ctx.setReadout('candidates', GRID_WIDTH * GRID_HEIGHT);

  return {
    getCompiledGraphs: () => [compiled] as CompiledGPUCommandGraph<never>[],
    encode(commandEncoder) {
      writeParameters();
      compiled.encode(commandEncoder, {parameters: undefined});
    },
    getLayers(): Layer[] {
      const options = ctx.options;
      const origin: [number, number, number] = [
        dataset.defaultOrigin[0],
        dataset.defaultOrigin[1],
        0
      ];
      const layers: Layer[] = [
        new SpatialAnalysisRasterLayer({
          id: 'source-finder-likelihood',
          coordinateOrigin: origin,
          gridSize: [GRID_WIDTH, GRID_HEIGHT],
          bounds,
          rowOrigin: 'south',
          values: likelihoodBuffer,
          valueFormat: 'float32',
          colormap: 'magma',
          valueRange: [0.05, 1],
          discardAtOrBelow: 0.03,
          color: [255, 255, 255, Math.round(options.opacity * 255)]
        }),
        new SpatialAnalysisPointLayer({
          id: 'source-finder-best',
          coordinateOrigin: origin,
          positions: bestPositionBuffer,
          instanceCount: 1,
          radiusPixels: 8,
          color: [0, 190, 255, 255]
        })
      ];
      if (options.showSensors) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'source-finder-sensors',
            coordinateOrigin: origin,
            positions: positionsBuffer,
            values: valuesBuffer,
            valueFormat: 'float32',
            colormap: 'cividis',
            valueRange: [0, 1],
            instanceCount: SENSOR_COUNT,
            radiusPixels: 4,
            color: [255, 255, 255, 255]
          })
        );
      }
      if (options.revealSource) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'source-finder-truth',
            coordinateOrigin: origin,
            positions: truePositionBuffer,
            instanceCount: 1,
            radiusPixels: 5,
            color: [255, 88, 70, 255]
          })
        );
      }
      return layers;
    },
    setOption(id) {
      if (id === 'showSensors' || id === 'revealSource' || id === 'opacity') ctx.requestLayers();
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      resources.destroy();
    }
  };
}
