// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisRasterLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
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
const ANALYTIC_OPTIONS = new Set<keyof SourceFinderOptions>([
  'windBearing',
  'emissionRate',
  'dispersion',
  'decayLength',
  'noiseSigma'
]);

const windVector = (bearing: number): [number, number] => {
  const radians = (bearing * Math.PI) / 180;
  return [Math.sin(radians), Math.cos(radians)];
};

/** Selects a deterministic, spatially distributed maximin design from eligible facility rows. */
export function selectSourceFinderSensorRows(
  positions: ArrayLike<number>,
  categories: ArrayLike<number>,
  rowCount: number,
  sensorCount: number
): number[] {
  const eligibleRows: number[] = [];
  let centerX = 0;
  let centerY = 0;
  for (let row = 0; row < rowCount; row++) {
    if (categories[row] !== 0 && categories[row] !== 3) continue;
    eligibleRows.push(row);
    centerX += positions[row * 2];
    centerY += positions[row * 2 + 1];
  }
  if (eligibleRows.length <= sensorCount) return eligibleRows;
  centerX /= eligibleRows.length;
  centerY /= eligibleRows.length;

  let firstRow = eligibleRows[0];
  let firstDistance = Number.POSITIVE_INFINITY;
  for (const row of eligibleRows) {
    const deltaX = positions[row * 2] - centerX;
    const deltaY = positions[row * 2 + 1] - centerY;
    const distance = deltaX * deltaX + deltaY * deltaY;
    if (distance < firstDistance) {
      firstDistance = distance;
      firstRow = row;
    }
  }

  const selectedRows = [firstRow];
  const selected = new Set(selectedRows);
  const nearestDistance = new Map<number, number>();
  while (selectedRows.length < sensorCount) {
    const latestRow = selectedRows[selectedRows.length - 1];
    let nextRow = -1;
    let nextDistance = -1;
    for (const row of eligibleRows) {
      if (selected.has(row)) continue;
      const deltaX = positions[row * 2] - positions[latestRow * 2];
      const deltaY = positions[row * 2 + 1] - positions[latestRow * 2 + 1];
      const distance = Math.min(
        nearestDistance.get(row) ?? Number.POSITIVE_INFINITY,
        deltaX * deltaX + deltaY * deltaY
      );
      nearestDistance.set(row, distance);
      if (distance > nextDistance) {
        nextDistance = distance;
        nextRow = row;
      }
    }
    selectedRows.push(nextRow);
    selected.add(nextRow);
  }
  return selectedRows;
}

/** Sparse real facility sites used for a synthetic source-localization experiment in Chicago. */
export async function createSourceFinder(
  ctx: SceneContext<SourceFinderOptions>
): Promise<SceneInstance<SourceFinderOptions>> {
  const dataset = ctx.datasets.get('chicago-facilities');
  const projection = dataset.getProjection();
  const allPositions = dataset.projectColumn('position');
  const categories = dataset.column<Uint8Array>('category');
  // Maximin selection makes this a reproducible placement experiment rather than a file-order sample.
  const sensorRows = selectSourceFinderSensorRows(
    allPositions,
    categories,
    dataset.count,
    SENSOR_COUNT
  );
  if (sensorRows.length !== SENSOR_COUNT) {
    throw new Error('Source finder requires 32 eligible facility sites');
  }
  const sensorPositions = new Float32Array(SENSOR_COUNT * 2);
  sensorRows.forEach((row, sensor) => {
    sensorPositions[sensor * 2] = allPositions[row * 2];
    sensorPositions[sensor * 2 + 1] = allPositions[row * 2 + 1];
  });
  const [trueX, trueY] = projection.project(...TRUE_SOURCE_LNG_LAT);
  const trueWind = windVector(TRUE_WIND_BEARING);
  const sensorValues = new Float32Array(SENSOR_COUNT);
  const downwindDistances = new Float32Array(SENSOR_COUNT);
  const sensorSides = new Uint8Array(SENSOR_COUNT);
  for (let sensor = 0; sensor < SENSOR_COUNT; sensor++) {
    const dx = sensorPositions[sensor * 2] - trueX;
    const dy = sensorPositions[sensor * 2 + 1] - trueY;
    const downwind = dx * trueWind[0] + dy * trueWind[1];
    downwindDistances[sensor] = downwind / 1000;
    sensorSides[sensor] = downwind > 0 ? 1 : 0;
    let value = 0.025;
    if (downwind > 0) {
      // The data-generating process intentionally differs from the fitted forward model.
      const sigma = 525 + 7.5 * Math.sqrt(downwind);
      const crosswind = dx * -trueWind[1] + dy * trueWind[0];
      value +=
        (1.04 / (1 + downwind / 11200)) *
        Math.exp((-0.5 * crosswind * crosswind) / (sigma * sigma));
    }
    // Reproducible heteroscedastic error combines a smooth spatial field and site-scale variation.
    const spatialError =
      Math.sin(dx / 2400 + 0.7) * Math.cos(dy / 3100 - 0.4) * 0.65 +
      Math.sin((sensor + 1) * 12.9898) * 0.35;
    value += (0.006 + 0.018 * Math.sqrt(Math.max(value, 0))) * spatialError;
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
  const scoreBuffer = resources.createBuffer('relative-fit-score', GRID_WIDTH * GRID_HEIGHT * 4);
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
      scores: importGraphBuffer(
        graph,
        'relative-fit-score',
        scoreBuffer,
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
  writeParameters();
  let analysisDirty = true;
  const summaryReader = new SummaryReader(
    resources,
    'source-finder-diagnostics',
    [{buffer: summaryBuffer, size: 8}],
    bytes => {
      const summaryWords = new Uint32Array(bytes);
      const scoreWords = new Float32Array(bytes);
      const winner = summaryWords[1];
      const cellWidth = (bounds[2] - bounds[0]) / GRID_WIDTH;
      const cellHeight = (bounds[3] - bounds[1]) / GRID_HEIGHT;
      const bestX = bounds[0] + ((winner % GRID_WIDTH) + 0.5) * cellWidth;
      const bestY = bounds[1] + (Math.floor(winner / GRID_WIDTH) + 0.5) * cellHeight;
      const localizationError = Math.hypot(bestX - trueX, bestY - trueY);
      ctx.setReadout('bestScore', scoreWords[0].toFixed(3));
      ctx.setReadout('localizationError', `${Math.round(localizationError).toLocaleString()} m`);
    }
  );
  ctx.setReadout('sensors', SENSOR_COUNT);
  ctx.setReadout('candidates', GRID_WIDTH * GRID_HEIGHT);
  let peakSensor = 0;
  for (let sensor = 1; sensor < SENSOR_COUNT; sensor++) {
    if (sensorValues[sensor] > sensorValues[peakSensor]) peakSensor = sensor;
  }
  ctx.setChart('sensorProfile', {
    kind: 'scatter',
    title: 'Synthetic readings along the generating wind axis',
    xLabel: 'Distance downwind from source (km)',
    yLabel: 'Synthetic concentration',
    x: downwindDistances,
    y: sensorValues,
    colorIndex: sensorSides,
    palette: [
      [122, 132, 154, 210],
      [255, 190, 70, 230]
    ],
    ringed: [peakSensor],
    markers: [{x: 0, label: 'source crosswind plane'}],
    guides: [{y: 0.025, label: 'background'}],
    description:
      'Synthetic sensor concentration against signed downwind distance from the known generating source. Grey sites are upwind, gold sites are downwind, and the highest reading is ringed.'
  });

  return {
    getCompiledGraphs: () => [compiled] as CompiledGPUCommandGraph<never>[],
    encode(commandEncoder) {
      if (analysisDirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        summaryReader.request(commandEncoder);
        analysisDirty = false;
      } else {
        summaryReader.flush(commandEncoder);
      }
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
          id: 'source-finder-relative-fit',
          coordinateOrigin: origin,
          gridSize: [GRID_WIDTH, GRID_HEIGHT],
          bounds,
          rowOrigin: 'south',
          values: scoreBuffer,
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
      if (ANALYTIC_OPTIONS.has(id)) {
        writeParameters();
        analysisDirty = true;
        return;
      }
      ctx.requestLayers();
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      summaryReader.stop();
      resources.destroy();
    }
  };
}
