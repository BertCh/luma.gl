// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {LoadedDataset, LoadedRaster} from '../../data/catalog';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {LocalMetricProjection} from '../../engine/projection';
import type {SceneContext, SceneInstance} from '../scene';
import {
  FRONTIER_NEURAL_PARAMETER_LENGTH,
  FrontierNeuralAutomaton,
  getFrontierNeuralParameterValues,
  getFrontierNeuralWeights
} from '../frontier/frontier-neural';

export type CloudRepairPlace = 'oahu' | 'venice';
export type CloudRepairDisplay = 'cloudy' | 'neural' | 'naive' | 'prior';

/** Live controls of the temporal cloud-repair field. */
export type NeuralGrowthOptions = {
  place: CloudRepairPlace;
  display: CloudRepairDisplay;
  play: boolean;
  stepScale: number;
  opacity: number;
};

type PlaceRasters = {
  title: string;
  targetDate: string;
  priorDate: string;
  target: LoadedRaster;
  prior: LoadedRaster;
  mask: LoadedRaster;
};

type PreparedPlace = PlaceRasters & {
  origin: [number, number];
  projection: LocalMetricProjection;
  bounds: [number, number, number, number];
  state: Float32Array;
  priorValues: Float32Array;
  cloudyDisplay: Uint32Array;
  neuralDisplay: Uint32Array;
  naiveDisplay: Uint32Array;
  priorDisplay: Uint32Array;
  maskedPixels: number;
};

const STEPS_PER_FRAME = 4;

function packRgba(red: number, green: number, blue: number): number {
  return (
    (Math.round(Math.max(0, Math.min(1, red)) * 255) |
      (Math.round(Math.max(0, Math.min(1, green)) * 255) << 8) |
      (Math.round(Math.max(0, Math.min(1, blue)) * 255) << 16) |
      (255 << 24)) >>>
    0
  );
}

function validatePlace(place: PlaceRasters): void {
  const {target, prior, mask} = place;
  if (target.encoding !== 'rgba8' || target.bands !== 4) {
    throw new Error(`${place.title} target is not an rgba8 raster`);
  }
  if (
    prior.encoding !== 'rgba8' ||
    prior.bands !== 4 ||
    prior.width !== target.width ||
    prior.height !== target.height
  ) {
    throw new Error(`${place.title} prior does not match its target`);
  }
  if (
    mask.encoding !== 'uint8-classes' ||
    mask.bands !== 1 ||
    mask.width !== target.width ||
    mask.height !== target.height
  ) {
    throw new Error(`${place.title} cloud mask does not match its target`);
  }
}

function preparePlace(dataset: LoadedDataset, place: PlaceRasters): PreparedPlace {
  validatePlace(place);
  const {target, prior, mask} = place;
  const cellCount = target.width * target.height;
  const after = target.values as Uint8Array;
  const before = prior.values as Uint8Array;
  const cloudMask = mask.values as Uint8Array;
  const state = new Float32Array(cellCount * 4);
  const priorValues = new Float32Array(cellCount * 3);
  const cloudyDisplay = new Uint32Array(cellCount);
  const neuralDisplay = new Uint32Array(cellCount);
  const naiveDisplay = new Uint32Array(cellCount);
  const priorDisplay = new Uint32Array(cellCount);
  const meanResidual = [0, 0, 0];
  let knownPixels = 0;
  let maskedPixels = 0;

  for (let cell = 0; cell < cellCount; cell++) {
    const pixelBase = cell * 4;
    if (cloudMask[cell]) {
      maskedPixels++;
      continue;
    }
    for (let channel = 0; channel < 3; channel++) {
      meanResidual[channel] += (after[pixelBase + channel] - before[pixelBase + channel]) / 255;
    }
    knownPixels++;
  }
  for (let channel = 0; channel < 3; channel++) meanResidual[channel] /= Math.max(knownPixels, 1);

  for (let cell = 0; cell < cellCount; cell++) {
    const pixelBase = cell * 4;
    const stateBase = cell * 4;
    const priorBase = cell * 3;
    const beforeRed = before[pixelBase] / 255;
    const beforeGreen = before[pixelBase + 1] / 255;
    const beforeBlue = before[pixelBase + 2] / 255;
    const afterRed = after[pixelBase] / 255;
    const afterGreen = after[pixelBase + 1] / 255;
    const afterBlue = after[pixelBase + 2] / 255;
    const masked = cloudMask[cell] !== 0;
    priorValues[priorBase] = beforeRed;
    priorValues[priorBase + 1] = beforeGreen;
    priorValues[priorBase + 2] = beforeBlue;
    state[stateBase] = masked ? meanResidual[0] : afterRed - beforeRed;
    state[stateBase + 1] = masked ? meanResidual[1] : afterGreen - beforeGreen;
    state[stateBase + 2] = masked ? meanResidual[2] : afterBlue - beforeBlue;
    state[stateBase + 3] = masked ? 0 : 1;
    cloudyDisplay[cell] = packRgba(afterRed, afterGreen, afterBlue);
    priorDisplay[cell] = packRgba(beforeRed, beforeGreen, beforeBlue);
    naiveDisplay[cell] = masked ? priorDisplay[cell] : cloudyDisplay[cell];
    neuralDisplay[cell] = masked
      ? packRgba(
          beforeRed + meanResidual[0],
          beforeGreen + meanResidual[1],
          beforeBlue + meanResidual[2]
        )
      : cloudyDisplay[cell];
  }

  const [west, south, east, north] = target.bounds;
  const origin: [number, number] = [(west + east) / 2, (south + north) / 2];
  const projection = dataset.getProjection(origin);
  const [minX, minY] = projection.project(west, south);
  const [maxX, maxY] = projection.project(east, north);
  return {
    ...place,
    origin,
    projection,
    bounds: [minX, minY, maxX, maxY],
    state,
    priorValues,
    cloudyDisplay,
    neuralDisplay,
    naiveDisplay,
    priorDisplay,
    maskedPixels
  };
}

/** Tiny neural cellular temporal inpainting over two real Sentinel-2 cloud/clear pairs. */
export async function createNeuralGrowth(
  ctx: SceneContext<NeuralGrowthOptions>
): Promise<SceneInstance<NeuralGrowthOptions>> {
  const dataset = ctx.datasets.get('sentinel-cloud-repair');
  const oahuTarget = dataset.raster;
  if (!oahuTarget) throw new Error('sentinel-cloud-repair has no primary Oahu target');
  const [oahuPrior, oahuMask, veniceTarget, venicePrior, veniceMask] = await Promise.all([
    dataset.loadRaster('oahuPrior', ctx.signal),
    dataset.loadRaster('oahuMask', ctx.signal),
    dataset.loadRaster('veniceTarget', ctx.signal),
    dataset.loadRaster('venicePrior', ctx.signal),
    dataset.loadRaster('veniceMask', ctx.signal)
  ]);
  ctx.signal.throwIfAborted();
  const places: Record<CloudRepairPlace, PreparedPlace> = {
    oahu: preparePlace(dataset, {
      title: 'Waimānalo, Oʻahu',
      targetDate: '29 Sep 2026',
      priorDate: '3 Feb 2026',
      target: oahuTarget,
      prior: oahuPrior,
      mask: oahuMask
    }),
    venice: preparePlace(dataset, {
      title: 'Venice Lagoon',
      targetDate: '25 Aug 2026',
      priorDate: '24 Aug 2026',
      target: veniceTarget,
      prior: venicePrior,
      mask: veniceMask
    })
  };
  const {width, height} = oahuTarget;
  const cellCount = width * height;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'neural-growth');
  const upload = <T extends Float32Array | Uint32Array>(name: string, values: T) => {
    const buffer = resources.createBuffer(name, values.byteLength);
    buffer.write(values);
    return buffer;
  };
  let active = places[ctx.options.place];
  const stateA = upload('state-a', active.state);
  const stateB = upload('state-b', active.state);
  const weights = upload('weights', getFrontierNeuralWeights());
  const habitat = upload('habitat', new Float32Array(cellCount).fill(1));
  const prior = upload('prior', active.priorValues);
  const cloudyDisplay = upload('cloudy-display', active.cloudyDisplay);
  const naiveDisplay = upload('naive-display', active.naiveDisplay);
  const referenceDisplay = upload('reference-display', active.priorDisplay);
  const display = upload('display', active.neuralDisplay);
  const parameters = resources.createParameterBuffer(
    'parameters',
    'float32',
    FRONTIER_NEURAL_PARAMETER_LENGTH
  );
  const buildGraph = (id: string, input: typeof stateA, output: typeof stateB) => {
    const graph = new GPUCommandGraph<void>(device, {id});
    graph.add(
      new FrontierNeuralAutomaton({
        id,
        width,
        height,
        state: importGraphBuffer(graph, `${id}-state`, input, 'float32', cellCount * 4),
        weights: importGraphBuffer(graph, `${id}-weights`, weights, 'float32', 212),
        parameters: parameters.importToGraph(graph),
        habitat: importGraphBuffer(graph, `${id}-habitat`, habitat, 'float32', cellCount),
        prior: importGraphBuffer(graph, `${id}-prior`, prior, 'float32', cellCount * 3),
        nextState: importGraphBuffer(graph, `${id}-next`, output, 'float32', cellCount * 4),
        display: importGraphBuffer(graph, `${id}-display`, display, 'uint32', cellCount),
        boundary: 'clamp'
      })
    );
    return resources.track(graph.compile());
  };
  const graphs = [
    buildGraph('neural-growth-a', stateA, stateB),
    buildGraph('neural-growth-b', stateB, stateA)
  ];
  let generation = 0;

  const updateReadouts = () => {
    ctx.setReadout('generation', generation);
    ctx.setReadout('cloud', `${((active.maskedPixels / cellCount) * 100).toFixed(1)}%`);
    ctx.setReadout('image', `${active.title} · ${active.targetDate} / ${active.priorDate}`);
  };
  const reset = () => {
    stateA.write(active.state);
    stateB.write(active.state);
    display.write(active.neuralDisplay);
    generation = 0;
    updateReadouts();
  };
  const selectPlace = (place: CloudRepairPlace, moveCamera: boolean) => {
    active = places[place];
    prior.write(active.priorValues);
    cloudyDisplay.write(active.cloudyDisplay);
    naiveDisplay.write(active.naiveDisplay);
    referenceDisplay.write(active.priorDisplay);
    reset();
    if (moveCamera) ctx.fitBounds(active.target.bounds, {transitionMs: 700, maxZoom: 13.6});
    ctx.requestLayers();
  };

  ctx.setReadout('weights', `${getFrontierNeuralWeights().byteLength} bytes`);
  ctx.setReadout('cells', cellCount);
  updateReadouts();
  ctx.setStatus('Cloud mask and both dated observations are resident; choose a repair method.');

  return {
    getCompiledGraphs: () => graphs as CompiledGPUCommandGraph<never>[],
    encode(commandEncoder, frame) {
      if (!ctx.options.play || ctx.options.display !== 'neural') return;
      parameters.write(
        getFrontierNeuralParameterValues({
          stepScale: ctx.options.stepScale,
          mutation: 0,
          damping: 0,
          generation
        })
      );
      for (let step = 0; step < STEPS_PER_FRAME; step++) {
        graphs[generation & 1].encode(commandEncoder, {parameters: undefined});
        generation++;
      }
      if (frame.frameIndex % 15 === 0) ctx.setReadout('generation', generation);
    },
    getLayers(): Layer[] {
      const values =
        ctx.options.display === 'cloudy'
          ? cloudyDisplay
          : ctx.options.display === 'naive'
            ? naiveDisplay
            : ctx.options.display === 'prior'
              ? referenceDisplay
              : display;
      return [
        new SpatialAnalysisRasterLayer({
          id: 'neural-growth-field',
          coordinateOrigin: [active.origin[0], active.origin[1], 0],
          gridSize: [width, height],
          bounds: active.bounds,
          rowOrigin: 'north',
          values,
          valueFormat: 'uint32',
          colormap: 'rgba',
          color: [255, 255, 255, Math.round(ctx.options.opacity * 255)]
        })
      ];
    },
    setOption(id) {
      if (id === 'place') selectPlace(ctx.options.place, true);
      if (id === 'display' || id === 'opacity') ctx.requestLayers();
    },
    onAction(id) {
      if (id === 'restart') reset();
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      resources.destroy();
    }
  };
}
