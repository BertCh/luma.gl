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
  FRONTIER_RESIDUAL_PARAMETER_LENGTH,
  FrontierResidualDiffusion,
  getFrontierResidualDiffusionParameterValues
} from './frontier-neural';
import {prepareCloudRepairStyle, type CloudRepairStyle} from './cloud-repair-style';

export type CloudRepairPlace = 'pilbara' | 'plateau';
export type CloudRepairDisplay = 'cloudy' | 'diffusion' | 'naive' | 'prior';

/** Live controls of the temporal cloud-repair field. */
export type ResidualRepairOptions = {
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
  diffusionDisplay: Uint32Array;
  naiveDisplay: Uint32Array;
  priorDisplay: Uint32Array;
  maskedPixels: number;
  cloudPixels: number;
  shadowPixels: number;
  guardedPixels: number;
  style: CloudRepairStyle;
};

const STEPS_PER_FRAME = 16;
/** Mask class written by the dataset build for cloud; 2 is cloud shadow, 0 clear. */
const CLOUD_CLASS = 1;

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
  const {state, priorValues, repairMask, style} = prepareCloudRepairStyle(
    after,
    before,
    cloudMask,
    target.width,
    target.height
  );
  const cloudyDisplay = new Uint32Array(cellCount);
  const diffusionDisplay = new Uint32Array(cellCount);
  const naiveDisplay = new Uint32Array(cellCount);
  const priorDisplay = new Uint32Array(cellCount);
  let maskedPixels = 0;
  let cloudPixels = 0;
  let shadowPixels = 0;
  let guardedPixels = 0;

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
    const masked = repairMask[cell] !== 0;
    if (masked) maskedPixels++;
    if (cloudMask[cell] === CLOUD_CLASS) cloudPixels++;
    else if (cloudMask[cell]) shadowPixels++;
    else if (masked) guardedPixels++;
    cloudyDisplay[cell] = packRgba(afterRed, afterGreen, afterBlue);
    priorDisplay[cell] = packRgba(beforeRed, beforeGreen, beforeBlue);
    naiveDisplay[cell] = masked ? priorDisplay[cell] : cloudyDisplay[cell];
    diffusionDisplay[cell] = masked
      ? packRgba(
          priorValues[priorBase] + state[stateBase],
          priorValues[priorBase + 1] + state[stateBase + 1],
          priorValues[priorBase + 2] + state[stateBase + 2]
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
    diffusionDisplay,
    naiveDisplay,
    priorDisplay,
    maskedPixels,
    cloudPixels,
    shadowPixels,
    guardedPixels,
    style
  };
}

/** Deterministic cellular residual diffusion over two Sentinel-2 cloud/clear pairs. */
export async function createResidualRepair(
  ctx: SceneContext<ResidualRepairOptions>
): Promise<SceneInstance<ResidualRepairOptions>> {
  const dataset = ctx.datasets.get('sentinel-cloud-repair');
  const pilbaraTarget = dataset.raster;
  if (!pilbaraTarget) throw new Error('sentinel-cloud-repair has no primary Pilbara target');
  const [pilbaraPrior, pilbaraMask, plateauTarget, plateauPrior, plateauMask] = await Promise.all([
    dataset.loadRaster('pilbaraPrior', ctx.signal),
    dataset.loadRaster('pilbaraMask', ctx.signal),
    dataset.loadRaster('plateauTarget', ctx.signal),
    dataset.loadRaster('plateauPrior', ctx.signal),
    dataset.loadRaster('plateauMask', ctx.signal)
  ]);
  ctx.signal.throwIfAborted();
  const places: Record<CloudRepairPlace, PreparedPlace> = {
    pilbara: preparePlace(dataset, {
      title: 'Hamersley Range, Pilbara',
      targetDate: '4 Feb 2026',
      priorDate: '30 Jan 2026',
      target: pilbaraTarget,
      prior: pilbaraPrior,
      mask: pilbaraMask
    }),
    plateau: preparePlace(dataset, {
      title: 'Colorado Plateau, Utah',
      targetDate: '29 Jul 2026',
      priorDate: '27 Jul 2026',
      target: plateauTarget,
      prior: plateauPrior,
      mask: plateauMask
    })
  };
  if (
    plateauTarget.width !== pilbaraTarget.width ||
    plateauTarget.height !== pilbaraTarget.height
  ) {
    throw new Error('sentinel-cloud-repair places must share one grid size');
  }
  const {width, height} = pilbaraTarget;
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
  const prior = upload('prior', active.priorValues);
  const cloudyDisplay = upload('cloudy-display', active.cloudyDisplay);
  const naiveDisplay = upload('naive-display', active.naiveDisplay);
  const referenceDisplay = upload('reference-display', active.priorDisplay);
  const display = upload('display', active.diffusionDisplay);
  const parameters = resources.createParameterBuffer(
    'parameters',
    'float32',
    FRONTIER_RESIDUAL_PARAMETER_LENGTH
  );
  const buildGraph = async (id: string, input: typeof stateA, output: typeof stateB) => {
    const graph = new GPUCommandGraph<void>(device, {id});
    graph.add(
      new FrontierResidualDiffusion({
        id,
        width,
        height,
        state: importGraphBuffer(graph, `${id}-state`, input, 'float32', cellCount * 4),
        parameters: parameters.importToGraph(graph),
        prior: importGraphBuffer(graph, `${id}-prior`, prior, 'float32', cellCount * 3),
        nextState: importGraphBuffer(graph, `${id}-next`, output, 'float32', cellCount * 4),
        display: importGraphBuffer(graph, `${id}-display`, display, 'uint32', cellCount),
        boundary: 'clamp'
      })
    );
    return resources.track(await graph.compileAsync());
  };
  const graphs = [
    await buildGraph('residual-repair-a', stateA, stateB),
    await buildGraph('residual-repair-b', stateB, stateA)
  ];
  let generation = 0;

  const updateReadouts = () => {
    ctx.setReadout('iterations', generation);
    const percent = (pixels: number) => `${((pixels / cellCount) * 100).toFixed(1)}%`;
    ctx.setReadout(
      'cloud',
      `${percent(active.maskedPixels)} (${percent(active.cloudPixels)} cloud · ${percent(active.shadowPixels)} shadow · ${percent(active.guardedPixels)} guard)`
    );
    ctx.setReadout(
      'style',
      `${active.style.sampleCount.toLocaleString()} clear pairs · RGB gains ${active.style.gain.map(value => value.toFixed(2)).join(' / ')}`
    );
    ctx.setReadout('image', `${active.title} · ${active.targetDate} / ${active.priorDate}`);
  };
  const reset = () => {
    stateA.write(active.state);
    stateB.write(active.state);
    display.write(active.diffusionDisplay);
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
    if (moveCamera) ctx.fitBounds(active.target.bounds, {transitionMs: 700, maxZoom: 12});
    ctx.requestLayers();
  };

  ctx.setReadout('cells', cellCount);
  updateReadouts();
  ctx.setStatus(
    'Current-day colour fit and guarded mask are ready; GPU diffusion refines local seams.'
  );

  return {
    getCompiledGraphs: () => graphs as CompiledGPUCommandGraph<never>[],
    encode(commandEncoder, frame) {
      if (!ctx.options.play || ctx.options.display !== 'diffusion') return;
      parameters.write(
        getFrontierResidualDiffusionParameterValues({
          stepScale: ctx.options.stepScale
        })
      );
      for (let step = 0; step < STEPS_PER_FRAME; step++) {
        graphs[generation & 1].encode(commandEncoder, {parameters: undefined});
        generation++;
      }
      if (frame.frameIndex % 15 === 0) ctx.setReadout('iterations', generation);
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
          id: 'residual-repair-field',
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
