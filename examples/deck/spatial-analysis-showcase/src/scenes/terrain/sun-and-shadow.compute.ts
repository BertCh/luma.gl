// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUReliefShadingParameterValues,
  getGPUSolarIrradianceParameterValues,
  getGPUSolarIrradianceSunTable,
  getGPUSolarPositionParameterValues,
  getGPUSolarShadowMaskParameterValues,
  getGPUTerrainCastShadowParameterValues,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainHorizonParameterValues,
  getSolarPosition,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH,
  GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES,
  GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
  GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE,
  GPU_SOLAR_POSITION_PARAMETER_LENGTH,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPUReliefShading,
  GPUSolarIrradiance,
  GPUSolarPosition,
  GPUSolarShadowMask,
  GPUTerrainCastShadow,
  GPUTerrainDerivatives,
  GPUTerrainHorizon
} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import type {PaintSpec} from './b14a-colorize';
import {loadAlpsGrid} from './b14a-grid';
import {Stage, TerrainSession, type ValueStats} from './b14a-session';
import {StageRegistry, type StageEntry} from './b14a-stage-registry';
import {ALPINE_ELEVATION_STOPS} from './relief-visualization.style';
import {
  formatDay,
  formatHour,
  getUtcFromZurich,
  getZurichZoneName,
  type SunOptions,
  type SunProduct
} from './sun-and-shadow.style';

export type {SunOptions, SunProduct};

type StageId = 'derivatives' | 'horizon' | 'sun' | 'cast' | 'irradiance' | 'look' | 'difference';

const DEPENDENCIES: Record<StageId, readonly StageId[]> = {
  derivatives: [],
  horizon: [],
  sun: ['horizon', 'derivatives'],
  cast: [],
  irradiance: ['horizon', 'derivatives'],
  look: ['horizon'],
  difference: ['sun', 'cast']
};

/** Sun table capacity (topology): one day of 5 minute rows. */
const SUN_TABLE_CAPACITY = 288;
const SUN_READ_INTERVAL_MILLISECONDS = 120;
const REBUILD_DELAY_MILLISECONDS = 250;

function getStageKey(id: StageId, state: SunOptions): string {
  switch (id) {
    case 'horizon':
      return `${state.horizonDirections}|${state.horizonRadius}|${state.horizonAlgorithm === 'sweep' ? 1 : state.horizonGrowth}|${state.horizonFormat}|${state.horizonAlgorithm}`;
    case 'sun':
      return `${state.refraction}|${state.horizonFormat}|${state.horizonDirections}`;
    case 'cast':
      return state.castRadius;
    case 'irradiance':
      return state.horizonFormat + state.horizonDirections;
    default:
      return id;
  }
}

/** Option id to the stages whose parameters it feeds. */
const PARAMETER_STAGES: Partial<Record<keyof SunOptions, readonly StageId[]>> = {
  dayOfYear: ['sun', 'cast', 'irradiance'],
  hour: ['sun', 'cast'],
  twilight: ['sun'],
  softness: ['sun', 'cast', 'irradiance'],
  ambient: ['sun'],
  sunIntensity: ['sun'],
  directIrradiance: ['irradiance'],
  diffuseIrradiance: ['irradiance'],
  tableStep: ['irradiance'],
  reliefShade: ['look'],
  elevationTint: ['look']
};

const COMPILE_OPTIONS = new Set<keyof SunOptions>([
  'refraction',
  'horizonDirections',
  'horizonRadius',
  'horizonGrowth',
  'horizonFormat',
  'horizonAlgorithm',
  'castRadius'
]);

/** Where each product's raster comes from. */
const SOURCES: Record<SunProduct, {stage: StageId; buffer: string; format: 'float32' | 'rgba8'}> = {
  composite: {stage: 'look', buffer: 'color', format: 'rgba8'},
  illumination: {stage: 'sun', buffer: 'illumination', format: 'float32'},
  shadow: {stage: 'sun', buffer: 'sunVisibility', format: 'float32'},
  'cast-shadow': {stage: 'cast', buffer: 'sunVisibility', format: 'float32'},
  'shadow-difference': {stage: 'difference', buffer: 'difference', format: 'float32'},
  'horizon-angle': {stage: 'cast', buffer: 'horizonAngle', format: 'float32'},
  'sun-hours': {stage: 'irradiance', buffer: 'sunHours', format: 'float32'},
  insolation: {stage: 'irradiance', buffer: 'insolation', format: 'float32'},
  'sky-view': {stage: 'horizon', buffer: 'svf', format: 'float32'}
};

/** Fixed color range and ramp of the scalar products. */
export const SUN_SCALARS: Partial<
  Record<
    SunProduct,
    {low: number; high: number; unit: string; label: string; ramp?: 'grayscale' | 'diverging'}
  >
> = {
  illumination: {low: 0, high: 1, unit: '', label: 'Illumination (sun + ambient)'},
  shadow: {low: 0, high: 1, unit: '', label: 'Sun visibility', ramp: 'grayscale'},
  'cast-shadow': {
    low: 0,
    high: 1,
    unit: '',
    label: 'Sun visibility (cast shadow)',
    ramp: 'grayscale'
  },
  'shadow-difference': {low: 0, high: 0.5, unit: '', label: 'Difference between the two methods'},
  'horizon-angle': {low: -10, high: 60, unit: '°', label: 'Terrain horizon angle toward the sun'},
  'sun-hours': {low: 0, high: 16, unit: 'h', label: 'Hours of direct sun'},
  insolation: {low: 0, high: 10, unit: 'kWh/m²', label: 'Clear-sky solar energy per day'},
  'sky-view': {low: 0.35, high: 1, unit: '', label: 'Sky-view factor'}
};

export function getSunPaint(state: SunOptions): Partial<PaintSpec> {
  const spec = SUN_SCALARS[state.product];
  if (!spec) return {};
  return {
    mode: 'ramp',
    ramp: spec.ramp ?? state.ramp,
    low: spec.low,
    high: spec.high,
    alpha: 1,
    fadeMiddle: false,
    // Insolation is stored in Wh/m^2; the ramp spans 0 to 10 kWh.
    ...(state.product === 'insolation' ? {low: 0, high: 10000} : {})
  };
}

function describeCell(state: SunOptions, value: number, elevation: number): string | null {
  const elevationText = Number.isFinite(elevation) ? `\nElevation ${elevation.toFixed(0)} m` : '';
  if (!Number.isFinite(value)) return `No value${elevationText}`;
  switch (state.product) {
    case 'illumination':
      return `Illumination ${value.toFixed(2)}${elevationText}`;
    case 'shadow':
    case 'cast-shadow':
      return value >= 0.99
        ? `In full sun${elevationText}`
        : value <= 0.01
          ? `In shadow${elevationText}`
          : `Penumbra: ${(value * 100).toFixed(0)} % of the sun disk visible${elevationText}`;
    case 'shadow-difference':
      return `The methods differ by ${(value * 100).toFixed(1)} % of full sun${elevationText}`;
    case 'horizon-angle':
      return `Terrain rises ${value.toFixed(1)}° toward the sun${elevationText}`;
    case 'sun-hours':
      return `${value.toFixed(1)} h of direct sun on ${formatDay(state.dayOfYear)}${elevationText}`;
    case 'insolation':
      return `${(value / 1000).toFixed(2)} kWh/m² on ${formatDay(state.dayOfYear)}${elevationText}`;
    case 'sky-view':
      return `Sky-view factor ${value.toFixed(3)}${elevationText}`;
    default:
      return null;
  }
}

/**
 * Sun and shadow over the Matterhorn tile. Stages (each re-encoded only when needed):
 *
 * - `derivatives` and `horizon` run once per elevation / horizon option: the horizon map does not
 *   depend on the sun, which is why moving the sun is cheap.
 * - `sun` runs every frame while animating: `GPUSolarPosition` writes the sun azimuth and altitude
 *   on the GPU, a one-thread kernel copies them into the shadow-mask settings, and
 *   `GPUSolarShadowMask` reads two horizon values per pixel.
 * - `cast` is the exact one-sun alternative (`GPUTerrainCastShadow`), `irradiance` integrates a whole
 *   day (`GPUSolarIrradiance`), `look` is the relief colour the composite lights, and `difference`
 *   compares the two shadow methods.
 */
export async function createSunAndShadow(
  ctx: SceneContext<SunOptions>
): Promise<SceneInstance<SunOptions>> {
  const {device} = ctx;
  const grid = await loadAlpsGrid(ctx.datasets.get('alps-dem'), ctx.signal);
  ctx.signal.throwIfAborted();
  const {width, height, pixelCount} = grid;
  const resources = new SpatialAnalysisResources(device, 'sun-and-shadow');
  const session = new TerrainSession(ctx, resources, grid);
  session.elevationBuffer.write(grid.cpuElevation);
  session.validityBuffer.write(new Uint32Array(pixelCount).fill(1));
  session.elevationChanged();
  const cell = grid.cellSettings;
  const [centerLongitude, centerLatitude] = grid.origin;
  let destroyed = false;
  let timers: ReturnType<typeof setTimeout>[] = [];
  let clockHours = ctx.options.hour;
  let lastSunRead = 0;

  const registry: StageRegistry<StageId, SunOptions> = new StageRegistry(
    session,
    DEPENDENCIES,
    getStageKey,
    (id, state, stages) => buildStage(id, state, stages)
  );

  /** The instant being shown: slider time, advanced by the animation, in Europe/Zurich civil time. */
  function getTimestamp(): number {
    return getUtcFromZurich(ctx.options.dayOfYear, clockHours);
  }
  function getSun() {
    return getSolarPosition(getTimestamp(), centerLongitude, centerLatitude, {
      refraction: ctx.options.refraction
    });
  }

  function horizonFormatInfo(state: SunOptions) {
    const directions = Number(state.horizonDirections);
    const unorm = state.horizonFormat === 'unorm16';
    return {
      directions,
      unorm,
      wordCount: unorm ? (pixelCount * directions) / 2 : pixelCount * directions,
      byteLength: pixelCount * directions * (unorm ? 2 : 4)
    };
  }

  function buildStage(
    id: StageId,
    state: SunOptions,
    stages: StageRegistry<StageId, SunOptions>
  ): StageEntry {
    const builder = session.builder(id);
    const elevation = builder.elevation();
    const importFrom = <Format extends 'float32' | 'uint32'>(
      stage: StageId,
      name: string,
      format: Format,
      count: number
    ) =>
      importGraphBuffer(
        builder.graph,
        name,
        stages.require(stage).builder.getBuffer(name),
        format,
        count
      );
    let stage: Stage;
    switch (id) {
      case 'derivatives': {
        const settings = builder.settings(GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUTerrainDerivatives({
            id: 'derivatives',
            width,
            height,
            elevation,
            settings: settings.view,
            slope: builder.floats('slope'),
            aspect: builder.floats('aspect'),
            cellSizeMode: 'web-mercator',
            rowDirection: 'south'
          })
        );
        stage = builder.finishStage({
          write: () => settings.parameters.write(getGPUTerrainDerivativesParameterValues({...cell}))
        });
        break;
      }
      case 'horizon': {
        const info = horizonFormatInfo(state);
        const settings = builder.settings(GPU_TERRAIN_HORIZON_PARAMETER_LENGTH);
        builder.buffer('horizon', info.byteLength);
        builder.graph.add(
          new GPUTerrainHorizon({
            id: 'horizon',
            width,
            height,
            elevation,
            settings: settings.view,
            directionCount: info.directions,
            maximumRadius: Math.min(Number(state.horizonRadius), Math.max(width, height) - 1),
            algorithm: state.horizonAlgorithm,
            ...(state.horizonAlgorithm === 'march' ? {stepGrowth: state.horizonGrowth} : {}),
            horizonFormat: state.horizonFormat,
            cellSizeMode: 'web-mercator',
            rowDirection: 'south',
            horizon: importGraphBuffer(
              builder.graph,
              'horizon',
              builder.getBuffer('horizon'),
              info.unorm ? 'uint32' : 'float32',
              info.wordCount
            ),
            skyViewFactor: builder.floats('svf')
          })
        );
        stage = builder.finishStage({
          write: () => settings.parameters.write(getGPUTerrainHorizonParameterValues({...cell}))
        });
        break;
      }
      case 'sun': {
        const info = horizonFormatInfo(state);
        const solarSettings = builder.settings(GPU_SOLAR_POSITION_PARAMETER_LENGTH);
        const positions = builder.buffer('positions', 4 * 2 * 4);
        const locations = [
          [centerLongitude, centerLatitude],
          ...['Matterhorn', 'Gornergrat', 'Zermatt'].map(name => {
            const observer = grid.observers.find(entry => entry.name === name);
            return [observer?.longitude ?? centerLongitude, observer?.latitude ?? centerLatitude];
          })
        ];
        positions.write(Float32Array.from(locations.flat()));
        const azimuth = builder.floats('azimuth', 4);
        const altitude = builder.floats('altitude', 4);
        const daylight = builder.words('daylight', 4);
        builder.graph.add(
          new GPUSolarPosition({
            id: 'solar-position',
            positions: importGraphBuffer(builder.graph, 'positions', positions, 'float32x2', 4),
            settings: solarSettings.view,
            azimuth,
            altitude,
            daylight,
            refraction: state.refraction
          })
        );
        // The GPU sun drives the shadow: copy its azimuth and altitude over the first two
        // settings of the mask. The CPU writes the rest (disk radius, intensities) each run.
        const maskSettingsBuffer = builder.buffer('mask-settings', 32);
        const maskSettings = importGraphBuffer(
          builder.graph,
          'mask-settings',
          maskSettingsBuffer,
          'float32',
          8
        );
        addKernelPass(builder.graph, {
          id: 'sun-to-mask-settings',
          invocationCount: 1,
          bindings: [
            {name: 'sunAzimuth', view: azimuth, type: 'f32', access: 'read'},
            {name: 'sunAltitude', view: altitude, type: 'f32', access: 'read'},
            {name: 'maskSettings', view: maskSettings, type: 'f32', access: 'read_write'}
          ],
          body: /* wgsl */ `
  maskSettings[maskSettingsOffset] = sunAzimuth[sunAzimuthOffset];
  maskSettings[maskSettingsOffset + 1u] = sunAltitude[sunAltitudeOffset];`
        });
        builder.graph.add(
          new GPUSolarShadowMask({
            id: 'shadow-mask',
            width,
            height,
            directionCount: info.directions,
            horizonFormat: state.horizonFormat,
            horizon: importFrom(
              'horizon',
              'horizon',
              info.unorm ? 'uint32' : 'float32',
              info.wordCount
            ),
            settings: maskSettings,
            slope: importFrom('derivatives', 'slope', 'float32', pixelCount),
            aspect: importFrom('derivatives', 'aspect', 'float32', pixelCount),
            skyViewFactor: importFrom('horizon', 'svf', 'float32', pixelCount),
            sunVisibility: builder.floats('sunVisibility'),
            illumination: builder.floats('illumination')
          })
        );
        stage = builder.finishStage({
          dependencies: [stages.require('horizon').stage, stages.require('derivatives').stage],
          write: () => {
            const o = ctx.options;
            const sun = getSun();
            maskSettingsBuffer.write(
              getGPUSolarShadowMaskParameterValues({
                azimuthDegrees: sun.azimuthDegrees,
                altitudeDegrees: sun.altitudeDegrees,
                angularRadiusDegrees: GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES * o.softness,
                sunIntensity: o.sunIntensity,
                ambientIntensity: o.ambient
              })
            );
            solarSettings.parameters.write(
              getGPUSolarPositionParameterValues({
                timestamp: getTimestamp(),
                daylightAltitudeDegrees: Number(o.twilight)
              })
            );
          }
        });
        break;
      }
      case 'cast': {
        const settings = builder.settings(GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUTerrainCastShadow({
            id: 'cast-shadow',
            width,
            height,
            elevation,
            settings: settings.view,
            maximumRadius: state.castRadius === 'tile' ? undefined : Number(state.castRadius),
            cellSizeMode: 'web-mercator',
            rowDirection: 'south',
            sunVisibility: builder.floats('sunVisibility'),
            horizonAngle: builder.floats('horizonAngle')
          })
        );
        stage = builder.finishStage({
          write: () => {
            const sun = getSun();
            settings.parameters.write(
              getGPUTerrainCastShadowParameterValues(
                {
                  ...cell,
                  azimuthDegrees: sun.azimuthDegrees,
                  altitudeDegrees: sun.altitudeDegrees,
                  angularRadiusDegrees: GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES * ctx.options.softness
                },
                'south'
              )
            );
          }
        });
        break;
      }
      case 'irradiance': {
        const info = horizonFormatInfo(state);
        const settings = builder.settings(GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH);
        const tableBuffer = builder.buffer(
          'sun-table',
          SUN_TABLE_CAPACITY * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE * 4
        );
        builder.graph.add(
          new GPUSolarIrradiance({
            id: 'solar-irradiance',
            width,
            height,
            directionCount: info.directions,
            horizonFormat: state.horizonFormat,
            horizon: importFrom(
              'horizon',
              'horizon',
              info.unorm ? 'uint32' : 'float32',
              info.wordCount
            ),
            sunTable: importGraphBuffer(
              builder.graph,
              'sun-table',
              tableBuffer,
              'float32',
              SUN_TABLE_CAPACITY * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE
            ),
            sampleCapacity: SUN_TABLE_CAPACITY,
            settings: settings.view,
            slope: importFrom('derivatives', 'slope', 'float32', pixelCount),
            aspect: importFrom('derivatives', 'aspect', 'float32', pixelCount),
            skyViewFactor: importFrom('horizon', 'svf', 'float32', pixelCount),
            sunHours: builder.floats('sunHours'),
            insolation: builder.floats('insolation')
          })
        );
        stage = builder.finishStage({
          dependencies: [stages.require('horizon').stage, stages.require('derivatives').stage],
          // The irradiance integral is the heaviest per-change job: let the GPU finish one run
          // before the next is queued while a slider is dragged.
          minIntervalMs: 300,
          write: () => {
            const o = ctx.options;
            const start = getUtcFromZurich(o.dayOfYear, 0);
            const table = getGPUSolarIrradianceSunTable({
              longitude: centerLongitude,
              latitude: centerLatitude,
              start,
              end: start + 24 * 3600 * 1000,
              stepMinutes: Number(o.tableStep),
              directNormalIrradiance:
                o.directIrradiance === 'meinel' ? 'meinel' : Number(o.directIrradiance),
              refraction: o.refraction
            });
            tableBuffer.write(table.values);
            settings.parameters.write(
              getGPUSolarIrradianceParameterValues({
                sampleCount: table.sampleCount,
                angularRadiusDegrees: GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES * o.softness,
                diffuseIrradiance: o.diffuseIrradiance
              })
            );
            let dayLength = 0;
            for (let row = 0; row < table.sampleCount; row++) {
              const base = row * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE;
              if (table.values[base + 1] > 0) dayLength += table.values[base + 2];
            }
            ctx.setReadout(
              'dayLength',
              `${Math.floor(dayLength)} h ${Math.round((dayLength % 1) * 60)} min`
            );
          }
        });
        break;
      }
      case 'look': {
        const settings = builder.settings(GPU_RELIEF_SHADING_PARAMETER_LENGTH);
        builder.graph.add(
          new GPUReliefShading({
            id: 'relief-shading',
            width,
            height,
            elevation,
            settings: settings.view,
            skyViewFactor: importFrom('horizon', 'svf', 'float32', pixelCount),
            cellSizeMode: 'web-mercator',
            rowDirection: 'south',
            color: builder.words('color')
          })
        );
        stage = builder.finishStage({
          dependencies: [stages.require('horizon').stage],
          write: () =>
            settings.parameters.write(
              getGPUReliefShadingParameterValues({
                ...cell,
                lights: 'mdow',
                lightWeighting: 'aspect',
                hillshadeStrength: ctx.options.reliefShade,
                skyViewStrength: 0.8,
                textureShadeStrength: 0,
                exposure: 1.1,
                tintStrength: 0.25,
                elevationStops: ctx.options.elevationTint ? ALPINE_ELEVATION_STOPS : []
              })
            )
        });
        break;
      }
      case 'difference': {
        const difference = builder.floats('difference');
        addKernelPass(builder.graph, {
          id: 'shadow-difference',
          invocationCount: pixelCount,
          bindings: [
            {
              name: 'horizonMap',
              view: importFrom('sun', 'sunVisibility', 'float32', pixelCount),
              type: 'f32',
              access: 'read'
            },
            {
              name: 'castShadow',
              view: importFrom('cast', 'sunVisibility', 'float32', pixelCount),
              type: 'f32',
              access: 'read'
            },
            {name: 'difference', view: difference, type: 'f32', access: 'read_write'}
          ],
          body: /* wgsl */ `
  difference[differenceOffset + index] = abs(horizonMap[horizonMapOffset + index] - castShadow[castShadowOffset + index]);`
        });
        stage = builder.finishStage({
          dependencies: [stages.require('sun').stage, stages.require('cast').stage],
          write: () => {}
        });
        break;
      }
    }
    return {key: '', stage, builder};
  }

  // --- Product display --------------------------------------------------------------------------
  /** Aggregates the stages the composite reads (relief colors and the sun illumination). */
  let compositeStage: Stage | null = null;
  let compositeKey = '';
  function getCompositeStage(state: SunOptions): Stage {
    const key = `${registry.getFullKey('look', state)}|${registry.getFullKey('sun', state)}`;
    if (compositeStage && compositeKey === key) return compositeStage;
    if (compositeStage) session.unregisterStage(compositeStage);
    compositeStage = new Stage('composite', [], () => {}, [
      registry.require('look').stage,
      registry.require('sun').stage
    ]);
    session.registerStage(compositeStage);
    compositeKey = key;
    return compositeStage;
  }

  function showProduct(): void {
    const state = ctx.options;
    const source = SOURCES[state.product];
    try {
      const key =
        state.product === 'composite'
          ? `${registry.getFullKey('look', state)}|${registry.getFullKey('sun', state)}`
          : registry.getFullKey(source.stage, state);
      let build = session.getBuild(state.product, key);
      if (!build) {
        const entry = registry.ensure(source.stage, state);
        if (state.product === 'composite') registry.ensure('sun', state);
        const stage = state.product === 'composite' ? getCompositeStage(state) : entry.stage;
        build = session.addBuild(
          session.createBuild(
            state.product,
            stage,
            entry.builder.getBuffer(source.buffer),
            source.format
          ),
          key
        );
      }
      ctx.setStatus('');
      session.activate(build, getSunPaint(state));
      ctx.setReadout('meanValue', null);
      ctx.requestLayers();
    } catch (error) {
      ctx.setStatus(`Cannot build ${state.product}: ${(error as Error).message}`);
    }
  }

  // --- Readouts ----------------------------------------------------------------------------------
  session.describeHover = ({value, elevation}) => describeCell(ctx.options, value, elevation);
  session.onStats = (_id, stats: ValueStats) => {
    if (stats.kind !== 'float') return;
    const state = ctx.options;
    const text =
      state.product === 'shadow' || state.product === 'cast-shadow'
        ? `${(stats.mean * 100).toFixed(1)} % of the tile in sun`
        : state.product === 'sun-hours'
          ? `${stats.mean.toFixed(1)} h`
          : state.product === 'insolation'
            ? `${(stats.mean / 1000).toFixed(2)} kWh/m²`
            : state.product === 'shadow-difference'
              ? `${(stats.mean * 100).toFixed(2)} % of full sun (mean)`
              : stats.mean.toPrecision(3);
    ctx.setReadout('meanValue', text);
  };

  function updateClockReadouts(): void {
    const timestamp = getTimestamp();
    ctx.setReadout(
      'localTime',
      `${formatDay(ctx.options.dayOfYear)}, ${formatHour(clockHours)} ${getZurichZoneName(timestamp)}`
    );
    const sun = getSun();
    ctx.setReadout(
      'sunCpu',
      `az ${sun.azimuthDegrees.toFixed(1)}°, alt ${sun.altitudeDegrees.toFixed(1)}° (${sun.altitudeDegrees > -0.833 ? 'day' : 'night'})`
    );
  }

  let sunReader: SummaryReader | null = null;
  let sunReaderEntry: StageEntry | null = null;
  function ensureSunReader(): SummaryReader | null {
    const entry = registry.get('sun');
    if (!entry) return null;
    if (sunReader && sunReaderEntry === entry) return sunReader;
    sunReader?.stop();
    const reader = new SummaryReader(
      resources,
      `sun-${entry.key}`,
      [
        {buffer: entry.builder.getBuffer('azimuth'), size: 16},
        {buffer: entry.builder.getBuffer('altitude'), size: 16},
        {buffer: entry.builder.getBuffer('daylight'), size: 16}
      ],
      bytes => {
        if (destroyed) return;
        const floats = new Float32Array(bytes, 0, 8);
        const daylight = new Uint32Array(bytes, 32, 4);
        // Row 3 is Zermatt (rows: tile center, Matterhorn, Gornergrat, Zermatt).
        const sun = getSun();
        ctx.setReadout(
          'sunGpu',
          `az ${floats[3].toFixed(1)}°, alt ${floats[7].toFixed(1)}° (${daylight[3] ? 'day' : 'night'}) · differs from float64 by ${Math.abs(floats[7] - sun.altitudeDegrees).toFixed(3)}° alt`
        );
      }
    );
    sunReader = reader;
    sunReaderEntry = entry;
    return reader;
  }

  const info = horizonFormatInfo(ctx.options);
  ctx.setReadout(
    'grid',
    `${width} x ${height} px, ${grid.groundCellSize.toFixed(2)} m ground per pixel`
  );
  ctx.setReadout(
    'horizonMemory',
    `${(info.byteLength / 1048576).toFixed(0)} MB (${info.directions} sectors, ${ctx.options.horizonFormat})`
  );
  updateClockReadouts();
  showProduct();

  let lastClockReadout = 0;
  let lastSunRunCount = -1;
  function markSunDirty(): void {
    registry.markDirty(['sun', 'cast']);
    const now = performance.now();
    if (now - lastClockReadout > 100) {
      lastClockReadout = now;
      updateClockReadouts();
    }
  }

  return {
    getCompiledGraphs: () => session.getCompiledGraphs(),

    setOption(id) {
      const key = id as keyof SunOptions;
      if (key === 'product') {
        showProduct();
        return;
      }
      if (key === 'hour') {
        clockHours = ctx.options.hour;
      }
      if (COMPILE_OPTIONS.has(key)) {
        const next = horizonFormatInfo(ctx.options);
        ctx.setReadout(
          'horizonMemory',
          `${(next.byteLength / 1048576).toFixed(0)} MB (${next.directions} sectors, ${ctx.options.horizonFormat})`
        );
        timers.push(setTimeout(() => !destroyed && showProduct(), REBUILD_DELAY_MILLISECONDS));
        return;
      }
      const stages = PARAMETER_STAGES[key];
      if (stages) {
        registry.markDirty(stages);
        if (key === 'hour' || key === 'dayOfYear') updateClockReadouts();
        return;
      }
      if (key === 'ramp') {
        session.setPaint(getSunPaint(ctx.options));
      } else {
        ctx.requestLayers();
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
            ? 'n/a'
            : `${total.toFixed(1)} ms, all stages of this view (${formatCount(pixelCount)} px, ${results[0].method === 'gpu-timestamps' ? 'GPU timestamps' : 'wall clock'})`
        );
      });
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip: event => session.getTooltip(event),

    encode(commandEncoder, frame) {
      if (ctx.options.animate) {
        clockHours = (clockHours + frame.deltaSeconds * ctx.options.animationSpeed) % 24;
        markSunDirty();
      }
      session.encode(commandEncoder);
      const reader = ensureSunReader();
      const sunStage = registry.get('sun')?.stage;
      if (reader && sunStage) {
        const now = performance.now();
        // Read the GPU sun back only when the sun stage ran, at most every 120 ms.
        if (
          sunStage.runCount !== lastSunRunCount &&
          now - lastSunRead > SUN_READ_INTERVAL_MILLISECONDS &&
          !reader.isPending
        ) {
          reader.request(commandEncoder);
          lastSunRead = now;
          lastSunRunCount = sunStage.runCount;
        } else {
          reader.flush(commandEncoder);
        }
      }
    },

    getLayers(): Layer[] {
      const state = ctx.options;
      const sun = registry.get('sun');
      const lit = state.product === 'composite' && sun;
      return session.getLayers({
        underlay: false,
        underlayAlpha: 1,
        alpha: state.opacity,
        light: lit
          ? {
              buffer: sun.builder.getBuffer('illumination'),
              strength: state.shadowStrength,
              gain: state.exposure,
              floor: state.lightFloor
            }
          : undefined
      });
    },

    destroy() {
      destroyed = true;
      for (const timer of timers) clearTimeout(timer);
      timers = [];
      sunReader?.stop();
      session.destroy();
      resources.destroy();
    }
  };
}
