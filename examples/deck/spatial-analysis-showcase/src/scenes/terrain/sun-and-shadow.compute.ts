// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUSolarIrradianceParameterValues,
  getGPUSolarIrradianceSunTable,
  getGPUSolarPositionParameterValues,
  getGPUSolarShadowMaskParameterValues,
  getGPUTerrainCastShadowParameterValues,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainHorizonParameterValues,
  getSolarPosition,
  GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES,
  GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
  GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE,
  GPU_SOLAR_POSITION_PARAMETER_LENGTH,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPUSolarIrradiance,
  GPUSolarPosition,
  GPUSolarShadowMask,
  GPUTerrainCastShadow,
  GPUTerrainDerivatives,
  GPUTerrainHorizon
} from '@luma.gl/experimental/gpu-terrain';
import {ALPS} from '../../cartography/gazetteer';
import {getTerrainInk} from './terrain-palettes';
import type {MapAnnotation} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {loadAlpsGrid} from './b14a-grid';
import {ColorRasterLayer} from './b14a-layers';
import {Stage, TerrainSession, type ValueStats} from './b14a-session';
import {StageRegistry, type StageEntry} from './b14a-stage-registry';
import {createDemProbe, createTerrainDemFromGrid} from './cpu-dem';
import {
  addSunPaintPass,
  getSunPaintParameters,
  SUN_PAINT_PARAMETER_LENGTH,
  type SunPaintSettings
} from './sun-and-shadow-paint';
import {getLightArrows, getSearchRing, getVillageNote} from './sun-and-shadow-places';
import {
  formatClockHour,
  getDayFacts,
  getPixelTimeline,
  getSkyDiagram,
  getSkyline,
  getSunSamples,
  getTimelineDiagram,
  interpolateHorizon,
  type DayFacts,
  type PixelTimeline,
  type SunSample
} from './sun-and-shadow-sky';
import {
  formatShare,
  getDifferenceShare,
  getSunTooltip,
  getValueHistogramChart,
  type HoverCell
} from './sun-and-shadow-summary';
import {
  DATE_PRESETS,
  DEFAULT_SUN_DAY,
  formatDay,
  formatHour,
  getCartoucheSubtitle,
  getInstantHours,
  getLightColors,
  getShadowColors,
  getTableAnchors,
  getUtcFromZurich,
  getZurichZoneName,
  makeSunTables,
  SUN_GOLD,
  type SunOptions,
  type SunProduct,
  type SunTables
} from './sun-and-shadow.style';
import {demSampleLine} from './terrain-furniture';
import {createTerrainGround, rasterizeGlacierMask} from './terrain-ground';
import {loadAlpsContext} from './terrain-places';

export type {SunOptions, SunProduct};

type StageId =
  | 'derivatives'
  | 'horizon'
  | 'sun'
  | 'cast'
  | 'irradiance'
  | 'difference'
  | 'paint-shadow'
  | 'paint-cast'
  | 'paint-light'
  | 'paint-hours'
  | 'paint-insolation';

const DEPENDENCIES: Record<StageId, readonly StageId[]> = {
  derivatives: [],
  horizon: [],
  sun: ['horizon', 'derivatives'],
  cast: [],
  irradiance: ['horizon', 'derivatives'],
  difference: ['sun', 'cast'],
  'paint-shadow': ['sun'],
  // The exact cast is shown with the difference stage beside it, so the readout of how far the two
  // methods disagree stays live while only one of them is drawn.
  'paint-cast': ['cast', 'difference'],
  'paint-light': ['sun'],
  'paint-hours': ['irradiance'],
  'paint-insolation': ['irradiance']
};

/** The paint stages, which turn a float raster of a compute stage into packed colours. */
const PAINT_STAGES: readonly StageId[] = [
  'paint-shadow',
  'paint-cast',
  'paint-light',
  'paint-hours',
  'paint-insolation',
  'difference'
];

/** The compute raster each paint stage reads. */
const PAINT_SOURCES: Partial<Record<StageId, {stage: StageId; buffer: string}>> = {
  'paint-shadow': {stage: 'sun', buffer: 'sunVisibility'},
  'paint-cast': {stage: 'cast', buffer: 'sunVisibility'},
  'paint-light': {stage: 'sun', buffer: 'illumination'},
  'paint-hours': {stage: 'irradiance', buffer: 'sunHours'},
  'paint-insolation': {stage: 'irradiance', buffer: 'insolation'}
};

/** Sun table capacity (topology): one day of 5 minute rows. */
const SUN_TABLE_CAPACITY = 288;
const SUN_READ_INTERVAL_MILLISECONDS = 120;
const REBUILD_DELAY_MILLISECONDS = 250;
const CHART_INTERVAL_MILLISECONDS = 90;
const DIFFERENCE_READ_INTERVAL_MILLISECONDS = 400;
/** The colorize pass of the session is unused here (the paint stages write the colours). */
const UNUSED_PAINT = {mode: 'ramp', ramp: 'grayscale', low: 0, high: 1, alpha: 0} as const;
/** Light azimuth of the chapter's relief ground, degrees clockwise from north. */
const MAP_LIGHT_AZIMUTH = 315;
/** Half an hour of light either side of the day while the animation runs. */
const ANIMATION_MARGIN_HOURS = 0.5;
const MEGABYTE = 1e6;

type CellPin = {
  lngLat: [number, number];
  column: number;
  row: number;
  isVillage: boolean;
};

function getStageKey(id: StageId, state: SunOptions, directions: number): string {
  switch (id) {
    case 'horizon':
      return `${directions}|${state.horizonRadius}|${state.horizonAlgorithm === 'sweep' ? 1 : state.horizonGrowth}|${state.horizonFormat}|${state.horizonAlgorithm}`;
    case 'sun':
      return `${state.refraction}|${state.horizonFormat}|${directions}`;
    case 'cast':
      return state.castRadius;
    case 'irradiance':
      return `${state.horizonFormat}${directions}`;
    default:
      return id;
  }
}

/** Option id to the compute stages whose parameters it feeds. */
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
  exposure: ['paint-light']
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

/** The stage that holds the colours of the displayed product. */
function getPaintStageId(state: SunOptions): StageId {
  switch (state.product) {
    case 'shadow':
      return state.method === 'horizon'
        ? 'paint-shadow'
        : state.method === 'cast'
          ? 'paint-cast'
          : 'difference';
    case 'light':
      return 'paint-light';
    case 'sun-hours':
      return 'paint-hours';
    case 'insolation':
      return 'paint-insolation';
  }
}

/** The raw raster of the displayed product (hover value, histogram) and the stage that owns it. */
function getValueSource(state: SunOptions): {stage: StageId; buffer: string; buildId: string} {
  switch (state.product) {
    case 'shadow':
      return state.method === 'horizon'
        ? {stage: 'sun', buffer: 'sunVisibility', buildId: 'shadow-horizon'}
        : state.method === 'cast'
          ? {stage: 'cast', buffer: 'sunVisibility', buildId: 'shadow-cast'}
          : {stage: 'difference', buffer: 'difference', buildId: 'shadow-difference'};
    case 'light':
      return {stage: 'sun', buffer: 'illumination', buildId: 'light'};
    case 'sun-hours':
      return {stage: 'irradiance', buffer: 'sunHours', buildId: 'sun-hours'};
    case 'insolation':
      return {stage: 'irradiance', buffer: 'insolation', buildId: 'insolation'};
  }
}

/**
 * Sun and shadow over the wide Zermatt window. Stages (each re-encoded only when needed):
 *
 * - `derivatives` and `horizon` run once per elevation and horizon option: the horizon map does not
 *   depend on the sun, which is why moving the sun is cheap.
 * - `sun` runs every frame while animating: `GPUSolarPosition` writes the sun azimuth and altitude
 *   on the GPU, a one-thread kernel copies them into the shadow-mask settings, and
 *   `GPUSolarShadowMask` reads two horizon values per pixel.
 * - `cast` is the exact one-sun alternative (`GPUTerrainCastShadow`), `difference` compares the two
 *   methods, and `irradiance` integrates a whole day (`GPUSolarIrradiance`).
 * - The `paint-*` stages turn a float raster into packed colours (the indigo veil, the light
 *   multiply colours, the class tables of the legends); they are cheap and re-run when the sun,
 *   the ground or a scale changes.
 *
 * The CPU repeats the same arithmetic for one cell: the sun path, the skyline of the pinned cell
 * and the lit / shadow / night timeline of the village, drawn as the two diagrams of the card.
 */
export async function createSunAndShadow(
  ctx: SceneContext<SunOptions>
): Promise<SceneInstance<SunOptions>> {
  const {device} = ctx;
  ctx.setStatus('Loading the elevation window and its OpenStreetMap context');
  const [grid, context] = await Promise.all([
    loadAlpsGrid(ctx.datasets.get('alps-dem-wide'), ctx.signal),
    loadAlpsContext(ctx.datasets.get('alps-context'), ctx.signal)
  ]);
  ctx.signal.throwIfAborted();
  const {width, height, pixelCount} = grid;
  const dem = createTerrainDemFromGrid(grid);
  const probe = createDemProbe(dem);
  const glacierMask = rasterizeGlacierMask(context.glacierGeoJson, dem);
  const resources = new SpatialAnalysisResources(device, 'sun-and-shadow');
  const session = new TerrainSession(ctx, resources, grid);
  const ground = createTerrainGround({dem, device, ground: ctx.ground(), glacierMask});
  session.setGround(ground);
  session.elevationBuffer.write(grid.cpuElevation);
  session.validityBuffer.write(new Uint32Array(pixelCount).fill(1));
  session.elevationChanged();
  ctx.setStatus('Building the shaded relief');
  await ground.prepare();
  ctx.signal.throwIfAborted();

  const cell = grid.cellSettings;
  const [centerLongitude, centerLatitude] = grid.origin;
  // The village: the gazetteer's Zermatt, on the DEM cell that contains it.
  const villagePixel = grid.getPixel(
    ALPS.places.zermatt.lngLat[0],
    ALPS.places.zermatt.lngLat[1]
  ) ?? [Math.floor(width / 2), Math.floor(height / 2)];
  const villageLngLat = grid.getLongitudeLatitude(villagePixel[0], villagePixel[1]);
  const villageIndex = villagePixel[1] * width + villagePixel[0];
  const villageCoordinate: [number, number] = [villageLngLat[0], villageLngLat[1]];
  const villageEye = resources.createBuffer(
    'village-eye',
    Float32Array.from(grid.projection.project(villageLngLat[0], villageLngLat[1]))
  );

  let destroyed = false;
  let timers: ReturnType<typeof setTimeout>[] = [];
  let clockHours = ctx.options.hour;
  let lastSunRead = 0;
  let lastChartTime = 0;
  let chartsDirty = true;
  let lastClockRefresh = 0;
  let lastSunRunCount = -1;
  let lastIrradianceRunCount = -1;
  // The difference raster is read back whole, rarely, to count where the two methods disagree.
  let lastDifferenceRun = -1;
  let lastDifferenceRead = 0;
  let differenceBusy = false;
  let lastFloatStats: Extract<ValueStats, {kind: 'float'}> | null = null;
  let gpuVillage: {sunHours: number; insolation: number} | null = null;

  // --- Horizon size and the GPU's limits ------------------------------------------------------------
  function getHorizonInfo(state: SunOptions) {
    const requested = Number(state.horizonDirections);
    const unorm = state.horizonFormat === 'unorm16';
    const bytesPerValue = unorm ? 2 : 4;
    const limit = Math.min(ctx.limits.maxStorageBufferBindingSize, ctx.limits.maxBufferSize);
    let directions = requested;
    while (directions > 4 && pixelCount * directions * bytesPerValue > limit) directions /= 2;
    return {
      requested,
      directions,
      unorm,
      bytesPerValue,
      wordCount: unorm ? (pixelCount * directions) / 2 : pixelCount * directions,
      byteLength: pixelCount * directions * bytesPerValue,
      requestedByteLength: pixelCount * requested * bytesPerValue
    };
  }
  const getDirections = (state: SunOptions) => getHorizonInfo(state).directions;

  const registry: StageRegistry<StageId, SunOptions> = new StageRegistry(
    session,
    DEPENDENCIES,
    (id, state) => getStageKey(id, state, getDirections(state)),
    (id, state, stages) => buildStage(id, state, stages)
  );

  // --- The sun and the CPU day ---------------------------------------------------------------------
  const getTimestamp = () => getUtcFromZurich(ctx.options.dayOfYear, clockHours);
  const getVillageSun = () =>
    getSolarPosition(getTimestamp(), villageLngLat[0], villageLngLat[1], {
      refraction: ctx.options.refraction
    });
  const getRadiusMeters = (row: number) =>
    Number(ctx.options.horizonRadius) * grid.getGroundCellSize(row);

  const referenceDays = [DATE_PRESETS.june, DATE_PRESETS.march, DATE_PRESETS.december];
  let referencePaths = referenceDays.map(dayOfYear => ({
    dayOfYear,
    samples: getSunSamples(dayOfYear, villageLngLat, ctx.options.refraction)
  }));
  let daySamples: SunSample[] = [];
  let dayFacts: DayFacts = {
    dayLengthHours: DEFAULT_SUN_DAY.dayLengthHours,
    sunriseHour: null,
    sunsetHour: null,
    directEnergyKilowattHours: DEFAULT_SUN_DAY.directEnergyKilowattHours
  };
  let timeline: PixelTimeline | null = null;
  let villageSkyline = getSkyline(probe, villageLngLat, getRadiusMeters(villagePixel[1]));
  let pinned: CellPin = {
    lngLat: villageCoordinate,
    column: villagePixel[0],
    row: villagePixel[1],
    isVillage: true
  };
  let pinnedSkyline = villageSkyline;
  let sunNow = {azimuth: 180, altitude: 0};
  let tables: SunTables = makeSunTables(ctx.options, ctx.ground(), DEFAULT_SUN_DAY);

  function refreshTables(): void {
    tables = makeSunTables(ctx.options, ctx.ground(), {
      dayLengthHours: dayFacts.dayLengthHours,
      directEnergyKilowattHours: dayFacts.directEnergyKilowattHours
    });
    ctx.setLegendData('tables', tables);
    publishHistogram();
  }

  function refreshSkylines(): void {
    villageSkyline = getSkyline(probe, villageLngLat, getRadiusMeters(villagePixel[1]));
    pinnedSkyline = pinned.isVillage
      ? villageSkyline
      : getSkyline(probe, pinned.lngLat, getRadiusMeters(pinned.row));
  }

  /** Everything that depends on the date, the skyline radius or the sun's disc, not on the hour. */
  function refreshDay(): void {
    const state = ctx.options;
    daySamples = getSunSamples(state.dayOfYear, villageLngLat, state.refraction);
    dayFacts = getDayFacts(state.dayOfYear, villageLngLat, state);
    timeline = getPixelTimeline(daySamples, villageSkyline, state.softness);
    const zone = getZurichZoneName(getUtcFromZurich(state.dayOfYear, 12));
    ctx.setReadout('zermattSunHours', `${timeline.sunHours.toFixed(1)} h`);
    ctx.setReadout(
      'firstSun',
      timeline.firstSun === null ? 'none' : `${formatClockHour(timeline.firstSun)} ${zone}`
    );
    refreshTables();
    publishAnnotations();
    chartsDirty = true;
  }

  function refreshClock(): void {
    const state = ctx.options;
    const sun = getVillageSun();
    sunNow = {azimuth: sun.azimuthDegrees, altitude: sun.altitudeDegrees};
    ctx.setReadout('sunAltitude', `${sun.altitudeDegrees.toFixed(1)}°`);
    ctx.setReadout('sunAzimuth', `${sun.azimuthDegrees.toFixed(0)}°`);
    if (timeline) {
      const index = Math.min(
        timeline.state.length - 1,
        Math.max(0, Math.floor(clockHours / timeline.stepHours))
      );
      const now = timeline.state[index];
      ctx.setReadout('zermattNow', now === 2 ? 'In sun' : now === 1 ? 'In shadow' : 'Night');
    }
    ctx.setReadout('localTime', `${formatDay(state.dayOfYear)}, ${formatHour(clockHours)}`);
    chartsDirty = true;
  }

  function publishCharts(): void {
    if (!timeline) return;
    const state = ctx.options;
    ctx.setChart(
      'pixelTimeline',
      getTimelineDiagram({timeline, hour: clockHours, dayOfYear: state.dayOfYear})
    );
    const directions = getHorizonInfo(state).directions;
    const referenceMatch = referencePaths.some(path => path.dayOfYear === state.dayOfYear);
    const paths = [
      ...referencePaths.map(path => ({
        ...path,
        current: path.dayOfYear === state.dayOfYear
      })),
      ...(referenceMatch ? [] : [{dayOfYear: state.dayOfYear, samples: daySamples, current: true}])
    ];
    ctx.setChart(
      'skyDiagram',
      getSkyDiagram({
        placeName: pinned.isVillage ? 'Zermatt' : 'The clicked cell',
        paths,
        now: sunNow,
        timeLabel: `${formatHour(clockHours)} ${getZurichZoneName(getTimestamp())}`,
        skyline: pinnedSkyline,
        sectors: state.showSearch
          ? Float64Array.from({length: directions}, (_, sector) =>
              interpolateHorizon(pinnedSkyline, (sector * 360) / directions)
            )
          : null
      })
    );
    // The arrow of the real sun follows it.
    if (state.product === 'light') publishAnnotations();
    chartsDirty = false;
    lastChartTime = performance.now();
  }

  function publishHistogram(): void {
    const state = ctx.options;
    const stats = lastFloatStats;
    if (!stats || (state.product !== 'sun-hours' && state.product !== 'insolation')) {
      ctx.setChart('sunHistogram', null);
      ctx.setLegendData('histogram', null);
      return;
    }
    const isHours = state.product === 'sun-hours';
    const table = isHours ? tables.hours : tables.insolation;
    const breaks = table.breaks;
    const classWidth = breaks[breaks.length - 1] - (breaks[breaks.length - 2] ?? 0);
    const marker = isHours
      ? timeline
        ? {value: timeline.sunHours, label: 'Zermatt'}
        : undefined
      : gpuVillage
        ? {value: gpuVillage.insolation / 1000, label: 'Zermatt'}
        : undefined;
    ctx.setChart(
      'sunHistogram',
      getValueHistogramChart(stats, table, {
        domainMaximum: breaks[breaks.length - 1] + classWidth,
        scale: isHours ? 1 : 0.001,
        xLabel: isHours ? 'Hours of direct sun' : 'kWh/m² per day',
        unit: isHours ? 'hours of direct sun' : 'kWh/m² per day',
        ...(marker ? {marker} : {})
      })
    );
  }

  // --- Paint settings of each stage ----------------------------------------------------------------
  function getPaintSettings(id: StageId): SunPaintSettings {
    const state = ctx.options;
    const tone = ctx.ground();
    switch (id) {
      case 'paint-shadow':
      case 'paint-cast':
        return {mode: 'veil', color: getShadowColors(tone).shadow};
      case 'paint-light': {
        const colors = getLightColors(tone);
        return {mode: 'light', exposure: state.exposure, lit: colors.lit, shade: colors.shade};
      }
      case 'difference':
        return {
          mode: 'classes',
          breaks: tables.difference.breaks,
          colors: tables.difference.colors
        };
      case 'paint-hours':
      case 'paint-insolation': {
        const table = id === 'paint-hours' ? tables.hours : tables.insolation;
        const scale = id === 'paint-hours' ? 1 : 0.001;
        return state.display === 'classes'
          ? {mode: 'classes', breaks: table.breaks, colors: table.colors, scale}
          : {mode: 'gradient', anchors: getTableAnchors(table), colors: table.colors, scale};
      }
      default:
        return {mode: 'veil', color: [0, 0, 0, 0]};
    }
  }
  const markPaintDirty = () => registry.markDirty(PAINT_STAGES);

  // --- Stages ----------------------------------------------------------------------------------------
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
        `${stage}-${name}`,
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
        const info = getHorizonInfo(state);
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
        const info = getHorizonInfo(state);
        const solarSettings = builder.settings(GPU_SOLAR_POSITION_PARAMETER_LENGTH);
        const positions = builder.buffer('positions', 4 * 2 * 4);
        // Row 0 is the village: it drives the shadow mask. The others are read for the comparison.
        const locations = [
          [villageLngLat[0], villageLngLat[1]],
          ALPS.places.matterhorn.lngLat,
          ALPS.places.gornergrat.lngLat,
          [centerLongitude, centerLatitude]
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
        // settings of the mask. The CPU writes the rest (disc radius, intensities) each run.
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
            const sun = getVillageSun();
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
            const sun = getVillageSun();
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
        const info = getHorizonInfo(state);
        const settings = builder.settings(GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH);
        const tableBuffer = builder.buffer(
          'sun-table',
          SUN_TABLE_CAPACITY * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE * 4
        );
        const sunHours = builder.floats('sunHours');
        const insolation = builder.floats('insolation');
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
            sunHours,
            insolation
          })
        );
        // One thread copies the village's two values into a 16 byte buffer the CPU can read back
        // cheaply, to check the CPU timeline against the map.
        const probeValues = builder.floats('probe', 4);
        addKernelPass(builder.graph, {
          id: 'village-probe',
          invocationCount: 1,
          bindings: [
            {name: 'sunHoursMap', view: sunHours, type: 'f32', access: 'read'},
            {name: 'insolationMap', view: insolation, type: 'f32', access: 'read'},
            {name: 'probeValues', view: probeValues, type: 'f32', access: 'read_write'}
          ],
          body: /* wgsl */ `
  probeValues[probeValuesOffset] = sunHoursMap[sunHoursMapOffset + ${villageIndex}u];
  probeValues[probeValuesOffset + 1u] = insolationMap[insolationMapOffset + ${villageIndex}u];`
        });
        stage = builder.finishStage({
          dependencies: [stages.require('horizon').stage, stages.require('derivatives').stage],
          // The irradiance integral is the heaviest per-change job: let the GPU finish one run
          // before the next is queued while a slider is dragged.
          minIntervalMs: 300,
          write: () => {
            const o = ctx.options;
            const start = getUtcFromZurich(o.dayOfYear, 0);
            const table = getGPUSolarIrradianceSunTable({
              longitude: villageLngLat[0],
              latitude: villageLngLat[1],
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
          }
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
        const settings = builder.settings(SUN_PAINT_PARAMETER_LENGTH);
        addSunPaintPass(builder.graph, {
          id: 'difference-paint',
          count: pixelCount,
          source: difference,
          settings: settings.view,
          output: builder.words('paint')
        });
        stage = builder.finishStage({
          dependencies: [stages.require('sun').stage, stages.require('cast').stage],
          write: () => settings.parameters.write(getSunPaintParameters(getPaintSettings(id)))
        });
        break;
      }
      default: {
        // The paint stages: one kernel from a compute raster to packed colours.
        const source = PAINT_SOURCES[id];
        if (!source) throw new Error(`No raster to paint for ${id}`);
        const settings = builder.settings(SUN_PAINT_PARAMETER_LENGTH);
        addSunPaintPass(builder.graph, {
          id: `${id}-kernel`,
          count: pixelCount,
          source: importFrom(source.stage, source.buffer, 'float32', pixelCount),
          settings: settings.view,
          output: builder.words('paint')
        });
        stage = builder.finishStage({
          dependencies: DEPENDENCIES[id].map(dependency => stages.require(dependency).stage),
          write: () => settings.parameters.write(getSunPaintParameters(getPaintSettings(id)))
        });
      }
    }
    return {key: '', stage, builder};
  }

  // --- Product display -----------------------------------------------------------------------------
  let activeStageId: StageId = 'paint-shadow';

  function showProduct(): void {
    const state = ctx.options;
    try {
      const stageId = getPaintStageId(state);
      const source = getValueSource(state);
      const key = registry.getFullKey(stageId, state);
      let build = session.getBuild(source.buildId, key);
      if (!build) {
        const entry = registry.ensure(stageId, state);
        const valueEntry = registry.require(source.stage);
        build = session.addBuild(
          session.createBuild(
            source.buildId,
            entry.stage,
            valueEntry.builder.getBuffer(source.buffer),
            'float32'
          ),
          key
        );
      }
      activeStageId = stageId;
      ctx.setStatus('');
      session.activate(build, UNUSED_PAINT);
      lastFloatStats = null;
      ctx.setReadout('tileMean', null);
      ctx.setChart('sunHistogram', null);
      lastDifferenceRun = -1;
      publishFurniture();
      publishAnnotations();
      publishMemory();
      ctx.requestLayers();
    } catch (error) {
      ctx.setStatus(`Cannot build ${state.product}: ${(error as Error).message}`);
    }
  }

  // --- Readouts and the hover probe --------------------------------------------------------------------
  let lastProbe: HoverCell | null = null;
  session.describeHover = hovered => {
    lastProbe = hovered;
    ctx.refreshTooltip();
    return null;
  };
  session.onStats = (_id, stats: ValueStats) => {
    if (stats.kind !== 'float') return;
    const state = ctx.options;
    lastFloatStats = stats;
    if (state.product === 'shadow' && state.method !== 'difference') {
      ctx.setReadout('tileMean', `${formatShare(stats.mean)} of the window in sun`);
    } else if (state.product === 'sun-hours') {
      ctx.setReadout('tileMean', `${stats.mean.toFixed(1)} h`);
    } else if (state.product === 'insolation') {
      ctx.setReadout('tileMean', `${(stats.mean / 1000).toFixed(2)} kWh/m²`);
    } else {
      ctx.setReadout('tileMean', null);
    }
    publishHistogram();
  };

  function publishMemory(): void {
    const info = getHorizonInfo(ctx.options);
    const megabytes = (info.byteLength / MEGABYTE).toFixed(0);
    ctx.setReadout(
      'horizonMemory',
      info.directions === info.requested
        ? `${megabytes} MB`
        : `${megabytes} MB (${info.directions} sectors: ${info.requested} need ${(info.requestedByteLength / MEGABYTE).toFixed(0)} MB)`
    );
    ctx.setReadout(
      'horizonSize',
      `${formatCount(width)} x ${formatCount(height)} cells x ${info.directions} sectors x ${info.bytesPerValue} bytes`
    );
  }

  function publishFurniture(): void {
    const state = ctx.options;
    const showsRadius =
      state.showSearch && state.product === 'shadow' && state.method === 'horizon';
    ctx.setFurniture({
      title: {
        subtitle: getCartoucheSubtitle(
          state,
          grid.groundCellSize,
          getZurichZoneName(getTimestamp())
        ),
        sample: demSampleLine(grid, 'swissALTI3D and other open DEMs via Mapterhorn')
      },
      scaleBar: {units: 'metric', ticks: showsRadius ? [getRadiusMeters(pinned.row)] : []}
    });
  }

  function publishAnnotations(): void {
    const state = ctx.options;
    const list: MapAnnotation[] = [];
    if (state.product === 'light') {
      list.push(
        ...getLightArrows(villageCoordinate, state.light === 'sun', sunNow, MAP_LIGHT_AZIMUTH)
      );
    } else if (timeline) {
      list.push(getVillageNote(villageCoordinate, timeline.sunHours, state.dayOfYear));
    }
    if (state.showSearch && state.product === 'shadow' && state.method === 'horizon') {
      list.push(getSearchRing(pinned.lngLat, getRadiusMeters(pinned.row)));
    }
    ctx.setAnnotations('sun', list);
    ctx.setHighlight(
      pinned.isVillage ? null : {kind: 'point', coordinate: pinned.lngLat, tone: 'signal'}
    );
  }

  let sunReader: SummaryReader | null = null;
  let sunReaderEntry: StageEntry | null = null;
  function ensureSunReader(): SummaryReader | null {
    const entry = registry.get('sun');
    if (!entry) return null;
    if (sunReader && sunReaderEntry === entry) return sunReader;
    sunReader?.stop();
    sunReader = new SummaryReader(
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
        // Row 0 is Zermatt, the cell that drives the shadow mask.
        const sun = getVillageSun();
        ctx.setReadout(
          'sunCpuGpu',
          `GPU ${floats[4].toFixed(2)}° up, ${floats[0].toFixed(2)}° az (${daylight[0] ? 'day' : 'night'}); float64 CPU differs by ${Math.abs(floats[4] - sun.altitudeDegrees).toFixed(4)}° and ${Math.abs(floats[0] - sun.azimuthDegrees).toFixed(4)}°`
        );
      }
    );
    sunReaderEntry = entry;
    return sunReader;
  }

  let probeReader: SummaryReader | null = null;
  let probeReaderEntry: StageEntry | null = null;
  function ensureProbeReader(): SummaryReader | null {
    const entry = registry.get('irradiance');
    if (!entry) return null;
    if (probeReader && probeReaderEntry === entry) return probeReader;
    probeReader?.stop();
    probeReader = new SummaryReader(
      resources,
      `probe-${entry.key}`,
      [{buffer: entry.builder.getBuffer('probe'), size: 16}],
      bytes => {
        if (destroyed) return;
        const values = new Float32Array(bytes, 0, 4);
        gpuVillage = {sunHours: values[0], insolation: values[1]};
        ctx.setReadout(
          'sunHoursCheck',
          timeline
            ? `Map ${values[0].toFixed(2)} h, CPU timeline ${timeline.sunHours.toFixed(2)} h`
            : `Map ${values[0].toFixed(2)} h`
        );
        publishHistogram();
      }
    );
    probeReaderEntry = entry;
    return probeReader;
  }

  // --- The pinned cell -----------------------------------------------------------------------------
  function setPinned(next: CellPin): void {
    pinned = next;
    pinnedSkyline = next.isVillage
      ? villageSkyline
      : getSkyline(probe, next.lngLat, getRadiusMeters(next.row));
    publishAnnotations();
    publishFurniture();
    chartsDirty = true;
  }
  const resetPinned = () =>
    setPinned({
      lngLat: villageCoordinate,
      column: villagePixel[0],
      row: villagePixel[1],
      isVillage: true
    });

  // --- Static readouts and first state -------------------------------------------------------------
  ctx.setReadout(
    'grid',
    `${formatCount(width)} x ${formatCount(height)} cells, ${grid.groundCellSize.toFixed(1)} m on the ground`
  );
  ctx.setReadout('castMemory', '0 MB');
  publishMemory();
  refreshDay();
  refreshClock();
  showProduct();
  publishCharts();

  // --- Animation ---------------------------------------------------------------------------------------
  /** The part of the day the animation plays: first to last light, with half an hour of margin. */
  function getAnimationWindow(): [number, number] {
    const rise = dayFacts.sunriseHour;
    const set = dayFacts.sunsetHour;
    return rise === null || set === null
      ? [6, 18]
      : [Math.max(0, rise - ANIMATION_MARGIN_HOURS), Math.min(24, set + ANIMATION_MARGIN_HOURS)];
  }

  return {
    getCompiledGraphs: () => session.getCompiledGraphs(),

    setOption(id) {
      const key = id as keyof SunOptions;
      const state = ctx.options;
      switch (key) {
        case 'product':
        case 'method':
          showProduct();
          return;
        case 'light':
          publishFurniture();
          publishAnnotations();
          ctx.requestLayers();
          return;
        case 'dayOfYear':
        case 'hour': {
          if (key === 'hour') clockHours = state.hour;
          registry.markDirty(PARAMETER_STAGES[key] ?? []);
          if (key === 'dayOfYear') refreshDay();
          refreshClock();
          ctx.setOptions({instant: getInstantHours(state.dayOfYear, clockHours)});
          publishFurniture();
          publishCharts();
          return;
        }
        case 'animate':
          if (state.animate) {
            clockHours = state.hour;
            const [start, end] = getAnimationWindow();
            if (clockHours < start || clockHours > end) clockHours = start;
          } else {
            refreshClock();
            publishCharts();
          }
          return;
        case 'showSearch':
          resetPinned();
          publishCharts();
          return;
        case 'scale':
        case 'display':
          refreshTables();
          markPaintDirty();
          ctx.requestLayers();
          return;
        case 'softness':
          registry.markDirty(PARAMETER_STAGES.softness ?? []);
          refreshDay();
          publishCharts();
          return;
        case 'directIrradiance':
        case 'diffuseIrradiance':
        case 'tableStep':
          registry.markDirty(PARAMETER_STAGES[key] ?? []);
          refreshDay();
          return;
        case 'refraction':
          referencePaths = referenceDays.map(dayOfYear => ({
            dayOfYear,
            samples: getSunSamples(dayOfYear, villageLngLat, state.refraction)
          }));
          break;
        case 'instant':
        case 'animationSpeed':
          return;
        default:
          break;
      }
      if (COMPILE_OPTIONS.has(key)) {
        if (key === 'horizonRadius') {
          refreshSkylines();
          refreshDay();
          publishAnnotations();
          publishFurniture();
          chartsDirty = true;
        }
        if (key === 'refraction') {
          refreshDay();
          refreshClock();
        }
        publishMemory();
        timers.push(setTimeout(() => !destroyed && showProduct(), REBUILD_DELAY_MILLISECONDS));
        return;
      }
      const stages = PARAMETER_STAGES[key];
      if (stages) {
        registry.markDirty(stages);
        return;
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'resetPin') {
        resetPinned();
        publishCharts();
        return;
      }
      if (id !== 'measure') return;
      ctx.setReadout('timing', 'measuring...');
      void session.measure().then(results => {
        if (destroyed) return;
        const total = results.reduce((sum, result) => sum + result.milliseconds, 0);
        ctx.setReadout(
          'timing',
          results.length === 0
            ? 'n/a'
            : `${total.toFixed(1)} ms, all stages of this view (${formatCount(pixelCount)} cells, ${results[0].method === 'gpu-timestamps' ? 'GPU timestamps' : 'wall clock'})`
        );
      });
    },

    onGroundChange(next) {
      ground.setGround(next);
      refreshTables();
      markPaintDirty();
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      session.getTooltip(event);
      if (!event.coordinate) return null;
      const pixel = grid.getPixel(event.coordinate[0], event.coordinate[1]);
      const hovered = lastProbe;
      if (!pixel || !hovered) return null;
      // The probe lags the pointer by a frame; a value from a neighbouring pixel is still honest.
      if (Math.abs(hovered.column - pixel[0]) > 3 || Math.abs(hovered.row - pixel[1]) > 3) {
        return null;
      }
      return getSunTooltip({
        state: ctx.options,
        tables,
        cell: hovered,
        probe,
        sun: sunNow,
        onGlacier: glacierMask[hovered.row * width + hovered.column] === 1
      });
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const pixel = grid.getPixel(event.coordinate[0], event.coordinate[1]);
      if (!pixel) return false;
      const lngLat = grid.getLongitudeLatitude(pixel[0], pixel[1]);
      setPinned({
        lngLat: [lngLat[0], lngLat[1]],
        column: pixel[0],
        row: pixel[1],
        isVillage: pixel[0] === villagePixel[0] && pixel[1] === villagePixel[1]
      });
      publishCharts();
      return true;
    },

    encode(commandEncoder, frame) {
      const state = ctx.options;
      if (state.animate && (state.product === 'shadow' || state.product === 'light')) {
        const [start, end] = getAnimationWindow();
        clockHours += frame.deltaSeconds * state.animationSpeed;
        if (clockHours > end || clockHours < start) clockHours = start;
        registry.markDirty(['sun', 'cast']);
        ctx.setOptions({hour: clockHours, instant: getInstantHours(state.dayOfYear, clockHours)});
        if (performance.now() - lastClockRefresh > CHART_INTERVAL_MILLISECONDS) {
          lastClockRefresh = performance.now();
          refreshClock();
        }
      }
      session.encode(commandEncoder);

      const sunStage = registry.get('sun')?.stage;
      const reader = ensureSunReader();
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
      const irradianceStage = registry.get('irradiance')?.stage;
      const probeBack = ensureProbeReader();
      if (probeBack && irradianceStage) {
        if (irradianceStage.runCount !== lastIrradianceRunCount) {
          probeBack.request(commandEncoder);
          lastIrradianceRunCount = irradianceStage.runCount;
        } else {
          probeBack.flush(commandEncoder);
        }
      }

      // How far the two shadow methods disagree: read the difference raster back when it has run.
      const differenceEntry = registry.get('difference');
      if (
        differenceEntry &&
        state.product === 'shadow' &&
        state.method !== 'horizon' &&
        !differenceBusy &&
        differenceEntry.stage.runCount !== lastDifferenceRun &&
        performance.now() - lastDifferenceRead > DIFFERENCE_READ_INTERVAL_MILLISECONDS
      ) {
        const runCount = differenceEntry.stage.runCount;
        const requested = session.readBulk(
          commandEncoder,
          differenceEntry.builder.getBuffer('difference'),
          bytes => {
            differenceBusy = false;
            if (destroyed) return;
            const share = getDifferenceShare(
              new Float32Array(bytes.buffer, bytes.byteOffset, pixelCount)
            );
            ctx.setReadout('differenceShare', formatShare(share));
          }
        );
        if (requested) {
          differenceBusy = true;
          lastDifferenceRun = runCount;
          lastDifferenceRead = performance.now();
        }
      }

      if (chartsDirty && performance.now() - lastChartTime > CHART_INTERVAL_MILLISECONDS) {
        publishCharts();
      }
    },

    getLayers(): Layer[] {
      const state = ctx.options;
      const tone = ctx.ground();
      const origin: [number, number, number] = [grid.origin[0], grid.origin[1], 0];
      const layers = session.getLayers({
        underlay: true,
        underlayAlpha: 1,
        alpha: 1,
        showProduct: false
      });
      const paintEntry = registry.get(activeStageId);
      const showsProduct = state.product !== 'light' || state.light === 'sun';
      if (paintEntry && showsProduct) {
        layers.push(
          new ColorRasterLayer({
            id: state.product === 'light' ? 'sun-light' : `sun-${state.product}`,
            coordinateOrigin: origin,
            gridSize: [width, height],
            bounds: grid.bounds,
            colors: paintEntry.builder.getBuffer('paint') as Buffer,
            alpha: 1,
            ...(state.product === 'light' ? {blendMode: 'multiply' as const} : {})
          })
        );
      }
      const colors = {
        eye: SUN_GOLD,
        outline: tone === 'dark' ? ([11, 13, 18, 255] as const) : ([31, 42, 77, 255] as const),
        halo: tone === 'dark' ? ([20, 23, 28] as const) : ([243, 239, 230] as const)
      } as const;
      const ink = getTerrainInk(tone);
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'sun-village-halo',
          coordinateOrigin: origin,
          positions: villageEye,
          instanceCount: 1,
          radiusPixels: 12,
          color: [colors.halo[0], colors.halo[1], colors.halo[2], Math.round(ink.haloAlpha * 255)]
        }),
        new SpatialAnalysisPointLayer({
          id: 'sun-village-eye',
          coordinateOrigin: origin,
          positions: villageEye,
          instanceCount: 1,
          radiusPixels: 7,
          color: colors.eye,
          outlineColor: colors.outline,
          outlineWidthPixels: 2
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const timer of timers) clearTimeout(timer);
      timers = [];
      sunReader?.stop();
      probeReader?.stop();
      session.destroy();
      ground.destroy();
      resources.destroy();
    }
  };
}
