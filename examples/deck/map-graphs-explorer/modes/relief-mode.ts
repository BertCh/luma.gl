// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Light and shadow in data space over the San Francisco elevation raster (the kepler.gl "Light &
 * Shadow" idea, computed on the GPU from the DEM instead of drawn by a lighting effect).
 *
 * Four compiled graphs, all built once in `create`:
 * - `setup` (encoded once, on the first frame): `GPUTerrainDerivatives` (slope and aspect) and
 *   `GPUTerrainHorizon` (16 azimuth sectors, sky-view factor). The horizon march is the expensive
 *   part (O(pixels x sectors x steps)) and does not depend on the sun, so it never reruns.
 * - `sun` (every frame): `GPUSolarPosition` for the map center (read back as a 12-byte summary and
 *   compared with the float64 CPU `getSolarPosition`) and `GPUSolarShadowMask`, which reads two
 *   horizon values per pixel and writes soft sun visibility plus illumination.
 * - `look` (only when a style control changed): `GPUTextureShading` and `GPUReliefShading`
 *   (single, multidirectional, USGS MDOW or Swiss/Imhof). Relief does not depend on the sun, so
 *   re-encoding it every frame would waste GPU time.
 *
 * Every slider, select and toggle is a parameter-buffer write; the rebuild counter stays 0.
 * Colors and illumination are drawn by a mode-local layer that unpacks the recipe's RGBA8 output.
 */

import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUReliefShadingParameterValues, getGPUSolarPositionParameterValues, getGPUSolarShadowMaskParameterValues, getGPUTerrainDerivativesParameterValues, getGPUTerrainHorizonParameterValues, getGPUTextureShadingParameterValues, getSolarPosition, GPU_RELIEF_SHADING_PARAMETER_LENGTH, GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES, GPU_SOLAR_POSITION_PARAMETER_LENGTH, GPU_SOLAR_SHADOW_MASK_PARAMETER_LENGTH, GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH, GPU_TERRAIN_HORIZON_PARAMETER_LENGTH, GPU_TEXTURE_SHADING_PARAMETER_LENGTH, GPUReliefShading, GPUSolarPosition, GPUSolarShadowMask, GPUTerrainDerivatives, GPUTerrainHorizon, GPUTextureShading, type GPUReliefShadingLight, type GPUReliefShadingStop} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import {LocalMetricProjection} from '../map-graphs-data';
import {MapGraphsRasterLayer, type MapGraphsRasterLayerProps} from '../map-graphs-layers';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {ReliefRasterLayer} from './relief-layers';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

/** Compile-time horizon sectors (`horizon[pixel * 16 + sector]`). */
const DIRECTION_COUNT = 16;
/** Compile-time horizon search radius in pixels (also the tile halo the recipe asks for). */
const HORIZON_RADIUS_PIXELS = 128;
/** Compile-time geometric growth of the ray-march step. */
const HORIZON_STEP_GROWTH = 1.08;
/** Compile-time number of texture-shading band-pass levels. */
const TEXTURE_LEVEL_COUNT = 5;
/** San Francisco is UTC-8 in standard time; the time slider is local standard time. */
const LOCAL_UTC_OFFSET_HOURS = -8;
/** Frames between sun readbacks. */
const READBACK_INTERVAL_FRAMES = 12;
/** Year used for the day-of-year slider. */
const YEAR = 2026;
/** Exposure of the composite: low winter sun gives small cosines, so brighten before clamping. */
const LIGHT_GAIN = 1.8;
/** Texture shade is in elevation meters times `gain`; 0.01 brings SF ridges to about +-0.5. */
const TEXTURE_GAIN = 0.01;

type Display =
  | 'composite'
  | 'relief'
  | 'illumination'
  | 'shadow'
  | 'hillshade'
  | 'skyview'
  | 'texture';
type ReliefStyle = 'single' | 'multidirectional' | 'mdow' | 'swiss';

/** Hypsometric tint, elevation meters to linear color. */
const ELEVATION_STOPS: readonly GPUReliefShadingStop[] = [
  {elevation: 0, color: [0.4, 0.62, 0.36]},
  {elevation: 40, color: [0.62, 0.74, 0.42]},
  {elevation: 100, color: [0.86, 0.8, 0.5]},
  {elevation: 180, color: [0.85, 0.65, 0.45]},
  {elevation: 260, color: [0.95, 0.88, 0.8]}
];

const MULTIDIRECTIONAL_LIGHTS: readonly GPUReliefShadingLight[] = [
  {azimuthDegrees: 225, altitudeDegrees: 30},
  {azimuthDegrees: 270, altitudeDegrees: 30},
  {azimuthDegrees: 315, altitudeDegrees: 30},
  {azimuthDegrees: 360, altitudeDegrees: 30}
];

export const reliefMode: MapGraphsModeDefinition = {
  id: 'relief',
  title: 'Relief',
  recipes: [
    'GPUSolarPosition',
    'GPUTerrainHorizon',
    'GPUSolarShadowMask',
    'GPUReliefShading',
    'GPUTextureShading'
  ],
  description:
    'Light and shadow over San Francisco terrain. Time of day and day of year drive the sun; ' +
    'soft shadows come from a horizon map computed once, so moving the sun is two buffer ' +
    'writes. Pick a relief style (single, multidirectional, MDOW, Swiss/Imhof) and tints.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.6},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const pixelCount = width * height;
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new MapGraphsResources(device, 'relief');
    const centerLongitudeLatitude = projection.unproject(
      (bounds[0] + bounds[2]) / 2,
      (bounds[1] + bounds[3]) / 2
    );

    // Sea (elevation 0) is invalid: derivatives, horizon and relief leave it transparent.
    const validityValues = new Uint32Array(pixelCount);
    let landCount = 0;
    for (let index = 0; index < pixelCount; index++) {
      const isLand = terrain.elevation[index] > 0.5;
      validityValues[index] = isLand ? 1 : 0;
      if (isLand) landCount++;
    }

    // --- Buffers -------------------------------------------------------------------------------
    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const slopeBuffer = resources.createBuffer('slope', pixelCount * 4);
    const aspectBuffer = resources.createBuffer('aspect', pixelCount * 4);
    const horizonByteLength = pixelCount * DIRECTION_COUNT * 4;
    const horizonBuffer = resources.createBuffer('horizon', horizonByteLength);
    const skyViewBuffer = resources.createBuffer('sky-view', pixelCount * 4);
    const sunVisibilityBuffer = resources.createBuffer('sun-visibility', pixelCount * 4);
    const illuminationBuffer = resources.createBuffer('illumination', pixelCount * 4);
    const textureShadeBuffer = resources.createBuffer('texture-shade', pixelCount * 4);
    const hillshadeBuffer = resources.createBuffer('hillshade', pixelCount * 4);
    const reliefColorBuffer = resources.createBuffer('relief-color', pixelCount * 4);
    const sunLocationBuffer = resources.createBuffer(
      'sun-location',
      Float32Array.of(centerLongitudeLatitude[0], centerLongitudeLatitude[1])
    );
    const sunAzimuthBuffer = resources.createBuffer('sun-azimuth', 4);
    const sunAltitudeBuffer = resources.createBuffer('sun-altitude', 4);
    const sunDaylightBuffer = resources.createBuffer('sun-daylight', 4);

    const derivativesSettings = resources.createParameterBuffer(
      'derivatives-settings',
      'float32',
      GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
    );
    const horizonSettings = resources.createParameterBuffer(
      'horizon-settings',
      'float32',
      GPU_TERRAIN_HORIZON_PARAMETER_LENGTH
    );
    const solarSettings = resources.createParameterBuffer(
      'solar-settings',
      'float32',
      GPU_SOLAR_POSITION_PARAMETER_LENGTH
    );
    const shadowSettings = resources.createParameterBuffer(
      'shadow-settings',
      'float32',
      GPU_SOLAR_SHADOW_MASK_PARAMETER_LENGTH
    );
    const textureSettings = resources.createParameterBuffer(
      'texture-settings',
      'float32',
      GPU_TEXTURE_SHADING_PARAMETER_LENGTH
    );
    const reliefSettings = resources.createParameterBuffer(
      'relief-settings',
      'float32',
      GPU_RELIEF_SHADING_PARAMETER_LENGTH
    );
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'relief-sun', byteLength: 12})
    );

    // --- Graphs --------------------------------------------------------------------------------
    /** Elevation band for one graph; every graph imports the same caller-owned buffers. */
    const importElevation = (graph: GPUCommandGraph<void>) => ({
      id: 'elevation',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
      },
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', pixelCount)
    });

    const setupGraph = new GPUCommandGraph<void>(device, {id: 'relief-setup'});
    const setupElevation = importElevation(setupGraph);
    setupGraph.add(
      new GPUTerrainDerivatives({
        id: 'derivatives',
        width,
        height,
        elevation: setupElevation,
        settings: derivativesSettings.importToGraph(setupGraph),
        slope: importGraphBuffer(setupGraph, 'slope', slopeBuffer, 'float32', pixelCount),
        aspect: importGraphBuffer(setupGraph, 'aspect', aspectBuffer, 'float32', pixelCount),
        cellSizeMode: 'uniform',
        rowDirection: 'south'
      })
    );
    setupGraph.add(
      new GPUTerrainHorizon({
        id: 'horizon',
        width,
        height,
        elevation: setupElevation,
        settings: horizonSettings.importToGraph(setupGraph),
        directionCount: DIRECTION_COUNT,
        maximumRadius: HORIZON_RADIUS_PIXELS,
        stepGrowth: HORIZON_STEP_GROWTH,
        cellSizeMode: 'uniform',
        rowDirection: 'south',
        horizon: importGraphBuffer(
          setupGraph,
          'horizon',
          horizonBuffer,
          'float32',
          pixelCount * DIRECTION_COUNT
        ),
        skyViewFactor: importGraphBuffer(
          setupGraph,
          'sky-view',
          skyViewBuffer,
          'float32',
          pixelCount
        )
      })
    );
    const compiledSetup: CompiledGPUCommandGraph<void> = resources.track(setupGraph.compile());

    const sunGraph = new GPUCommandGraph<void>(device, {id: 'relief-sun'});
    sunGraph.add(
      new GPUSolarPosition({
        id: 'solar-position',
        positions: importGraphBuffer(sunGraph, 'sun-location', sunLocationBuffer, 'float32x2', 1),
        settings: solarSettings.importToGraph(sunGraph),
        azimuth: importGraphBuffer(sunGraph, 'sun-azimuth', sunAzimuthBuffer, 'float32', 1),
        altitude: importGraphBuffer(sunGraph, 'sun-altitude', sunAltitudeBuffer, 'float32', 1),
        daylight: importGraphBuffer(sunGraph, 'sun-daylight', sunDaylightBuffer, 'uint32', 1)
      })
    );
    sunGraph.add(
      new GPUSolarShadowMask({
        id: 'shadow-mask',
        width,
        height,
        directionCount: DIRECTION_COUNT,
        horizon: importGraphBuffer(
          sunGraph,
          'horizon',
          horizonBuffer,
          'float32',
          pixelCount * DIRECTION_COUNT
        ),
        settings: shadowSettings.importToGraph(sunGraph),
        slope: importGraphBuffer(sunGraph, 'slope', slopeBuffer, 'float32', pixelCount),
        aspect: importGraphBuffer(sunGraph, 'aspect', aspectBuffer, 'float32', pixelCount),
        skyViewFactor: importGraphBuffer(
          sunGraph,
          'sky-view',
          skyViewBuffer,
          'float32',
          pixelCount
        ),
        sunVisibility: importGraphBuffer(
          sunGraph,
          'sun-visibility',
          sunVisibilityBuffer,
          'float32',
          pixelCount
        ),
        illumination: importGraphBuffer(
          sunGraph,
          'illumination',
          illuminationBuffer,
          'float32',
          pixelCount
        )
      })
    );
    const compiledSun: CompiledGPUCommandGraph<void> = resources.track(sunGraph.compile());

    const lookGraph = new GPUCommandGraph<void>(device, {id: 'relief-look'});
    const lookElevation = importElevation(lookGraph);
    const textureShadeView = importGraphBuffer(
      lookGraph,
      'texture-shade',
      textureShadeBuffer,
      'float32',
      pixelCount
    );
    lookGraph.add(
      new GPUTextureShading({
        id: 'texture-shading',
        width,
        height,
        elevation: lookElevation,
        settings: textureSettings.importToGraph(lookGraph),
        levelCount: TEXTURE_LEVEL_COUNT,
        baseSigma: 1,
        textureShade: textureShadeView
      })
    );
    lookGraph.add(
      new GPUReliefShading({
        id: 'relief-shading',
        width,
        height,
        elevation: lookElevation,
        settings: reliefSettings.importToGraph(lookGraph),
        skyViewFactor: importGraphBuffer(
          lookGraph,
          'sky-view',
          skyViewBuffer,
          'float32',
          pixelCount
        ),
        textureShade: textureShadeView,
        cellSizeMode: 'uniform',
        rowDirection: 'south',
        hillshade: importGraphBuffer(
          lookGraph,
          'hillshade',
          hillshadeBuffer,
          'float32',
          pixelCount
        ),
        color: importGraphBuffer(lookGraph, 'relief-color', reliefColorBuffer, 'uint32', pixelCount)
      })
    );
    const compiledLook: CompiledGPUCommandGraph<void> = resources.track(lookGraph.compile());

    // --- State ---------------------------------------------------------------------------------
    let display: Display = 'composite';
    let style: ReliefStyle = 'swiss';
    let hour = 15.75;
    let dayOfYear = 355;
    let animate = false;
    let animationSpeed = 2;
    let softness = 2;
    let ambient = 0.15;
    let shadowStrength = 1;
    let lightAzimuth = 315;
    let exposure = 1.1;
    let textureDetail = 0.5;
    let textureStrength = 0.4;
    let elevationTint = true;
    let aspectTint = true;
    let skyViewShading = true;
    let textureShading = true;
    let opacity = 0.92;
    let lookDirty = true;
    let readbackPending = false;
    let destroyed = false;
    let sun = getSun();

    function getTimestamp(): number {
      // Day 1 is January 1 local; hours are local standard time.
      return Date.UTC(YEAR, 0, dayOfYear, hour - LOCAL_UTC_OFFSET_HOURS, 0, 0);
    }

    function getSun() {
      return getSolarPosition(
        getTimestamp(),
        centerLongitudeLatitude[0],
        centerLongitudeLatitude[1]
      );
    }

    function formatClock(): string {
      const date = new Date(Date.UTC(YEAR, 0, dayOfYear));
      const month = date.toLocaleString('en-US', {month: 'short', timeZone: 'UTC'});
      return `${month} ${date.getUTCDate()}, ${formatHour(hour)} PST`;
    }

    function writeStatic(): void {
      derivativesSettings.write(
        getGPUTerrainDerivativesParameterValues({
          cellSize,
          azimuthDegrees: 315,
          altitudeDegrees: 45
        })
      );
      horizonSettings.write(getGPUTerrainHorizonParameterValues({cellSize}));
    }

    function writeSun(): void {
      sun = getSun();
      solarSettings.write(getGPUSolarPositionParameterValues({timestamp: getTimestamp()}));
      shadowSettings.write(
        getGPUSolarShadowMaskParameterValues({
          azimuthDegrees: sun.azimuthDegrees,
          altitudeDegrees: sun.altitudeDegrees,
          angularRadiusDegrees: GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES * softness,
          sunIntensity: 1,
          ambientIntensity: ambient
        })
      );
    }

    function writeLook(): void {
      const lightSettings =
        style === 'single'
          ? {
              lights: [{azimuthDegrees: lightAzimuth, altitudeDegrees: 45}] as const,
              lightWeighting: 'fixed' as const
            }
          : style === 'multidirectional'
            ? {lights: MULTIDIRECTIONAL_LIGHTS, lightWeighting: 'fixed' as const}
            : {lights: 'mdow' as const, lightWeighting: 'aspect' as const};
      reliefSettings.write(
        getGPUReliefShadingParameterValues({
          cellSize,
          ...lightSettings,
          hillshadeStrength: 1,
          skyViewStrength: skyViewShading ? 1 : 0,
          textureShadeStrength: textureShading ? textureStrength : 0,
          exposure,
          tintStrength: aspectTint ? 0.35 : 0,
          elevationStops: elevationTint ? ELEVATION_STOPS : []
        })
      );
      textureSettings.write(
        getGPUTextureShadingParameterValues({detail: textureDetail, gain: TEXTURE_GAIN})
      );
      lookDirty = true;
    }

    /** Swiss/Imhof is the cartographic preset: every optional blend input on. */
    function applySwissPreset(): void {
      elevationTint = true;
      aspectTint = true;
      skyViewShading = true;
      textureShading = true;
      elevationTintToggle.setValue(true);
      aspectTintToggle.setValue(true);
      skyViewToggle.setValue(true);
      textureToggle.setValue(true);
    }

    writeStatic();
    writeSun();

    // --- Controls ------------------------------------------------------------------------------
    const sunReadout = context.controls.addReadout('Sun (float64 CPU)');
    const gpuSunReadout = context.controls.addReadout('Sun (GPUSolarPosition)', '...');
    const clockReadout = context.controls.addReadout('Local standard time', formatClock());

    const updateSunReadouts = () => {
      clockReadout.setValue(formatClock());
      sunReadout.setValue(
        `az ${sun.azimuthDegrees.toFixed(1)}°, alt ${sun.altitudeDegrees.toFixed(1)}° ` +
          (sun.geometricAltitudeDegrees > -0.833 ? '(day)' : '(night)')
      );
    };
    updateSunReadouts();

    context.controls.addSelect<Display>({
      label: 'Display',
      options: [
        {value: 'composite', label: 'Relief color x sun illumination'},
        {value: 'relief', label: 'Relief color only'},
        {value: 'illumination', label: 'Illumination (sun + ambient)'},
        {value: 'shadow', label: 'Sun visibility (soft shadow)'},
        {value: 'hillshade', label: 'Hillshade'},
        {value: 'skyview', label: 'Sky-view factor'},
        {value: 'texture', label: 'Texture shade'}
      ],
      value: display,
      onChange: value => {
        display = value;
        context.updateLayers();
      }
    });
    const hourSlider = context.controls.addSlider({
      label: 'Time of day (per-frame)',
      min: 0,
      max: 24,
      step: 0.05,
      value: hour,
      format: formatHour,
      onChange: value => {
        hour = value;
        writeSun();
        updateSunReadouts();
      }
    });
    context.controls.addSlider({
      label: 'Day of year (per-frame)',
      min: 1,
      max: 365,
      step: 1,
      value: dayOfYear,
      format: value => formatDay(value),
      onChange: value => {
        dayOfYear = value;
        writeSun();
        updateSunReadouts();
      }
    });
    context.controls.addToggle({
      label: 'Animate time of day',
      value: animate,
      onChange: value => {
        animate = value;
      }
    });
    context.controls.addSlider({
      label: 'Animation speed',
      min: 0.5,
      max: 8,
      step: 0.5,
      value: animationSpeed,
      format: value => `${value} h/s`,
      onChange: value => {
        animationSpeed = value;
      }
    });
    context.controls.addSlider({
      label: 'Shadow softness (solar disk radius)',
      min: 0,
      max: 8,
      step: 0.5,
      value: softness,
      format: value =>
        value === 0 ? 'hard' : `${(value * GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES).toFixed(2)}°`,
      onChange: value => {
        softness = value;
        writeSun();
      }
    });
    context.controls.addSlider({
      label: 'Ambient light (x sky-view factor)',
      min: 0,
      max: 0.8,
      step: 0.05,
      value: ambient,
      format: value => value.toFixed(2),
      onChange: value => {
        ambient = value;
        writeSun();
      }
    });
    context.controls.addSlider({
      label: 'Shadow strength in composite',
      min: 0,
      max: 1,
      step: 0.05,
      value: shadowStrength,
      format: value => value.toFixed(2),
      onChange: value => {
        shadowStrength = value;
        context.updateLayers();
      }
    });
    context.controls.addSelect<ReliefStyle>({
      label: 'Relief style (GPUReliefShading, per-frame)',
      options: [
        {value: 'single', label: 'Single light'},
        {value: 'multidirectional', label: 'Multidirectional (4 fixed lights)'},
        {value: 'mdow', label: 'USGS MDOW (aspect-weighted)'},
        {value: 'swiss', label: 'Swiss / Imhof (MDOW + tints)'}
      ],
      value: style,
      onChange: value => {
        style = value;
        if (style === 'swiss') applySwissPreset();
        writeLook();
      }
    });
    context.controls.addSlider({
      label: 'Single-light azimuth',
      min: 0,
      max: 360,
      step: 5,
      value: lightAzimuth,
      format: value => `${value}°`,
      onChange: value => {
        lightAzimuth = value;
        writeLook();
      }
    });
    context.controls.addSlider({
      label: 'Relief exposure',
      min: 0.6,
      max: 1.6,
      step: 0.05,
      value: exposure,
      format: value => value.toFixed(2),
      onChange: value => {
        exposure = value;
        writeLook();
      }
    });
    const elevationTintToggle = context.controls.addToggle({
      label: 'Elevation tint',
      value: elevationTint,
      onChange: value => {
        elevationTint = value;
        writeLook();
      }
    });
    const aspectTintToggle = context.controls.addToggle({
      label: 'Warm / cool aspect tint',
      value: aspectTint,
      onChange: value => {
        aspectTint = value;
        writeLook();
      }
    });
    const skyViewToggle = context.controls.addToggle({
      label: 'Sky-view shading in relief',
      value: skyViewShading,
      onChange: value => {
        skyViewShading = value;
        writeLook();
      }
    });
    const textureToggle = context.controls.addToggle({
      label: 'Texture shading in relief',
      value: textureShading,
      onChange: value => {
        textureShading = value;
        writeLook();
      }
    });
    context.controls.addSlider({
      label: 'Texture detail (alpha)',
      min: 0,
      max: 1,
      step: 0.05,
      value: textureDetail,
      format: value => value.toFixed(2),
      onChange: value => {
        textureDetail = value;
        writeLook();
      }
    });
    context.controls.addSlider({
      label: 'Texture strength',
      min: 0,
      max: 1.5,
      step: 0.1,
      value: textureStrength,
      format: value => value.toFixed(1),
      onChange: value => {
        textureStrength = value;
        writeLook();
      }
    });
    context.controls.addSlider({
      label: 'Layer opacity',
      min: 0.3,
      max: 1,
      step: 0.05,
      value: opacity,
      format: value => value.toFixed(2),
      onChange: value => {
        opacity = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Sun illumination (shadow to lit)',
      gradient: {
        colors: [
          [10, 10, 14],
          [120, 100, 90],
          [255, 245, 225]
        ],
        minimumLabel: 'ambient only',
        maximumLabel: 'full sun'
      }
    });
    context.controls.addLegend({
      title: 'Elevation tint (Swiss style)',
      entries: ELEVATION_STOPS.map(stop => ({
        color: [
          Math.round(stop.color[0] * 255),
          Math.round(stop.color[1] * 255),
          Math.round(stop.color[2] * 255),
          255
        ],
        label: `${stop.elevation} m`
      }))
    });
    context.controls.addNote(
      'The horizon map is computed once at startup. Time, day, softness and ambient are buffer ' +
        'writes read by the shadow mask every frame; style and tint changes re-encode only the ' +
        'relief graph. Sea is masked as invalid.'
    );
    context.controls.addReadout(
      'Raster',
      `${width} × ${height} cells, ${formatCount(landCount)} land`
    );
    context.controls.addReadout(
      'Cell size',
      `${cellSize[0].toFixed(1)} × ${cellSize[1].toFixed(1)} m`
    );
    context.controls.addReadout(
      'Horizon map',
      `${DIRECTION_COUNT} sectors × ${HORIZON_RADIUS_PIXELS} px (${((HORIZON_RADIUS_PIXELS * cellSize[0]) / 1000).toFixed(1)} km), ` +
        `${(horizonByteLength / 1048576).toFixed(1)} MB`
    );
    const setupTimingReadout = context.controls.addReadout('Setup graph (once)', 'measuring...');
    const sunTimingReadout = context.controls.addReadout('Sun graph (every frame)', 'measuring...');
    const lookTimingReadout = context.controls.addReadout('Look graph (on change)', 'measuring...');
    context.controls.addReadout('Data', terrain.attribution);

    const measureAll = async () => {
      try {
        const options = {
          parameters: undefined,
          completionBuffer: sunVisibilityBuffer,
          signal: context.signal
        };
        const setupTiming = await measureCompiledGraph(device, compiledSetup, {
          ...options,
          runs: 3,
          warmUpRuns: 1,
          repetitions: 2
        });
        setupTimingReadout.setValue(
          `${compiledSetup.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(setupTiming)}`
        );
        const sunTiming = await measureCompiledGraph(device, compiledSun, options);
        sunTimingReadout.setValue(
          `${compiledSun.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(sunTiming)}`
        );
        const lookTiming = await measureCompiledGraph(device, compiledLook, options);
        lookTimingReadout.setValue(
          `${compiledLook.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(lookTiming)}`
        );
      } catch (error) {
        if (!destroyed) setupTimingReadout.setValue(`failed: ${(error as Error).message}`);
      }
    };
    context.controls.addButton({label: 'Measure GPU cost', onClick: () => void measureAll()});

    writeLook();

    // --- Per-frame encode ----------------------------------------------------------------------
    const readSun = async (commandEncoder: Parameters<MapGraphsModeInstance['encode']>[0]) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: sunAzimuthBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 0,
        size: 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: sunAltitudeBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 4,
        size: 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: sunDaylightBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 8,
        size: 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: 12});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const floats = new Float32Array(bytes.buffer, bytes.byteOffset, 2);
        const daylight = new Uint32Array(bytes.buffer, bytes.byteOffset + 8, 1)[0];
        const azimuthError = ((floats[0] - sun.azimuthDegrees + 540) % 360) - 180;
        gpuSunReadout.setValue(
          `az ${floats[0].toFixed(1)}°, alt ${floats[1].toFixed(1)}° ${daylight ? '(day)' : '(night)'}` +
            ` · Δ ${azimuthError.toFixed(2)}° / ${(floats[1] - sun.altitudeDegrees).toFixed(2)}°`
        );
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [compiledSetup, compiledSun, compiledLook],
      encode(commandEncoder, frame) {
        if (frame.frameIndex === 0) {
          // The horizon map depends only on the elevation raster: encode it exactly once.
          compiledSetup.encode(commandEncoder, {parameters: undefined});
        }
        if (animate) {
          hour = (hour + frame.deltaSeconds * animationSpeed) % 24;
          hourSlider.setValue(hour);
          writeSun();
          if (frame.frameIndex % 4 === 0) updateSunReadouts();
        }
        if (lookDirty) {
          // Relief and texture shading do not depend on the sun, so they re-encode only on change.
          compiledLook.encode(commandEncoder, {parameters: undefined});
          lookDirty = false;
        }
        compiledSun.encode(commandEncoder, {parameters: undefined});
        if (!readbackPending && frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) {
          void readSun(commandEncoder);
        }
      },
      getLayers() {
        const rasterProps = {
          coordinateOrigin: origin,
          gridSize: [width, height] as const,
          bounds,
          rowOrigin: 'north' as const,
          valueFormat: 'float32' as const
        } satisfies Partial<MapGraphsRasterLayerProps>;
        if (display === 'composite' || display === 'relief') {
          return [
            new ReliefRasterLayer({
              id: 'relief-composite',
              coordinateOrigin: origin,
              colors: reliefColorBuffer,
              light: illuminationBuffer,
              lightStrength: display === 'composite' ? shadowStrength : 0,
              gridSize: [width, height],
              bounds,
              lightGain: LIGHT_GAIN,
              opacity
            })
          ];
        }
        const scalars: Record<
          Exclude<Display, 'composite' | 'relief'>,
          Partial<MapGraphsRasterLayerProps> & {id: string}
        > = {
          illumination: {
            id: 'relief-illumination',
            values: illuminationBuffer,
            colormap: 'inferno',
            valueRange: [0, 1.2]
          },
          shadow: {
            id: 'relief-shadow',
            values: sunVisibilityBuffer,
            colormap: 'grayscale',
            valueRange: [0, 1]
          },
          hillshade: {
            id: 'relief-hillshade',
            values: hillshadeBuffer,
            colormap: 'grayscale',
            valueRange: [0, 1]
          },
          skyview: {
            id: 'relief-skyview',
            values: skyViewBuffer,
            colormap: 'viridis',
            valueRange: [0.6, 1]
          },
          texture: {
            id: 'relief-texture',
            values: textureShadeBuffer,
            colormap: 'grayscale',
            valueRange: [-0.5, 0.5]
          }
        };
        const {id, ...scalarStyle} = scalars[display];
        return [
          new MapGraphsRasterLayer({
            ...rasterProps,
            id,
            ...scalarStyle,
            gridSize: [width, height],
            bounds,
            opacity
          } as MapGraphsRasterLayerProps)
        ];
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };

    // Measure once the first frames have run, outside Deck's frame encoder.
    setTimeout(() => {
      if (!destroyed) void measureAll();
    }, 1500);
    return instance;
  }
};

/** Formats decimal hours as `"15:45"`. */
function formatHour(hours: number): string {
  const totalMinutes = Math.round(hours * 60) % 1440;
  const minutes = totalMinutes % 60;
  return `${String(Math.floor(totalMinutes / 60)).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
}

/** Formats a 1-based day of year as `"Dec 21"`. */
function formatDay(dayOfYear: number): string {
  const date = new Date(Date.UTC(YEAR, 0, dayOfYear));
  return `${date.toLocaleString('en-US', {month: 'short', timeZone: 'UTC'})} ${date.getUTCDate()}`;
}
