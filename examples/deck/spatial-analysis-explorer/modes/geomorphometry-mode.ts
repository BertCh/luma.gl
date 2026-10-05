// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Landforms of the San Francisco elevation raster, entirely on the GPU. One product is active at a
 * time and each product is its own compiled graph: the geomorphometry contributor(s) followed by a
 * display kernel that turns the raw measure into a packed RGBA8 color and a float value for the
 * hover readout.
 *
 * - Curvature: one of the Florinsky kinds, the partial-derivative method, and the multi-radius ring.
 * - Geomorphons: ten landform classes from lines of sight (search radius, flatness angle).
 * - Ruggedness: TPI, TRI (Riley or Wilson) and roughness from one contributor.
 * - Vector ruggedness (VRM): surface-normal dispersion in a square window.
 * - Multiscale DEV: deviation from mean elevation over twelve scales from one summed-area table.
 * - Weiss landforms: ten classes from small- and large-scale TPI plus slope.
 *
 * Per-frame parameter writes (no recompile): the symmetric color clamp, the quantity shown by the
 * ruggedness and multiscale products, the multiscale scale, the geomorphon flatness angle and the
 * Weiss thresholds. Compile-time choices (curvature kind, method, ring radius, search and window
 * radii, standardization, algorithm) rebuild that product's graph, which the shell counts.
 * Signed and sequential products are calibrated once per rebuild from a single 98th percentile
 * readback so the clamp slider starts at a useful range.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUGeomorphonsParameterValues,
  getGPUTerrainCurvatureParameterValues,
  getGPUTerrainVectorRuggednessParameterValues,
  getGPUTerrainWeissLandformsParameterValues,
  GPU_GEOMORPHONS_PARAMETER_LENGTH,
  GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH,
  GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH,
  GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH,
  GPUGeomorphons,
  GPUTerrainCurvature,
  GPUTerrainRuggedness,
  GPUTerrainTopographicPosition,
  GPUTerrainVectorRuggedness,
  GPUTerrainWeissLandforms,
  type GPUGeomorphonComparison,
  type GPUTerrainCurvatureKind,
  type GPUTerrainCurvatureMethod,
  type GPUTerrainRuggednessAlgorithm,
  type GPUTerrainWeissStandardization
} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance,
  SpatialAnalysisPointerEvent
} from '../spatial-analysis-mode';
import {SpatialAnalysisResources} from '../spatial-analysis-resources';
import {addKernelPass, type KernelBinding} from './mode-kernels';
import {ReliefRasterLayer} from './relief-layers';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

type Product = 'curvature' | 'geomorphons' | 'ruggedness' | 'vrm' | 'multiscale' | 'weiss';
type CurvatureChoice = GPUTerrainCurvatureKind | 'ring-multi-radius';
type RuggednessQuantity = 'tpi' | 'tri' | 'roughness';
type MultiscaleQuantity = 'dev' | 'devmax' | 'devmax-scale';
type Color = readonly [number, number, number];

/** Compile-time scales (cells) of the multiscale product; one summed-area table serves all. */
const MULTISCALE_RADII = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64] as const;
/** Color mode written to the display kernel. */
const COLOR_MODE = {diverging: 0, sequential: 1, categorical: 2} as const;
const REBUILD_DELAY_MILLISECONDS = 250;
const CALIBRATION_PERCENTILE = 0.98;

const DIVERGING_RAMP: readonly Color[] = [
  [5, 48, 97],
  [67, 147, 195],
  [247, 247, 247],
  [214, 96, 77],
  [103, 0, 31]
];
const SEQUENTIAL_RAMP: readonly Color[] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37]
];

/** Standard GRASS `r.geomorphon` colors, indexed by class 1 to 10. */
const GEOMORPHON_CLASSES: readonly {label: string; color: Color}[] = [
  {label: 'Flat', color: [220, 220, 220]},
  {label: 'Peak', color: [56, 0, 0]},
  {label: 'Ridge', color: [200, 0, 0]},
  {label: 'Shoulder', color: [255, 80, 20]},
  {label: 'Spur', color: [250, 210, 60]},
  {label: 'Slope', color: [255, 255, 60]},
  {label: 'Hollow', color: [180, 230, 20]},
  {label: 'Footslope', color: [60, 250, 150]},
  {label: 'Valley', color: [0, 0, 255]},
  {label: 'Pit', color: [0, 0, 56]}
];
/** Weiss (2001) classes in `GPU_TERRAIN_WEISS_LANDFORMS` order. */
const WEISS_CLASSES: readonly {label: string; color: Color}[] = [
  {label: 'Canyon', color: [36, 0, 120]},
  {label: 'Midslope drainage', color: [40, 90, 200]},
  {label: 'Upland drainage', color: [110, 170, 230]},
  {label: 'U-shaped valley', color: [40, 170, 150]},
  {label: 'Plain', color: [240, 235, 170]},
  {label: 'Open slope', color: [220, 190, 110]},
  {label: 'Upper slope', color: [190, 140, 60]},
  {label: 'Local ridge', color: [240, 140, 90]},
  {label: 'Midslope ridge', color: [210, 70, 60]},
  {label: 'Mountain top', color: [130, 0, 10]}
];

const CURVATURE_OPTIONS: readonly {value: CurvatureChoice; label: string}[] = [
  {value: 'profile', label: 'Profile'},
  {value: 'plan', label: 'Plan'},
  {value: 'tangential', label: 'Tangential'},
  {value: 'mean', label: 'Mean'},
  {value: 'gaussian', label: 'Gaussian'},
  {value: 'minimal', label: 'Minimal'},
  {value: 'maximal', label: 'Maximal'},
  {value: 'unsphericity', label: 'Unsphericity'},
  {value: 'difference', label: 'Difference'},
  {value: 'horizontal-excess', label: 'Horizontal excess'},
  {value: 'vertical-excess', label: 'Vertical excess'},
  {value: 'accumulation', label: 'Accumulation'},
  {value: 'ring', label: 'Ring (excess product)'},
  {value: 'rotor', label: 'Rotor'},
  {value: 'laplacian', label: 'Laplacian'},
  {value: 'ring-multi-radius', label: 'Ring (multi-radius)'}
];

/** One compiled product graph and everything that must be released with it. */
type ProductBuild = {
  compiled: CompiledGPUCommandGraph<undefined>;
  release: () => void;
  /** Writes the per-frame settings of the product; called on every relevant slider move. */
  writeSettings: () => void;
};

export const geomorphometryMode: SpatialAnalysisModeDefinition = {
  id: 'geomorphometry',
  title: 'Landforms',
  contributors: [
    'GPUTerrainCurvature',
    'GPUGeomorphons',
    'GPUTerrainRuggedness',
    'GPUTerrainVectorRuggedness',
    'GPUTerrainTopographicPosition',
    'GPUTerrainWeissLandforms'
  ],
  description:
    'Landform measures of one elevation raster: curvature kinds, geomorphons, TPI / TRI / ' +
    'roughness, vector ruggedness, multiscale deviation from mean elevation and Weiss classes. ' +
    'The color clamp, thresholds and scale are per-frame parameters; radii and methods rebuild ' +
    'one graph. Hover for the value or class under the pointer.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.6},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const pixelCount = width * height;
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new SpatialAnalysisResources(device, 'landforms');

    // Sea (elevation 0) is invalid so every contributor leaves it transparent.
    const validityValues = new Uint32Array(pixelCount);
    for (let index = 0; index < pixelCount; index++) {
      validityValues[index] = terrain.elevation[index] > 0.5 ? 1 : 0;
    }
    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const valuesBuffer = resources.createBuffer('values', pixelCount * 4);
    const colorsBuffer = resources.createBuffer('colors', pixelCount * 4);
    /** `[clamp, scalePlane, quantity, colorMode, 0, 0, 0, 0]`, read by the display kernel. */
    const displayParameters = resources.createParameterBuffer('display', 'float32', 8);
    const hoverRing = resources.track(
      new GPUReadbackRing(device, {id: 'landforms-hover', byteLength: 4})
    );
    const calibrationRing = resources.track(
      new GPUReadbackRing(device, {id: 'landforms-calibration', byteLength: pixelCount * 4})
    );

    // --- State ---------------------------------------------------------------------------------
    let product: Product = 'curvature';
    let curvatureChoice: CurvatureChoice = 'profile';
    let curvatureMethod: GPUTerrainCurvatureMethod = 'evans-young';
    let ringRadius = 2;
    let geomorphonRadius = 20;
    let geomorphonComparison: GPUGeomorphonComparison = 'anglev1';
    let flatAngle = 1;
    let ruggednessQuantity: RuggednessQuantity = 'tpi';
    let ruggednessAlgorithm: GPUTerrainRuggednessAlgorithm = 'riley';
    let vectorRadius = 2;
    let multiscaleQuantity: MultiscaleQuantity = 'dev';
    let multiscaleIndex = 5;
    let smallRadius = 3;
    let largeRadius = 15;
    let standardization: GPUTerrainWeissStandardization = 'global';
    let standardThreshold = 1;
    let slopeThreshold = 5;
    let clampExponent = 0;
    let nominalClamp = 1;
    let opacity = 0.85;
    let build: ProductBuild | null = null;
    let generation = 0;
    let rebuildTimer: ReturnType<typeof setTimeout> | undefined;
    let destroyed = false;
    let calibrationRequested = false;
    let calibrationPending = false;
    let hoverPending = false;
    let hoverIndex = -1;
    let hoverValue = Number.NaN;
    let hoverResolvedIndex = -1;
    let hoverResolvedGeneration = -1;

    function getColorMode(): number {
      if (product === 'geomorphons' || product === 'weiss') return COLOR_MODE.categorical;
      if (product === 'curvature') {
        return curvatureChoice === 'unsphericity' ? COLOR_MODE.sequential : COLOR_MODE.diverging;
      }
      if (product === 'ruggedness') {
        return ruggednessQuantity === 'tpi' ? COLOR_MODE.diverging : COLOR_MODE.sequential;
      }
      if (product === 'vrm') return COLOR_MODE.sequential;
      return multiscaleQuantity === 'devmax-scale' ? COLOR_MODE.sequential : COLOR_MODE.diverging;
    }

    /** Fixed nominal clamp for products with a known range; `null` means calibrate. */
    function getFixedNominalClamp(): number | null {
      if (product === 'multiscale') {
        return multiscaleQuantity === 'devmax-scale'
          ? MULTISCALE_RADII[MULTISCALE_RADII.length - 1]
          : 2.5;
      }
      if (product === 'geomorphons' || product === 'weiss') return 1;
      return null;
    }

    function getClamp(): number {
      return nominalClamp * 10 ** clampExponent;
    }

    function writeDisplay(): void {
      hoverResolvedIndex = -1;
      displayParameters.write(
        Float32Array.of(
          getClamp(),
          multiscaleIndex,
          product === 'ruggedness'
            ? ['tpi', 'tri', 'roughness'].indexOf(ruggednessQuantity)
            : ['dev', 'devmax', 'devmax-scale'].indexOf(multiscaleQuantity),
          getColorMode(),
          0,
          0,
          0,
          0
        )
      );
    }

    // --- Graph construction --------------------------------------------------------------------
    function makeElevation(graph: GPUCommandGraph<undefined>) {
      return {
        id: 'elevation',
        format: 'float32' as const,
        storage: {
          kind: 'buffer' as const,
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        },
        validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', pixelCount)
      };
    }

    /**
     * Adds the display kernel: `fetch` assigns `value` (and may read the `display` words), the rest
     * maps it to a hover value and a packed color through the diverging, sequential or categorical
     * ramp chosen by the per-frame color mode.
     */
    function addDisplayKernel(
      graph: GPUCommandGraph<undefined>,
      options: {
        classes?: readonly {label: string; color: Color}[];
        inputs: readonly KernelBinding[];
        fetch: string;
      }
    ): void {
      const palette = options.classes ?? GEOMORPHON_CLASSES;
      addKernelPass(graph, {
        id: `${product}-display`,
        invocationCount: pixelCount,
        bindings: [
          ...options.inputs,
          {
            name: 'display',
            view: displayParameters.importToGraph(graph),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'values',
            view: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', pixelCount),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'colors',
            view: importGraphBuffer(graph, 'colors', colorsBuffer, 'uint32', pixelCount),
            type: 'u32',
            access: 'read_write'
          }
        ],
        declarations: /* wgsl */ `
const PIXEL_COUNT: u32 = ${pixelCount}u;
var<private> DIVERGING: array<vec3<f32>, 5> = array<vec3<f32>, 5>(${DIVERGING_RAMP.map(formatColor).join(', ')});
var<private> SEQUENTIAL: array<vec3<f32>, 5> = array<vec3<f32>, 5>(${SEQUENTIAL_RAMP.map(formatColor).join(', ')});
var<private> CLASSES: array<vec3<f32>, 10> = array<vec3<f32>, 10>(${palette.map(entry => formatColor(entry.color)).join(', ')});
fn isBad(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}
fn rampColor(diverging: bool, t: f32) -> vec3<f32> {
  let scaled = clamp(t, 0.0, 1.0) * 4.0;
  let i = min(u32(scaled), 3u);
  let f = scaled - f32(i);
  if (diverging) {
    return mix(DIVERGING[i], DIVERGING[i + 1u], f);
  }
  return mix(SEQUENTIAL[i], SEQUENTIAL[i + 1u], f);
}`,
        body: /* wgsl */ `
  var value = 0.0;
  ${options.fetch}
  let clampValue = max(display[displayOffset], 1e-30);
  let mode = u32(display[displayOffset + 3u]);
  var color = vec4<f32>(0.0);
  if (mode == 2u) {
    let classIndex = u32(max(value, 0.0));
    if (classIndex >= 1u && classIndex <= 10u) {
      color = vec4<f32>(CLASSES[classIndex - 1u], 1.0);
    }
  } else if (!isBad(value)) {
    if (mode == 0u) {
      let signedPosition = clamp(value / clampValue, -1.0, 1.0);
      color = vec4<f32>(
        rampColor(true, 0.5 + 0.5 * signedPosition),
        mix(0.3, 1.0, abs(signedPosition))
      );
    } else {
      color = vec4<f32>(rampColor(false, value / clampValue), 1.0);
    }
  }
  values[valuesOffset + index] = value;
  colors[colorsOffset + index] = pack4x8unorm(color);`
      });
    }

    function buildProduct(): ProductBuild {
      const graph = new GPUCommandGraph<undefined>(device, {id: `landforms-${product}`});
      const elevation = makeElevation(graph);
      const owned: {destroy: () => void}[] = [];
      const makeBuffer = (name: string, byteLength: number): Buffer => {
        const buffer = resources.createBuffer(`${product}-${name}`, byteLength);
        owned.push({destroy: () => resources.release(buffer)});
        return buffer;
      };
      const makeSettings = (
        length: number
      ): ReturnType<typeof resources.createParameterBuffer<'float32'>> => {
        const settings = resources.createParameterBuffer(`${product}-settings`, 'float32', length);
        owned.push({destroy: () => resources.release(settings)});
        return settings;
      };
      const floats = (name: string, count = pixelCount) =>
        importGraphBuffer(graph, name, makeBuffer(name, count * 4), 'float32', count);
      const words = (name: string) =>
        importGraphBuffer(graph, name, makeBuffer(name, pixelCount * 4), 'uint32', pixelCount);
      const input = (name: string, view: ReturnType<typeof floats>): KernelBinding => ({
        name,
        view,
        type: 'f32',
        access: 'read'
      });
      let writeSettings: () => void = () => {};

      if (product === 'curvature') {
        const settings = makeSettings(GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH);
        const settingsView = settings.importToGraph(graph);
        const curvature = floats('curvature');
        const isRing = curvatureChoice === 'ring-multi-radius';
        graph.add(
          new GPUTerrainCurvature({
            id: 'curvature',
            width,
            height,
            elevation,
            settings: settingsView,
            method: curvatureMethod,
            ...(isRing
              ? {ringCurvature: curvature, ringRadii: [ringRadius, ringRadius * 4]}
              : {curvatures: {[curvatureChoice as GPUTerrainCurvatureKind]: curvature}}),
            cellSizeMode: 'uniform',
            rowDirection: 'south'
          })
        );
        addDisplayKernel(graph, {
          inputs: [input('curvature', curvature)],
          fetch: 'value = curvature[curvatureOffset + index];'
        });
        writeSettings = () =>
          settings.write(getGPUTerrainCurvatureParameterValues({cellSize, zFactor: 1}));
      } else if (product === 'geomorphons') {
        const settings = makeSettings(GPU_GEOMORPHONS_PARAMETER_LENGTH);
        const forms = words('forms');
        graph.add(
          new GPUGeomorphons({
            id: 'geomorphons',
            width,
            height,
            elevation,
            settings: settings.importToGraph(graph),
            searchRadius: geomorphonRadius,
            comparison: geomorphonComparison,
            forms,
            cellSizeMode: 'uniform',
            rowDirection: 'south'
          })
        );
        addDisplayKernel(graph, {
          classes: GEOMORPHON_CLASSES,
          inputs: [{name: 'forms', view: forms, type: 'u32', access: 'read'}],
          fetch: 'value = f32(forms[formsOffset + index]);'
        });
        writeSettings = () =>
          settings.write(
            getGPUGeomorphonsParameterValues({cellSize, flatThresholdDegrees: flatAngle})
          );
      } else if (product === 'ruggedness') {
        const tpi = floats('tpi');
        const tri = floats('tri');
        const roughness = floats('roughness');
        graph.add(
          new GPUTerrainRuggedness({
            id: 'ruggedness',
            width,
            height,
            elevation,
            topographicPositionIndex: tpi,
            terrainRuggednessIndex: tri,
            roughness,
            terrainRuggednessAlgorithm: ruggednessAlgorithm
          })
        );
        addDisplayKernel(graph, {
          inputs: [input('tpi', tpi), input('tri', tri), input('roughness', roughness)],
          fetch: /* wgsl */ `
  let quantity = u32(display[displayOffset + 2u]);
  if (quantity == 0u) {
    value = tpi[tpiOffset + index];
  } else if (quantity == 1u) {
    value = tri[triOffset + index];
  } else {
    value = roughness[roughnessOffset + index];
  }`
        });
      } else if (product === 'vrm') {
        const settings = makeSettings(GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH);
        const vrm = floats('vrm');
        graph.add(
          new GPUTerrainVectorRuggedness({
            id: 'vector-ruggedness',
            width,
            height,
            elevation,
            settings: settings.importToGraph(graph),
            radius: vectorRadius,
            vectorRuggedness: vrm,
            cellSizeMode: 'uniform',
            rowDirection: 'south'
          })
        );
        addDisplayKernel(graph, {
          inputs: [input('vrm', vrm)],
          fetch: 'value = vrm[vrmOffset + index];'
        });
        writeSettings = () =>
          settings.write(getGPUTerrainVectorRuggednessParameterValues({cellSize}));
      } else if (product === 'multiscale') {
        const planes = floats('dev-planes', MULTISCALE_RADII.length * pixelCount);
        const maximumDeviation = floats('devmax');
        const maximumRadius = words('devmax-radius');
        graph.add(
          new GPUTerrainTopographicPosition({
            id: 'topographic-position',
            width,
            height,
            elevation,
            scales: MULTISCALE_RADII.map(radius => ({radius})),
            deviationFromMean: planes,
            maximumDeviation,
            maximumDeviationRadius: maximumRadius
          })
        );
        addDisplayKernel(graph, {
          inputs: [
            input('planes', planes),
            input('devmax', maximumDeviation),
            {name: 'devmaxRadius', view: maximumRadius, type: 'u32', access: 'read'}
          ],
          fetch: /* wgsl */ `
  let quantity = u32(display[displayOffset + 2u]);
  if (quantity == 0u) {
    value = planes[planesOffset + u32(display[displayOffset + 1u]) * PIXEL_COUNT + index];
  } else if (quantity == 1u) {
    value = devmax[devmaxOffset + index];
  } else {
    let radius = devmaxRadius[devmaxRadiusOffset + index];
    if (radius == 0u || isBad(devmax[devmaxOffset + index])) {
      var notANumber = 0x7fc00000u;
      value = bitcast<f32>(notANumber);
    } else {
      value = f32(radius);
    }
  }`
        });
      } else {
        const settings = makeSettings(GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH);
        const landforms = words('landforms');
        graph.add(
          new GPUTerrainWeissLandforms({
            id: 'weiss',
            width,
            height,
            elevation,
            settings: settings.importToGraph(graph),
            smallScale: {radius: smallRadius},
            largeScale: {radius: largeRadius},
            standardization,
            landforms,
            cellSizeMode: 'uniform'
          })
        );
        addDisplayKernel(graph, {
          classes: WEISS_CLASSES,
          inputs: [{name: 'landforms', view: landforms, type: 'u32', access: 'read'}],
          fetch: 'value = f32(landforms[landformsOffset + index]);'
        });
        writeSettings = () =>
          settings.write(
            getGPUTerrainWeissLandformsParameterValues({
              cellSize,
              standardThreshold,
              slopeThresholdDegrees: slopeThreshold
            })
          );
      }

      const compiled = resources.track(graph.compile());
      writeSettings();
      return {
        compiled,
        writeSettings,
        release: () => {
          resources.release(compiled);
          for (let index = owned.length - 1; index >= 0; index--) owned[index].destroy();
        }
      };
    }

    // --- Rebuild and calibration ---------------------------------------------------------------
    function resetClamp(): void {
      const fixed = getFixedNominalClamp();
      clampExponent = 0;
      clampHandle.setValue(0);
      if (fixed !== null) nominalClamp = fixed;
      writeDisplay();
      updateRangeReadout();
      if (fixed === null) calibrationRequested = true;
    }

    function rebuildProduct(): void {
      if (destroyed) return;
      generation++;
      try {
        const previous = build;
        build = buildProduct();
        previous?.release();
        hoverResolvedIndex = -1;
        resetClamp();
        context.setStatus('');
        context.updateLayers();
      } catch (error) {
        context.setStatus(`Error: ${(error as Error).message}`);
      }
    }

    function scheduleRebuild(): void {
      clearTimeout(rebuildTimer);
      rebuildTimer = setTimeout(rebuildProduct, REBUILD_DELAY_MILLISECONDS);
    }

    function updateRangeReadout(): void {
      const clamp = getClamp();
      const mode = getColorMode();
      if (mode === COLOR_MODE.categorical) rangeReadout.setValue('classes');
      else if (mode === COLOR_MODE.diverging) {
        rangeReadout.setValue(`${formatNumber(-clamp)} to ${formatNumber(clamp)}`);
      } else rangeReadout.setValue(`0 to ${formatNumber(clamp)}`);
    }

    async function calibrate(
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ): Promise<void> {
      const ticket = calibrationRing.tryAcquire();
      if (!ticket) return;
      const requestGeneration = generation;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: valuesBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 0,
        size: pixelCount * 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: pixelCount * 4});
      calibrationPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed || requestGeneration !== generation) return;
        const values = new Float32Array(
          bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + pixelCount * 4)
        );
        const magnitudes = new Float32Array(pixelCount);
        let count = 0;
        const diverging = getColorMode() === COLOR_MODE.diverging;
        for (const value of values) {
          if (Number.isFinite(value)) magnitudes[count++] = diverging ? Math.abs(value) : value;
        }
        if (count === 0) return;
        const sorted = magnitudes.subarray(0, count).sort();
        const percentile = sorted[Math.min(count - 1, Math.floor(count * CALIBRATION_PERCENTILE))];
        if (percentile > 0) {
          nominalClamp = Number(percentile.toPrecision(2));
          writeDisplay();
          clampHandle.setValue(clampExponent);
          updateRangeReadout();
        }
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        calibrationPending = false;
      }
    }

    // --- Hover ---------------------------------------------------------------------------------
    function readHover(commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]) {
      const ticket = hoverRing.tryAcquire();
      if (!ticket) return;
      const index = hoverIndex;
      const requestGeneration = generation;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: valuesBuffer,
        sourceOffset: index * 4,
        destinationBuffer: ticket.buffer,
        destinationOffset: 0,
        size: 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: 4});
      hoverPending = true;
      void ticket
        .read()
        .then(bytes => {
          if (destroyed) return;
          hoverValue = new Float32Array(
            bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + 4)
          )[0];
          hoverResolvedIndex = index;
          hoverResolvedGeneration = requestGeneration;
        })
        .catch(() => {})
        .finally(() => {
          hoverPending = false;
        });
    }

    function describeValue(value: number): string {
      if (product === 'geomorphons' || product === 'weiss') {
        const classes = product === 'geomorphons' ? GEOMORPHON_CLASSES : WEISS_CLASSES;
        return classes[Math.round(value) - 1]?.label ?? 'no data';
      }
      if (!Number.isFinite(value)) return 'no data';
      if (product === 'curvature') {
        const unit =
          curvatureChoice === 'gaussian'
            ? ' 1/m²'
            : curvatureChoice.startsWith('ring-')
              ? ''
              : ' 1/m';
        return `${curvatureChoice} curvature ${value.toExponential(3)}${unit}`;
      }
      if (product === 'ruggedness') {
        return `${ruggednessQuantity.toUpperCase()} ${value.toFixed(2)} m`;
      }
      if (product === 'vrm') return `VRM ${value.toFixed(4)}`;
      if (multiscaleQuantity === 'devmax-scale') {
        const radius = Math.round(value);
        return `DEVmax scale ${radius} cells (${(radius * cellSize[0]).toFixed(0)} m)`;
      }
      if (multiscaleQuantity === 'devmax') return `DEVmax ${value.toFixed(2)} σ`;
      const radius = MULTISCALE_RADII[multiscaleIndex];
      return `DEV ${value.toFixed(2)} σ at ${radius} cells (${(radius * cellSize[0]).toFixed(0)} m)`;
    }

    // --- Controls ------------------------------------------------------------------------------
    const {controls} = context;
    const productControl = controls.addSelect<Product>({
      label: 'Product',
      options: [
        {value: 'curvature', label: 'Curvature'},
        {value: 'geomorphons', label: 'Geomorphons'},
        {value: 'ruggedness', label: 'TPI / TRI / roughness'},
        {value: 'vrm', label: 'Vector ruggedness (VRM)'},
        {value: 'multiscale', label: 'Multiscale DEV'},
        {value: 'weiss', label: 'Weiss landforms'}
      ],
      value: product,
      onChange: value => {
        product = value;
        syncDisabled();
        clearTimeout(rebuildTimer);
        rebuildProduct();
      }
    });
    void productControl;
    const disabledGroups: [Product, {setDisabled: (disabled: boolean) => void}][] = [];
    const group = <T extends {setDisabled: (disabled: boolean) => void}>(
      groupProduct: Product,
      control: T
    ): T => {
      disabledGroups.push([groupProduct, control]);
      return control;
    };
    function syncDisabled(): void {
      for (const [groupProduct, control] of disabledGroups) {
        control.setDisabled(groupProduct !== product);
      }
      ringControl.setDisabled(product !== 'curvature' || curvatureChoice !== 'ring-multi-radius');
      scaleControl.setDisabled(product !== 'multiscale' || multiscaleQuantity !== 'dev');
    }

    group(
      'curvature',
      controls.addSelect<CurvatureChoice>({
        label: 'Curvature kind',
        options: CURVATURE_OPTIONS,
        value: curvatureChoice,
        onChange: value => {
          curvatureChoice = value;
          syncDisabled();
          rebuildProduct();
        }
      })
    );
    group(
      'curvature',
      controls.addSelect<GPUTerrainCurvatureMethod>({
        label: 'Partials method',
        options: [
          {value: 'evans-young', label: 'Evans-Young (3x3)'},
          {value: 'zevenbergen-thorne', label: 'Zevenbergen-Thorne (3x3)'},
          {value: 'florinsky', label: 'Florinsky (5x5)'}
        ],
        value: curvatureMethod,
        onChange: value => {
          curvatureMethod = value;
          rebuildProduct();
        }
      })
    );
    const ringControl = controls.addSlider({
      label: 'Ring radius (rebuild)',
      min: 1,
      max: 8,
      step: 1,
      value: ringRadius,
      format: value => `${value} and ${value * 4} cells`,
      onChange: value => {
        ringRadius = value;
        scheduleRebuild();
      }
    });
    group(
      'geomorphons',
      controls.addSlider({
        label: 'Search radius (rebuild)',
        min: 3,
        max: 80,
        step: 1,
        value: geomorphonRadius,
        format: value => `${value} cells · ${(value * cellSize[1]).toFixed(0)} m`,
        onChange: value => {
          geomorphonRadius = value;
          scheduleRebuild();
        }
      })
    );
    group(
      'geomorphons',
      controls.addSlider({
        label: 'Flatness angle (per-frame)',
        min: 0.1,
        max: 12,
        step: 0.1,
        value: flatAngle,
        format: value => `${value.toFixed(1)}°`,
        onChange: value => {
          flatAngle = value;
          build?.writeSettings();
        }
      })
    );
    group(
      'geomorphons',
      controls.addSelect<GPUGeomorphonComparison>({
        label: 'Comparison',
        options: [
          {value: 'anglev1', label: 'anglev1 (GRASS)'},
          {value: 'anglev2', label: 'anglev2'},
          {value: 'anglev2-distance', label: 'anglev2, far tie'}
        ],
        value: geomorphonComparison,
        onChange: value => {
          geomorphonComparison = value;
          rebuildProduct();
        }
      })
    );
    group(
      'ruggedness',
      controls.addSelect<RuggednessQuantity>({
        label: 'Index (per-frame)',
        options: [
          {value: 'tpi', label: 'TPI (centre minus 8 neighbours)'},
          {value: 'tri', label: 'TRI'},
          {value: 'roughness', label: 'Roughness (max minus min)'}
        ],
        value: ruggednessQuantity,
        onChange: value => {
          ruggednessQuantity = value;
          resetClamp();
        }
      })
    );
    group(
      'ruggedness',
      controls.addSelect<GPUTerrainRuggednessAlgorithm>({
        label: 'TRI algorithm',
        options: [
          {value: 'riley', label: 'Riley (root of squares)'},
          {value: 'wilson', label: 'Wilson (mean absolute)'}
        ],
        value: ruggednessAlgorithm,
        onChange: value => {
          ruggednessAlgorithm = value;
          rebuildProduct();
        }
      })
    );
    group(
      'vrm',
      controls.addSlider({
        label: 'VRM window radius (rebuild)',
        min: 1,
        max: 8,
        step: 1,
        value: vectorRadius,
        format: value => `${value} cells (${2 * value + 1} square)`,
        onChange: value => {
          vectorRadius = value;
          scheduleRebuild();
        }
      })
    );
    group(
      'multiscale',
      controls.addSelect<MultiscaleQuantity>({
        label: 'Display (per-frame)',
        options: [
          {value: 'dev', label: 'DEV at one scale'},
          {value: 'devmax', label: 'DEVmax (largest |DEV|)'},
          {value: 'devmax-scale', label: 'Scale of DEVmax'}
        ],
        value: multiscaleQuantity,
        onChange: value => {
          multiscaleQuantity = value;
          syncDisabled();
          resetClamp();
        }
      })
    );
    const scaleControl = controls.addSlider({
      label: 'Scale (per-frame)',
      min: 0,
      max: MULTISCALE_RADII.length - 1,
      step: 1,
      value: multiscaleIndex,
      format: value =>
        `${MULTISCALE_RADII[value]} cells · ${(MULTISCALE_RADII[value] * cellSize[0]).toFixed(0)} m`,
      onChange: value => {
        multiscaleIndex = value;
        writeDisplay();
      }
    });
    group(
      'weiss',
      controls.addSlider({
        label: 'Small scale (rebuild)',
        min: 1,
        max: 12,
        step: 1,
        value: smallRadius,
        format: value => `${value} cells`,
        onChange: value => {
          smallRadius = value;
          scheduleRebuild();
        }
      })
    );
    group(
      'weiss',
      controls.addSlider({
        label: 'Large scale (rebuild)',
        min: 8,
        max: 64,
        step: 1,
        value: largeRadius,
        format: value => `${value} cells`,
        onChange: value => {
          largeRadius = value;
          scheduleRebuild();
        }
      })
    );
    group(
      'weiss',
      controls.addSelect<GPUTerrainWeissStandardization>({
        label: 'Standardization',
        options: [
          {value: 'global', label: 'Global z-score (Weiss)'},
          {value: 'local', label: 'Local DEV'}
        ],
        value: standardization,
        onChange: value => {
          standardization = value;
          rebuildProduct();
        }
      })
    );
    group(
      'weiss',
      controls.addSlider({
        label: 'Position threshold (per-frame)',
        min: 0.25,
        max: 2,
        step: 0.05,
        value: standardThreshold,
        format: value => `${value.toFixed(2)} σ`,
        onChange: value => {
          standardThreshold = value;
          build?.writeSettings();
        }
      })
    );
    group(
      'weiss',
      controls.addSlider({
        label: 'Plain slope limit (per-frame)',
        min: 1,
        max: 20,
        step: 0.5,
        value: slopeThreshold,
        format: value => `${value.toFixed(1)}°`,
        onChange: value => {
          slopeThreshold = value;
          build?.writeSettings();
        }
      })
    );
    const clampHandle = controls.addSlider({
      label: 'Symmetric color clamp (per-frame)',
      min: -2,
      max: 2,
      step: 0.05,
      value: clampExponent,
      format: value => formatNumber(nominalClamp * 10 ** value),
      onChange: value => {
        clampExponent = value;
        writeDisplay();
        updateRangeReadout();
      }
    });
    controls.addSlider({
      label: 'Layer opacity',
      min: 0.2,
      max: 1,
      step: 0.05,
      value: opacity,
      format: value => `${Math.round(value * 100)}%`,
      onChange: value => {
        opacity = value;
        context.updateLayers();
      }
    });
    controls.addLegend({
      title: 'Signed measures (curvature, TPI, DEV): clamp to clamp',
      gradient: {
        colors: DIVERGING_RAMP,
        minimumLabel: 'concave / low',
        maximumLabel: 'convex / high'
      }
    });
    controls.addLegend({
      title: 'Unsigned measures (TRI, roughness, VRM, scale): 0 to clamp',
      gradient: {colors: SEQUENTIAL_RAMP, minimumLabel: '0', maximumLabel: 'clamp'}
    });
    controls.addLegend({
      title: 'Geomorphons',
      entries: GEOMORPHON_CLASSES.map(entry => ({color: [...entry.color, 255], label: entry.label}))
    });
    controls.addLegend({
      title: 'Weiss landforms',
      entries: WEISS_CLASSES.map(entry => ({color: [...entry.color, 255], label: entry.label}))
    });
    controls.addNote(
      'Signed and unsigned ranges start at the 98th percentile of the current measure, read back ' +
        'once per rebuild. Hover the map for the value or class under the pointer.'
    );
    const rangeReadout = controls.addReadout('Color range', '');
    controls.addReadout('Raster', `${width} × ${height} cells`);
    controls.addReadout('Cell size', `${cellSize[0].toFixed(1)} × ${cellSize[1].toFixed(1)} m`);
    const timingReadout = controls.addReadout('Graph cost', 'measuring...');
    controls.addReadout('Data', terrain.attribution);

    const measureActive = async () => {
      const active = build;
      if (!active) return;
      try {
        const timing = await measureCompiledGraph(device, active.compiled, {
          parameters: undefined,
          completionBuffer: colorsBuffer,
          signal: context.signal
        });
        if (!destroyed && active === build) {
          timingReadout.setValue(
            `${active.compiled.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(timing)}`
          );
        }
      } catch (error) {
        if (!destroyed) timingReadout.setValue(`failed: ${(error as Error).message}`);
      }
    };
    controls.addButton({label: 'Measure GPU cost', onClick: () => void measureActive()});

    syncDisabled();
    build = buildProduct();
    resetClamp();

    // --- Instance ------------------------------------------------------------------------------
    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => (build ? [build.compiled] : []),
      encode(commandEncoder) {
        if (!build) return;
        build.compiled.encode(commandEncoder, {parameters: undefined});
        if (calibrationRequested && !calibrationPending) {
          calibrationRequested = false;
          void calibrate(commandEncoder);
        }
        if (!hoverPending && hoverIndex >= 0 && hoverIndex !== hoverResolvedIndex) {
          readHover(commandEncoder);
        }
      },
      getLayers(): Layer[] {
        return [
          new ReliefRasterLayer({
            id: 'landforms-raster',
            coordinateOrigin: origin,
            colors: colorsBuffer,
            gridSize: [width, height],
            bounds,
            opacity
          })
        ];
      },
      getTooltip(event: SpatialAnalysisPointerEvent) {
        if (!event.coordinate) {
          hoverIndex = -1;
          return null;
        }
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const column = Math.floor((x - bounds[0]) / cellSize[0]);
        const row = Math.floor((bounds[3] - y) / cellSize[1]);
        if (column < 0 || column >= width || row < 0 || row >= height) {
          hoverIndex = -1;
          return null;
        }
        hoverIndex = row * width + column;
        const elevation = terrain.elevation[hoverIndex];
        if (elevation <= 0.5) return 'Sea / no data';
        if (hoverResolvedIndex !== hoverIndex || hoverResolvedGeneration !== generation) {
          return `Elevation ${elevation.toFixed(0)} m`;
        }
        return `${describeValue(hoverValue)} · elevation ${elevation.toFixed(0)} m`;
      },
      destroy() {
        destroyed = true;
        clearTimeout(rebuildTimer);
        resources.destroy();
      }
    };
    return instance;
  }
};

/** WGSL `vec3<f32>` literal of an 8-bit color. */
function formatColor(color: Color): string {
  return `vec3<f32>(${color.map(channel => (channel / 255).toFixed(4)).join(', ')})`;
}

/** Compact number with enough digits for curvature (1e-3) up to elevations (1e3). */
function formatNumber(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return '0';
  return magnitude < 0.01 || magnitude >= 10000 ? value.toExponential(2) : value.toPrecision(3);
}
