// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  assertCompatibleGPUVerticalDatums,
  type GPUTerrainHeightReference
} from '../../analysis-core/contracts/vertical-datum';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {type GPURasterBand, type GPURasterMetadata} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getGPURasterExtremaPyramidLayout,
  type GPURasterExtremaPyramidLayout,
  type GPURasterExtremaPyramidOutput
} from '../../gpu-raster/raster-pyramid/index';
import {
  createRasterExtremaPyramidNodes,
  validateRasterExtremaPyramidOutput
} from '../../gpu-raster/raster-pyramid/gpu-raster-extrema-pyramid';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateParameterTarget,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters
} from '../../gpu-raster/raster-band-utils';
import {
  getRasterGridGeoreference,
  type GPURasterCellSizeMode,
  validateRasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {
  getTerrainSightLineGroundCellWGSL,
  getTerrainSightLineWGSL,
  type GPUTerrainSightLineTraversal
} from './terrain-sight-line';

/**
 * Number of float32 values read from the `parameters` of {@link GPUTerrainLineOfSight} and
 * `GPUTerrainCumulativeViewshed`.
 */
export const GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH = 12;

/**
 * CPU-side description packed by {@link getGPUTerrainLineOfSightParameterValues}; shared by
 * {@link GPUTerrainLineOfSight} and `GPUTerrainCumulativeViewshed`, which read the same layout.
 *
 * Cell-size model: `'uniform'` (default) treats grid world units as ground meters on a planar
 * grid. `'geographic'` and `'web-mercator'` use a WGS84 ground cell evaluated at each ray's middle
 * row, held constant along that ray. Along-ray scale change is ignored, so range boundaries and
 * curvature drops are approximate on long rays, especially at high latitudes.
 * Rays join pixel positions, with row order supplied by the affine; there is no azimuth input,
 * so no `getPixelDirection` conversion or `rowDirection` prop is needed.
 * Curvature convention: drop at distance `d` is `c * d^2`, with
 * `c = getGPUTerrainCurvatureCoefficient(k)`; mt-image uses `k = 0.13`, GDAL `-cc 0.85714` is
 * `k = 1 / 7`.
 */
export type GPUTerrainLineOfSightSettings = {
  /**
   * Eye height above terrain, or absolute eye elevation with `heightReference: 'absolute'`.
   * In elevation-band units. Defaults to 1.7; set explicitly in absolute mode.
   */
  observerHeight?: number;
  /** Height added to the target elevation. Defaults to 0. */
  targetHeight?: number;
  /** Ground radius in metres; `<= 0` means unlimited. Defaults to 0. */
  maximumDistance?: number;
  /** Drop coefficient `c` in 1/metres. Defaults to 0. */
  curvatureCoefficient?: number;
  /** Constant tolerance band half-width in metres. Defaults to 0. */
  toleranceMeters?: number;
  /** Additional band half-width per kilometre of distance, in metres. Defaults to 0. */
  tolerancePerKilometer?: number;
  /** Metres at the target end that are not tested. Defaults to 0. */
  targetIgnoreDistance?: number;
  /** Fraction of the distance at the target end that is not tested. Defaults to 0. */
  targetIgnoreFraction?: number;
};

/**
 * Packs line-of-sight settings into the 12 float32 values
 * `[observerHeight, targetHeight, maximumDistance, cellSizeX, cellSizeY, curvatureCoefficient,
 * toleranceMeters, tolerancePerKilometer, targetIgnoreDistance, targetIgnoreFraction, northEdge, southEdge]`.
 * `observerHeight` is an absolute eye elevation when the contributor uses `heightReference: 'absolute'`;
 * otherwise it is added to bilinear ground. `targetHeight` is always added to target ground.
 * Cell sizes and row edges always come from `grid`, in world units (meters or degrees), regardless
 * of cellSizeMode. Edges are the top of row 0 and bottom of the last row, including point-pixel
 * half-row offsets. The kernel converts cell sizes to ground meters according to cellSizeMode.
 *
 * @throws If a grid cell size is not finite and positive, a tolerance value is negative or not
 *   finite, or `target` is too short.
 */
export function getGPUTerrainLineOfSightParameterValues(
  settings: GPUTerrainLineOfSightSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUTerrainLineOfSightParameterValues',
    target,
    GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH
  );
  const {cellSize, northEdge, southEdge} = getRasterGridGeoreference(grid);
  if (!cellSize.every(size => Number.isFinite(size) && size > 0)) {
    throw new Error(
      'getGPUTerrainLineOfSightParameterValues grid cell size must be finite and positive'
    );
  }
  const tolerance = [
    settings.toleranceMeters ?? 0,
    settings.tolerancePerKilometer ?? 0,
    settings.targetIgnoreDistance ?? 0,
    settings.targetIgnoreFraction ?? 0
  ];
  if (!tolerance.every(value => Number.isFinite(value) && value >= 0)) {
    throw new Error(
      'getGPUTerrainLineOfSightParameterValues tolerance values must be finite and non-negative'
    );
  }
  target.set([
    settings.observerHeight ?? 1.7,
    settings.targetHeight ?? 0,
    settings.maximumDistance ?? 0,
    cellSize[0],
    cellSize[1],
    settings.curvatureCoefficient ?? 0,
    ...tolerance,
    northEdge,
    southEdge
  ]);
  return target;
}

/** Creates the optional pyramid nodes and binding shared by the sight-line contributors. @internal */
export function getSightLinePyramid<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  traversal: GPUTerrainSightLineTraversal | undefined,
  layout: GPURasterExtremaPyramidLayout | undefined,
  values: GraphDataView<'float32'>,
  validity: GraphDataView<'uint32'>,
  shared?: GPURasterExtremaPyramidOutput
): {nodes: GPUCommandNode<Parameters>[]; binding?: WGSLKernelBinding} {
  if (traversal !== 'pyramid' || !layout) {
    return {nodes: []};
  }
  if (shared) {
    // The caller built the pyramid (once, elsewhere): read it, build nothing.
    return {
      nodes: [],
      binding: {name: 'pyramid', view: shared.combined, type: 'f32', access: 'read'}
    };
  }
  const combined = createTransientView(graph, `${id}-pyramid`, 'float32', 2 * layout.length);
  return {
    nodes: createRasterExtremaPyramidNodes(graph, {
      id: `${id}-pyramid`,
      layout,
      values,
      validity,
      combined
    }),
    binding: {name: 'pyramid', view: combined, type: 'f32', access: 'read'}
  };
}

/** Validates the compile-time traversal and returns the pyramid layout it needs. @internal */
export function getSightLinePyramidLayout(
  id: string,
  traversal: GPUTerrainSightLineTraversal | undefined,
  grid: GPURasterMetadata,
  shared?: GPURasterExtremaPyramidOutput
): GPURasterExtremaPyramidLayout | undefined {
  if (traversal !== undefined && traversal !== 'march' && traversal !== 'pyramid') {
    throw new Error(`${id} traversal must be 'march' or 'pyramid'`);
  }
  if (shared) {
    if (traversal !== 'pyramid') {
      throw new Error(`${id} pyramid requires traversal 'pyramid'`);
    }
    return validateRasterExtremaPyramidOutput(id, shared, grid);
  }
  return traversal === 'pyramid'
    ? getGPURasterExtremaPyramidLayout(grid, {firstBlockSize: 4, footprint: 'bilinear'})
    : undefined;
}

/** Caller-owned outputs of {@link GPUTerrainLineOfSight}, one row per pair. */
export type GPUTerrainLineOfSightOutput = {
  /** `GPU_TERRAIN_VISIBILITY` code per pair. */
  visibility: GraphDataView<'uint32'>;
  /**
   * Optional metres per pair by which the target could sink and stay visible (negative when
   * hidden): `(targetSlope - maxSampleSlope) * distance`. `3.4028234663852886e38` when no sample was
   * tested; NaN for `noData` and `outOfRange` pairs. Disables the early exit of hidden pairs.
   */
  clearance?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUTerrainLineOfSight}.
 *
 * Cell-size model: projected metres on a planar grid, see {@link GPUTerrainLineOfSightSettings}
 * (curvature conventions: mt-image `k = 0.13`, GDAL `-cc 0.85714` is `k = 1 / 7`).
 */
export type GPUTerrainLineOfSightProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-line-of-sight'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /**
   * Cell-size interpretation. Defaults to `'uniform'`: grid world units are ground meters
   * (projected planar grid). `'geographic'` uses a degree grid; `'web-mercator'` uses EPSG:3857.
   * In both non-uniform modes, maximumDistance, curvature drop `c * d^2`, tolerance bands,
   * targetIgnoreDistance and clearance are in WGS84 ground meters, using the ray's middle row.
   */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Elevation band, buffer or texture. */
  elevation: GPURasterBand;
  /**
   * Compile-time eye reference. `'ground'` (default) adds the observer height to bilinear ground.
   * `'absolute'` uses the observer height itself in the elevation band's units and vertical datum,
   * without sampling ground under the observer; no-data ground is allowed, but the eye must be
   * inside the grid. Set the height explicitly in absolute mode: the 1.7 default is meaningless.
   * An eye below ground hides most targets. Targets always remain ground-relative.
   */
  heightReference?: GPUTerrainHeightReference;
  /** Vertical datum of absolute observer heights; requires `heightReference: 'absolute'`. */
  observerVerticalDatum?: string;
  /**
   * One row `[observerColumn, observerRow, targetColumn, targetRow]` per pair, in pixel-center
   * index space; fractional values allowed.
   */
  pairs: GraphDataView<'float32x4'>;
  /**
   * Optional rows `[observerHeight, targetHeight]` per pair, overriding the settings.
   * Observer heights are absolute eye elevations with `heightReference: 'absolute'`;
   * target heights always remain ground-relative.
   */
  pairHeights?: GraphDataView<'float32x2'>;
  /** Per-frame parameters, see {@link getGPUTerrainLineOfSightParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** `'march'` (default) or `'pyramid'`, which gives bit-identical results. */
  traversal?: GPUTerrainSightLineTraversal;
  /**
   * Optional prebuilt min-max pyramid to read instead of building one inside this contributor.
   * Requires `traversal: 'pyramid'`. Pass `GPURasterExtremaPyramid.output` of a pyramid created with
   * `combined` over the same elevation, `footprint: 'bilinear'` and the same grid size
   * (otherwise the constructor throws), added to the same graph before this contributor or run in
   * an earlier graph. One pyramid then serves every consumer and is rebuilt only when the
   * elevation changes. Results are bit-identical to a self-built pyramid and to `'march'`.
   */
  pyramid?: GPURasterExtremaPyramidOutput;
  /** Caller-owned outputs. */
  output: GPUTerrainLineOfSightOutput;
};

/**
 * Tests explicit observer-target pairs against an elevation tile.
 *
 * Observer and target must lie inside the grid with valid bilinear samples, otherwise the pair is
 * `noData`; the target elevation is the bilinear sample at the target plus the target height.
 * This differs from {@link GPUTerrainViewshed} and `GPUTerrainCumulativeViewshed`, which use the
 * raw pixel value at integer pixel targets: the two agree at pixel centres, and a fractional
 * line-of-sight target sees the interpolated surface rather than its containing pixel.
 * Then `distance > maximumDistance > 0` is `outOfRange`, a zero distance is `visible`, and otherwise
 * the tolerance-aware sight-line model of {@link GPUTerrainViewshed} applies (codes
 * `hidden`, `visible`, `marginal`). One invocation handles one pair.
 *
 * Cell-size model: see {@link GPUTerrainLineOfSightSettings} and `cellSizeMode`.
 */
export class GPUTerrainLineOfSight implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainLineOfSightProps;
  /** Min-max pyramid layout when `traversal` is `'pyramid'`. */
  readonly pyramidLayout?: GPURasterExtremaPyramidLayout;

  constructor(props: GPUTerrainLineOfSightProps) {
    this.id = props.id ?? 'terrain-line-of-sight';
    this.props = props;
    const {id} = this;
    if (
      props.heightReference !== undefined &&
      props.heightReference !== 'ground' &&
      props.heightReference !== 'absolute'
    ) {
      throw new Error(`${id} heightReference must be 'ground' or 'absolute'`);
    }
    if (props.observerVerticalDatum !== undefined && props.heightReference !== 'absolute') {
      throw new Error(`${id} observerVerticalDatum needs heightReference absolute`);
    }
    assertCompatibleGPUVerticalDatums(
      id,
      props.observerVerticalDatum,
      props.elevation.verticalDatum
    );
    const {output} = props;
    validateRasterGrid(id, props.grid);
    validateRasterCellSizeMode(id, props.cellSizeMode);
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    validatePackedView(props.pairs, ['float32x4'], `${id} pairs`);
    const pairCount = props.pairs.length;
    if (props.pairHeights) {
      validatePackedView(props.pairHeights, ['float32x2'], `${id} pairHeights`);
      if (props.pairHeights.length !== pairCount) {
        throw new Error(`${id} pairHeights must contain one row per pair`);
      }
    }
    validatePackedView(output.visibility, ['uint32'], `${id} output.visibility`);
    if (output.visibility.length !== pairCount) {
      throw new Error(`${id} output.visibility must contain one value per pair`);
    }
    if (output.clearance) {
      validatePackedView(output.clearance, ['float32'], `${id} output.clearance`);
      if (output.clearance.length !== pairCount) {
        throw new Error(`${id} output.clearance must contain one value per pair`);
      }
    }
    validateRasterParameters(id, props.parameters, GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH);
    this.pyramidLayout = getSightLinePyramidLayout(id, props.traversal, props.grid, props.pyramid);
    validateRasterBuffersDistinct(
      id,
      [output.visibility, output.clearance],
      [
        ...getRasterBandViews(props.elevation),
        props.parameters,
        props.pairs,
        ...(props.pairHeights ? [props.pairHeights] : [])
      ]
    );
  }

  /** Returns canonical elevation, optional pyramid, and the line-of-sight nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, pyramidLayout} = this;
    const {output} = props;
    const {width, height} = props.grid;
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const groundCellDeclarations =
      cellSizeMode === 'uniform' ? '' : getTerrainSightLineGroundCellWGSL(cellSizeMode);
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.pyramid?.combined,
      props.parameters,
      props.pairs,
      props.pairHeights,
      output.visibility,
      output.clearance
    ]);
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const values = source.band.storage.values as GraphDataView<'float32'>;
    const validity = source.band.validity as GraphDataView<'uint32'>;
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const pyramid = getSightLinePyramid(
      graph,
      id,
      props.traversal,
      pyramidLayout,
      values,
      validity,
      props.pyramid
    );
    nodes.push(...pyramid.nodes);
    const bindings: WGSLKernelBinding[] = [
      {name: 'elevationValues', view: values, type: 'f32', access: 'read'},
      {name: 'elevationValidity', view: validity, type: 'u32', access: 'read'},
      ...(pyramid.binding ? [pyramid.binding] : []),
      {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'pairs', view: props.pairs, type: 'f32', access: 'read'},
      ...(props.pairHeights
        ? [
            {
              name: 'pairHeights',
              view: props.pairHeights,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : []),
      {name: 'visibilityValues', view: output.visibility, type: 'u32', access: 'read_write'},
      ...(output.clearance
        ? [
            {
              name: 'clearanceValues',
              view: output.clearance,
              type: 'f32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-line-of-sight`,
        operation: 'GPUTerrainLineOfSight',
        variant: `${props.traversal === 'pyramid' ? 'pyramid' : 'march'}${cellSizeMode === 'uniform' ? '' : `-${cellSizeMode}`}${props.heightReference === 'absolute' ? '-absolute' : ''}`,
        bindings,
        invocationCount: props.pairs.length,
        declarations: `${getTerrainSightLineWGSL({
          width,
          height,
          traversal: props.traversal ?? 'march',
          layout: pyramidLayout,
          clearance: Boolean(output.clearance)
        })}${groundCellDeclarations}
fn isInsideGrid(position: vec2<f32>) -> bool {
  return isFiniteValue(position.x) && isFiniteValue(position.y) &&
    position.x >= 0.0 && position.y >= 0.0 &&
    position.x <= f32(WIDTH - 1u) && position.y <= f32(HEIGHT - 1u);
}`,
        body: `let pairBase = pairsOffset + 4u * index;
  let observer = vec2<f32>(pairs[pairBase], pairs[pairBase + 1u]);
  let targetPosition = vec2<f32>(pairs[pairBase + 2u], pairs[pairBase + 3u]);
  ${
    props.pairHeights
      ? `let observerHeight = pairHeights[pairHeightsOffset + 2u * index];
  let targetHeight = pairHeights[pairHeightsOffset + 2u * index + 1u];`
      : `let observerHeight = settings[settingsOffset];
  let targetHeight = settings[settingsOffset + 1u];`
  }
  let maxDistance = settings[settingsOffset + 2u];
  let cellSize = ${
    cellSizeMode === 'uniform'
      ? 'vec2<f32>(settings[settingsOffset + 3u], settings[settingsOffset + 4u])'
      : 'getRayGroundCell(vec2<f32>(settings[settingsOffset + 3u], settings[settingsOffset + 4u]), settings[settingsOffset + 10u], settings[settingsOffset + 11u], 0.5 * (observer.y + targetPosition.y) + 0.5)'
  };
  let curvature = settings[settingsOffset + 5u];
  let tolerance = vec4<f32>(
    settings[settingsOffset + 6u], settings[settingsOffset + 7u],
    settings[settingsOffset + 8u], settings[settingsOffset + 9u]);
  let observerInside = isInsideGrid(observer);
  let targetInside = isInsideGrid(targetPosition);
${props.heightReference === 'absolute' ? '' : '  let observerSample = sampleElevation(select(vec2<f32>(0.0), observer, observerInside));\n'}  let targetSample = sampleElevation(select(vec2<f32>(0.0), targetPosition, targetInside));
  var code = NO_DATA;
  // A runtime-dependent bit pattern: WGSL rejects constant-evaluated NaN.
  var clearance = bitcast<f32>(0x7fc00000u | (index & 0u));
  if (observerInside && targetInside && ${props.heightReference === 'absolute' ? '' : 'observerSample.y != 0.0 && '}targetSample.y != 0.0) {
    let result = traceSightLine(
      observer, targetPosition, ${props.heightReference === 'absolute' ? 'observerHeight' : 'observerSample.x + observerHeight'}, targetSample.x + targetHeight,
      cellSize, curvature, maxDistance, tolerance);
    code = result.code;
    if (result.code != OUT_OF_RANGE) {
      clearance = result.clearance;
    }
  }
  visibilityValues[visibilityValuesOffset + index] = code;${
    output.clearance
      ? `
  clearanceValues[clearanceValuesOffset + index] = clearance;`
      : ''
  }`
      })
    );
    return nodes;
  }
}
