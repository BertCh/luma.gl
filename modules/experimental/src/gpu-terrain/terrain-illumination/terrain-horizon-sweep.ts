// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {TERRAIN_ILLUMINATION_WGSL_CONSTANTS} from './terrain-illumination-utils';
import {
  getRasterGroundCellSizeWGSL,
  type GPURasterCellSizeMode,
  type GPURasterRowDirection
} from '../../analysis-core/raster-grid';

/** Fixed-point scale of the digital line slope: `slopeFixed = round(minor / major * 65536)`. @internal */
export const TERRAIN_SWEEP_SLOPE_SCALE = 65536;

/** Largest grid extent supported by the 32-bit fixed-point line geometry. @internal */
export const TERRAIN_SWEEP_MAXIMUM_EXTENT = 32767;

/** Cell size and row-edge slots of the sweep's `settings` layout. @internal */
export const TERRAIN_SWEEP_GRID_SLOTS = {cellSizeIndex: 0, northEdgeIndex: 4} as const;

/** Workgroup size of the sweep kernels (one invocation per digital line). @internal */
const TERRAIN_SWEEP_WORKGROUP_SIZE = 64;

/**
 * Exact integer description of the digital line family for one direction.
 *
 * The lines `L_m = {(a, m + F(a))}` with `F(a) = (slopeFixed * a + 32768) >> 16` partition all
 * pixels of the grid. `a` is the column for x-major directions (`|dx| >= |dy|`, minor index = row)
 * and the row for y-major directions (minor index = column). The same integers are used on CPU
 * oracles and in WGSL, so both agree on which pixels lie on which line.
 *
 * @internal
 */
export type TerrainSweepLineGeometry = {
  /** True when columns are the major axis. */
  xMajor: boolean;
  /** Fixed-point (16 fractional bits) minor offset per major step, an integer in `[-65536, 65536]`. */
  slopeFixed: number;
  /** `+1` when the ray travels toward increasing major index, else `-1`. */
  travelSign: 1 | -1;
  /** Number of lines that contain at least one pixel. */
  lineCount: number;
  /** Minor offset `m` of the first line (lines run `m = firstLine .. firstLine + lineCount - 1`). */
  firstLine: number;
  /** `Math.fround(hypot(1, slopeFixed / 65536))`: pixel length of one major step. */
  stepPixelLength: number;
};

/**
 * Returns `F(a) = (slopeFixed * a + 32768) >> 16` (arithmetic shift, i.e. floor division), the
 * minor offset of major index `a`. Exact in float64 for grid extents up to 32767.
 *
 * @internal
 */
export function getTerrainSweepMinorOffset(slopeFixed: number, major: number): number {
  return Math.floor((slopeFixed * major + 32768) / TERRAIN_SWEEP_SLOPE_SCALE);
}

/**
 * Computes the digital line family of a pixel-space direction on a `width x height` grid.
 *
 * Direction components are used as given (callers pass float32 sector directions or float64 sun
 * directions). The slope is rounded once, in float64, to the 16-bit fixed point `slopeFixed`; the
 * geometry that follows is exact integer arithmetic.
 *
 * @throws If the direction is zero or not finite, or an extent exceeds 32767.
 * @internal
 */
export function getTerrainSweepLineGeometry(
  direction: readonly [number, number],
  width: number,
  height: number
): TerrainSweepLineGeometry {
  const [dx, dy] = direction;
  if (!Number.isFinite(dx) || !Number.isFinite(dy) || (dx === 0 && dy === 0)) {
    throw new Error('Terrain sweep direction must be finite and non-zero');
  }
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width < 1 ||
    height < 1 ||
    width > TERRAIN_SWEEP_MAXIMUM_EXTENT ||
    height > TERRAIN_SWEEP_MAXIMUM_EXTENT
  ) {
    throw new Error(
      `Terrain sweep extents must be integers in [1, ${TERRAIN_SWEEP_MAXIMUM_EXTENT}]`
    );
  }
  const xMajor = Math.abs(dx) >= Math.abs(dy);
  const major = xMajor ? dx : dy;
  const minor = xMajor ? dy : dx;
  // `+ 0` turns -0 into 0.
  const slopeFixed = Math.round((minor / major) * TERRAIN_SWEEP_SLOPE_SCALE) + 0;
  const majorExtent = xMajor ? width : height;
  const minorExtent = xMajor ? height : width;
  const lastOffset = getTerrainSweepMinorOffset(slopeFixed, majorExtent - 1);
  return {
    xMajor,
    slopeFixed,
    travelSign: major > 0 ? 1 : -1,
    // F(0) is always 0.
    firstLine: -Math.max(0, lastOffset) + 0,
    lineCount: minorExtent + Math.abs(lastOffset),
    stepPixelLength: Math.fround(Math.hypot(1, slopeFixed / TERRAIN_SWEEP_SLOPE_SCALE))
  };
}

/**
 * Mirrors a line family across the row axis (negated row component), exactly as the sweep kernel
 * does for a `'settings'` source on north-advancing rows: the slope negates, and so does the travel
 * sign of y-major lines. CPU twin for specs and oracles.
 *
 * @internal
 */
export function getTerrainSweepMirroredLineGeometry(
  geometry: TerrainSweepLineGeometry,
  width: number,
  height: number
): TerrainSweepLineGeometry {
  const slopeFixed = -geometry.slopeFixed + 0;
  const majorExtent = geometry.xMajor ? width : height;
  const minorExtent = geometry.xMajor ? height : width;
  const lastOffset = getTerrainSweepMinorOffset(slopeFixed, majorExtent - 1);
  return {
    xMajor: geometry.xMajor,
    slopeFixed,
    travelSign: geometry.xMajor ? geometry.travelSign : geometry.travelSign === 1 ? -1 : 1,
    firstLine: -Math.max(0, lastOffset) + 0,
    lineCount: minorExtent + Math.abs(lastOffset),
    stepPixelLength: geometry.stepPixelLength
  };
}

/**
 * Computes the digital line family that follows a ground-space unit direction `(east, row-ward)`
 * on cells of `groundCell = [x, y]` ground meters, exactly as the sweep kernel does it in WGSL.
 *
 * Square cells use {@link getTerrainSweepLineGeometry} on `groundDirection` unchanged (float64 slope),
 * so isotropic grids keep their exact geometry. Anisotropic cells use the pixel direction
 * `groundDirection / groundCell`, rounded to float32 at each step like the kernel, and round the
 * slope half to even like WGSL `round`.
 *
 * @throws If the direction or cell is zero or not finite, or an extent exceeds 32767.
 * @internal
 */
export function getTerrainSweepGroundLineGeometry(
  groundDirection: readonly [number, number],
  groundCell: readonly [number, number],
  width: number,
  height: number
): TerrainSweepLineGeometry {
  const geometry = getTerrainSweepLineGeometry(groundDirection, width, height);
  const [cellX, cellY] = groundCell.map(Math.fround);
  if (cellX === cellY || !(cellX > 0 && cellY > 0 && Number.isFinite(cellX + cellY))) {
    return geometry;
  }
  const pixelX = Math.fround(Math.fround(groundDirection[0]) / cellX);
  const pixelY = Math.fround(Math.fround(groundDirection[1]) / cellY);
  const xMajor = Math.abs(pixelX) >= Math.abs(pixelY);
  const major = xMajor ? pixelX : pixelY;
  const minor = xMajor ? pixelY : pixelX;
  const scaled = Math.fround(minor / major) * TERRAIN_SWEEP_SLOPE_SCALE;
  // WGSL round() rounds half to even; `+ 0` turns -0 into 0.
  const nearest = Math.round(scaled);
  const slopeFixed =
    (Math.abs(scaled % 1) === 0.5 && nearest % 2 !== 0 ? nearest - 1 : nearest) + 0;
  const majorExtent = xMajor ? width : height;
  const minorExtent = xMajor ? height : width;
  const lastOffset = getTerrainSweepMinorOffset(slopeFixed, majorExtent - 1);
  return {
    xMajor,
    slopeFixed,
    travelSign: major > 0 ? 1 : -1,
    firstLine: -Math.max(0, lastOffset) + 0,
    lineCount: minorExtent + Math.abs(lastOffset),
    stepPixelLength: Math.fround(Math.hypot(1, slopeFixed / TERRAIN_SWEEP_SLOPE_SCALE))
  };
}

/** Elements between sector records: 256 bytes, so every record binds at element offset 0. */
const SWEEP_SECTOR_RECORD_STRIDE = 64;

/** Words of one sweep sector record, see {@link createTerrainSweepSectorTable}. */
const SWEEP_SECTOR_RECORD_LENGTH = 7;

/**
 * Graph-owned sweep geometry of every horizon sector, written once per encoding.
 *
 * Record `s` (at element `s * 64`) holds `[xMajor (0/1), slopeFixed (i32 bits), travelSign (i32
 * bits), stepPixelLength (f32 bits), groundEast (f32 bits), groundRowward (f32 bits), s]`: the
 * float64 square-cell geometry of {@link getTerrainSweepLineGeometry} and the ground direction it
 * came from. A sweep kernel with the `'table'` geometry source reads its record, so every
 * sector's kernel has the same WGSL (library rule 12).
 *
 * @internal
 */
export type TerrainSweepSectorTable<Parameters> = {
  /** One-thread node that writes every record; schedule it before the first sweep. */
  writeNode: GPUCommandNode<Parameters>;
  /** Returns the record view of `sector`. */
  getRecord: (sector: number) => GraphDataView<'uint32'>;
};

/**
 * Creates the {@link TerrainSweepSectorTable} for ground-space sector directions.
 *
 * @internal
 */
export function createTerrainSweepSectorTable<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    /** Ground-space unit direction of every sector, see `getGPUTerrainHorizonDirection`. */
    directions: readonly (readonly [number, number])[];
  }
): TerrainSweepSectorTable<Parameters> {
  const sectorCount = props.directions.length;
  const words = new Uint32Array(sectorCount * SWEEP_SECTOR_RECORD_LENGTH);
  const floats = new Float32Array(1);
  const integers = new Int32Array(1);
  const getFloatBits = (value: number): number => {
    floats[0] = value;
    return new Uint32Array(floats.buffer)[0];
  };
  const getIntegerBits = (value: number): number => {
    integers[0] = value;
    return new Uint32Array(integers.buffer)[0];
  };
  for (const [sector, direction] of props.directions.entries()) {
    const geometry = getTerrainSweepLineGeometry(direction, props.width, props.height);
    words.set(
      [
        geometry.xMajor ? 1 : 0,
        getIntegerBits(geometry.slopeFixed),
        getIntegerBits(geometry.travelSign),
        getFloatBits(geometry.stepPixelLength),
        getFloatBits(direction[0]),
        getFloatBits(direction[1]),
        sector
      ],
      sector * SWEEP_SECTOR_RECORD_LENGTH
    );
  }
  const table = createTransientView(
    graph,
    `${props.id}-sweep-sector-records`,
    'uint32',
    Math.max(sectorCount, 1) * SWEEP_SECTOR_RECORD_STRIDE
  );
  const writeNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-sweep-sectors`,
    operation: 'GPUTerrainHorizon',
    variant: 'sweep-sector-table',
    bindings: [{name: 'records', view: table, type: 'u32', access: 'read_write'}],
    invocationCount: 1,
    declarations: `const SECTOR_WORDS = array<u32, ${words.length}>(${Array.from(words, word => `${word}u`).join(', ')});`,
    body: `for (var sector = 0u; sector < ${sectorCount}u; sector++) {
    for (var word = 0u; word < ${SWEEP_SECTOR_RECORD_LENGTH}u; word++) {
      records[recordsOffset + sector * ${SWEEP_SECTOR_RECORD_STRIDE}u + word] =
        SECTOR_WORDS[sector * ${SWEEP_SECTOR_RECORD_LENGTH}u + word];
    }
  }`
  });
  const getRecord = (sector: number): GraphDataView<'uint32'> =>
    graph.createDataView(table.buffer, {
      format: 'uint32',
      length: SWEEP_SECTOR_RECORD_LENGTH,
      byteOffset: table.byteOffset + sector * SWEEP_SECTOR_RECORD_STRIDE * 4
    });
  return {writeNode, getRecord};
}

/** WGSL `u32` expression of the sector index in a `'table'` sweep kernel. @internal */
export const TERRAIN_SWEEP_SECTOR_WGSL = 'sectorRecord[sectorRecordOffset + 6u]';

/** Where the sweep reads its line geometry. @internal */
export type TerrainHorizonSweepGeometrySource =
  | {
      /**
       * Geometry of one sector read from a {@link TerrainSweepSectorTable} record (binding
       * `sectorRecord`), so all sectors share one pipeline. Conversion on anisotropic reference
       * cells is that of the baked source.
       */
      kind: 'table';
      /** Record view from {@link TerrainSweepSectorTable.getRecord}. */
      record: GraphDataView<'uint32'>;
    }
  | {
      /**
       * Geometry baked from a ground-space direction (one node per sector). Square reference
       * cells use the baked geometry of `direction`; anisotropic ones convert it in the kernel to
       * the pixel direction of the reference row, see {@link getTerrainSweepGroundLineGeometry}.
       */
      kind: 'baked';
      /** Ground-space unit direction `(east, row-ward)`. */
      direction: readonly [number, number];
    }
  | {
      /**
       * Geometry read from `settings[index .. index + 3] = [xMajor (0/1), slopeFixed, travelSign,
       * stepPixelLength]`, packed for square cells and south-advancing rows from the ground
       * direction `(east, south)` stored at `settings[groundDirectionIndex .. + 1]`. The kernel
       * mirrors it for `rowDirection: 'north'` and converts it on anisotropic reference cells
       * exactly like the baked source, so the packer needs neither the row direction nor the cell
       * size mode. The line count is derived in the shader.
       */
      kind: 'settings';
      /** Settings index of `xMajor`. */
      index: number;
      /** Settings index of the ground-space unit direction `(east, south)`. */
      groundDirectionIndex: number;
      /** Row direction of the grid; the packed geometry assumes `'south'`. */
      rowDirection: GPURasterRowDirection;
    };

/**
 * Properties for {@link getTerrainHorizonSweepNode}.
 *
 * The cell-size model is that of `GPUTerrainHorizon`: projected metres (`uniform`) or
 * latitude-dependent spacing (`web-mercator`, `geographic`) read from `settings`.
 *
 * @internal
 */
export type TerrainHorizonSweepNodeProps = {
  /** Graph-wide node ID. */
  id: string;
  /** Grid width in pixels, at most 32767. */
  width: number;
  /** Grid height in pixels, at most 32767. */
  height: number;
  /** Ground-space unit direction of the sector, from `getGPUTerrainHorizonDirection`. */
  direction: readonly [number, number];
  /** Radius window in pixels (the contributor's `maximumRadius`). */
  maximumRadius: number;
  /** Cell size interpretation. */
  cellSizeMode: GPURasterCellSizeMode;
  /** 1 for horizons, -1 for the nadir pass (inverted DEM; curvature still drops). */
  zFactorSign: 1 | -1;
  /** Canonical elevation values, one per pixel: NaN exactly where a pixel is invalid. */
  elevationValues: GraphDataView<'float32'>;
  /** GPUTerrainHorizon settings: `[cellX, cellY, zFactor, curvature, northEdge, southEdge, maximumDistance, _]`. */
  settings: GraphDataView<'float32'>;
  /** Pixel-count u32 scratch, reused by every sector (sectors run in order). */
  hull: GraphDataView<'uint32'>;
  /** Up to 4 extra bindings written by `outputWGSL` (3 with the `'table'` geometry source). */
  outputBindings: readonly WGSLKernelBinding[];
  /** Extra module-scope WGSL (helpers and constants) for `outputWGSL`. */
  outputDeclarations?: string;
  /**
   * WGSL statements run once per pixel with `pixel: u32` and `horizonAngle: f32` (degrees, NaN
   * when the centre is invalid) in scope. They run inside a block, so `let` names are local.
   */
  outputWGSL: string;
};

/** {@link TerrainHorizonSweepNodeProps} with a selectable geometry source. @internal */
export type TerrainHorizonSweepKernelProps = Omit<TerrainHorizonSweepNodeProps, 'direction'> & {
  /** Baked direction or per-frame settings geometry. */
  geometry: TerrainHorizonSweepGeometrySource;
};

/**
 * Builds the exact upper-hull sweep kernel for one direction: the horizon angle of every pixel
 * along one digital line family, in amortised O(1) per pixel.
 *
 * ## Algorithm (Stewart-style hull pointers)
 *
 * Each invocation owns one digital line and walks its pixels from the far ("ahead") end back
 * toward the start. For pixel `p` with ahead samples `r` at distance index `k = |a_r - a_p|`, the
 * horizon is the sample maximising `rise(r) / k` with
 * `rise(r) = z * (h_r - h_p) - c * (k * Lref)^2`. Writing `G(x) = z * h_x - b * a_x^2` with
 * `b = c * Lref^2`, `rise / k = (G(r) - G(p)) / k + 2 * b * a_p * t`, so the best sample is the
 * tangent from `p` to the upper convex hull of the points ahead and does not depend on the
 * viewpoint's offset; this holds for any sign of `z` and any `c`. The hull of the points ahead is
 * the chain `p + 1 -> hull[p + 1] -> ...` stored in a per-pixel `hull` pointer: each pixel
 * points at its tangent vertex, found by following the chain from the next valid pixel while the
 * slope keeps growing. The unbounded walk is amortised O(1) per pixel.
 *
 * Decisions use cross-multiplication with the integer `k` (no division), so ties are exact and
 * the nearest sample wins. Radius (`k * stepPixelLength <= maximumRadius`) and distance
 * (`k * Lp <= maximumDistance`) windows are monotone in `k`; when the unbounded tangent lies
 * outside the window an exact windowed query re-walks hull chains restarted after each in-window
 * chain end (worst case O(window), typically a few steps).
 *
 * One digital line family has one pixel slope, so the sweep follows the sector's ground azimuth
 * with the pixel direction of a reference row: row 0 in `uniform` mode (every row is identical)
 * and row `height / 2` otherwise. Square reference cells keep the baked geometry; anisotropic ones
 * (geographic tiles, non-square `cellSize`) derive it in the kernel from the per-frame cell size,
 * so the dispatch always covers `width + height` lines and surplus invocations exit. Rows away from
 * the reference row see a slightly rotated ground azimuth on geographic tiles (cos(latitude)
 * changes along the tile); see `GPUTerrainHorizonAlgorithm`.
 *
 * `Lref` is constant for the whole sweep: the ground step of the reference row. The argmax is exact for
 * `c = 0` in every mode and for any `c` in uniform mode; with `c != 0` and latitude-dependent
 * cells it uses the middle row's spacing. The reported angle always uses the pixel's own row
 * spacing `Lp`.
 *
 * Elevation, validity, and hull pointers of invalid pixels follow the march semantics: invalid
 * samples are skipped, a ray without a valid in-window sample reports 0, an invalid centre reports
 * NaN.
 *
 * @internal
 */
export function getTerrainHorizonSweepNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TerrainHorizonSweepNodeProps
): GPUCommandNode<Parameters> {
  const {direction, ...rest} = props;
  return createTerrainHorizonSweepNode(graph, {
    ...rest,
    geometry: {kind: 'baked', direction}
  });
}

/**
 * Like {@link getTerrainHorizonSweepNode} with a selectable geometry source; used by the cast
 * shadow, which reads the per-frame sun direction from settings.
 *
 * @internal
 */
export function createTerrainHorizonSweepNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TerrainHorizonSweepKernelProps
): GPUCommandNode<Parameters> {
  const {width, height, geometry} = props;
  const outputLimit = geometry.kind === 'table' ? 4 : 5;
  if (props.outputBindings.length > outputLimit) {
    throw new Error(`${props.id} sweep supports at most ${outputLimit} output bindings`);
  }
  if (!Number.isFinite(props.maximumRadius) || props.maximumRadius <= 0) {
    throw new Error(`${props.id} sweep maximumRadius must be positive`);
  }
  const bakedGeometry =
    geometry.kind === 'baked'
      ? getTerrainSweepLineGeometry(geometry.direction, width, height)
      : undefined;
  if (width > TERRAIN_SWEEP_MAXIMUM_EXTENT || height > TERRAIN_SWEEP_MAXIMUM_EXTENT) {
    throw new Error(`${props.id} sweep extents must not exceed ${TERRAIN_SWEEP_MAXIMUM_EXTENT}`);
  }
  const referenceRow = props.cellSizeMode === 'uniform' ? '0u' : 'HEIGHT / 2u';
  const settingsIndex = geometry.kind === 'settings' ? geometry.index : 0;
  const groundDirectionIndex = geometry.kind === 'settings' ? geometry.groundDirectionIndex : 0;
  const geometryWGSL = `${
    geometry.kind === 'table'
      ? `var xMajor = sectorRecord[sectorRecordOffset] != 0u;
  var slopeFixed = bitcast<i32>(sectorRecord[sectorRecordOffset + 1u]);
  var travelSign = bitcast<i32>(sectorRecord[sectorRecordOffset + 2u]);
  var stepPixelLength = bitcast<f32>(sectorRecord[sectorRecordOffset + 3u]);
  let groundDirection = vec2<f32>(
    bitcast<f32>(sectorRecord[sectorRecordOffset + 4u]),
    bitcast<f32>(sectorRecord[sectorRecordOffset + 5u])
  );`
      : bakedGeometry
        ? `var xMajor = ${bakedGeometry.xMajor ? 'true' : 'false'};
  var slopeFixed: i32 = ${bakedGeometry.slopeFixed};
  var travelSign: i32 = ${bakedGeometry.travelSign};
  var stepPixelLength: f32 = ${getWGSLFloatLiteral(bakedGeometry.stepPixelLength)};
  let groundDirection = GROUND_DIRECTION;`
        : `var xMajor = settings[settingsOffset + ${settingsIndex}u] != 0.0;
  var slopeFixed = i32(settings[settingsOffset + ${settingsIndex + 1}u]);
  var travelSign = select(-1, 1, settings[settingsOffset + ${settingsIndex + 2}u] > 0.0);
  var stepPixelLength = settings[settingsOffset + ${settingsIndex + 3}u];
  ${geometry.kind === 'settings' && geometry.rowDirection === 'north' ? 'var' : 'let'} groundDirection = vec2<f32>(
    settings[settingsOffset + ${groundDirectionIndex}u],
    settings[settingsOffset + ${groundDirectionIndex + 1}u]
  );${
    geometry.kind === 'settings' && geometry.rowDirection === 'north'
      ? `
  // The packed line assumes south-advancing rows; mirror it for north-advancing rows. Negating the
  // row component negates the slope, and the travel sign of y-major lines.
  groundDirection.y = -groundDirection.y;
  slopeFixed = -slopeFixed;
  if (!xMajor) { travelSign = -travelSign; }`
      : ''
  }`
  }
  // The direction is a ground azimuth; on anisotropic cells the line follows the pixel direction
  // of the reference row instead (CPU twin: getTerrainSweepGroundLineGeometry).
  let referenceCell = getGroundCellSize(${referenceRow});
  if (referenceCell.x != referenceCell.y && all(referenceCell > vec2<f32>(0.0)) &&
      isFiniteValue(referenceCell.x + referenceCell.y) && any(groundDirection != vec2<f32>(0.0))) {
    let pixelDirection = groundDirection / referenceCell;
    xMajor = abs(pixelDirection.x) >= abs(pixelDirection.y);
    let majorComponent = select(pixelDirection.y, pixelDirection.x, xMajor);
    let minorComponent = select(pixelDirection.x, pixelDirection.y, xMajor);
    slopeFixed = i32(round((minorComponent / majorComponent) * ${TERRAIN_SWEEP_SLOPE_SCALE}.0));
    travelSign = select(-1, 1, majorComponent > 0.0);
    stepPixelLength = length(vec2<f32>(1.0, f32(slopeFixed) / ${TERRAIN_SWEEP_SLOPE_SCALE}.0));
  }`;
  const bindings: WGSLKernelBinding[] = [
    {name: 'elevationValues', view: props.elevationValues, type: 'f32', access: 'read'},
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
    {name: 'hull', view: props.hull, type: 'u32', access: 'read_write'},
    ...(geometry.kind === 'table'
      ? [
          {
            name: 'sectorRecord',
            view: geometry.record,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]
      : []),
    ...props.outputBindings
  ];
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTerrainHorizon',
    variant: `sweep-${props.cellSizeMode}`,
    bindings,
    workgroupSize: TERRAIN_SWEEP_WORKGROUP_SIZE,
    // Anisotropic cells change the line count per frame, so every source dispatches the bound.
    invocationCount: width + height,
    declarations: `${getTerrainHorizonSweepDeclarations(props)}
${
  geometry.kind === 'baked'
    ? `const GROUND_DIRECTION: vec2<f32> = vec2<f32>(${getWGSLFloatLiteral(geometry.direction[0])}, ${getWGSLFloatLiteral(geometry.direction[1])});`
    : ''
}`,
    body: `${geometryWGSL}
  let majorExtent = select(HEIGHT, WIDTH, xMajor);
  let minorExtent = select(WIDTH, HEIGHT, xMajor);
  let lastOffset = getFixedOffset(slopeFixed, i32(majorExtent) - 1);
  let lineCount = i32(minorExtent) + abs(lastOffset);
  if (i32(index) >= lineCount) {
    return;
  }
  sweepXMajor = xMajor;
  sweepSlopeFixed = slopeFixed;
  sweepTravel = travelSign;
  sweepMajorExtent = i32(majorExtent);
  sweepMinorExtent = i32(minorExtent);
  sweepLine = i32(index) - max(0, lastOffset);
  sweepZ = f32(${props.zFactorSign}) * settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  sweepMaximumDistance = settings[settingsOffset + 6u];
  sweepStepPixelLength = stepPixelLength;
  let referenceStep = getStepLength(${referenceRow}, xMajor, slopeFixed);
  sweepCurvature = curvature * referenceStep * referenceStep;
  var previous = NONE;
  for (var stepIndex = 0; stepIndex < sweepMajorExtent; stepIndex++) {
    let major = select(stepIndex, sweepMajorExtent - 1 - stepIndex, travelSign > 0);
    let minor = sweepLine + getFixedOffset(slopeFixed, major);
    if (minor < 0 || minor >= sweepMinorExtent) {
      continue;
    }
    let pixel = select(u32(major) * WIDTH + u32(minor), u32(minor) * WIDTH + u32(major), xMajor);
    let next = previous;
    previous = pixel;
    var firstAhead = NONE;
    if (next != NONE) {
      firstAhead = select(hull[hullOffset + next], next, isValidPixel(next));
    }
    var horizonAngle = getNaN(pixel);
    if (!isValidPixel(pixel)) {
      hull[hullOffset + pixel] = firstAhead;
    } else {
      sweepElevation = elevationValues[elevationValuesOffset + pixel];
      sweepMajor = major;
      var tangent = firstAhead;
      if (tangent != NONE) {
        var chain = hull[hullOffset + tangent];
        while (chain != NONE && isGrowing(chain, tangent)) {
          tangent = chain;
          chain = hull[hullOffset + chain];
        }
      }
      hull[hullOffset + pixel] = tangent;
      let pixelStep = getStepLength(pixel / WIDTH, xMajor, slopeFixed);
      if (pixelStep > 0.0 && isFiniteValue(pixelStep)) {
        sweepPixelStep = pixelStep;
        var best = NONE;
        if (firstAhead != NONE && isInsideWindow(firstAhead)) {
          if (isInsideWindow(tangent)) {
            best = tangent;
          } else {
            best = queryWindow(firstAhead);
          }
        }
        if (best == NONE) {
          horizonAngle = 0.0;
        } else {
          let groundDistance = getDistanceIndex(best) * pixelStep;
          let rise = sweepZ * (elevationValues[elevationValuesOffset + best] - sweepElevation) -
            curvature * groundDistance * groundDistance;
          horizonAngle = atan(rise / groundDistance) * RADIANS_TO_DEGREES;
        }
      }
    }
    {
      ${props.outputWGSL}
    }
  }`
  });
}

/** Module-scope WGSL of the sweep kernel. */
function getTerrainHorizonSweepDeclarations(props: TerrainHorizonSweepKernelProps): string {
  return `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const NONE: u32 = 0xffffffffu;
const MAXIMUM_RADIUS: f32 = ${getWGSLFloatLiteral(props.maximumRadius)};
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${getRasterGroundCellSizeWGSL(props.cellSizeMode, TERRAIN_SWEEP_GRID_SLOTS)}
var<private> sweepXMajor: bool;
var<private> sweepSlopeFixed: i32;
var<private> sweepTravel: i32;
var<private> sweepMajorExtent: i32;
var<private> sweepMinorExtent: i32;
var<private> sweepLine: i32;
var<private> sweepZ: f32;
var<private> sweepCurvature: f32;
var<private> sweepMaximumDistance: f32;
var<private> sweepStepPixelLength: f32;
var<private> sweepPixelStep: f32;
var<private> sweepElevation: f32;
var<private> sweepMajor: i32;
fn getFixedOffset(slopeFixed: i32, major: i32) -> i32 {
  return (slopeFixed * major + 32768) >> 16u;
}
// Ground length of one major step on the pixel row, using the row's cell size.
fn getStepLength(row: u32, xMajor: bool, slopeFixed: i32) -> f32 {
  let slope = f32(slopeFixed) / 65536.0;
  let cell = getGroundCellSize(row);
  return length(select(vec2<f32>(slope, 1.0), vec2<f32>(1.0, slope), xMajor) * cell);
}
// Canonical elevation is NaN exactly where a pixel is invalid.
fn isValidPixel(pixel: u32) -> bool {
  return isFiniteValue(elevationValues[elevationValuesOffset + pixel]);
}
fn getMajorIndex(pixel: u32) -> i32 {
  return i32(select(pixel / WIDTH, pixel % WIDTH, sweepXMajor));
}
fn getDistanceIndex(pixel: u32) -> f32 {
  return f32(abs(getMajorIndex(pixel) - sweepMajor));
}
// Hull-metric rise for the sample at distance index kf (curvature uses the reference step).
fn getRise(pixel: u32, kf: f32) -> f32 {
  return sweepZ * (elevationValues[elevationValuesOffset + pixel] - sweepElevation) -
    sweepCurvature * (kf * kf);
}
// True when sample r has a strictly larger tangent than sample c (cross-multiplied, exact ties).
fn isGrowing(r: u32, c: u32) -> bool {
  let kr = getDistanceIndex(r);
  let kc = getDistanceIndex(c);
  return getRise(r, kr) * kc > getRise(c, kc) * kr;
}
fn isInsideWindow(pixel: u32) -> bool {
  let kf = getDistanceIndex(pixel);
  return kf * sweepStepPixelLength <= MAXIMUM_RADIUS &&
    (sweepMaximumDistance <= 0.0 || kf * sweepPixelStep <= sweepMaximumDistance);
}
// First valid pixel strictly after c on the line, NONE at the end.
fn getValidAfter(c: u32) -> u32 {
  let major = getMajorIndex(c) + sweepTravel;
  if (major < 0 || major >= sweepMajorExtent) { return NONE; }
  let minor = sweepLine + getFixedOffset(sweepSlopeFixed, major);
  if (minor < 0 || minor >= sweepMinorExtent) { return NONE; }
  let pixel = select(u32(major) * WIDTH + u32(minor), u32(minor) * WIDTH + u32(major), sweepXMajor);
  return select(hull[hullOffset + pixel], pixel, isValidPixel(pixel));
}
// Exact maximum over the in-window samples at or after start (start is valid and inside).
fn queryWindow(start: u32) -> u32 {
  var best = NONE;
  var t = start;
  loop {
    if (t == NONE || !isInsideWindow(t)) { break; }
    var c = t;
    var r = hull[hullOffset + c];
    while (r != NONE && isInsideWindow(r) && isGrowing(r, c)) {
      c = r;
      r = hull[hullOffset + r];
    }
    if (best == NONE) {
      best = c;
    } else if (isGrowing(c, best)) {
      best = c;
    }
    if (r == NONE || isInsideWindow(r)) { break; }
    t = getValidAfter(c);
  }
  return best;
}
${props.outputDeclarations ?? ''}`;
}
