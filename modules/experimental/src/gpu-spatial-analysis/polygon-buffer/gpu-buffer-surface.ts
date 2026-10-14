// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
  validatePowerOfTwo
} from '../../utils/gpu-contributor-utils';
import type {
  GPULineGeometryPort,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';
import {
  GPUGeometryOrientation,
  GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH
} from '../geometry-edit/index';
import {
  getGPUOffsetCurveRowsPerVertex,
  GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS,
  GPUOffsetCurve,
  GPU_OFFSET_CURVE_PARAMETER_LENGTH,
  type GPUOffsetCurveJoinStyle
} from '../outline-geometry/index';
import {
  GPUPolygonOverlay,
  type GPUPolygonOverlayCapacity,
  type GPUPolygonOverlayOutput
} from '../polygon-overlay/index';
import type {GPUPolygonOverlayFillRule} from '../polygon-overlay/gpu-polygon-overlay';
import {getGPUSnapRoundedNodingMaximumVertexCount} from '../geometry-topology/gpu-snap-rounded-noding';
import type {Device} from '@luma.gl/core';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUBufferSurface';

/**
 * Largest sum of boundary rows and `capacity.intersections` of {@link GPUBufferSurface} on
 * `device`: the raw boundary rows plus one closing vertex per raw boundary ring are noded by
 * `GPUPolygonOverlay`, which snap-rounds them and keeps four diagonal vertices of 32 bytes per hot
 * pixel (a vertex or a crossing) in one segment table. The limit is
 * `floor(device.limits.maxStorageBufferBindingSize / 128)`: 1048576 with 128 MiB bindings,
 * or 8388608 with 1 GiB bindings. The boundary rows may use what `capacity.intersections`
 * leaves. For polygons the rows are `(1 + 2 * rowsPerVertex) * vertices + 3 * rings`, and for lines
 * `2 * rowsPerVertex * vertices + (2 * capRows + 1) * lines`, where `rowsPerVertex` is
 * `getGPUOffsetCurveRowsPerVertex(joinStyle, quadSegments, 'buffer')` (17 for the default round
 * joins).
 */
export function getGPUBufferSurfaceMaximumBoundaryRowCount(device: Device): number {
  return getGPUSnapRoundedNodingMaximumVertexCount(device, 0);
}

/** End treatment for buffered lines. */
export type GPUBufferCapStyle = 'round' | 'flat' | 'square';

/** Properties for {@link GPUBufferSurface}. */
export type GPUBufferSurfaceProps = {
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Lines or polygons to buffer; a generated geometry port can be passed as is. The raw boundary
   * rows derived from it (see {@link getGPUBufferSurfaceMaximumBoundaryRowCount}) are at most that
   * maximum minus `capacity.intersections` (1048576 before that subtraction on a default device).
   */
  geometry: GPUPackedGeometryPort<GPULineGeometryPort | GPUPolygonGeometryPort>;
  /** `[distance, mitreLimit, 0, 0]`, matching `GPUOffsetCurve`. */
  parameters: GraphDataView<'float32'>;
  joinStyle?: GPUOffsetCurveJoinStyle;
  capStyle?: GPUBufferCapStyle;
  quadSegments?: number;
  vertexTolerance: number;
  capacity: GPUPolygonOverlayCapacity;
  /** Caller-owned outputs; `output.status.uncertainCount` is the optional uncertain-pair counter. */
  output: GPUPolygonOverlayOutput;
  spatialSort?: boolean;
  leafCapacity?: number;
};

/**
 * Builds polygon buffer surfaces for lines and polygons.
 *
 * This is deliberately distinct from `GPUOffsetCurve`: offset curves are generated for both
 * sides of a line (with round, flat or square caps) or for every normalized polygon ring, then
 * the resulting raw boundaries are noded and dissolved by `GPUPolygonOverlay` into polygons.
 *
 * Ring preparation: consecutive duplicate vertices are dropped from every line and ring, and a
 * polygon ring's closing vertex is dropped when it repeats the first one (GeoJSON and GeoArrow
 * rings are closed; the offset curve requires open rings, and a repeated closing vertex would leave
 * a notch there). Lines keep their endpoints, so a closed line stays a line with coincident ends.
 *
 * Fill rule: raw boundaries are dissolved with a WINDING rule (`'negative'`), not even-odd, so
 * regions the raw boundaries cover more than once (hairpins and switchbacks closer than twice the
 * distance, self-crossing lines, polygon slots narrower than twice the distance) stay inside, as in
 * `shapely.buffer`. Lines emit the clockwise outline of their two offsets and caps.
 *
 * Polygons are a union of simple pieces: the polygon (normalized to counter-clockwise shells and
 * clockwise holes, then reversed) and the stroke of every ring: its two closed offset rings at
 * `+|d|` and `-|d|` (joins at every vertex, inside turns routed through the vertex where the
 * offsets do not meet, as GEOS does; a ring narrower than `2|d|` drops its interior-side offset,
 * the GEOS eroded-ring test). A positive distance keeps `polygon + strokes`, a negative one
 * `polygon - strokes` (the stroke rings are then read the other way, so their winding cancels the
 * polygon's), so collapsed holes vanish and necks thinner than twice the distance split, as in
 * `shapely.buffer`. Raw boundary rows per ring of `n` vertices: `n (1 + 2 rowsPerVertex)`, so
 * size `capacity` for about twice the segments of the ring offset alone.
 *
 * Planar only: the distance is in position units, so buffering longitude/latitude degrees gives an
 * ellipse on the ground and a longitude-latitude `spatialContext` throws. Project first, for example
 * with `createGPULocalFrame` and `GPULocalFrameProjection`, and buffer in the frame's meters.
 *
 * Polygon distance is signed. Positive grows shells and shrinks holes; negative does the reverse.
 * A non-positive line distance emits an empty surface, matching ordinary planar buffer semantics.
 * `output.status.requiredCount` is the unclamped ring requirement of the assembly.
 */
export class GPUBufferSurface implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUBufferSurfaceProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;
  readonly joinStyle: GPUOffsetCurveJoinStyle;
  readonly capStyle: GPUBufferCapStyle;
  readonly quadSegments: number;
  readonly rowsPerVertex: number;

  constructor(props: GPUBufferSurfaceProps) {
    this.id = props.id ?? 'buffer-surface';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.props = props;
    this.joinStyle = props.joinStyle ?? 'round';
    this.capStyle = props.capStyle ?? 'round';
    this.quadSegments = props.quadSegments ?? 8;
    // Buffer-style inside turns keep the raw boundary's winding inside the buffer.
    this.rowsPerVertex = getGPUOffsetCurveRowsPerVertex(
      this.joinStyle,
      this.quadSegments,
      'buffer'
    );
    const {id} = this;
    const {geometry} = props;
    validatePackedView(geometry.positions, ['float32x2'], `${id} geometry.positions`);
    if (geometry.positions.length < 1) {
      throw new Error(`${id} geometry.positions must not be empty`);
    }
    if (geometry.kind === 'lines') {
      validatePackedUint32View(geometry.lineOffsets, `${id} geometry.lineOffsets`);
      if (geometry.lineOffsets.length < 2) {
        throw new Error(`${id} lineOffsets must contain at least two entries`);
      }
    } else {
      for (const [name, view] of [
        ['featureOffsets', geometry.featureOffsets],
        ['polygonOffsets', geometry.polygonOffsets],
        ['ringOffsets', geometry.ringOffsets]
      ] as const) {
        validatePackedUint32View(view, `${id} geometry.${name}`);
        if (view.length < 2) {
          throw new Error(`${id} geometry.${name} must contain at least two entries`);
        }
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_OFFSET_CURVE_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must contain four float32 values`);
    }
    if (!['round', 'mitre', 'bevel'].includes(this.joinStyle)) {
      throw new Error(`${id} joinStyle must be round, mitre or bevel`);
    }
    if (!['round', 'flat', 'square'].includes(this.capStyle)) {
      throw new Error(`${id} capStyle must be round, flat or square`);
    }
    validateCapacity(
      id,
      'quadSegments',
      this.quadSegments,
      1,
      GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS
    );
    // Delegates are built later, in getCommandNodes: check forwarded options now, under this id.
    if (this.props.leafCapacity !== undefined) {
      validatePowerOfTwo(id, 'leafCapacity', this.props.leafCapacity);
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {geometry} = this.props;
    validateGraphViewsBelongToGraph(this.id, graph, [
      geometry.positions,
      geometry.kind === 'lines' ? geometry.lineOffsets : geometry.featureOffsets,
      geometry.kind === 'lines' ? undefined : geometry.polygonOffsets,
      geometry.kind === 'lines' ? undefined : geometry.ringOffsets,
      this.props.parameters
    ]);
    const capRows =
      this.capStyle === 'round' ? 2 * this.quadSegments - 1 : this.capStyle === 'square' ? 2 : 0;
    validateMaximumCount(
      this.id,
      'boundary rows',
      geometry.kind === 'lines'
        ? 2 * this.rowsPerVertex * geometry.positions.length +
            (2 * capRows + 1) * (geometry.lineOffsets.length - 1)
        : (1 + 2 * this.rowsPerVertex) * geometry.positions.length +
            3 * (geometry.ringOffsets.length - 1),
      getGPUBufferSurfaceMaximumBoundaryRowCount(graph.device) - this.props.capacity.intersections
    );
    return geometry.kind === 'lines'
      ? this._getLineNodes(graph, geometry)
      : this._getPolygonNodes(graph, geometry);
  }

  /**
   * Polygon buffer as a union of simple pieces: the polygon itself and the stroke (two-sided
   * buffer of width `2|d|`) of every ring, `polygon + stroke` for positive and `polygon - stroke`
   * for negative distances. Every piece is written as raw rings of one polygon per source polygon:
   * the normalized rings reversed (winding -1 inside the polygon) and, per ring, its two closed
   * offset rings at `+|d|` (forward) and `-|d|` (reversed), whose windings sum to <= -1 inside the
   * stroke for `d > 0`; for `d < 0` both are read backwards (>= +1). The `'negative'` fill rule then
   * keeps exactly the union, or the polygon minus the stroke, so a hole the buffer collapses
   * disappears instead of leaving a reflected hole. Closed ring offsets have a join at every vertex
   * and no caps or repeated edges, whose separately rounded copies would form slivers.
   */
  private _getPolygonNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    polygons: GPUPackedGeometryPort<GPUPolygonGeometryPort>
  ): GPUCommandNode<Parameters>[] {
    const {id, props, rowsPerVertex} = this;
    const prepared = this._getCompactionNodes(
      graph,
      'ring',
      polygons.positions,
      polygons.ringOffsets,
      true
    );
    const ringCount = polygons.ringOffsets.length - 1;
    const polygonCount = polygons.polygonOffsets.length - 1;
    const vertexCapacity = polygons.positions.length;
    const rawRowCount = (1 + 2 * rowsPerVertex) * vertexCapacity;
    const transient = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const normalized = transient('normalized', 'float32x2', vertexCapacity);
    const orientationParameters = transient(
      'orientation-parameters',
      'float32',
      GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH
    );
    const leftParameters = transient('stroke-left-parameters', 'float32', 4);
    const rightParameters = transient('stroke-right-parameters', 'float32', 4);
    const leftPositions = transient('stroke-left', 'float32x2', vertexCapacity * rowsPerVertex);
    const rightPositions = transient('stroke-right', 'float32x2', vertexCapacity * rowsPerVertex);
    const rawPositions = transient('raw-positions', 'float32x2', rawRowCount);
    const rawRingOffsets = transient('raw-ring-offsets', 'uint32', 3 * ringCount + 1);
    const rawPolygonOffsets = transient('raw-polygon-offsets', 'uint32', polygonCount + 1);
    // Ring layout of the raw pieces: polygon p owns raw rings [3 r0, 3 r1) for its source rings
    // [r0, r1): first the reversed rings, then for each ring its +|d| and -|d| offset rings. A ring
    // of n vertices gives n reversed rows and n ROWS_PER_VERTEX rows per offset ring.
    const layoutWGSL = /* wgsl */ `const RING_COUNT: u32 = ${ringCount}u;
const POLYGON_COUNT: u32 = ${polygonCount}u;
const ROWS_PER_VERTEX: u32 = ${rowsPerVertex}u;

fn findPolygonOfRing(ring: u32) -> u32 {
  var low = 0u;
  var high = POLYGON_COUNT;
  while (low + 1u < high) {
    let middle = (low + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= ring) { low = middle; } else { high = middle; }
  }
  return low;
}`;
    const offsetCurve = (side: 'left' | 'right') =>
      new GPUOffsetCurve({
        id: `${id}-stroke-${side}-offset`,
        spatialContext: this.spatialContext,
        positions: normalized,
        featureOffsets: polygons.featureOffsets,
        polygonOffsets: polygons.polygonOffsets,
        ringOffsets: prepared.offsets,
        geometryType: 'polygons',
        joinStyle: this.joinStyle,
        quadSegments: this.quadSegments,
        insideTurns: 'buffer',
        parameters: side === 'left' ? leftParameters : rightParameters,
        output: {positions: side === 'left' ? leftPositions : rightPositions}
      }).getCommandNodes(graph);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...prepared.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-polygon-parameters`,
        operation: OPERATION,
        variant: 'polygon-parameters',
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'orientation', view: orientationParameters, type: 'f32', access: 'read_write'},
          {name: 'left', view: leftParameters, type: 'f32', access: 'read_write'},
          {name: 'right', view: rightParameters, type: 'f32', access: 'read_write'}
        ],
        invocationCount: GPU_OFFSET_CURVE_PARAMETER_LENGTH,
        body: `orientation[orientationOffset + index] = 0.0;
  let distance = abs(parameters[parametersOffset]);
  let value = parameters[parametersOffset + index];
  left[leftOffset + index] = select(value, distance, index == 0u);
  right[rightOffset + index] = select(value, -distance, index == 0u);`
      }),
      ...new GPUGeometryOrientation({
        id: `${id}-orient`,
        spatialContext: this.spatialContext,
        positions: prepared.positions,
        ringOffsets: prepared.offsets,
        polygonOffsets: polygons.polygonOffsets,
        featureOffsets: polygons.featureOffsets,
        mode: 'orient-polygons',
        parameters: orientationParameters,
        output: {positions: normalized}
      }).getCommandNodes(graph),
      ...offsetCurve('left'),
      ...offsetCurve('right'),
      // A ring whose envelope is narrower than 2|d| is entirely within |d| of itself (the GEOS
      // eroded-ring test): its offset toward its own interior inverts, and an inverted ring would
      // cancel the stroke there. That offset ring is collapsed to one point, so the stroke covers
      // the whole ring interior (a hole a positive buffer collapses disappears).
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-stroke-collapse`,
        operation: OPERATION,
        variant: 'stroke-collapse',
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: prepared.offsets, type: 'u32', access: 'read'},
          {name: 'normalized', view: normalized, type: 'f32', access: 'read'},
          {name: 'leftPositions', view: leftPositions, type: 'f32', access: 'read_write'},
          {name: 'rightPositions', view: rightPositions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: Math.max(ringCount, 1),
        declarations: `const RING_COUNT: u32 = ${ringCount}u;
const ROWS_PER_VERTEX: u32 = ${rowsPerVertex}u;
fn readNormalized(row: u32) -> vec2f {
  return vec2f(normalized[normalizedOffset + 2u * row], normalized[normalizedOffset + 2u * row + 1u]);
}`,
        body: `if (index >= RING_COUNT) { return; }
  let begin = ringOffsets[ringOffsetsOffset + index];
  let end = ringOffsets[ringOffsetsOffset + index + 1u];
  if (end < begin + 3u) { return; }
  let origin = readNormalized(begin);
  var low = vec2f(0.0);
  var high = vec2f(0.0);
  var doubledArea = 0.0;
  var previous = readNormalized(end - 1u) - origin;
  for (var row = begin; row < end; row++) {
    let current = readNormalized(row) - origin;
    low = min(low, current);
    high = max(high, current);
    doubledArea += previous.x * current.y - previous.y * current.x;
    previous = current;
  }
  let extent = high - low;
  if (min(extent.x, extent.y) >= 2.0 * abs(parameters[parametersOffset])) { return; }
  // The interior side is left of a counter-clockwise ring and right of a clockwise one.
  let rowBegin = begin * ROWS_PER_VERTEX;
  let rowEnd = end * ROWS_PER_VERTEX;
  if (doubledArea > 0.0) {
    let point = vec2f(leftPositions[leftPositionsOffset + 2u * rowBegin], leftPositions[leftPositionsOffset + 2u * rowBegin + 1u]);
    for (var row = rowBegin; row < rowEnd; row++) {
      leftPositions[leftPositionsOffset + 2u * row] = point.x;
      leftPositions[leftPositionsOffset + 2u * row + 1u] = point.y;
    }
  } else {
    let point = vec2f(rightPositions[rightPositionsOffset + 2u * rowBegin], rightPositions[rightPositionsOffset + 2u * rowBegin + 1u]);
    for (var row = rowBegin; row < rowEnd; row++) {
      rightPositions[rightPositionsOffset + 2u * row] = point.x;
      rightPositions[rightPositionsOffset + 2u * row + 1u] = point.y;
    }
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-raw-layout`,
        operation: OPERATION,
        variant: 'raw-layout',
        bindings: [
          {name: 'ringOffsets', view: prepared.offsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: polygons.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'rawRingOffsets', view: rawRingOffsets, type: 'u32', access: 'read_write'},
          {name: 'rawPolygonOffsets', view: rawPolygonOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: Math.max(3 * ringCount, polygonCount) + 1,
        declarations: layoutWGSL,
        body: `if (index <= POLYGON_COUNT) {
    rawPolygonOffsets[rawPolygonOffsetsOffset + index] = 3u * polygonOffsets[polygonOffsetsOffset + index];
  }
  if (index > 3u * RING_COUNT) { return; }
  var start = 0u;
  if (index == 3u * RING_COUNT) {
    start = (1u + 2u * ROWS_PER_VERTEX) * ringOffsets[ringOffsetsOffset + RING_COUNT];
  } else {
    let polygon = findPolygonOfRing(index / 3u);
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let lastRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    let firstVertex = ringOffsets[ringOffsetsOffset + firstRing];
    let lastVertex = ringOffsets[ringOffsetsOffset + lastRing];
    let local = index - 3u * firstRing;
    let polygonStart = (1u + 2u * ROWS_PER_VERTEX) * firstVertex;
    if (local < lastRing - firstRing) {
      start = polygonStart + ringOffsets[ringOffsetsOffset + firstRing + local] - firstVertex;
    } else {
      let band = local - (lastRing - firstRing);
      let ring = firstRing + band / 2u;
      let ringBegin = ringOffsets[ringOffsetsOffset + ring];
      let ringLength = ringOffsets[ringOffsetsOffset + ring + 1u] - ringBegin;
      start = polygonStart + (lastVertex - firstVertex) +
        2u * ROWS_PER_VERTEX * (ringBegin - firstVertex) + (band % 2u) * ROWS_PER_VERTEX * ringLength;
    }
  }
  rawRingOffsets[rawRingOffsetsOffset + index] = start;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-raw-pieces`,
        operation: OPERATION,
        variant: 'raw-pieces',
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: prepared.offsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: polygons.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'rawRingOffsets', view: rawRingOffsets, type: 'u32', access: 'read'},
          {name: 'normalized', view: normalized, type: 'f32', access: 'read'},
          {name: 'leftPositions', view: leftPositions, type: 'f32', access: 'read'},
          {name: 'rightPositions', view: rightPositions, type: 'f32', access: 'read'},
          {name: 'output', view: rawPositions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rawRowCount,
        declarations: layoutWGSL,
        body: `var value = vec2f(0.0);
  if (index < rawRingOffsets[rawRingOffsetsOffset + 3u * RING_COUNT]) {
    var rawRing = 0u;
    var high = 3u * RING_COUNT;
    while (rawRing + 1u < high) {
      let middle = (rawRing + high) / 2u;
      if (rawRingOffsets[rawRingOffsetsOffset + middle] <= index) { rawRing = middle; } else { high = middle; }
    }
    let polygon = findPolygonOfRing(rawRing / 3u);
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let ringsInPolygon = polygonOffsets[polygonOffsetsOffset + polygon + 1u] - firstRing;
    let local = rawRing - 3u * firstRing;
    let row = index - rawRingOffsets[rawRingOffsetsOffset + rawRing];
    if (local < ringsInPolygon) {
      // Reversed normalized ring: winding -1 inside the polygon.
      let ring = firstRing + local;
      let source = ringOffsets[ringOffsetsOffset + ring + 1u] - 1u - row;
      value = vec2f(normalized[normalizedOffset + 2u * source], normalized[normalizedOffset + 2u * source + 1u]);
    } else {
      // Stroke rings: the +|d| offset forward and the -|d| offset backward (stroke winding -1 for
      // a positive distance); a negative distance reads both the other way (+1).
      let band = local - ringsInPolygon;
      let ring = firstRing + band / 2u;
      let rowBegin = ringOffsets[ringOffsetsOffset + ring] * ROWS_PER_VERTEX;
      let rowCount = (ringOffsets[ringOffsetsOffset + ring + 1u] - ringOffsets[ringOffsetsOffset + ring]) * ROWS_PER_VERTEX;
      let isLeft = (band % 2u) == 0u;
      let forward = isLeft == (parameters[parametersOffset] >= 0.0);
      let source = rowBegin + select(rowCount - 1u - row, row, forward);
      if (isLeft) {
        value = vec2f(leftPositions[leftPositionsOffset + 2u * source], leftPositions[leftPositionsOffset + 2u * source + 1u]);
      } else {
        value = vec2f(rightPositions[rightPositionsOffset + 2u * source], rightPositions[rightPositionsOffset + 2u * source + 1u]);
      }
    }
  }
  output[outputOffset + 2u * index] = value.x;
  output[outputOffset + 2u * index + 1u] = value.y;`
      })
    ];
    nodes.push(
      ...this._getDissolveNodes(
        graph,
        {
          kind: 'polygons',
          positions: rawPositions,
          featureOffsets: polygons.featureOffsets,
          polygonOffsets: rawPolygonOffsets,
          ringOffsets: rawRingOffsets,
          sourceIds: polygons.sourceIds
        },
        'negative'
      )
    );
    return nodes;
  }

  private _getLineNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    lines: GPUPackedGeometryPort<GPULineGeometryPort>
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const prepared = this._getCompactionNodes(
      graph,
      'line',
      lines.positions,
      lines.lineOffsets,
      false
    );
    const lineCount = lines.lineOffsets.length - 1;
    const sideRows = lines.positions.length * this.rowsPerVertex;
    const capRows =
      this.capStyle === 'round' ? 2 * this.quadSegments - 1 : this.capStyle === 'square' ? 2 : 0;
    const rawRowCount = sideRows * 2 + lineCount * capRows * 2;
    const positiveParameters = createTransientView(graph, `${id}-positive`, 'float32', 4);
    const negativeParameters = createTransientView(graph, `${id}-negative`, 'float32', 4);
    const leftPositions = createTransientView(graph, `${id}-left`, 'float32x2', sideRows);
    const rightPositions = createTransientView(graph, `${id}-right`, 'float32x2', sideRows);
    const rawPositions = createTransientView(
      graph,
      `${id}-raw-positions`,
      'float32x2',
      rawRowCount
    );
    const rawOffsets = createTransientView(graph, `${id}-raw-offsets`, 'uint32', lineCount + 1);
    const identityOffsets = createTransientView(
      graph,
      `${id}-identity-offsets`,
      'uint32',
      lineCount + 1
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      ...prepared.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-line-parameters`,
        operation: OPERATION,
        variant: 'line-parameters',
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'positive', view: positiveParameters, type: 'f32', access: 'read_write'},
          {name: 'negative', view: negativeParameters, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 4,
        body: `let distance = max(parameters[parametersOffset], 0.0);
  let value = parameters[parametersOffset + index];
  positive[positiveOffset + index] = select(value, distance, index == 0u);
  negative[negativeOffset + index] = select(value, -distance, index == 0u);`
      }),
      ...new GPUOffsetCurve({
        id: `${id}-left-offset`,
        spatialContext: this.spatialContext,
        positions: prepared.positions,
        lineOffsets: prepared.offsets,
        geometryType: 'lines',
        joinStyle: this.joinStyle,
        quadSegments: this.quadSegments,
        insideTurns: 'buffer',
        parameters: positiveParameters,
        output: {positions: leftPositions}
      }).getCommandNodes(graph),
      ...new GPUOffsetCurve({
        id: `${id}-right-offset`,
        spatialContext: this.spatialContext,
        positions: prepared.positions,
        lineOffsets: prepared.offsets,
        geometryType: 'lines',
        joinStyle: this.joinStyle,
        quadSegments: this.quadSegments,
        insideTurns: 'buffer',
        parameters: negativeParameters,
        output: {positions: rightPositions}
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-line-topology`,
        operation: OPERATION,
        variant: 'line-topology',
        bindings: [
          {name: 'lineOffsets', view: prepared.offsets, type: 'u32', access: 'read'},
          {name: 'ringOffsets', view: rawOffsets, type: 'u32', access: 'read_write'},
          {name: 'identityOffsets', view: identityOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: lineCount + 1,
        declarations: `const ROWS_PER_VERTEX: u32 = ${this.rowsPerVertex}u;
const CAP_ROWS: u32 = ${capRows}u;`,
        body: `ringOffsets[ringOffsetsOffset + index] = 2u * ROWS_PER_VERTEX * lineOffsets[lineOffsetsOffset + index] + 2u * CAP_ROWS * index;
  identityOffsets[identityOffsetsOffset + index] = index;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-line-boundaries`,
        operation: OPERATION,
        variant: `line-${this.capStyle}`,
        bindings: [
          {name: 'positions', view: prepared.positions, type: 'f32', access: 'read'},
          {name: 'lineOffsets', view: prepared.offsets, type: 'u32', access: 'read'},
          {name: 'ringOffsets', view: rawOffsets, type: 'u32', access: 'read'},
          {name: 'parameters', view: positiveParameters, type: 'f32', access: 'read'},
          {name: 'leftPositions', view: leftPositions, type: 'f32', access: 'read'},
          {name: 'rightPositions', view: rightPositions, type: 'f32', access: 'read'},
          {name: 'output', view: rawPositions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rawRowCount,
        declarations: `const LINE_COUNT: u32 = ${lineCount}u;
const ROWS_PER_VERTEX: u32 = ${this.rowsPerVertex}u;
const CAP_ROWS: u32 = ${capRows}u;
const PI: f32 = 3.141592653589793;

fn point(view: ptr<storage, array<f32>, read>, base: u32, row: u32) -> vec2f {
  return vec2f((*view)[base + 2u * row], (*view)[base + 2u * row + 1u]);
}

fn unitDirection(a: vec2f, b: vec2f) -> vec2f {
  let delta = b - a;
  return delta / max(length(delta), 1e-20);
}`,
        body: `var line = 0u;
  var high = LINE_COUNT;
  while (line + 1u < high) {
    let middle = (line + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) { line = middle; } else { high = middle; }
  }
  let firstVertex = lineOffsets[lineOffsetsOffset + line];
  let lastVertex = lineOffsets[lineOffsetsOffset + line + 1u];
  let vertexCount = lastVertex - firstVertex;
  let sideCount = vertexCount * ROWS_PER_VERTEX;
  let local = index - ringOffsets[ringOffsetsOffset + line];
  if (vertexCount < 2u) {
    output[outputOffset + index * 2u] = 0.0;
    output[outputOffset + index * 2u + 1u] = 0.0;
    return;
  }
  var value = vec2f(0.0);
  if (local < sideCount) {
    value = point(&leftPositions, leftPositionsOffset, firstVertex * ROWS_PER_VERTEX + local);
  } else if (local < sideCount + CAP_ROWS) {
    let cap = local - sideCount;
    let center = point(&positions, positionsOffset, lastVertex - 1u);
    let direction = unitDirection(point(&positions, positionsOffset, lastVertex - 2u), center);
    let normal = vec2f(-direction.y, direction.x);
    ${this._getCapWGSL('end', 'cap', 'center', 'direction', 'normal')}
  } else if (local < 2u * sideCount + CAP_ROWS) {
    let rightRow = local - sideCount - CAP_ROWS;
    value = point(&rightPositions, rightPositionsOffset, (lastVertex * ROWS_PER_VERTEX - 1u) - rightRow);
  } else {
    let cap = local - 2u * sideCount - CAP_ROWS;
    let center = point(&positions, positionsOffset, firstVertex);
    let direction = unitDirection(center, point(&positions, positionsOffset, firstVertex + 1u));
    let normal = vec2f(-direction.y, direction.x);
    ${this._getCapWGSL('start', 'cap', 'center', 'direction', 'normal')}
  }
  output[outputOffset + index * 2u] = value.x;
  output[outputOffset + index * 2u + 1u] = value.y;`
      })
    ];
    nodes.push(
      ...this._getDissolveNodes(
        graph,
        {
          kind: 'polygons',
          positions: rawPositions,
          featureOffsets: identityOffsets,
          polygonOffsets: identityOffsets,
          ringOffsets: rawOffsets,
          sourceIds: lines.sourceIds
        },
        'negative'
      )
    );
    return nodes;
  }

  /**
   * Drops consecutive duplicate vertices (and, for rings, a closing vertex equal to the first)
   * with a keep flag, an inclusive scan and a scatter, publishing compacted positions and offsets.
   * Rows past the last offset are unused.
   */
  private _getCompactionNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    kind: 'line' | 'ring',
    positions: GraphDataView<'float32x2'>,
    offsets: GraphDataView<'uint32'>,
    dropClosingVertex: boolean
  ): {
    nodes: GPUCommandNode<Parameters>[];
    positions: GraphDataView<'float32x2'>;
    offsets: GraphDataView<'uint32'>;
  } {
    const {id} = this;
    const prefix = `${id}-${kind}-compact`;
    const keep = createTransientView(graph, `${prefix}-keep`, 'uint32', positions.length);
    const kept = createTransientView(graph, `${prefix}-kept`, 'uint32', positions.length);
    const compactPositions = createTransientView(
      graph,
      `${prefix}-positions`,
      'float32x2',
      positions.length
    );
    const compactOffsets = createTransientView(
      graph,
      `${prefix}-offsets`,
      'uint32',
      offsets.length
    );
    const lineCount = offsets.length - 1;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-flags`,
        operation: OPERATION,
        variant: `${kind}-compact-flags`,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'keep', view: keep, type: 'u32', access: 'read_write'}
        ],
        invocationCount: positions.length,
        declarations: `const LINE_COUNT: u32 = ${lineCount}u;
const DROP_CLOSING: bool = ${dropClosingVertex};

fn isSame(first: u32, second: u32) -> bool {
  return positions[positionsOffset + 2u * first] == positions[positionsOffset + 2u * second] &&
    positions[positionsOffset + 2u * first + 1u] == positions[positionsOffset + 2u * second + 1u];
}`,
        body: `var line = 0u;
  var high = LINE_COUNT;
  while (line + 1u < high) {
    let middle = (line + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) { line = middle; } else { high = middle; }
  }
  let begin = offsets[offsetsOffset + line];
  let end = offsets[offsetsOffset + line + 1u];
  var flag = index < end;
  if (flag && index > begin && isSame(index - 1u, index)) { flag = false; }
  if (flag && DROP_CLOSING && index + 1u == end && index > begin && isSame(begin, index)) { flag = false; }
  keep[keepOffset + index] = select(0u, 1u, flag);`
      }),
      ...new GPUScan({
        id: `${prefix}-scan`,
        input: keep,
        output: kept,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-offsets`,
        operation: OPERATION,
        variant: `${kind}-compact-offsets`,
        bindings: [
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'kept', view: kept, type: 'u32', access: 'read'},
          {name: 'output', view: compactOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: offsets.length,
        body: `let begin = offsets[offsetsOffset + index];
  output[outputOffset + index] = select(kept[keptOffset + begin - 1u], 0u, begin == 0u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-scatter`,
        operation: OPERATION,
        variant: `${kind}-compact-scatter`,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'keep', view: keep, type: 'u32', access: 'read'},
          {name: 'kept', view: kept, type: 'u32', access: 'read'},
          {name: 'output', view: compactPositions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: positions.length,
        body: `if (keep[keepOffset + index] == 0u) { return; }
  let row = kept[keptOffset + index] - 1u;
  output[outputOffset + 2u * row] = positions[positionsOffset + 2u * index];
  output[outputOffset + 2u * row + 1u] = positions[positionsOffset + 2u * index + 1u];`
      })
    ];
    return {nodes, positions: compactPositions, offsets: compactOffsets};
  }

  private _getCapWGSL(
    end: 'start' | 'end',
    cap: string,
    center: string,
    direction: string,
    normal: string
  ): string {
    if (this.capStyle === 'round') {
      const sign = '-';
      const start = end === 'end' ? normal : `-${normal}`;
      return `let angle = ${sign}PI * f32(${cap} + 1u) / f32(CAP_ROWS + 1u);
    let c = cos(angle); let s = sin(angle);
    let radial = vec2f(c * (${start}).x - s * (${start}).y, s * (${start}).x + c * (${start}).y);
    value = ${center} + radial * parameters[parametersOffset];`;
    }
    if (this.capStyle === 'square') {
      const tangent = end === 'end' ? direction : `-${direction}`;
      const side =
        end === 'end'
          ? `select(${normal}, -${normal}, ${cap} == 1u)`
          : `select(-${normal}, ${normal}, ${cap} == 1u)`;
      return `value = ${center} + (${tangent} + ${side}) * parameters[parametersOffset];`;
    }
    return `value = ${center};`;
  }

  private _getDissolveNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    raw: GPUPackedGeometryPort<GPUPolygonGeometryPort>,
    fillRule: GPUPolygonOverlayFillRule
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    return new GPUPolygonOverlay({
      id: `${id}-repair`,
      spatialContext: this.spatialContext,
      left: raw,
      operation: 'dissolve',
      fillRule,
      capacity: props.capacity,
      vertexTolerance: props.vertexTolerance,
      output: props.output,
      spatialSort: props.spatialSort,
      leafCapacity: props.leafCapacity
    }).getCommandNodes(graph);
  }
}
