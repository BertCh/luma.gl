// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import type {GPULineGeometryPort, GPUPolygonGeometryPort} from '../contracts/index';
import {
  GPUGeometryOrientation,
  GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH
} from '../geometry-edit/index';
import {
  getGPUOffsetCurveRowsPerVertex,
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
import {resolveGPUSpatialContext, type GPUSpatialContext} from '../contracts/spatial-context';

const OPERATION = 'GPUBufferSurface';

/** End treatment for buffered lines. */
export type GPUBufferCapStyle = 'round' | 'flat' | 'square';

type PackedLines = Omit<GPULineGeometryPort, 'positions' | 'sourceIds'> & {
  positions: GraphDataView<'float32x2'>;
  sourceIds?: GraphDataView<'uint32'>;
};

type PackedPolygons = Omit<GPUPolygonGeometryPort, 'positions' | 'sourceIds'> & {
  positions: GraphDataView<'float32x2'>;
  sourceIds?: GraphDataView<'uint32'>;
};

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
  geometry: PackedLines | PackedPolygons;
  /** `[distance, mitreLimit, 0, 0]`, matching `GPUOffsetCurve`. */
  parameters: GraphDataView<'float32'>;
  joinStyle?: GPUOffsetCurveJoinStyle;
  capStyle?: GPUBufferCapStyle;
  quadSegments?: number;
  vertexTolerance: number;
  capacity: GPUPolygonOverlayCapacity;
  output: GPUPolygonOverlayOutput;
  uncertainCount?: GraphDataView<'uint32'>;
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
 * Fill rule: the raw offset boundary is dissolved with a WINDING rule, not even-odd, so regions
 * the raw offset covers more than once (hairpins and switchbacks closer than twice the distance,
 * self-crossing lines, polygon slots narrower than twice the distance) stay inside, as in
 * `shapely.buffer`. Polygons are normalized to counter-clockwise shells and clockwise holes and
 * dissolved with `'positive'` winding, which also discards the inverted loops a negative buffer
 * leaves where a neck is thinner than twice the distance; lines emit a clockwise outline and use
 * `'negative'`. Checked against `shapely.buffer` on hairpin, figure-eight, slot and thin-neck
 * shapes with mitre joins.
 *
 * Known limitation: a polygon hole that the offset collapses (hole narrower than twice the
 * distance for a positive buffer) is not removed. The raw offset of the hole keeps its clockwise
 * orientation when it inverts, so it is carved out as a smaller reflected hole instead of
 * disappearing. A "union of simple pieces" construction (polygon union ring strokes) would fix it
 * and is deferred.
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
    this.rowsPerVertex = getGPUOffsetCurveRowsPerVertex(this.joinStyle, this.quadSegments);
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
    if (!Number.isSafeInteger(this.quadSegments) || this.quadSegments < 1) {
      throw new Error(`${id} quadSegments must be a positive integer`);
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
      geometry.sourceIds,
      this.props.parameters
    ]);
    return geometry.kind === 'lines'
      ? this._getLineNodes(graph, geometry)
      : this._getPolygonNodes(graph, geometry);
  }

  private _getPolygonNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    polygons: PackedPolygons
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const prepared = this._getCompactionNodes(
      graph,
      'ring',
      polygons.positions,
      polygons.ringOffsets,
      true
    );
    const normalized = createTransientView(
      graph,
      `${id}-normalized`,
      'float32x2',
      polygons.positions.length
    );
    const orientationParameters = createTransientView(
      graph,
      `${id}-orientation-parameters`,
      'float32',
      GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH
    );
    const offsetParameters = createTransientView(
      graph,
      `${id}-offset-parameters`,
      'float32',
      GPU_OFFSET_CURVE_PARAMETER_LENGTH
    );
    const rawPositions = createTransientView(
      graph,
      `${id}-raw-positions`,
      'float32x2',
      polygons.positions.length * this.rowsPerVertex
    );
    const rawRingOffsets = createTransientView(
      graph,
      `${id}-raw-ring-offsets`,
      'uint32',
      polygons.ringOffsets.length
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      ...prepared.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-polygon-parameters`,
        operation: OPERATION,
        variant: 'polygon-parameters',
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'orientation', view: orientationParameters, type: 'f32', access: 'read_write'},
          {name: 'offset', view: offsetParameters, type: 'f32', access: 'read_write'}
        ],
        invocationCount: GPU_OFFSET_CURVE_PARAMETER_LENGTH,
        body: `orientation[orientationOffset + index] = 0.0;
  offset[offsetOffset + index] = select(parameters[parametersOffset + index], -parameters[parametersOffset], index == 0u);`
      }),
      ...new GPUGeometryOrientation({
        id: `${id}-orient`,
        spatialContext: this.spatialContext,
        positions: prepared.positions,
        ringOffsets: prepared.offsets,
        polygonOffsets: polygons.polygonOffsets,
        mode: 'orient-polygons',
        parameters: orientationParameters,
        output: {positions: normalized}
      }).getCommandNodes(graph),
      ...new GPUOffsetCurve({
        id: `${id}-offset`,
        spatialContext: this.spatialContext,
        positions: normalized,
        pathOffsets: prepared.offsets,
        geometryType: 'rings',
        joinStyle: this.joinStyle,
        quadSegments: this.quadSegments,
        parameters: offsetParameters,
        output: {positions: rawPositions}
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-polygon-offsets`,
        operation: OPERATION,
        variant: 'polygon-offsets',
        bindings: [
          {name: 'input', view: prepared.offsets, type: 'u32', access: 'read'},
          {name: 'output', view: rawRingOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: polygons.ringOffsets.length,
        declarations: `const ROWS_PER_VERTEX: u32 = ${this.rowsPerVertex}u;`,
        body: 'output[outputOffset + index] = input[inputOffset + index] * ROWS_PER_VERTEX;'
      })
    ];
    nodes.push(
      ...this._getDissolveNodes(
        graph,
        {
          kind: 'polygons',
          positions: rawPositions,
          featureOffsets: polygons.featureOffsets,
          polygonOffsets: polygons.polygonOffsets,
          ringOffsets: rawRingOffsets,
          sourceIds: polygons.sourceIds
        },
        'positive'
      )
    );
    return nodes;
  }

  private _getLineNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    lines: PackedLines
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
        pathOffsets: prepared.offsets,
        geometryType: 'lines',
        joinStyle: this.joinStyle,
        quadSegments: this.quadSegments,
        parameters: positiveParameters,
        output: {positions: leftPositions}
      }).getCommandNodes(graph),
      ...new GPUOffsetCurve({
        id: `${id}-right-offset`,
        spatialContext: this.spatialContext,
        positions: prepared.positions,
        pathOffsets: prepared.offsets,
        geometryType: 'lines',
        joinStyle: this.joinStyle,
        quadSegments: this.quadSegments,
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
    const pathCount = offsets.length - 1;
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
        declarations: `const PATH_COUNT: u32 = ${pathCount}u;
const DROP_CLOSING: bool = ${dropClosingVertex};

fn isSame(first: u32, second: u32) -> bool {
  return positions[positionsOffset + 2u * first] == positions[positionsOffset + 2u * second] &&
    positions[positionsOffset + 2u * first + 1u] == positions[positionsOffset + 2u * second + 1u];
}`,
        body: `var path = 0u;
  var high = PATH_COUNT;
  while (path + 1u < high) {
    let middle = (path + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) { path = middle; } else { high = middle; }
  }
  let begin = offsets[offsetsOffset + path];
  let end = offsets[offsetsOffset + path + 1u];
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
    raw: PackedPolygons,
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
      uncertainCount: props.uncertainCount,
      spatialSort: props.spatialSort,
      leafCapacity: props.leafCapacity
    }).getCommandNodes(graph);
  }
}
