// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer, LayerProps} from '@deck.gl/core';
import type {Buffer, CommandEncoder, RenderPass} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUHilbertKeys,
  GPU_HILBERT_MAXIMUM_ORDER
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  getGPUShapeGeneratorParameterValues,
  getGPUShapeVertexCount,
  GPUShapeGenerator,
  GPU_SHAPE_GENERATOR_PARAMETER_LENGTH,
  type GPUShapeType
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {SpatialAnalysisBikeParking} from '../spatial-analysis-data';
import type {SpatialAnalysisModeContext} from '../spatial-analysis-mode';
import {formatCount, type SpatialAnalysisResources} from '../spatial-analysis-resources';
import {
  GEOMETRY_COMMON_WGSL,
  GeometryBaseLayer,
  PolygonFanLayer,
  type GeometryStyleProps
} from './geometry-layers';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';

/**
 * Bespoke layer of the Geometry tools mode: {@link TriangleListLayer} draws a non-indexed triangle
 * list of `float32x2` positions straight from a GPU buffer, for example the triangles that
 * `GPUOutlineGeometry` writes.
 */

const TRIANGLE_SHADER = /* wgsl */ `
${GEOMETRY_COMMON_WGSL}
@group(0) @binding(auto) var<storage, read> triangleVertices: array<vec2<f32>>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let position = triangleVertices[instanceIndex * 3u + vertexIndex];
  if (position.x != position.x || position.y != position.y) {
    return getHiddenGeometryVertex();
  }
  var output: GeometryVertexOutput;
  output.position = projectGeometryPosition(position);
  output.side = 0.0;
  output.color = geometryStyle.color;
  return output;
}

@fragment fn fragmentMain(input: GeometryVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * geometryStyle.opacity);
}
`;

/** Props of {@link TriangleListLayer}. */
export type TriangleListLayerProps = LayerProps &
  GeometryStyleProps & {
    /** `float32x2` triangle corners, three rows per triangle. */
    positions: Buffer;
    /** Number of triangles to draw. */
    triangleCount: number;
  };

/** Draws a triangle list of GPU-resident positions in one flat color. */
export class TriangleListLayer extends GeometryBaseLayer<TriangleListLayerProps> {
  static override layerName = 'TriangleListLayer';

  protected getShaderSource(): string {
    return TRIANGLE_SHADER;
  }
  protected override getVertexCount(): number {
    return 3;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {geometryValues: placeholder, triangleVertices: this.props.positions};
  }
  protected getValueSource(): number {
    return 0;
  }
  protected drawInstances(model: Model, renderPass: RenderPass): void {
    model.setInstanceCount(this.props.triangleCount);
    model.draw(renderPass);
  }
}

const SHAPE_TYPES: readonly GPUShapeType[] = ['circle', 'sector', 'ellipse'];
/** Compile-time maximum of the segment slider. */
const MAXIMUM_SEGMENTS = 128;

/** What the Geometry tools mode needs from the shape and Hilbert overlay. */
export type ShapeOverlay = {
  /** Compiled graphs the overlay encodes. */
  getCompiledGraphs: () => CompiledGPUCommandGraph<void>[];
  /** Encodes the static Hilbert graph once and the shape graph whenever a control moved. */
  encode: (commandEncoder: CommandEncoder, frameIndex: number) => void;
  /** Layers of the overlay. */
  getLayers: (coordinateOrigin: [number, number, number]) => Layer[];
  /** Stops readbacks (buffers and graphs are released by the owning resources). */
  stop: () => void;
};

/**
 * Circles, sectors and ellipses around the bike-parking points (`GPUShapeGenerator`) colored by
 * the Hilbert curve position of their centers (`GPUHilbertKeys`), with the curve itself drawn
 * through the points in curve order. The shape kind is compiled up front (three graphs); radius,
 * segment count, sector sweep and Hilbert order are buffer writes.
 */
export function createShapeOverlay(
  context: SpatialAnalysisModeContext,
  resources: SpatialAnalysisResources,
  parking: SpatialAnalysisBikeParking
): ShapeOverlay {
  const {device} = context;
  const pointCount = parking.positions.length / 2;
  const maximumVertices = getGPUShapeVertexCount('sector', MAXIMUM_SEGMENTS);
  const slotCount = pointCount * maximumVertices;

  // State.
  let shape: GPUShapeType = 'circle';
  let radius = 180;
  let segments = 40;
  let sweep = 110;
  let hilbertOrder = 6;
  let showShapes = true;
  let showCurve = true;
  let shapeDirty = true;
  let curveRatio = NaN;

  // Per-feature inputs: radii follow the parking spaces, sectors and ellipses get a hashed angle.
  let meanSpaces = 0;
  for (const spaces of parking.spaces) meanSpaces += spaces;
  meanSpaces = Math.max(meanSpaces / Math.max(pointCount, 1), 1);
  const circleRadii = new Float32Array(pointCount);
  const ellipseRadii = new Float32Array(pointCount * 2);
  const rotations = new Float32Array(pointCount);
  const startBearings = new Float32Array(pointCount);
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let row = 0; row < pointCount; row++) {
    const factor = Math.min(Math.max(Math.sqrt(parking.spaces[row] / meanSpaces), 0.5), 2.5);
    circleRadii[row] = 100 * factor;
    ellipseRadii[2 * row] = 160 * factor;
    ellipseRadii[2 * row + 1] = 60 * factor;
    const hash = Math.imul(row + 1, 2654435761) >>> 0;
    rotations[row] = (hash / 4294967296) * 180;
    startBearings[row] = ((hash >>> 8) / 16777216) * 360;
    minimumX = Math.min(minimumX, parking.positions[2 * row]);
    maximumX = Math.max(maximumX, parking.positions[2 * row]);
    minimumY = Math.min(minimumY, parking.positions[2 * row + 1]);
    maximumY = Math.max(maximumY, parking.positions[2 * row + 1]);
  }
  const writeBearings = () => {
    const bearings = new Float32Array(pointCount * 2);
    for (let row = 0; row < pointCount; row++) {
      bearings[2 * row] = startBearings[row];
      bearings[2 * row + 1] = startBearings[row] + sweep;
    }
    bearingsBuffer.write(bearings);
  };

  // Inputs.
  const centersBuffer = resources.createBuffer('shape-centers', parking.positions);
  const circleRadiiBuffer = resources.createBuffer('shape-circle-radii', circleRadii);
  const ellipseRadiiBuffer = resources.createBuffer('shape-ellipse-radii', ellipseRadii);
  const rotationsBuffer = resources.createBuffer('shape-rotations', rotations);
  const bearingsBuffer = resources.createBuffer('shape-bearings', pointCount * 8);
  const shapeParameters = resources.createParameterBuffer(
    'shape-parameters',
    'float32',
    GPU_SHAPE_GENERATOR_PARAMETER_LENGTH
  );
  const padding = 50;
  const hilbertBounds = resources.createParameterBuffer(
    'hilbert-bounds',
    'float32',
    4,
    Float32Array.of(minimumX - padding, minimumY - padding, maximumX + padding, maximumY + padding)
  );
  const colorParameters = resources.createParameterBuffer('hilbert-color-parameters', 'float32', 2);

  // Outputs.
  const ringPositions = resources.createBuffer('shape-positions', slotCount * 8);
  const ringOffsets = resources.createBuffer('shape-offsets', (pointCount + 1) * 4);
  const ringVertexCount = resources.createBuffer('shape-vertex-count', 4);
  const hilbertKeys = resources.createBuffer('hilbert-keys', pointCount * 4);
  const hilbertRows = resources.createBuffer('hilbert-rows', pointCount * 4);
  const orderValues = resources.createBuffer('hilbert-order-values', pointCount * 4);
  const ringSegments = resources.createBuffer('shape-segments', slotCount * 16);
  const fanSegments = resources.createBuffer('shape-fan-segments', slotCount * 16);
  const featureRows = resources.createBuffer('shape-feature-rows', slotCount * 4);
  const curveSegments = resources.createBuffer('hilbert-curve-segments', pointCount * 16);
  const curveValues = resources.createBuffer('hilbert-curve-values', pointCount * 4);

  // Shape graphs, one per kind (compile-time), all writing the same output buffers.
  const shapeGraphs = Object.fromEntries(
    SHAPE_TYPES.map(kind => {
      const graph = new GPUCommandGraph<void>(device, {id: `geometry-tools-${kind}`});
      graph.add(
        new GPUShapeGenerator({
          id: `shape-${kind}`,
          shape: kind,
          coordinateSystem: 'planar',
          maximumSegments: MAXIMUM_SEGMENTS,
          centers: importGraphBuffer(graph, 'centers', centersBuffer, 'float32x2', pointCount),
          radii:
            kind === 'ellipse'
              ? importGraphBuffer(graph, 'radii', ellipseRadiiBuffer, 'float32x2', pointCount)
              : importGraphBuffer(graph, 'radii', circleRadiiBuffer, 'float32', pointCount),
          bearings:
            kind === 'sector'
              ? importGraphBuffer(graph, 'bearings', bearingsBuffer, 'float32x2', pointCount)
              : undefined,
          rotations:
            kind === 'ellipse'
              ? importGraphBuffer(graph, 'rotations', rotationsBuffer, 'float32', pointCount)
              : undefined,
          parameters: shapeParameters.importToGraph(graph),
          output: {
            positions: importGraphBuffer(graph, 'positions', ringPositions, 'float32x2', slotCount),
            offsets: importGraphBuffer(graph, 'offsets', ringOffsets, 'uint32', pointCount + 1),
            vertexCount: importGraphBuffer(graph, 'vertex-count', ringVertexCount, 'uint32', 1)
          }
        })
      );
      return [kind, resources.track(graph.compile())];
    })
  ) as Record<GPUShapeType, CompiledGPUCommandGraph<void>>;

  // Hilbert graph: keys and the sorted permutation of the point centers (static), the curve
  // segments through the sorted points, all at order 16.
  const hilbertGraph = new GPUCommandGraph<void>(device, {id: 'geometry-tools-hilbert'});
  {
    const centers = importGraphBuffer(
      hilbertGraph,
      'centers',
      centersBuffer,
      'float32x2',
      pointCount
    );
    const keys = importGraphBuffer(hilbertGraph, 'keys', hilbertKeys, 'uint32', pointCount);
    const rows = importGraphBuffer(hilbertGraph, 'rows', hilbertRows, 'uint32', pointCount);
    hilbertGraph.add(
      new GPUHilbertKeys({
        id: 'hilbert',
        order: GPU_HILBERT_MAXIMUM_ORDER,
        points: centers,
        bounds: hilbertBounds.importToGraph(hilbertGraph),
        output: {keys, sortedRows: rows}
      })
    );
    addKernelPass(hilbertGraph, {
      id: 'hilbert-curve',
      invocationCount: pointCount,
      bindings: [
        {name: 'rows', view: rows, type: 'u32', access: 'read'},
        {name: 'centers', view: centers, type: 'f32', access: 'read'},
        {
          name: 'segments',
          view: importGraphBuffer(
            hilbertGraph,
            'curve-segments',
            curveSegments,
            'float32',
            pointCount * 4
          ),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'values',
          view: importGraphBuffer(hilbertGraph, 'curve-values', curveValues, 'float32', pointCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `const POINT_COUNT: u32 = ${pointCount}u;`,
      body: /* wgsl */ `
  let zero = f32(index) * 0.0;
  var start = vec2<f32>(zero / zero);
  var end = start;
  if (index + 1u < POINT_COUNT) {
    let first = rows[rowsOffset + index];
    let second = rows[rowsOffset + index + 1u];
    start = vec2<f32>(centers[centersOffset + first * 2u], centers[centersOffset + first * 2u + 1u]);
    end = vec2<f32>(centers[centersOffset + second * 2u], centers[centersOffset + second * 2u + 1u]);
  }
  segments[segmentsOffset + index * 4u] = start.x;
  segments[segmentsOffset + index * 4u + 1u] = start.y;
  segments[segmentsOffset + index * 4u + 2u] = end.x;
  segments[segmentsOffset + index * 4u + 3u] = end.y;
  values[valuesOffset + index] = f32(index) / f32(max(POINT_COUNT - 1u, 1u));`
    });
  }
  const compiledHilbert = resources.track(hilbertGraph.compile());

  // Derive graph: ring segments (and fan triangles) from the rings and the order coloring from
  // the keys, re-encoded when a control moved.
  const deriveGraph = new GPUCommandGraph<void>(device, {id: 'geometry-tools-shape-derive'});
  {
    const centers = importGraphBuffer(
      deriveGraph,
      'centers',
      centersBuffer,
      'float32x2',
      pointCount
    );
    addKernelPass(deriveGraph, {
      id: 'ring-segments',
      invocationCount: slotCount,
      bindings: [
        {
          name: 'positions',
          view: importGraphBuffer(deriveGraph, 'positions', ringPositions, 'float32x2', slotCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'offsets',
          view: importGraphBuffer(deriveGraph, 'offsets', ringOffsets, 'uint32', pointCount + 1),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'vertexCount',
          view: importGraphBuffer(deriveGraph, 'vertex-count', ringVertexCount, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {name: 'centers', view: centers, type: 'f32', access: 'read'},
        {
          name: 'ringSegments',
          view: importGraphBuffer(
            deriveGraph,
            'ring-segments',
            ringSegments,
            'float32',
            slotCount * 4
          ),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'fanSegments',
          view: importGraphBuffer(
            deriveGraph,
            'fan-segments',
            fanSegments,
            'float32',
            slotCount * 4
          ),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'featureRows',
          view: importGraphBuffer(deriveGraph, 'feature-rows', featureRows, 'uint32', slotCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      declarations: `const POINT_COUNT: u32 = ${pointCount}u;`,
      body: /* wgsl */ `
  // Ring vertices are packed at a stride of V = offsets[1]; slot i joins vertex i to i + 1 except
  // for the last vertex of each ring, and slots past the written vertices are hidden.
  let total = vertexCount[vertexCountOffset];
  let stride = max(offsets[offsetsOffset + 1u], 1u);
  let feature = min(index / stride, POINT_COUNT - 1u);
  let center = vec2<f32>(centers[centersOffset + feature * 2u], centers[centersOffset + feature * 2u + 1u]);
  let zero = f32(index) * 0.0;
  var start = vec2<f32>(zero / zero);
  var end = start;
  var fanStart = center;
  var fanEnd = center;
  if (index < total && (index + 1u) % stride != 0u) {
    start = vec2<f32>(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
    end = vec2<f32>(positions[positionsOffset + index * 2u + 2u], positions[positionsOffset + index * 2u + 3u]);
    fanStart = start;
    fanEnd = end;
  }
  ringSegments[ringSegmentsOffset + index * 4u] = start.x;
  ringSegments[ringSegmentsOffset + index * 4u + 1u] = start.y;
  ringSegments[ringSegmentsOffset + index * 4u + 2u] = end.x;
  ringSegments[ringSegmentsOffset + index * 4u + 3u] = end.y;
  fanSegments[fanSegmentsOffset + index * 4u] = fanStart.x;
  fanSegments[fanSegmentsOffset + index * 4u + 1u] = fanStart.y;
  fanSegments[fanSegmentsOffset + index * 4u + 2u] = fanEnd.x;
  fanSegments[fanSegmentsOffset + index * 4u + 3u] = fanEnd.y;
  featureRows[featureRowsOffset + index] = feature;`
    });
    addKernelPass(deriveGraph, {
      id: 'hilbert-order-values',
      invocationCount: pointCount,
      bindings: [
        {
          name: 'keys',
          view: importGraphBuffer(deriveGraph, 'keys', hilbertKeys, 'uint32', pointCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'colorParameters',
          view: colorParameters.importToGraph(deriveGraph),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'orderValues',
          view: importGraphBuffer(deriveGraph, 'order-values', orderValues, 'float32', pointCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  // Order-k index is the order-16 index shifted right by 2 * (16 - k) bits.
  let shift = u32(colorParameters[colorParametersOffset]);
  orderValues[orderValuesOffset + index] =
    f32(keys[keysOffset + index] >> shift) * colorParameters[colorParametersOffset + 1u];`
    });
  }
  const compiledDerive = resources.track(deriveGraph.compile());

  // Parameter writes.
  const writeShape = () => {
    shapeParameters.write(
      getGPUShapeGeneratorParameterValues({segmentCount: segments, radiusScale: radius / 100})
    );
    shapeDirty = true;
  };
  const writeColor = () => {
    colorParameters.write(
      Float32Array.of(2 * (GPU_HILBERT_MAXIMUM_ORDER - hilbertOrder), 1 / 4 ** hilbertOrder)
    );
    shapeDirty = true;
  };
  writeBearings();
  writeShape();
  writeColor();

  // Controls.
  context.controls.addSelect<GPUShapeType>({
    label: 'Shape around each bike-parking point (GPUShapeGenerator, compiled up front)',
    options: [
      {value: 'circle', label: 'Circle (radius follows the parking spaces)'},
      {value: 'sector', label: 'Sector (pie slice with a hashed start bearing)'},
      {value: 'ellipse', label: 'Ellipse (hashed rotation)'}
    ],
    value: shape,
    onChange: value => {
      shape = value;
      shapeDirty = true;
      shapeReader.markStale();
      context.updateLayers();
    }
  });
  context.controls.addSlider({
    label: 'Radius (per-frame scale)',
    min: 20,
    max: 500,
    step: 10,
    value: radius,
    format: value => `${value} m for an average site`,
    onChange: value => {
      radius = value;
      writeShape();
    }
  });
  context.controls.addSlider({
    label: 'Segments per ring (per-frame, up to the compiled maximum)',
    min: 3,
    max: MAXIMUM_SEGMENTS,
    step: 1,
    value: segments,
    format: value => `${value}`,
    onChange: value => {
      segments = value;
      writeShape();
      shapeReader.markStale();
    }
  });
  context.controls.addSlider({
    label: 'Sector sweep (bearings buffer write)',
    min: 10,
    max: 360,
    step: 5,
    value: sweep,
    format: value => `${value} degrees`,
    onChange: value => {
      sweep = value;
      writeBearings();
      shapeDirty = true;
    }
  });
  context.controls.addSlider({
    label: 'Hilbert order shown (keys are order 16; coarser orders shift the key)',
    min: 1,
    max: GPU_HILBERT_MAXIMUM_ORDER,
    step: 1,
    value: hilbertOrder,
    format: value => `order ${value}: ${formatCount(4 ** value)} cells`,
    onChange: value => {
      hilbertOrder = value;
      writeColor();
      context.updateLayers();
    }
  });
  for (const [label, getValue, setValue] of [
    ['Shapes colored by Hilbert order', () => showShapes, (value: boolean) => (showShapes = value)],
    [
      'Hilbert curve through the sorted points',
      () => showCurve,
      (value: boolean) => (showCurve = value)
    ]
  ] as const) {
    context.controls.addToggle({
      label,
      value: getValue(),
      onChange: value => {
        setValue(value);
        context.updateLayers();
      }
    });
  }
  context.controls.addNote(
    'Shapes are generated on the GPU around every bike-parking point and colored by the position ' +
      'of the point on a Hilbert curve (viridis, start to end); the thin curve connects the ' +
      'points in curve order, so short hops mean good spatial locality.'
  );
  context.controls.addReadout('Bike-parking points', formatCount(pointCount));
  const ringReadout = context.controls.addReadout('Vertices per ring / total');
  const curveReadout = context.controls.addReadout('Hilbert path vs input-order path length');

  // Readbacks: ring size (when the shape or segments change) and the sorted permutation (once).
  const shapeReader = new SummaryReader(
    resources,
    'geometry-tools-shape-size',
    [
      {buffer: ringOffsets, size: 8},
      {buffer: ringVertexCount, size: 4}
    ],
    bytes => {
      const words = new Uint32Array(bytes);
      ringReadout.setValue(`${words[1]} / ${formatCount(words[2])}`);
    }
  );
  const curveReader = new SummaryReader(
    resources,
    'geometry-tools-hilbert-rows',
    [{buffer: hilbertRows, size: pointCount * 4}],
    bytes => {
      const rows = new Uint32Array(bytes);
      const length = (first: number, second: number) =>
        Math.hypot(
          parking.positions[2 * first] - parking.positions[2 * second],
          parking.positions[2 * first + 1] - parking.positions[2 * second + 1]
        );
      let hilbertLength = 0;
      let inputLength = 0;
      for (let i = 0; i + 1 < pointCount; i++) {
        hilbertLength += length(rows[i], rows[i + 1]);
        inputLength += length(i, i + 1);
      }
      curveRatio = hilbertLength / Math.max(inputLength, 1);
      curveReadout.setValue(
        `${(hilbertLength / 1000).toFixed(1)} km vs ${(inputLength / 1000).toFixed(1)} km ` +
          `(${(100 * curveRatio).toFixed(0)}%)`
      );
    }
  );

  return {
    getCompiledGraphs: () => [...Object.values(shapeGraphs), compiledHilbert, compiledDerive],
    encode(commandEncoder, frameIndex) {
      if (frameIndex < 2) {
        compiledHilbert.encode(commandEncoder, {parameters: undefined});
        curveReader.request(commandEncoder);
        shapeReader.markStale();
      }
      if (shapeDirty || frameIndex < 2) {
        shapeGraphs[shape].encode(commandEncoder, {parameters: undefined});
        compiledDerive.encode(commandEncoder, {parameters: undefined});
        shapeReader.request(commandEncoder);
        shapeDirty = false;
      }
      curveReader.flush(commandEncoder);
      shapeReader.flush(commandEncoder);
    },
    getLayers(coordinateOrigin) {
      const layers: Layer[] = [];
      if (showShapes) {
        layers.push(
          new PolygonFanLayer({
            id: 'geometry-tools-shape-fill',
            coordinateOrigin,
            segments: fanSegments,
            featureRows,
            centroids: centersBuffer,
            instanceCount: slotCount,
            values: orderValues,
            colormap: 'viridis',
            valueRange: [0, 1],
            color: [255, 255, 255, 235]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'geometry-tools-shape-rings',
            coordinateOrigin,
            segments: ringSegments,
            instanceCount: slotCount,
            widthPixels: 1,
            color: [255, 255, 255, 190]
          })
        );
      }
      if (showCurve) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'geometry-tools-hilbert-curve',
            coordinateOrigin,
            segments: curveSegments,
            instanceCount: pointCount,
            widthPixels: 1.5,
            values: curveValues,
            colormap: 'inferno',
            valueRange: [0, 1]
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'geometry-tools-shape-centers',
          coordinateOrigin,
          positions: centersBuffer,
          instanceCount: pointCount,
          radiusPixels: 1.5,
          color: [255, 255, 255, 200]
        })
      );
      return layers;
    },
    stop() {
      shapeReader.stop();
      curveReader.stop();
    }
  };
}
