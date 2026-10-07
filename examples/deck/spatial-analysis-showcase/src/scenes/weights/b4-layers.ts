// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {SpatialAnalysisResources} from '../../engine/resources';
import type {Geography} from './b4-geography';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisPointLayerProps,
  type SpatialAnalysisSegmentLayerProps
} from '../../engine/layers';

/**
 * Chapter-local layers of the weights scenes. Both reuse the style uniform, value lookup and ramp
 * code of the shared layers by keeping the shared shader prefix and appending their own entry
 * points, so a colour never leaves the GPU and every colormap of the shared layers works.
 */

const SEGMENT_MARKER = '@group(0) @binding(auto) var<storage, read> segmentPositions';
const POINT_MARKER = '@group(0) @binding(auto) var<storage, read> pointPositions';

const POLYGON_BODY = /* wgsl */ `
@group(0) @binding(auto) var<storage, read> polygonVertices: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> polygonFeatures: array<u32>;

struct PolygonVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> PolygonVertexOutput {
  var output: PolygonVertexOutput;
  let source = polygonVertices[vertexIndex];
  let color = getSpatialAnalysisColor(getSpatialAnalysisValueRow(polygonFeatures[vertexIndex]));
  if (source.x != source.x || color.a <= 0.0) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    return output;
  }
  output.position = projectSpatialAnalysisPosition(getSpatialAnalysisPosition(source));
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: PolygonVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * spatialAnalysisStyle.opacity);
}
`;

/** Props of {@link B4PolygonFillLayer}. */
export type B4PolygonFillLayerProps = Omit<SpatialAnalysisSegmentLayerProps, 'segments'> & {
  /** `float32x2` triangle-list vertices in planar metres. */
  triangles: Buffer;
  /** `uint32` feature (value row) of every triangle vertex. */
  features: Buffer;
  /** Number of triangle vertices (three per triangle). */
  triangleVertexCount: number;
};

/**
 * Filled polygons coloured per feature straight from a GPU value buffer. The mesh is triangulated
 * once on the CPU; a recolour is a buffer write plus a draw. Reads `values[feature]` through the
 * shared style (any ramp, category palette or mask), and alpha 0 hides a feature.
 */
export class B4PolygonFillLayer extends SpatialAnalysisSegmentLayer {
  static override layerName = 'B4PolygonFillLayer';

  constructor(props: B4PolygonFillLayerProps) {
    super({
      ...props,
      segments: props.triangles,
      instanceCount: 1
    } as SpatialAnalysisSegmentLayerProps);
  }

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    const split = source.indexOf(SEGMENT_MARKER);
    if (split < 0) throw new Error('B4PolygonFillLayer: the shared segment shader changed');
    return source.slice(0, split) + POLYGON_BODY;
  }

  protected override getVertexCount(): number {
    return (this.props as unknown as B4PolygonFillLayerProps).triangleVertexCount;
  }

  protected override getBindings(placeholder: Buffer): Record<string, Buffer> {
    const props = this.props as unknown as B4PolygonFillLayerProps;
    return {
      ...this.getStyleBindings(placeholder),
      polygonVertices: props.triangles,
      polygonFeatures: props.features
    };
  }
}

/** Shape of a {@link B4CellLayer} cell. */
export type B4CellShape = 'square' | 'hexagon';

/** Props of {@link B4CellLayer}. */
export type B4CellLayerProps = Omit<SpatialAnalysisPointLayerProps, 'radiusPixels'> & {
  /** Cell shape. Baked into the shader, so changing it needs a new layer id. */
  shape: B4CellShape;
  /** Half the cell width (square) or the centre-to-vertex radius (hexagon), in planar metres. */
  radiusMeters: number;
};

const SQUARE_BODY = /* wgsl */ `
@group(0) @binding(auto) var<storage, read> pointPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pointIds: array<u32>;

struct CellVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> CellVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: CellVertexOutput;
  var row = instanceIndex;
  if (spatialAnalysisStyle.useIds != 0u) {
    row = pointIds[instanceIndex];
  }
  let source = pointPositions[row];
  let color = getSpatialAnalysisColor(getSpatialAnalysisValueRow(row));
  if (source.x != source.x || source.y != source.y || color.a <= 0.0) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    output.corner = vec2<f32>(0.0);
    return output;
  }
  let corner = corners[vertexIndex];
  output.position = projectSpatialAnalysisPosition(source + corner * spatialAnalysisStyle.sizePixels);
  output.corner = corner;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: CellVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * spatialAnalysisStyle.opacity);
}
`;

// A flat-topped hexagon of circumradius 1 spans x in [-1, 1] and y in [-0.866, 0.866]. The quad
// covers [-1, 1]^2 and the fragment shader keeps the hexagon.
const HEXAGON_BODY = SQUARE_BODY.replace(
  'return vec4<f32>(input.color.rgb, input.color.a * spatialAnalysisStyle.opacity);',
  `let p = abs(input.corner);
  let inside = p.y <= 0.8660254 && (0.8660254 * p.x + 0.5 * p.y) <= 0.8660254;
  if (!inside) { discard; }
  return vec4<f32>(input.color.rgb, input.color.a * spatialAnalysisStyle.opacity);`
);

/**
 * Square or hexagonal cells at GPU-resident centres, sized in metres so they tile at every zoom.
 * Extends the shared point layer: `valueIndices`, `ids`, colormaps and the no-data rules all work.
 */
export class B4CellLayer extends SpatialAnalysisPointLayer {
  static override layerName = 'B4CellLayer';

  constructor(props: B4CellLayerProps) {
    super({...props, radiusPixels: props.radiusMeters} as SpatialAnalysisPointLayerProps);
  }

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    const split = source.indexOf(POINT_MARKER);
    if (split < 0) throw new Error('B4CellLayer: the shared point shader changed');
    const shape = (this.props as unknown as B4CellLayerProps).shape;
    return source.slice(0, split) + (shape === 'hexagon' ? HEXAGON_BODY : SQUARE_BODY);
  }
}

/** GPU buffers of a geography's polygon fill and outlines. */
export type GeographyBuffers = {
  triangles: Buffer;
  features: Buffer;
  triangleVertexCount: number;
  outline: Buffer;
  outlineSegmentCount: number;
};

/** Uploads the triangle mesh and the outline segments of a geography. */
export function createGeographyBuffers(
  resources: SpatialAnalysisResources,
  geography: Geography,
  id: string
): GeographyBuffers {
  return {
    triangles: resources.createBuffer(`${id}-triangles`, geography.triangles),
    features: resources.createBuffer(`${id}-triangle-features`, geography.triangleFeatures),
    triangleVertexCount: geography.triangleFeatures.length,
    outline: resources.createBuffer(`${id}-outline`, geography.outlineSegments),
    outlineSegmentCount: geography.outlineSegments.length / 4
  };
}
