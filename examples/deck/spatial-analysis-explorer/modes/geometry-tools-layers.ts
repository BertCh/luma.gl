// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LayerProps} from '@deck.gl/core';
import type {Buffer, RenderPass} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import {GEOMETRY_COMMON_WGSL, GeometryBaseLayer, type GeometryStyleProps} from './geometry-layers';

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
