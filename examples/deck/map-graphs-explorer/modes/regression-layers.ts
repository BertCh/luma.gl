// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LayerProps} from '@deck.gl/core';
import type {Buffer, RenderPass} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import {GEOMETRY_COMMON_WGSL, GeometryBaseLayer, type GeometryStyleProps} from './geometry-layers';

/**
 * Cell grid layer of the Regression mode: one quad per grid cell, colored from a storage buffer.
 *
 * Instance `i` is the cell at column `i % columns`, row `i / columns` of a grid whose south-west
 * corner is the `positionOffset` prop (planar meters, `METER_OFFSETS`). Its color comes from
 * `values[valueRow * valueStride + valueOffset]`, where `valueRow` is `i` or, with `indices`,
 * `indices[i]` (`0xffffffff` hides the cell). That lets one layer draw a compacted per-row result,
 * one column of an interleaved coefficient table, or a per-zone value, without repacking.
 */
const CELL_SHADER = /* wgsl */ `
${GEOMETRY_COMMON_WGSL}
@group(0) @binding(auto) var<storage, read> cellIndices: array<u32>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let columns = max(geometryStyle.extraWords.x, 1u);
  let stride = max(geometryStyle.extraWords.y, 1u);
  let offset = geometryStyle.extraWords.z;
  var valueRow = instanceIndex;
  if (geometryStyle.extraWords.w != 0u) {
    valueRow = cellIndices[instanceIndex];
    if (valueRow == 0xffffffffu) {
      return getHiddenGeometryVertex();
    }
  }
  let color = getGeometryColor(geometryValues[valueRow * stride + offset]);
  if (color.a <= 0.0) {
    return getHiddenGeometryVertex();
  }
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  let cellSize = geometryStyle.extraFloats.xy;
  let inset = geometryStyle.extraFloats.z;
  let cell = vec2<f32>(f32(instanceIndex % columns), f32(instanceIndex / columns));
  let local = (cell + vec2<f32>(inset) + corners[vertexIndex] * (1.0 - 2.0 * inset)) * cellSize;
  var output: GeometryVertexOutput;
  output.position = projectGeometryPosition(local);
  output.side = 0.0;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: GeometryVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * geometryStyle.opacity);
}
`;

/** Props of {@link CellGridLayer}. */
export type CellGridLayerProps = LayerProps &
  GeometryStyleProps & {
    /** Pass the grid's south-west corner (planar meters) as `positionOffset`. */
    /** Cell width and height in meters. */
    cellSize: number;
    /** Number of grid columns. */
    columns: number;
    /** Number of cells (`columns * rows`). */
    cellCount: number;
    /** Float32 values buffer; required. */
    values: Buffer;
    /** Float32 values per value row (a coefficient table has several). Defaults to 1. */
    valueStride?: number;
    /** Column of the stride to color by. Defaults to 0. */
    valueOffset?: number;
    /** Optional uint32 cell to value-row map; `0xffffffff` hides the cell. */
    indices?: Buffer | null;
    /** Fraction of the cell width left empty on every side. Defaults to 0.04. */
    inset?: number;
  };

/** One quad per grid cell colored from a storage buffer. See the file comment. */
export class CellGridLayer extends GeometryBaseLayer<CellGridLayerProps> {
  static override layerName = 'CellGridLayer';

  protected getShaderSource(): string {
    return CELL_SHADER;
  }
  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      geometryValues: this.props.values,
      cellIndices: this.props.indices ?? placeholder
    };
  }
  protected getValueSource(): number {
    return 2;
  }
  protected override getExtraFloats(): readonly number[] {
    return [this.props.cellSize, this.props.cellSize, this.props.inset ?? 0.04, 0];
  }
  protected override getExtraWords(): readonly number[] {
    return [
      this.props.columns,
      this.props.valueStride ?? 1,
      this.props.valueOffset ?? 0,
      this.props.indices ? 1 : 0
    ];
  }
  protected drawInstances(model: Model, renderPass: RenderPass): void {
    model.setInstanceCount(this.props.cellCount);
    model.draw(renderPass);
  }
}
