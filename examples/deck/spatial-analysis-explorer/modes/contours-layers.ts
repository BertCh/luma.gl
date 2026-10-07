// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, RenderPass} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import {
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisInstanceProps,
  type SpatialAnalysisRasterLayerProps,
  type SpatialAnalysisSegmentLayerProps
} from '../spatial-analysis-layers';

/*
 * Bespoke layers for the Contours mode. They reuse the shared layers' style uniform, bindings and
 * lifecycle and replace only the shader body, so no shared file changes:
 *
 * - `PackedColorRasterLayer` draws a raster whose cells are already packed rgba8 words (the
 *   `colors` output of GPURasterStretch).
 * - `IsobandTriangleLayer` draws the GPUIsobands triangle soup with a GPU-written indirect vertex
 *   count and colors each triangle by its band index through a packed palette.
 * - `ZoneBandRasterLayer` draws the polygon-rasterized fill of the GPUIsobandRings polygons, by band.
 * - `PolylineLayer` draws the stitched GPUIsolines polylines: one instance per vertex, segment
 *   `(vertex i, vertex i + 1)`, skipped where `i + 1` starts the next polyline.
 */

const RASTER_BINDINGS_MARKER = '@group(0) @binding(auto) var<storage, read> rasterBounds';

type LayerState = {model: Model | null; styleBuffer: Buffer | null};

/** Draws packed rgba8 cells (`r | g << 8 | b << 16 | a << 24`); zero is transparent. */
export class PackedColorRasterLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'PackedColorRasterLayer';

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    const patched = source.replace(
      'let color = getSpatialAnalysisColor(getSpatialAnalysisValueRow(u32(row * columns + column)));',
      `let packedColor = styleValues[u32(row * columns + column)];
  let color = vec4<f32>(
    f32(packedColor & 255u),
    f32((packedColor >> 8u) & 255u),
    f32((packedColor >> 16u) & 255u),
    f32(packedColor >> 24u)
  ) / 255.0;`
    );
    if (patched === source) {
      throw new Error('PackedColorRasterLayer could not patch the raster fragment shader');
    }
    return patched;
  }
}

/** Extra props of {@link IsobandTriangleLayer}. */
export type IsobandTriangleLayerProps = SpatialAnalysisRasterLayerProps &
  SpatialAnalysisInstanceProps & {
    /** `float32x2` triangle vertices, three per triangle. */
    triangles: Buffer;
    /** Band index per triangle. */
    triangleBands: Buffer;
    /**
     * Optional facility row per triangle (`GPUNetworkIsochrones` `triangleFacilities`). When set,
     * the triangle color is `palette[facility]` shaded from light (band 0) to dark (last band)
     * instead of the band palette.
     */
    triangleFacilities?: Buffer;
  };

/**
 * Isoband triangles colored by band. `values` is the packed rgba8 palette (uint32), and `extent`
 * a float32 buffer whose first value is the break count: band `b` samples the palette at
 * `(b + 0.5) / (breakCount + 1)`. `drawCommands[drawCommandIndex].vertexCount` must hold `3 *
 * triangleCount`; `positionScale`/`positionOffset` map the contributor's frame to meters. With
 * `triangleFacilities` the color comes from the style `palette` by facility, shaded by band.
 */
export class IsobandTriangleLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'IsobandTriangleLayer';

  constructor(props: IsobandTriangleLayerProps) {
    super(props);
  }

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    const markerIndex = source.indexOf(RASTER_BINDINGS_MARKER);
    if (markerIndex < 0) {
      throw new Error('IsobandTriangleLayer could not find the raster binding block');
    }
    const {triangleFacilities} = this.props as unknown as IsobandTriangleLayerProps;
    const facilityBinding = triangleFacilities
      ? '@group(0) @binding(auto) var<storage, read> bandFacilities: array<u32>;'
      : '';
    const colorSource = triangleFacilities
      ? `let facility = bandFacilities[vertexIndex / 3u];
  let base = spatialAnalysisStyle.palette[facility % max(spatialAnalysisStyle.paletteSize, 1u)];
  let shade = mix(1.15, 0.5, clamp(f32(band) / (breakCount + 1.0), 0.0, 1.0));
  output.color = vec4<f32>(clamp(base.rgb * shade, vec3<f32>(0.0), vec3<f32>(1.0)), base.a);`
      : `let t = clamp((f32(band) + 0.5) / (breakCount + 1.0), 0.0, 0.9999);
  let packedColor = styleValues[u32(t * f32(paletteSize))];
  output.color = vec4<f32>(
    f32(packedColor & 255u),
    f32((packedColor >> 8u) & 255u),
    f32((packedColor >> 16u) & 255u),
    f32(packedColor >> 24u)
  ) / 255.0;`;
    return `${source.slice(0, markerIndex)}
@group(0) @binding(auto) var<storage, read> bandTriangles: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> bandIds: array<u32>;
${facilityBinding}

struct BandVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> BandVertexOutput {
  var output: BandVertexOutput;
  let band = bandIds[vertexIndex / 3u];
  let breakCount = max(styleExtent[0], 0.0);
  let paletteSize = max(arrayLength(&styleValues), 1u);
  ${colorSource}
  output.position = projectSpatialAnalysisPosition(getSpatialAnalysisPosition(bandTriangles[vertexIndex]));
  return output;
}

@fragment fn fragmentMain(input: BandVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * spatialAnalysisStyle.opacity);
}
`;
  }

  protected override getBindings(placeholder: Buffer): Record<string, Buffer> {
    const {triangles, triangleBands, triangleFacilities} = this
      .props as unknown as IsobandTriangleLayerProps;
    return {
      ...this.getStyleBindings(placeholder),
      bandTriangles: triangles,
      bandIds: triangleBands,
      ...(triangleFacilities ? {bandFacilities: triangleFacilities} : {})
    };
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as LayerState;
    if (!model || !styleBuffer) return;
    this.writeLayerStyle(styleBuffer);
    const {drawCommands, drawCommandIndex = 0} = this.props as SpatialAnalysisInstanceProps;
    if (!drawCommands) return;
    // Bind pipeline and bindings with an empty draw, then replay the GPU-written record.
    model.setInstanceCount(0);
    model.draw(renderPass);
    drawCommands.draw(renderPass, drawCommandIndex);
  }
}

/** Extra props of {@link PolylineLayer}. */
export type PolylineLayerProps = SpatialAnalysisSegmentLayerProps &
  SpatialAnalysisInstanceProps & {
    /** `uint32` vertex offset of each polyline (`polylineCount + 1` valid rows). */
    polylineOffsets: Buffer;
  };

/**
 * Stitched polylines as screen-space-width lines. `segments` is the `float32x2` vertex buffer,
 * `extent` the one-word `polylineCount` buffer, `valueIndices` the per-polyline level buffer and
 * `values` a uint32 per-level style table (`category` colormap). The instance count (the vertex
 * count) comes from `drawCommands`; the last vertex of each polyline draws no segment.
 */
export class PolylineLayer extends SpatialAnalysisSegmentLayer {
  static override layerName = 'PolylineLayer';

  constructor(props: PolylineLayerProps) {
    super(props);
  }

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    let patched = source.replace(
      'var<storage, read> segmentPositions: array<vec4<f32>>',
      'var<storage, read> segmentPositions: array<vec2<f32>>'
    );
    patched = patched.replace(
      /let segment = segmentPositions\[row\];\s*var color = getSpatialAnalysisColor\(getSpatialAnalysisValueRow\(row\)\);/,
      `let polylineCount = bitcast<u32>(styleExtent[0]);
  var segment = vec4<f32>(0.0);
  var polylineValid = false;
  var polylineLevel = 0u;
  if (polylineCount > 0u) {
    // Last polyline whose first vertex is at or before this instance.
    var low = 0u;
    var high = polylineCount;
    while (low + 1u < high) {
      let middle = (low + high) / 2u;
      if (segmentIds[middle] <= row) {
        low = middle;
      } else {
        high = middle;
      }
    }
    if (row + 1u < segmentIds[low + 1u]) {
      segment = vec4<f32>(segmentPositions[row], segmentPositions[row + 1u]);
      polylineLevel = styleValueIndices[low];
      polylineValid = true;
    }
  }
  var color = getSpatialAnalysisColor(polylineLevel);`
    );
    patched = patched.replace('segment.x != segment.x || segment.z != segment.z', '!polylineValid');
    if (!patched.includes('polylineValid') || patched.includes('segment.x != segment.x')) {
      throw new Error('PolylineLayer could not patch the segment vertex shader');
    }
    return patched;
  }

  protected override getBindings(placeholder: Buffer): Record<string, Buffer> {
    const {polylineOffsets} = this.props as unknown as PolylineLayerProps;
    return {...super.getBindings(placeholder), segmentIds: polylineOffsets};
  }
}

/** Extra props of {@link ZoneBandRasterLayer}. */
export type ZoneBandRasterLayerProps = SpatialAnalysisRasterLayerProps &
  SpatialAnalysisInstanceProps & {
    /** `uint32` band per zone row (for example the polygon groups of the ring assembly). */
    zoneBands: Buffer;
    /** Packed rgba8 `uint32` band palette (a ramp sampled by band). */
    bandPalette: Buffer;
  };

/**
 * Draws a `GPUPolygonRasterization` zone raster (`values`, uint32, `0xffffffff` where no polygon
 * covers the cell) colored by band: the zone row indexes `zoneBands`, and the band samples the
 * packed palette (`bandPalette`, uint32) at `(band + 0.5) / (breakCount + 1)`. `extent` is a float32
 * buffer whose first value is the break count. Cells are drawn as the exact even-odd fill of the
 * GPU-assembled shell and hole rings, at raster resolution.
 */
export class ZoneBandRasterLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'ZoneBandRasterLayer';

  constructor(props: ZoneBandRasterLayerProps) {
    super(props);
  }

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    const markerIndex = source.indexOf(RASTER_BINDINGS_MARKER);
    if (markerIndex < 0) {
      throw new Error('ZoneBandRasterLayer could not find the raster binding block');
    }
    const withBindings = `${source.slice(0, markerIndex)}
@group(0) @binding(auto) var<storage, read> zoneBandIds: array<u32>;
@group(0) @binding(auto) var<storage, read> zoneBandPalette: array<u32>;
${source.slice(markerIndex)}`;
    const patched = withBindings.replace(
      'let color = getSpatialAnalysisColor(getSpatialAnalysisValueRow(u32(row * columns + column)));',
      `let zone = styleValues[u32(row * columns + column)];
  if (zone == 0xffffffffu) {
    discard;
  }
  let band = zoneBandIds[zone];
  let breakCount = max(styleExtent[0], 0.0);
  let paletteSize = max(arrayLength(&zoneBandPalette), 1u);
  let t = clamp((f32(band) + 0.5) / (breakCount + 1.0), 0.0, 0.9999);
  let packedColor = zoneBandPalette[u32(t * f32(paletteSize))];
  let color = vec4<f32>(
    f32(packedColor & 255u),
    f32((packedColor >> 8u) & 255u),
    f32((packedColor >> 16u) & 255u),
    f32(packedColor >> 24u)
  ) / 255.0;`
    );
    if (patched === withBindings) {
      throw new Error('ZoneBandRasterLayer could not patch the raster fragment shader');
    }
    return patched;
  }

  protected override getBindings(placeholder: Buffer): Record<string, Buffer> {
    const {zoneBands, bandPalette} = this.props as unknown as ZoneBandRasterLayerProps;
    return {
      ...super.getBindings(placeholder),
      zoneBandIds: zoneBands,
      zoneBandPalette: bandPalette
    };
  }
}
