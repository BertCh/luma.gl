// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LayerContext, UpdateParameters} from '@deck.gl/core';
import {Buffer, type RenderPass} from '@luma.gl/core';
import {
  SpatialAnalysisBaseLayer,
  getSpatialAnalysisStyleWgsl,
  writeStyle,
  type SpatialAnalysisColor,
  type SpatialAnalysisCommonLayerProps,
  type SpatialAnalysisLayerState,
  type SpatialAnalysisShaderFeatures
} from './layers';

/**
 * Time-windowed trajectories (SYNTHESIS G30, section 1.3 "Trails"): the comet trails of vessels,
 * birds and drifters, drawn from GPU buffers.
 *
 * The layer draws each vertex-to-next segment whose time lies in `[currentTime - tail,
 * currentTime]`, clipped to that window. Age is alpha only: `headAlpha * (1 - age / tail)^2`
 * (0.9 at the head, 0 at the tail), with the width tapering from `headWidthPixels` to
 * `tailWidthPixels`. Optional head dots mark the interpolated position at `currentTime`, with a
 * ground-colour halo, and draw above every trail. A casing (`outlineColor`) is for light grounds
 * only.
 *
 * Time-decay is a draw-time uniform: playback changes `currentTime` (a prop, no rebuild, no
 * buffer write), so stepping the playhead every frame is cheap. Timestamps are `float32`: keep
 * them small (seconds or days since a dataset epoch, below 2^24 for exact seconds).
 *
 * **Feeding it.** `vertices` are planar metres (`float32x2`) of all paths back to back and
 * `timestamps` one `float32` per vertex, in the same units as `currentTime` and `tail`. Paths
 * are given either as `pathOffsets` (path `p` owns vertices `pathOffsets[p]` up to
 * `pathOffsets[p + 1]`; the layer builds the id buffer) or as a per-vertex `pathIds` buffer. From
 * a trajectory dataset (see `poopdeck-ais-us` or `gull-migration`):
 *
 * ```ts
 * const origin = dataset.defaultOrigin;
 * const offsets = dataset.column<Uint32Array>('pathOffsets');
 * const times = Float32Array.from(dataset.column<Uint32Array>('timestamp'), t => t - epoch);
 * new SpatialAnalysisTrailLayer({
 *   id: 'trails',
 *   vertices: resources.createBuffer('vertices', dataset.projectColumn('vertices', origin)),
 *   timestamps: resources.createBuffer('times', times),
 *   vertexCount: times.length,
 *   pathOffsets: offsets,
 *   currentTime: playhead,
 *   tail: 0.06 * loopLength
 * });
 * ```
 *
 * The `b12-tracks` `TrackSet` already has exactly these arrays (`positions`, `timestamps`,
 * `offsets`).
 *
 * **Colour.** The shared style, per vertex (`values` has one entry per vertex, so a speed or
 * heading column colours the step that starts at that vertex) or per path (`colorBy: 'path'`, with
 * `values` holding one entry per path). Segments of different paths are never joined.
 *
 * Storage bindings per stage: 7 with default props (3 style, vertices, times, path ids, `ids`);
 * `instanceChannels` adds one for 8, the ceiling. One extra uniform buffer carries the trail
 * parameters.
 */

/** Props of {@link SpatialAnalysisTrailLayer}. */
export type SpatialAnalysisTrailLayerProps = SpatialAnalysisCommonLayerProps & {
  /** `float32x2` planar metres of every vertex of every path, paths back to back. */
  vertices: Buffer;
  /** `float32` time of each vertex, ascending within a path. */
  timestamps: Buffer;
  /** Number of vertices in `vertices` and `timestamps`. */
  vertexCount: number;
  /** `uint32` path id per vertex. Segments join only vertices with equal ids. */
  pathIds?: Buffer | null;
  /**
   * `pathCount + 1` vertex offsets (`pathOffsets[p]` is the first vertex of path `p`). The layer
   * builds the `pathIds` buffer from it; give either this or `pathIds`. Without either, all
   * vertices are one path.
   */
  pathOffsets?: ArrayLike<number> | null;
  /** The playhead, in the units of `timestamps`. A prop change is a uniform write only. */
  currentTime: number;
  /** Length of the visible tail, in the units of `timestamps`: 6 percent of the loop for fast things, 15-25 percent for slow. */
  tail: number;
  /**
   * What `values` is indexed by: `'vertex'` (default) one entry per vertex, or `'path'` one entry
   * per path (the path id picks the value).
   */
  colorBy?: 'vertex' | 'path';
  /** Width at the head in CSS pixels. Defaults to 2.2. */
  headWidthPixels?: number;
  /** Width at the end of the tail in CSS pixels. Defaults to 1.2. */
  tailWidthPixels?: number;
  /** Alpha at the head. Defaults to 0.9. */
  headAlpha?: number;
  /** Radius in CSS pixels of the head dot at the interpolated position; 0 (default) draws none. */
  headRadiusPixels?: number;
  /** Width of the head dot's ground-colour halo in CSS pixels. Defaults to 1.2. */
  headHaloPixels?: number;
  /** Colour of the head halo: the ground colour. Defaults to the light paper at 0.92. */
  headHaloColor?: SpatialAnalysisColor;
  /**
   * Segments whose time step is longer than this are not drawn (a vessel that went dark, a gap in
   * a GPS fix), in the units of `timestamps`. Unset: every segment is drawn.
   */
  maximumGapTime?: number;
  /** Casing colour (light grounds only; omit on dark grounds). */
  outlineColor?: SpatialAnalysisColor;
  /** Casing width in CSS pixels on each side. Defaults to 0 (no casing). */
  outlineWidthPixels?: number;
};

/**
 * The per-vertex path ids of `pathOffsets`: vertex `v` of path `p` gets `p`. Useful when several
 * layers share one id buffer (`resources.createBuffer('path-ids', getPathIds(offsets, count))`).
 */
export function getPathIds(offsets: ArrayLike<number>, vertexCount: number): Uint32Array {
  const pathIds = new Uint32Array(vertexCount);
  for (let path = 0; path + 1 < offsets.length; path++) {
    const first = Math.max(0, offsets[path]);
    const last = Math.min(vertexCount, offsets[path + 1]);
    for (let vertex = first; vertex < last; vertex++) pathIds[vertex] = path;
  }
  return pathIds;
}

const DEFAULT_HEAD_HALO_COLOR: SpatialAnalysisColor = [243, 239, 230, 235];

/** Shader source of {@link SpatialAnalysisTrailLayer}: one quad per segment, then one per head. */
export function buildTrailShader(features: SpatialAnalysisShaderFeatures): string {
  return /* wgsl */ `
${getSpatialAnalysisStyleWgsl(features)}
@group(0) @binding(auto) var<storage, read> trailVertices: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> trailTimes: array<f32>;
@group(0) @binding(auto) var<storage, read> trailPathIds: array<u32>;
@group(0) @binding(auto) var<storage, read> trailIds: array<u32>;
// [0] currentTime, tail, headAlpha, maximumGap (0 = none); [1] headWidth, tailWidth, headRadius,
// headHalo; [2] headHaloColor; [3] bit patterns of u32: vertexCount, rowCount, usePathIds.
// Counts are a u32 vector (not bitcast floats): some drivers flush f32 denormals on load.
struct TrailParams {
  values: array<vec4<f32>, 3>,
  counts: vec4<u32>,
};
@group(0) @binding(auto) var<uniform> trailParams: TrailParams;

struct TrailVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) local: vec2<f32>,
  @location(2) @interpolate(flat) shape: vec4<f32>,
};

// Tail position of a time: 1 at the playhead, 0 at the end of the tail.
fn getTrailFreshness(time: f32) -> f32 {
  let tail = max(trailParams.values[0].y, 1e-20);
  return clamp(1.0 - (trailParams.values[0].x - time) / tail, 0.0, 1.0);
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> TrailVertexOutput {
  var output: TrailVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.local = vec2<f32>(0.0);
  output.shape = vec4<f32>(0.0);
  let counts = trailParams.counts;
  let rowCount = counts.y;
  let isHead = instanceIndex >= rowCount;
  var row = select(instanceIndex, instanceIndex - rowCount, isHead);
  if (spatialAnalysisStyle.useIds != 0u) {
    row = trailIds[row];
  }
  if (row + 1u >= counts.x) {
    return output;
  }
  if (counts.z != 0u && trailPathIds[row] != trailPathIds[row + 1u]) {
    return output;
  }
  let now = trailParams.values[0].x;
  let tail = trailParams.values[0].y;
  let time0 = trailTimes[row];
  let time1 = trailTimes[row + 1u];
  // Outside the window (or NaN): nothing to draw.
  if (!(time1 >= now - tail) || !(time0 <= now)) {
    return output;
  }
  if (trailParams.values[0].w > 0.0 && time1 - time0 > trailParams.values[0].w) {
    return output;
  }
  let valueRow = getSpatialAnalysisValueRow(row);
  let color = getSpatialAnalysisColor(valueRow);
  if (color.a <= 0.0) {
    return output;
  }
  let point0 = getSpatialAnalysisPosition(trailVertices[row]);
  let point1 = getSpatialAnalysisPosition(trailVertices[row + 1u]);
  let span = max(time1 - time0, 1e-20);
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex];

  if (isHead) {
    // The head dot, at the interpolated position, only on the segment that contains the playhead.
    let radius = trailParams.values[1].z;
    if (radius <= 0.0 || now >= time1) {
      return output;
    }
    let headPoint = mix(point0, point1, clamp((now - time0) / span, 0.0, 1.0));
    let halo = trailParams.values[1].w;
    let extent = radius + halo + 1.0;
    let clip = projectSpatialAnalysisPosition(headPoint);
    output.position = vec4<f32>(clip.xy + project_pixel_size_to_clipspace(vec2<f32>(corner.x * 2.0 - 1.0, corner.y) * extent), clip.z, clip.w);
    output.local = vec2<f32>(corner.x * 2.0 - 1.0, corner.y) * extent;
    output.shape = vec4<f32>(1.0, radius, halo, 0.0);
    output.color = color;
    return output;
  }

  // The part of the segment inside the window [now - tail, now].
  let fraction0 = clamp((now - tail - time0) / span, 0.0, 1.0);
  let fraction1 = clamp((now - time0) / span, 0.0, 1.0);
  let fraction = mix(fraction0, fraction1, corner.x);
  let freshness = getTrailFreshness(time0 + fraction * span);
  let width = mix(trailParams.values[1].y, trailParams.values[1].x, freshness);
  let totalWidth = width + 2.0 * max(spatialAnalysisStyle.outlineWidthPixels, 0.0);
  let startClip = projectSpatialAnalysisPosition(mix(point0, point1, fraction0));
  let endClip = projectSpatialAnalysisPosition(mix(point0, point1, fraction1));
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  // Extend only the ends that are real vertices (not the clipped window ends) so polylines join.
  var extension = 0.0;
  if (corner.x < 0.5 && fraction0 <= 0.0) {
    extension = -0.25 * totalWidth;
  } else if (corner.x > 0.5 && fraction1 >= 1.0) {
    extension = 0.25 * totalWidth;
  }
  let clip = mix(startClip, endClip, corner.x);
  output.position = vec4<f32>(
    clip.xy + project_pixel_size_to_clipspace(normal * corner.y * totalWidth * 0.5 + direction * extension),
    clip.z,
    clip.w
  );
  output.color = color;
  output.local = vec2<f32>(corner.y, freshness);
  output.shape = vec4<f32>(0.0, width / max(totalWidth, 1e-6), totalWidth, 0.0);
  return output;
}

@fragment fn fragmentMain(input: TrailVertexOutput) -> @location(0) vec4<f32> {
  if (isSpatialAnalysisCompareHidden(input.position)) { discard; }
  if (input.shape.x > 0.5) {
    // Head dot: disc with a ground halo.
    let radius = input.shape.y;
    let halo = input.shape.z;
    let fromCenter = length(input.local);
    let fillCoverage = clamp(radius + 0.5 - fromCenter, 0.0, 1.0);
    let haloCoverage = clamp(radius + halo + 0.5 - fromCenter, 0.0, 1.0);
    let haloColor = trailParams.values[2];
    let alpha = mix(haloColor.a * haloCoverage, input.color.a, fillCoverage) * spatialAnalysisStyle.opacity;
    if (alpha <= 0.0) { discard; }
    return finishSpatialAnalysisColor(vec4<f32>(mix(haloColor.rgb, input.color.rgb, fillCoverage), alpha));
  }
  let ageAlpha = trailParams.values[0].z * input.local.y * input.local.y;
  if (ageAlpha <= 0.0) { discard; }
  let distance = abs(input.local.x);
  if (spatialAnalysisStyle.outlineWidthPixels <= 0.0) {
    let coverage = 1.0 - smoothstep(0.55, 1.0, distance);
    return finishSpatialAnalysisColor(vec4<f32>(input.color.rgb, input.color.a * ageAlpha * spatialAnalysisStyle.opacity * coverage));
  }
  // Casing: the fill core, then the outline colour out to the full width.
  let pixel = 2.0 / max(input.shape.z, 0.5);
  let coverage = 1.0 - smoothstep(1.0 - pixel, 1.0, distance);
  let outline = smoothstep(input.shape.y - pixel * 0.5, input.shape.y + pixel * 0.5, distance);
  let color = mix(input.color, spatialAnalysisStyle.outlineColor, outline);
  return finishSpatialAnalysisColor(vec4<f32>(color.rgb, color.a * ageAlpha * spatialAnalysisStyle.opacity * coverage));
}
`;
}

/**
 * Comet trails over time from GPU trajectory buffers. See the module notes above and
 * {@link SpatialAnalysisTrailLayerProps}.
 */
export class SpatialAnalysisTrailLayer extends SpatialAnalysisBaseLayer<SpatialAnalysisTrailLayerProps> {
  static override layerName = 'SpatialAnalysisTrailLayer';

  private paramsBuffer: Buffer | null = null;
  private pathIdsBuffer: Buffer | null = null;
  private pathIdsSource: ArrayLike<number> | null = null;
  private pathIdsVertexCount = -1;

  protected override getShaderFeatures(): SpatialAnalysisShaderFeatures {
    return {channels: Boolean(this.props.instanceChannels)};
  }

  protected getShaderSource(): string {
    return buildTrailShader(this.getShaderFeatures());
  }

  protected getVertexCount(): number {
    return 6;
  }

  private getParamsBuffer(): Buffer {
    this.paramsBuffer ??= this.context.device.createBuffer({
      id: `${this.id}-trail-params`,
      byteLength: 64,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    return this.paramsBuffer;
  }

  /** The path id buffer in use: the `pathIds` prop, or the one built from `pathOffsets`. */
  private getPathIdsBuffer(): Buffer | null {
    return this.props.pathIds ?? this.pathIdsBuffer;
  }

  /** The buffer that maps a row to its `values` entry for `colorBy: 'path'`, else `valueIndices`. */
  private getValueIndicesBuffer(): Buffer | null {
    return this.props.colorBy === 'path'
      ? this.getPathIdsBuffer()
      : (this.props.valueIndices ?? null);
  }

  protected override getStyleBindings(placeholder: Buffer): Record<string, Buffer> {
    return {
      ...super.getStyleBindings(placeholder),
      styleValueIndices: this.getValueIndicesBuffer() ?? placeholder
    };
  }

  protected getBindings(placeholder: Buffer): Record<string, Buffer> {
    const {vertices, timestamps, ids} = this.props;
    return {
      ...this.getStyleBindings(placeholder),
      trailVertices: vertices,
      trailTimes: timestamps,
      trailPathIds: this.getPathIdsBuffer() ?? placeholder,
      trailIds: ids ?? placeholder,
      trailParams: this.getParamsBuffer()
    };
  }

  override updateState(parameters: UpdateParameters<this>): void {
    const {pathOffsets, pathIds, vertexCount} = this.props;
    if (pathOffsets && !pathIds) {
      if (pathOffsets !== this.pathIdsSource || vertexCount !== this.pathIdsVertexCount) {
        this.pathIdsBuffer?.destroy();
        this.pathIdsBuffer = this.context.device.createBuffer({
          id: `${this.id}-path-ids`,
          usage: Buffer.STORAGE | Buffer.COPY_DST,
          data: getPathIds(pathOffsets, Math.max(1, vertexCount))
        });
        this.pathIdsSource = pathOffsets;
        this.pathIdsVertexCount = vertexCount;
      }
    } else if (this.pathIdsBuffer) {
      this.pathIdsBuffer.destroy();
      this.pathIdsBuffer = null;
      this.pathIdsSource = null;
    }
    super.updateState(parameters);
  }

  protected writeLayerStyle(styleBuffer: Buffer): void {
    const {props} = this;
    const frame = this.getStyleFrame();
    const valueIndices = this.getValueIndicesBuffer();
    writeStyle(
      styleBuffer,
      {...props, valueIndices},
      {
        sizePixels: props.headWidthPixels ?? 2.2,
        outlineColor: props.outlineColor,
        outlineWidthPixels: props.outlineWidthPixels,
        useIds: Boolean(props.ids),
        frame
      }
    );
    const data = new ArrayBuffer(64);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    const haloColor = props.headHaloColor ?? DEFAULT_HEAD_HALO_COLOR;
    floats.set([
      props.currentTime,
      Math.max(props.tail, 1e-20),
      props.headAlpha ?? 0.9,
      props.maximumGapTime ?? 0,
      props.headWidthPixels ?? 2.2,
      props.tailWidthPixels ?? 1.2,
      Math.max(0, props.headRadiusPixels ?? 0),
      Math.max(0, props.headHaloPixels ?? 1.2),
      haloColor[0] / 255,
      haloColor[1] / 255,
      haloColor[2] / 255,
      (haloColor[3] ?? 255) / 255
    ]);
    words[12] = Math.max(0, Math.floor(props.vertexCount));
    words[13] = this.getRowCount();
    words[14] = this.getPathIdsBuffer() ? 1 : 0;
    this.getParamsBuffer().write(new Uint8Array(data));
  }

  /** Segment rows drawn: `instanceCount`, else the `ids` length, else `vertexCount - 1`. */
  private getRowCount(): number {
    const {instanceCount, ids, vertexCount} = this.props;
    if (instanceCount !== undefined) return Math.max(0, Math.floor(instanceCount));
    if (ids) return Math.floor(ids.byteLength / 4);
    return Math.max(0, Math.floor(vertexCount) - 1);
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as SpatialAnalysisLayerState;
    if (!model || !styleBuffer) return;
    this.writeLayerStyle(styleBuffer);
    const rows = this.getRowCount();
    // Rows draw first, then (when a head radius is set) the heads of the same rows on top.
    const instances = (this.props.headRadiusPixels ?? 0) > 0 ? rows * 2 : rows;
    if (instances <= 0) return;
    model.setInstanceCount(instances);
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    this.paramsBuffer?.destroy();
    this.pathIdsBuffer?.destroy();
    this.paramsBuffer = null;
    this.pathIdsBuffer = null;
    super.finalizeState(context);
  }
}
