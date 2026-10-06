// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, RenderPass} from '@luma.gl/core';
import type {Model} from '@luma.gl/engine';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  type SpatialAnalysisInstanceProps,
  type SpatialAnalysisPointLayerProps,
  type SpatialAnalysisRasterLayerProps
} from '../spatial-analysis-layers';

/*
 * Bespoke layers for the Zone events mode. Like the contours layers they reuse the shared style
 * uniform, bindings and lifecycle and replace only the shader body:
 *
 * - `ZoneEventMarkerLayer` draws one pulse per GPUZoneEvents event. The crossing position is
 *   interpolated on the GPU from the event row and the event time, so no position is ever read back.
 * - `ZoneChoroplethLayer` fills fan-triangulated zones and colors each by a row of the per-zone
 *   statistics table that GPUGroupStatistics wrote (keys ascending, so the shader scans for it).
 */

/** Triangles per zone: one per boundary edge of an eight-vertex ring. */
const ZONE_TRIANGLES = 8;
const POINT_POSITIONS_MARKER = '@group(0) @binding(auto) var<storage, read> pointPositions';
const RASTER_BINDINGS_MARKER = '@group(0) @binding(auto) var<storage, read> rasterBounds';

/**
 * Removes the style value bindings (`styleValues`, `styleValueIndices`) and the helpers that read
 * them from a shared layer shader. WebGPU allows 8 storage buffers per stage and the bespoke layers
 * here need more of their own, so they drop what they do not use.
 */
function stripStyleValueBindings(source: string): string {
  const stripped = source
    .replace('@group(0) @binding(auto) var<storage, read> styleValues: array<u32>;\n', '')
    .replace('@group(0) @binding(auto) var<storage, read> styleValueIndices: array<u32>;\n', '');
  const valueRowStart = stripped.indexOf('// Maps a drawn row');
  const viridisStart = stripped.indexOf('fn spatialAnalysisViridis');
  const colorStart = stripped.indexOf('// Returns the style color');
  const positionStart = stripped.indexOf('fn getSpatialAnalysisPosition');
  if (
    stripped === source ||
    valueRowStart < 0 ||
    viridisStart < valueRowStart ||
    colorStart < viridisStart ||
    positionStart < colorStart
  ) {
    throw new Error('Zone events layers could not strip the style value bindings');
  }
  return (
    stripped.slice(0, valueRowStart) +
    stripped.slice(viridisStart, colorStart) +
    stripped.slice(positionStart)
  );
}

/** Extra props of {@link ZoneEventMarkerLayer}. */
export type ZoneEventMarkerLayerProps = SpatialAnalysisPointLayerProps & {
  /** `float32` sample times relative to the dataset epoch, one per `positions` row. */
  timestamps: Buffer;
  /** `uint32` first-row offsets per track (`trackCount + 1` rows). */
  trackOffsets: Buffer;
  /** `uint32` track index per event (`events.output.ids`). */
  eventTracks: Buffer;
  /** `uint32` segment end row per event. */
  eventRows: Buffer;
  /** `float32` crossing time per event, relative to the track's first timestamp. */
  eventTimes: Buffer;
  /** `uint32` event type per event (0 enter, 1 exit). */
  eventTypes: Buffer;
};

/**
 * Enter and exit pulses at the interpolated crossing position. `extent` holds `[playhead,
 * fadeSeconds, showAll]` (seconds on the dataset clock): an event pulses and fades during the
 * `fadeSeconds` after the playhead passes it; with `showAll` the other events stay as faint dots.
 * The instance count comes from `drawCommands`.
 */
export class ZoneEventMarkerLayer extends SpatialAnalysisPointLayer {
  static override layerName = 'ZoneEventMarkerLayer';

  constructor(props: ZoneEventMarkerLayerProps & SpatialAnalysisInstanceProps) {
    super(props);
  }

  protected override getShaderSource(): string {
    const source = stripStyleValueBindings(super.getShaderSource())
      .replace('@group(0) @binding(auto) var<storage, read> pointIds: array<u32>;\n', '')
      .replace('row = pointIds[instanceIndex];', 'row = instanceIndex;');
    const markerIndex = source.indexOf(POINT_POSITIONS_MARKER);
    const original = `let source = pointPositions[row];
  let color = getSpatialAnalysisColor(getSpatialAnalysisValueRow(row));`;
    if (markerIndex < 0 || !source.includes(original)) {
      throw new Error('ZoneEventMarkerLayer could not patch the point shader');
    }
    const withBindings = `${source.slice(0, markerIndex)}
@group(0) @binding(auto) var<storage, read> eventTimestamps: array<f32>;
@group(0) @binding(auto) var<storage, read> eventOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> eventTracks: array<u32>;
@group(0) @binding(auto) var<storage, read> eventRows: array<u32>;
@group(0) @binding(auto) var<storage, read> eventTimes: array<f32>;
@group(0) @binding(auto) var<storage, read> eventTypes: array<u32>;
${source.slice(markerIndex)}`;
    return withBindings
      .replace(
        original,
        `let eventTrack = eventTracks[instanceIndex];
  let eventRow = max(eventRows[instanceIndex], 1u);
  let trackStart = eventOffsets[eventTrack];
  let epoch = eventTimestamps[trackStart];
  let startTime = eventTimestamps[eventRow - 1u] - epoch;
  let endTime = eventTimestamps[eventRow] - epoch;
  let along = clamp((eventTimes[instanceIndex] - startTime) / max(endTime - startTime, 1e-6), 0.0, 1.0);
  let source = mix(pointPositions[eventRow - 1u], pointPositions[eventRow], along);
  let playhead = styleExtent[0];
  let fadeSeconds = max(styleExtent[1], 1.0);
  let age = playhead - (epoch + eventTimes[instanceIndex]);
  let isExit = eventTypes[instanceIndex] == 1u;
  var color = select(vec4<f32>(0.3, 0.9, 0.55, 1.0), vec4<f32>(1.0, 0.45, 0.25, 1.0), isExit);
  var pulse = 0.5;
  if (age >= 0.0 && age <= fadeSeconds) {
    let phase = age / fadeSeconds;
    pulse = 1.0 + 2.5 * phase;
    color.a = 1.0 - phase;
  } else if (styleExtent[2] > 0.5) {
    color.a = 0.3;
  } else {
    color.a = 0.0;
  }`
      )
      .replace(
        'corner * spatialAnalysisStyle.sizePixels),',
        'corner * spatialAnalysisStyle.sizePixels * pulse),'
      );
  }

  protected override getBindings(placeholder: Buffer): Record<string, Buffer> {
    const props = this.props as unknown as ZoneEventMarkerLayerProps;
    return {
      styleExtent: this.props.extent ?? placeholder,
      pointPositions: this.props.positions,
      eventTimestamps: props.timestamps,
      eventOffsets: props.trackOffsets,
      eventTracks: props.eventTracks,
      eventRows: props.eventRows,
      eventTimes: props.eventTimes,
      eventTypes: props.eventTypes
    };
  }
}

type LayerState = {model: Model | null; styleBuffer: Buffer | null};

/** Extra props of {@link ZoneChoroplethLayer}. */
export type ZoneChoroplethLayerProps = SpatialAnalysisRasterLayerProps &
  SpatialAnalysisInstanceProps & {
    /** `float32x2` triangle vertices, three per triangle. */
    triangles: Buffer;
    /** Per-zone table keys (zone index, ascending, `0xffffffff` in unused rows). */
    tableKeys: Buffer;
    /** Per-zone table: total dwell time. */
    tableSums: Buffer;
    /** Per-zone table: number of visiting tracks. */
    tableCounts: Buffer;
  };

/**
 * Zone fill colored by a statistic. `extent` holds `[metric, maximum]` with metric 0 total dwell,
 * 1 mean dwell and 2 visiting tracks; the value maps through viridis over `[0, maximum]`. Zones
 * with no table row (no visits) draw dim. Zone `z` owns triangles `[8z, 8z + 8)`. `drawCommands[drawCommandIndex].vertexCount` must hold
 * `3 * triangleCount`.
 */
export class ZoneChoroplethLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'ZoneChoroplethLayer';

  constructor(props: ZoneChoroplethLayerProps) {
    super(props);
  }

  protected override getShaderSource(): string {
    const source = stripStyleValueBindings(super.getShaderSource());
    const markerIndex = source.indexOf(RASTER_BINDINGS_MARKER);
    if (markerIndex < 0) {
      throw new Error('ZoneChoroplethLayer could not find the raster binding block');
    }
    return `${source.slice(0, markerIndex)}
@group(0) @binding(auto) var<storage, read> zoneTriangles: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> tableKeys: array<u32>;
@group(0) @binding(auto) var<storage, read> tableSums: array<f32>;
@group(0) @binding(auto) var<storage, read> tableCounts: array<u32>;

struct ZoneVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> ZoneVertexOutput {
  var output: ZoneVertexOutput;
  // Every zone is a fan of ${ZONE_TRIANGLES} triangles.
  let zone = vertexIndex / ${ZONE_TRIANGLES * 3}u;
  let metric = u32(styleExtent[0]);
  var value = -1.0;
  for (var row = 0u; row < arrayLength(&tableKeys); row++) {
    if (tableKeys[row] == zone) {
      if (metric == 0u) {
        value = tableSums[row];
      } else if (metric == 1u) {
        value = tableSums[row] / max(f32(tableCounts[row]), 1.0);
      } else {
        value = f32(tableCounts[row]);
      }
    }
  }
  if (value < 0.0) {
    output.color = vec4<f32>(0.2, 0.22, 0.28, 0.25);
  } else {
    let t = clamp(value / max(styleExtent[1], 1e-9), 0.0, 1.0);
    output.color = vec4<f32>(spatialAnalysisViridis(sqrt(t)), 0.62);
  }
  output.position = projectSpatialAnalysisPosition(getSpatialAnalysisPosition(zoneTriangles[vertexIndex]));
  return output;
}

@fragment fn fragmentMain(input: ZoneVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * spatialAnalysisStyle.opacity);
}
`;
  }

  protected override getBindings(placeholder: Buffer): Record<string, Buffer> {
    const props = this.props as unknown as ZoneChoroplethLayerProps;
    return {
      styleExtent: this.props.extent ?? placeholder,
      zoneTriangles: props.triangles,
      tableKeys: props.tableKeys,
      tableSums: props.tableSums,
      tableCounts: props.tableCounts
    };
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as LayerState;
    if (!model || !styleBuffer) return;
    this.writeLayerStyle(styleBuffer);
    const {drawCommands, drawCommandIndex = 0} = this.props as SpatialAnalysisInstanceProps;
    if (!drawCommands) return;
    // Bind pipeline and bindings with an empty draw, then replay the record.
    model.setInstanceCount(0);
    model.draw(renderPass);
    drawCommands.draw(renderPass, drawCommandIndex);
  }
}
