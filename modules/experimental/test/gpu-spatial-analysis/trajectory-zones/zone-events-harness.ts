// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {getInt64TimeWords} from '../../../src/gpu-dataframe/time-window-filter/time-words';
import {GPUZoneEvents} from '../../../src/gpu-spatial-analysis/trajectory-zones/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import type {OracleTracks} from '../trajectory-interpolation/trajectory-interpolation-oracle';
import type {ZoneEdges} from './zone-events-oracle';

/** GPU readback of one `GPUZoneEvents` run. */
export type ZoneGPUResult = {
  count: number;
  overflow: number;
  totalCount: number;
  tracks: number[];
  zones: number[];
  types: number[];
  times: number[];
  rows: number[];
  dwellTimes: number[];
  visitCounts: number[];
  trackEventCounts: number[];
  candidateCount: number;
  candidateOverflow: number;
  trackOverflow: number;
  eventOverflow: number;
};

/** Runs one zone-events graph and reads every output. */
export async function runZoneEvents(
  device: Device,
  tracks: OracleTracks,
  edges: ZoneEdges,
  options: {
    zoneCount: number;
    candidateCapacity: number;
    maxEventsPerTrack: number;
    eventCapacity: number;
    spatialSort?: boolean;
  }
): Promise<ZoneGPUResult> {
  const isWordMode = tracks.times.kind === 'words';
  const rowCount = tracks.positions.length / 2;
  const trackCount = tracks.trackOffsets.length - 1;
  const cellCount = trackCount * options.zoneCount;
  const {eventCapacity} = options;
  const graph = new GPUCommandGraph(device, {id: 'zone-events-test'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffer.write(new Uint32Array(Math.max(length, 1)).fill(0x7f7f7f7f));
    buffers.push(buffer);
    return buffer;
  };
  const timestampData =
    tracks.times.kind === 'words'
      ? getInt64TimeWords(BigInt64Array.from(tracks.times.values))
      : tracks.times.values;
  const out = {
    ids: output(eventCapacity),
    zones: output(eventCapacity),
    types: output(eventCapacity),
    times: output(eventCapacity),
    rows: output(eventCapacity),
    count: output(1),
    overflow: output(1),
    total: output(1),
    dwell: output(cellCount),
    visits: output(cellCount),
    trackCounts: output(trackCount),
    candidateCount: output(1),
    candidateOverflow: output(1),
    trackOverflow: output(1),
    eventOverflow: output(1)
  };
  const timestampBuffer = input(timestampData);
  const edgeCount = edges.zones.length;
  graph.add(
    new GPUZoneEvents({
      id: 'zone-events',
      positions: importGraphBuffer(
        graph,
        'positions',
        input(tracks.positions),
        'float32x2',
        rowCount
      ),
      timestamps: isWordMode
        ? importGraphBuffer(graph, 'timestamps', timestampBuffer, 'uint32x2', rowCount)
        : importGraphBuffer(graph, 'timestamps', timestampBuffer, 'float32', rowCount),
      trackOffsets: importGraphBuffer(
        graph,
        'offsets',
        input(Uint32Array.from(tracks.trackOffsets)),
        'uint32',
        trackCount + 1
      ),
      edgeStarts: importGraphBuffer(
        graph,
        'edge-starts',
        input(edges.starts),
        'float32x2',
        edgeCount
      ),
      edgeEnds: importGraphBuffer(graph, 'edge-ends', input(edges.ends), 'float32x2', edgeCount),
      edgeZones: importGraphBuffer(graph, 'edge-zones', input(edges.zones), 'uint32', edgeCount),
      zoneCount: options.zoneCount,
      candidateCapacity: options.candidateCapacity,
      maxEventsPerTrack: options.maxEventsPerTrack,
      spatialSort: options.spatialSort,
      events: {
        output: {
          ids: importGraphBuffer(graph, 'o-ids', out.ids, 'uint32', eventCapacity),
          count: importGraphBuffer(graph, 'o-count', out.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'o-overflow', out.overflow, 'uint32', 1),
          totalCount: importGraphBuffer(graph, 'o-total', out.total, 'uint32', 1)
        },
        eventZones: importGraphBuffer(graph, 'o-zones', out.zones, 'uint32', eventCapacity),
        eventTypes: importGraphBuffer(graph, 'o-types', out.types, 'uint32', eventCapacity),
        eventTimes: importGraphBuffer(graph, 'o-times', out.times, 'float32', eventCapacity),
        eventRows: importGraphBuffer(graph, 'o-rows', out.rows, 'uint32', eventCapacity)
      },
      dwellTimes: importGraphBuffer(graph, 'o-dwell', out.dwell, 'float32', cellCount),
      visitCounts: importGraphBuffer(graph, 'o-visits', out.visits, 'uint32', cellCount),
      trackEventCounts: importGraphBuffer(
        graph,
        'o-track-counts',
        out.trackCounts,
        'uint32',
        trackCount
      ),
      diagnostics: {
        candidateCount: importGraphBuffer(graph, 'o-cand-count', out.candidateCount, 'uint32', 1),
        candidateOverflow: importGraphBuffer(
          graph,
          'o-cand-over',
          out.candidateOverflow,
          'uint32',
          1
        ),
        trackOverflow: importGraphBuffer(graph, 'o-track-over', out.trackOverflow, 'uint32', 1),
        eventOverflow: importGraphBuffer(graph, 'o-event-over', out.eventOverflow, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(out.count, 1);
  const [overflow] = await readUint32(out.overflow, 1);
  const [totalCount] = await readUint32(out.total, 1);
  const result: ZoneGPUResult = {
    count,
    overflow,
    totalCount,
    tracks: await readUint32(out.ids, eventCapacity),
    zones: await readUint32(out.zones, eventCapacity),
    types: await readUint32(out.types, eventCapacity),
    times: await readFloat32(out.times, eventCapacity),
    rows: await readUint32(out.rows, eventCapacity),
    dwellTimes: await readFloat32(out.dwell, cellCount),
    visitCounts: await readUint32(out.visits, cellCount),
    trackEventCounts: await readUint32(out.trackCounts, trackCount),
    candidateCount: (await readUint32(out.candidateCount, 1))[0],
    candidateOverflow: (await readUint32(out.candidateOverflow, 1))[0],
    trackOverflow: (await readUint32(out.trackOverflow, 1))[0],
    eventOverflow: (await readUint32(out.eventOverflow, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}
