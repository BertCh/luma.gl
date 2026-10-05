// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, CommandEncoder, Device} from '@luma.gl/core';
import {GPUReadbackRing, type GPUReadbackTicket} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_REGION_STATISTICS_FLAGS,
  GPU_REGION_STATISTICS_HEADER_LENGTH,
  GPU_REGION_STATISTICS_SUMMARY_LAYOUT as LAYOUT,
  type GPURegionStatisticsResult
} from './region-statistics-types';

/** Returns the summary length in `uint32` words for a histogram bin count. */
export function getGPURegionStatisticsSummaryLength(binCount: number = 0): number {
  return GPU_REGION_STATISTICS_HEADER_LENGTH + binCount;
}

/** Decodes summary bytes copied from a `GPURegionStatistics` summary view. */
export function decodeGPURegionStatistics(
  data: ArrayBuffer | ArrayBufferView
): GPURegionStatisticsResult {
  const bytes = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  if (bytes.byteLength < GPU_REGION_STATISTICS_HEADER_LENGTH * 4 || bytes.byteLength % 4 !== 0) {
    throw new Error('Region statistics summaries must be at least 32 bytes and 4-byte aligned');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const getWord = (index: number) => view.getUint32(index * 4, true);
  const getFloat = (index: number) => view.getFloat32(index * 4, true);
  const binCount = bytes.byteLength / 4 - GPU_REGION_STATISTICS_HEADER_LENGTH;
  const histogram = new Uint32Array(binCount);
  for (let bin = 0; bin < binCount; bin++) {
    histogram[bin] = getWord(LAYOUT.histogram + bin);
  }
  const flags = getWord(LAYOUT.flags);
  return {
    selectedCount: getWord(LAYOUT.selectedCount),
    valueCount: getWord(LAYOUT.valueCount),
    sum: getFloat(LAYOUT.sum),
    mean: getFloat(LAYOUT.mean),
    minimum: getFloat(LAYOUT.minimum),
    maximum: getFloat(LAYOUT.maximum),
    histogram,
    histogramOutsideCount: getWord(LAYOUT.histogramOutsideCount),
    selectionTruncated: (flags & GPU_REGION_STATISTICS_FLAGS.selectionTruncated) !== 0,
    regionTruncated: (flags & GPU_REGION_STATISTICS_FLAGS.regionTruncated) !== 0,
    candidatesTruncated: (flags & GPU_REGION_STATISTICS_FLAGS.candidatesTruncated) !== 0
  };
}

/** Properties for {@link GPURegionStatisticsReadback}. */
export type GPURegionStatisticsReadbackProps = {
  /** Staging buffer ID prefix. */
  id?: string;
  /** Histogram bin count of the summary. Defaults to 0. */
  binCount?: number;
  /** Number of in-flight staging buffers. Defaults to 3. */
  slotCount?: number;
};

/**
 * Bounded asynchronous summary readback through a `GPUReadbackRing`.
 *
 * Application-side helper: it owns staging buffers, so destroy it when done.
 */
export class GPURegionStatisticsReadback {
  /** Bytes copied per readback. */
  readonly byteLength: number;
  /** Underlying ring. */
  readonly ring: GPUReadbackRing;

  constructor(device: Device, props: GPURegionStatisticsReadbackProps = {}) {
    this.byteLength = getGPURegionStatisticsSummaryLength(props.binCount ?? 0) * 4;
    this.ring = new GPUReadbackRing(device, {
      id: props.id ?? 'region-statistics-readback',
      byteLength: this.byteLength,
      slotCount: props.slotCount ?? 3
    });
  }

  /**
   * Records a copy of the summary after `compiled.encode(...)`.
   *
   * @returns The ticket, or `null` under backpressure (drop the frame).
   */
  encodeRead(
    commandEncoder: CommandEncoder,
    summaryBuffer: Buffer,
    byteOffset: number = 0
  ): GPUReadbackTicket | null {
    const ticket = this.ring.tryAcquire();
    ticket?.copyFrom(commandEncoder, summaryBuffer, {
      sourceOffset: byteOffset,
      byteLength: this.byteLength
    });
    return ticket;
  }

  /** Awaits a submitted ticket and decodes it. */
  async read(ticket: GPUReadbackTicket): Promise<GPURegionStatisticsResult> {
    return decodeGPURegionStatistics(await ticket.read());
  }

  /** Destroys the ring. */
  destroy(): void {
    this.ring.destroy();
  }
}
