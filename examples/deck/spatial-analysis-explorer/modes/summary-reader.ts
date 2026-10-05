// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {GPUReadbackRing} from '@luma.gl/gpgpu/gpu-core';
import type {SpatialAnalysisResources} from '../spatial-analysis-resources';

/**
 * Small summary readback: copies `sources` back to back into a ring ticket and hands the bytes to
 * `onResult`. At most one read is in flight.
 */
export class SummaryReader {
  private pending = false;
  private stale = false;
  private destroyed = false;
  private readonly ring: GPUReadbackRing;
  private readonly sources: readonly {buffer: Buffer; size: number}[];
  private readonly byteLength: number;
  private readonly onResult: (bytes: ArrayBuffer) => void;

  constructor(
    resources: SpatialAnalysisResources,
    id: string,
    sources: readonly {buffer: Buffer; size: number}[],
    onResult: (bytes: ArrayBuffer) => void
  ) {
    this.sources = sources;
    this.byteLength = sources.reduce((total, source) => total + source.size, 0);
    this.onResult = onResult;
    this.ring = resources.track(
      new GPUReadbackRing(resources.device, {id: `summary-${id}`, byteLength: this.byteLength})
    );
  }

  /** True while a read is in flight. */
  get isPending(): boolean {
    return this.pending;
  }

  /** Marks the summary stale without recording a copy; {@link flush} reads it when possible. */
  markStale(): void {
    this.stale = true;
  }

  /** Marks the summary stale and reads it now, or on the next {@link flush} if a read is in flight. */
  request(commandEncoder: CommandEncoder): void {
    this.stale = true;
    this.flush(commandEncoder);
  }

  /**
   * Records the copies into `commandEncoder` when the summary is stale and no read is in flight,
   * and delivers the bytes when they are ready. Call it every frame after the graphs are encoded
   * so a request made while a read was in flight is not lost.
   */
  flush(commandEncoder: CommandEncoder): void {
    if (!this.stale || this.pending || this.destroyed) return;
    const ticket = this.ring.tryAcquire();
    if (!ticket) return;
    let offset = 0;
    for (const {buffer, size} of this.sources) {
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: buffer,
        destinationBuffer: ticket.buffer,
        destinationOffset: offset,
        size
      });
      offset += size;
    }
    ticket.markEncoded({byteOffset: 0, byteLength: this.byteLength});
    this.stale = false;
    this.pending = true;
    void ticket
      .read()
      .then(bytes => {
        if (!this.destroyed) this.onResult(bytes.slice().buffer);
      })
      .catch(() => {
        // The ring or device was destroyed while the read was in flight.
      })
      .finally(() => {
        this.pending = false;
      });
  }

  /** Stops delivering results. */
  stop(): void {
    this.destroyed = true;
  }
}
