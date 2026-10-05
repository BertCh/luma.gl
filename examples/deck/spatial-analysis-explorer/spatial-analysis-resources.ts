// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Viewport} from '@deck.gl/core';
import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUParameterBuffer,
  type GPUParameterFormat
} from '@luma.gl/experimental/gpu-spatial-analysis';
import type {LocalMetricProjection} from './spatial-analysis-data';

type Destroyable = {destroy: () => void};

/**
 * Tracks the GPU resources one mode creates and destroys them in reverse creation order, so
 * compiled graphs created after their imported buffers are destroyed first.
 */
export class SpatialAnalysisResources {
  readonly device: Device;
  private readonly prefix: string;
  private readonly resources: Destroyable[] = [];
  private destroyed = false;

  constructor(device: Device, prefix: string) {
    this.device = device;
    this.prefix = prefix;
  }

  /**
   * Creates a storage buffer usable by compute graphs and by spatial-analysis layers.
   *
   * @param name Suffix for the buffer ID.
   * @param data Initial contents, or a byte length for a zero-filled buffer.
   * @param additionalUsage Extra usage flags such as `Buffer.VERTEX` or `Buffer.INDIRECT`.
   */
  createBuffer(
    name: string,
    data: Float32Array | Uint32Array | Int32Array | number,
    additionalUsage = 0
  ): Buffer {
    const usage = Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC | additionalUsage;
    const buffer =
      typeof data === 'number'
        ? this.device.createBuffer({
            id: `${this.prefix}-${name}`,
            byteLength: Math.max(4, data),
            usage
          })
        : this.device.createBuffer({
            id: `${this.prefix}-${name}`,
            usage,
            ...(data.byteLength > 0 ? {data} : {byteLength: 4})
          });
    return this.track(buffer);
  }

  /** Creates and tracks a per-frame parameter buffer. */
  createParameterBuffer<Format extends GPUParameterFormat>(
    name: string,
    format: Format,
    length: number,
    values?: Float32Array | Uint32Array | Int32Array
  ): GPUParameterBuffer<Format> {
    return this.track(
      new GPUParameterBuffer(this.device, {
        id: `${this.prefix}-${name}`,
        format,
        length,
        values
      })
    );
  }

  /** Tracks any resource with `destroy()`, such as a compiled graph or `DrawCommandBuffer`. */
  track<T extends Destroyable>(resource: T): T {
    this.resources.push(resource);
    return resource;
  }

  /** Destroys and untracks one resource early, for example a graph replaced by a rebuild. */
  release(resource: Destroyable): void {
    const index = this.resources.indexOf(resource);
    if (index >= 0) {
      this.resources.splice(index, 1);
      resource.destroy();
    }
  }

  /** Destroys every tracked resource in reverse creation order. Idempotent. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (let index = this.resources.length - 1; index >= 0; index--) {
      this.resources[index].destroy();
    }
    this.resources.length = 0;
  }
}

/** Returns the viewport's visible `[minX, minY, maxX, maxY]` in a projection's planar meters. */
export function getViewportMetricBounds(
  viewport: Viewport,
  projection: LocalMetricProjection
): [number, number, number, number] {
  const corners = [
    viewport.unproject([0, 0]),
    viewport.unproject([viewport.width, 0]),
    viewport.unproject([0, viewport.height]),
    viewport.unproject([viewport.width, viewport.height])
  ].map(([longitude, latitude]) => projection.project(longitude, latitude));
  return [
    Math.min(...corners.map(corner => corner[0])),
    Math.min(...corners.map(corner => corner[1])),
    Math.max(...corners.map(corner => corner[0])),
    Math.max(...corners.map(corner => corner[1]))
  ];
}

/** Formats an integer with thousands separators. */
export function formatCount(value: number): string {
  return Math.round(value).toLocaleString('en-US');
}
