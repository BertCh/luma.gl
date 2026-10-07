// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Effect, EffectContext, PreRenderOptions, Viewport} from '@deck.gl/core';
import type {Device} from '@luma.gl/core';
import type {EncodableInstance} from './types';

/** CPU-side frame statistics published by {@link SpatialAnalysisDeckEffect}. */
export type SpatialAnalysisEffectStats = {
  /** Frames encoded for the active mode. */
  frameCount: number;
  /** Smoothed frames per second. */
  framesPerSecond: number;
  /** CPU milliseconds spent in the mode's `encode` (parameter writes plus graph encoding). */
  encodeMilliseconds: number;
};

/**
 * Encodes the active mode's compiled graphs into Deck's own frame encoder before layers draw.
 *
 * Deck remains the sole owner of queue submission, so compute and the layers that read its outputs
 * land in one command buffer per frame with no CPU synchronization in between.
 */
export class SpatialAnalysisDeckEffect implements Effect {
  readonly id = 'spatial-analysis-effect';
  readonly props = {};
  readonly useInPicking = false;
  readonly order = 0;
  private readonly device: Device;
  private mode: EncodableInstance | null = null;
  private frameIndex = 0;
  private previousTime = 0;
  private smoothedFramesPerSecond = 0;
  private lastEncodeMilliseconds = 0;
  private latestViewport: Viewport | null = null;
  private readonly onError: (error: Error) => void;

  constructor(device: Device, onError: (error: Error) => void) {
    this.device = device;
    this.onError = onError;
  }

  /** Switches the encoded mode. `null` stops encoding. */
  setMode(mode: EncodableInstance | null): void {
    this.mode = mode;
    this.frameIndex = 0;
    this.previousTime = 0;
  }

  /** Latest Deck viewport seen in `preRender`. */
  get viewport(): Viewport | null {
    return this.latestViewport;
  }

  /** Current CPU-side frame statistics. */
  get stats(): SpatialAnalysisEffectStats {
    return {
      frameCount: this.frameIndex,
      framesPerSecond: this.smoothedFramesPerSecond,
      encodeMilliseconds: this.lastEncodeMilliseconds
    };
  }

  setup(_context: EffectContext): void {}

  preRender(options: PreRenderOptions): void {
    const viewport = options.viewports[0];
    if (!viewport) return;
    this.latestViewport = viewport;
    const mode = this.mode;
    if (!mode) return;
    const now = performance.now() / 1000;
    const deltaSeconds =
      this.previousTime > 0 ? Math.min(0.25, Math.max(0, now - this.previousTime)) : 0;
    if (deltaSeconds > 0) {
      const framesPerSecond = 1 / deltaSeconds;
      this.smoothedFramesPerSecond =
        this.smoothedFramesPerSecond === 0
          ? framesPerSecond
          : this.smoothedFramesPerSecond * 0.9 + framesPerSecond * 0.1;
    }
    this.previousTime = now;
    const start = performance.now();
    try {
      mode.encode(this.device.commandEncoder, {
        viewport,
        timeSeconds: now,
        deltaSeconds,
        frameIndex: this.frameIndex
      });
    } catch (error) {
      this.mode = null;
      this.onError(error instanceof Error ? error : new Error(String(error)));
      return;
    }
    this.lastEncodeMilliseconds = performance.now() - start;
    this.frameIndex++;
  }

  cleanup(_context: EffectContext): void {
    this.mode = null;
  }
}
