// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The stage helper (SYNTHESIS G28): a story with stages (dataframe, cells, joins, flows, terrain,
 * raster) keeps the layers of every stage mounted and cross-fades their opacity, so Back is
 * instant and nothing is rebuilt.
 *
 * ```ts
 * const fader = createStageFader({
 *   stages: ['points', 'cells'],
 *   initial: 'points',
 *   onFrame: () => ctx.requestLayers()
 * });
 * // in the layer list: new SpatialAnalysisPointLayer({..., opacity: fader.opacity('points')})
 * //                    new SpatialAnalysisRasterLayer({..., opacity: fader.opacity('cells')})
 * // in the step handler: fader.set(step.stage);
 * ```
 *
 * A layer whose `opacity` is 0 still draws nothing but stays alive; pass `visible:
 * fader.opacity(stage) > 0` as well if the layer is expensive to draw.
 */

/** Default cross-fade duration in milliseconds. */
export const STAGE_FADE_MILLISECONDS = 400;

/** Options of {@link createStageFader}. */
export type StageFaderOptions = {
  /** Every stage name, in any order. */
  stages: readonly string[];
  /** The stage that is fully shown at the start. */
  initial: string;
  /** Cross-fade duration. Defaults to {@link STAGE_FADE_MILLISECONDS}. */
  durationMs?: number;
  /** Called after every opacity change, once per animation frame while fading (ask for layers). */
  onFrame: () => void;
  /**
   * Whether reduced motion is requested; when it returns `true` a stage change jumps. Defaults to
   * the `prefers-reduced-motion: reduce` media query.
   */
  reducedMotion?: () => boolean;
};

/** A stage cross-fader; see {@link createStageFader}. */
export type StageFader = {
  /** Fades to `stage` (the other stages fade out); jumps when reduced motion is requested. */
  set(stage: string): void;
  /** Current opacity of a stage's layers, 0-1 (0 for an unknown stage). */
  opacity(stage: string): number;
  /** The stage most recently asked for. */
  getStage(): string;
  /** Stops any animation; later `set` calls jump without calling `onFrame`. */
  destroy(): void;
};

function prefersReducedMotion(): boolean {
  try {
    return globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  } catch {
    return false;
  }
}

/** Ease-out cubic, close to the app's `--ease` curve. */
function ease(progress: number): number {
  return 1 - (1 - progress) ** 3;
}

/**
 * Creates a fader whose stages all stay mounted. `opacity(stage)` is 1 for the current stage and
 * 0 for the others at rest, and moves between them over `durationMs` after `set`. A `set` during a
 * fade starts from the current opacities, so quick Back and Next never pop.
 */
export function createStageFader(options: StageFaderOptions): StageFader {
  const duration = Math.max(0, options.durationMs ?? STAGE_FADE_MILLISECONDS);
  const isReduced = options.reducedMotion ?? prefersReducedMotion;
  const opacities = new Map<string, number>();
  for (const stage of options.stages) opacities.set(stage, stage === options.initial ? 1 : 0);
  if (!opacities.has(options.initial)) opacities.set(options.initial, 1);
  let current = options.initial;
  let from = new Map(opacities);
  let startTime = 0;
  let frameHandle: number | null = null;
  let destroyed = false;

  const stop = () => {
    if (frameHandle !== null) {
      cancelAnimationFrame(frameHandle);
      frameHandle = null;
    }
  };

  const jump = () => {
    for (const stage of opacities.keys()) opacities.set(stage, stage === current ? 1 : 0);
  };

  const step = (time: number) => {
    frameHandle = null;
    if (destroyed) return;
    const progress = duration > 0 ? Math.min(1, (time - startTime) / duration) : 1;
    const eased = ease(Math.max(0, progress));
    for (const [stage, start] of from) {
      const target = stage === current ? 1 : 0;
      opacities.set(stage, start + (target - start) * eased);
    }
    if (progress >= 1) jump();
    options.onFrame();
    if (progress < 1) frameHandle = requestAnimationFrame(step);
  };

  return {
    set(stage) {
      if (destroyed || stage === current) return;
      current = stage;
      if (!opacities.has(stage)) opacities.set(stage, 0);
      stop();
      if (isReduced() || duration === 0 || typeof requestAnimationFrame !== 'function') {
        jump();
        options.onFrame();
        return;
      }
      from = new Map(opacities);
      startTime = performance.now();
      frameHandle = requestAnimationFrame(step);
    },
    opacity(stage) {
      return opacities.get(stage) ?? 0;
    },
    getStage() {
      return current;
    },
    destroy() {
      destroyed = true;
      stop();
    }
  };
}
