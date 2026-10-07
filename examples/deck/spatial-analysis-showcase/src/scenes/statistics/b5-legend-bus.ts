// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Hands GPU-derived legend data (class breaks and counts, colors) from a scene's compute module
 * to its `legends(state)` function, which the shell calls again after every
 * `ctx.setLegendExtent`. No luma.gl imports here: scene files stay light.
 */

/** One class row of a dynamic legend. */
export type LegendClass = {
  color: readonly [number, number, number, number?];
  label: string;
};

const store = new Map<string, unknown>();

/** Stores the latest data for a scene. */
export function setLegendData<T>(sceneId: string, data: T): void {
  store.set(sceneId, data);
}

/** Reads the latest data of a scene, or `undefined` before the first readback. */
export function getLegendData<T>(sceneId: string): T | undefined {
  return store.get(sceneId) as T | undefined;
}

/** Drops a scene's data when it is destroyed. */
export function clearLegendData(sceneId: string): void {
  store.delete(sceneId);
}

/** Formats a number compactly: 1.2M, 34.5k, 12.3, 0.042. */
export function formatCompact(value: number): string {
  if (!Number.isFinite(value)) return value > 0 ? '+inf' : value < 0 ? '-inf' : 'n/a';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (magnitude >= 1e4) return `${(value / 1e3).toFixed(1)}k`;
  if (magnitude >= 100 || Number.isInteger(value)) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  if (magnitude === 0) return '0';
  return value.toPrecision(2);
}

/** Renders a histogram as a short line of unicode bars. */
export function formatSparkline(bins: ArrayLike<number>): string {
  const bars = '▁▂▃▄▅▆▇█';
  let maximum = 0;
  for (let index = 0; index < bins.length; index++) maximum = Math.max(maximum, bins[index]);
  let text = '';
  for (let index = 0; index < bins.length; index++) {
    text +=
      bins[index] === 0 || maximum === 0
        ? '▁'
        : bars[
            Math.min(bars.length - 1, Math.floor((bins[index] / maximum) * (bars.length - 1) + 0.5))
          ];
  }
  return text;
}

/** Packed rgba8 (r in the low byte) to `[r, g, b, a]` 0-255. */
export function unpackColor(packed: number): [number, number, number, number] {
  return [packed & 255, (packed >>> 8) & 255, (packed >>> 16) & 255, (packed >>> 24) & 255];
}
