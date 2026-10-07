// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUNetworkLineGraph} parameter view. */
export const GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH = 8;

/** Per-frame turn settings of {@link GPUNetworkLineGraph}. */
export type GPUNetworkLineGraphSettings = {
  /** Cost per radian of absolute turn angle, so a right angle costs `angleCost * pi / 2`. Defaults to 0. */
  angleCost?: number;
  /** Extra cost of a left turn (counter-clockwise, `angle > straightAngle`). Defaults to 0. */
  leftTurnCost?: number;
  /** Extra cost of a right turn (clockwise, `angle < -straightAngle`). Defaults to 0. */
  rightTurnCost?: number;
  /**
   * Cost of a U-turn (absolute angle at or above `uTurnAngle`), replacing the angle cost. A
   * negative value bans U-turns. Defaults to -1 (banned).
   */
  uTurnCost?: number;
  /** Absolute angle in radians below which a turn counts as straight for left and right extras. Defaults to 0.5. */
  straightAngle?: number;
  /** Absolute angle in radians at or above which a turn is a U-turn. Defaults to 2.9 (about 166 degrees). */
  uTurnAngle?: number;
  /** Number of active banned turns, at most half the length of `bannedTurns`. Defaults to 0. */
  bannedTurnCount?: number;
};

/**
 * Packs per-frame {@link GPUNetworkLineGraph} settings.
 *
 * Layout (float32): `[angleCost, leftTurnCost, rightTurnCost, uTurnCost, straightAngle,
 * bannedTurnCount, uTurnAngle, 0]`.
 *
 * @param settings Turn settings.
 * @param target Optional destination of at least 8 elements.
 * @throws If the target is too short or a value is invalid.
 */
export function getGPUNetworkLineGraphParameterValues(
  settings: GPUNetworkLineGraphSettings = {},
  target: Float32Array = new Float32Array(GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH) {
    throw new Error(
      `Line graph parameter target must hold ${GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH} elements`
    );
  }
  const angleCost = settings.angleCost ?? 0;
  const leftTurnCost = settings.leftTurnCost ?? 0;
  const rightTurnCost = settings.rightTurnCost ?? 0;
  for (const [name, value] of [
    ['angleCost', angleCost],
    ['leftTurnCost', leftTurnCost],
    ['rightTurnCost', rightTurnCost]
  ] as const) {
    if (!(value >= 0) || !Number.isFinite(value)) {
      throw new Error(`Line graph ${name} must be a non-negative finite number`);
    }
  }
  const bannedTurnCount = settings.bannedTurnCount ?? 0;
  if (!Number.isSafeInteger(bannedTurnCount) || bannedTurnCount < 0) {
    throw new Error('Line graph bannedTurnCount must be a non-negative integer');
  }
  target[0] = angleCost;
  target[1] = leftTurnCost;
  target[2] = rightTurnCost;
  target[3] = settings.uTurnCost ?? -1;
  target[4] = settings.straightAngle ?? 0.5;
  target[5] = bannedTurnCount;
  target[6] = settings.uTurnAngle ?? 2.9;
  target[7] = 0;
  return target;
}
