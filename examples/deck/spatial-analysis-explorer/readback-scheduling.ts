// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Starts a readback in the next task, after submission of the current Deck frame.
 *
 * The effect encodes copies into `device.commandEncoder` during Deck's render callback.
 * `AnimationLoop._renderFrame` synchronously calls `device.submit()` after that callback returns.
 * Waiting one task therefore keeps ring slots unmapped until their copies have been submitted.
 * Call this during frame encoding, passing a callback that starts the read, not an active promise.
 */
export async function scheduleReadback<Result>(read: () => Promise<Result>): Promise<Result> {
  await new Promise<void>(resolve => setTimeout(resolve, 0));
  return read();
}
