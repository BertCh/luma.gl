// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {TestRunner, type TestContext} from 'vitest';

/**
 * Requires a WebGPU test device in a GPU spec, replacing `if (!device) return;` guards that report
 * green with zero assertions when no device exists.
 *
 * - Outside a browser (the `node` project), WebGPU cannot exist, so the current test is skipped
 *   through `context.skip()` and shows up in the skip count.
 * - In a browser-backed project (`headless`, `browser`), a missing device means a mis-provisioned
 *   runner, so this throws and the test fails.
 *
 * Skipping and throwing both stop the caller, so helper functions that acquire their own device
 * can call this too: the error propagates to the running test.
 *
 * @param device Result of `getWebGPUTestDevice()` (or an equivalent device getter).
 * @param context Test context; defaults to the context of the test that is currently running.
 */
export function requireTestDevice<DeviceT extends object>(
  device: DeviceT | null | undefined,
  context?: TestContext
): asserts device is DeviceT {
  if (device) {
    return;
  }
  const testContext = context ?? TestRunner.getCurrentTest()?.context;
  const isBrowser = typeof window !== 'undefined' && typeof document !== 'undefined';
  if (!isBrowser && testContext) {
    testContext.skip('no WebGPU test device outside a browser');
  }
  const projectName = testContext?.task.file.projectName ?? 'unknown';
  throw new Error(`No WebGPU test device in the '${projectName}' project`);
}
