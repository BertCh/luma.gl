// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {commands} from 'vitest/browser';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {listLensScenarios, toRecordedLensFile} from '../../src/computation-lenses/index';

// Registers every lens scenario descriptor (each loads its GPU code only when run).
import '../../src/computation-lenses/scenarios/index';
// browser-process-shim.mjs does not forward shell env. Run with VITE_CAPTURE_LENS_RECORDS=1.
const captureEnabled =
  (import.meta as ImportMeta & {env?: Record<string, string | undefined>}).env?.[
    'VITE_CAPTURE_LENS_RECORDS'
  ] === '1';
const scenarios = listLensScenarios();

if (!scenarios.length) test.skip('capture records: no lens scenarios registered', () => {});
for (const scenario of scenarios) {
  test.skipIf(!captureEnabled)(`capture lens record ${scenario.id}`, async () => {
    const device = await getWebGPUTestDevice();
    expect(device, 'Record capture requires WebGPU').toBeTruthy();
    if (!device) throw new Error('Record capture requires WebGPU');
    const result = await (await scenario.load()).run(device);
    const adapterLabel = `${device.info.renderer || device.info.vendor || device.info.type}${device.info.gpuBackend ? ` (${device.info.gpuBackend})` : ''}`;
    const recorded = toRecordedLensFile(result, {
      scenarioId: scenario.id,
      recordedAt: new Date().toISOString(),
      adapterLabel,
      variant: result.adapter.variant
    });
    await commands.writeFile(
      `docs/lens-records/${scenario.id}.json`,
      `${JSON.stringify(recorded)}\n`
    );
  });
}
