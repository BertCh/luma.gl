// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, onTestFinished} from 'vitest';
import {requireTestDevice} from './require-test-device';

it('requireTestDevice returns for a device and narrows its type', () => {
  const device: {label: string} | null = {label: 'device'};
  requireTestDevice(device);
  expect(device.label).toBe('device');
});

it('requireTestDevice skips the running test outside a browser', context => {
  onTestFinished(({task}) => {
    // A skip, not a pass: the guarded spec must show up in the skip count.
    expect(task.result?.state).toBe('skip');
  });
  requireTestDevice(null);
  context.expect.unreachable('requireTestDevice must stop the test');
});
