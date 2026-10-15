// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {describeReachabilityStructure, runReachabilityScenario} from './network-reachability.lens';

/** Captures the default 32-hop chaining policy at the end, with global queue stamps only. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  return runReachabilityScenario(device, true, options);
}

/** Declares the default chained graph on a null device without execution. */
export async function describeStructure(device: Device) {
  return describeReachabilityStructure(device, true);
}
