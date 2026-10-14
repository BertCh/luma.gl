// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GraphResourceUse} from '@luma.gl/gpgpu/gpu-core';

/**
 * Returns the buffer view a node declares for a resource use, or undefined for textures. The
 * specs bind views, so the buffer of a use is compared and measured as one.
 */
export function getResourceBuffer(resource: GraphResourceUse): GraphDataView | undefined {
  return 'buffer' in resource ? (resource.buffer as GraphDataView) : undefined;
}
