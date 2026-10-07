// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisSegmentLayerProps
} from '../../engine/layers';

/** The weight line of the shared segment shader that {@link AgeFadeTrailLayer} patches. */
const WEIGHT_LINE = 'color.a = color.a * segmentWeights[row];';

/**
 * The trail segments of `GPUTimeWindowFilter`, drawn with age as alpha only: the contributor's
 * fade weight runs linearly from 1 at the head to 0 at the tail of the window (one minus age over
 * the trail length), and this layer squares it, so a trail has alpha `(1 - age / tail)^2`, the
 * curve the trail diagram in the story draws. Everything else is the shared segment layer: the
 * compact ids, the clip fractions at the window edges, the style channels and `blending`.
 */
export class AgeFadeTrailLayer extends SpatialAnalysisSegmentLayer {
  static override layerName = 'AgeFadeTrailLayer';

  constructor(props: SpatialAnalysisSegmentLayerProps) {
    super(props);
  }

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    if (!source.includes(WEIGHT_LINE)) {
      throw new Error('AgeFadeTrailLayer could not find the segment weight line');
    }
    return source.replace(
      WEIGHT_LINE,
      'color.a = color.a * segmentWeights[row] * segmentWeights[row];'
    );
  }
}
