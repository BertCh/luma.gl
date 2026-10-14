// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPU_PLANAR_SPATIAL_CONTEXT} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUAffineTransform} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-affine-transform';
import {GPUGeometryCleanup} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-geometry-cleanup';
import {GPUGeometryOrientation} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-geometry-orientation';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import {GPULineNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-line-noding';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPUGeometryValidity} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {GPULineClipByPolygon} from '../../../src/gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon';
import {GPUSharedPaths} from '../../../src/gpu-spatial-analysis/line-clip/gpu-shared-paths';
import {GPULineMerge} from '../../../src/gpu-spatial-analysis/line-merge/gpu-line-merge';
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/gpu-line-split';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification';
import {GPUPolygonOverlay} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {GPUPolygonTriangulation} from '../../../src/gpu-spatial-analysis/polygon-triangulation/gpu-polygon-triangulation';
import {GPURectangleClip} from '../../../src/gpu-spatial-analysis/rectangle-clip/gpu-rectangle-clip';
import {GPUSegmentIntersection} from '../../../src/gpu-spatial-analysis/segment-intersection/gpu-segment-intersection';
import {GPUVertexSnap} from '../../../src/gpu-spatial-analysis/vertex-snap/gpu-vertex-snap';

// Every planar-only planar-sweep operator resolves its spatial context before it reads any other
// property, so a context-only property bag is enough to observe the context check.
type PlanarConstructor = new (props: never) => {readonly spatialContext: unknown};
// GPULineSimplification has a great-circle path; its context matrix is in line-simplification/.
const PLANAR_OPERATORS: Record<string, unknown> = {
  GPUAffineTransform,
  GPUBufferSurface,
  GPUCoverageSimplification,
  GPUGeometryCleanup,
  GPUGeometryOrientation,
  GPUGeometryPredicates,
  GPUGeometryValidity,
  GPULineClipByPolygon,
  GPULineMerge,
  GPULineNoding,
  GPULineSplit,
  GPUMakeValid,
  GPUPolygonOverlay,
  GPUPolygonTriangulation,
  GPUPolygonize,
  GPURectangleClip,
  GPUSegmentIntersection,
  GPUSharedPaths,
  GPUVertexSnap
};

for (const [name, operator] of Object.entries(PLANAR_OPERATORS)) {
  it(`${name} rejects longitude-latitude spatial contexts and accepts planar ones`, () => {
    const Operator = operator as PlanarConstructor;
    for (const metric of ['none', 'great-circle'] as const) {
      expect(
        () =>
          new Operator({spatialContext: {coordinateSpace: 'longitude-latitude', metric}} as never)
      ).toThrow(/does not support longitude-latitude coordinates; project positions/);
    }
    // The planar context passes the context check; other property validation may still throw.
    for (const spatialContext of [undefined, GPU_PLANAR_SPATIAL_CONTEXT]) {
      try {
        const instance = new Operator({spatialContext} as never);
        expect(instance.spatialContext).toBeTruthy();
      } catch (error) {
        expect(String((error as Error).message)).not.toMatch(/does not support|spatial context/);
      }
    }
  });
}
