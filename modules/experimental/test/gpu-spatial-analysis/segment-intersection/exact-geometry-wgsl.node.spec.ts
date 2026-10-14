// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {GPUDelaunayTessellation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface';
import {GPUZoneEvents} from '../../../src/gpu-spatial-analysis/trajectory-zones';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

// The symbolic and exact-sign kernels of the 2026-10-13 correctness round (Delaunay ghost
// triangles, overlay winding membership, buffer ring compaction, make-valid source parity, zone
// tie rule) are generated text; parse every kernel so a WGSL syntax error fails in node, before a
// GPU run.
type Format = 'float32' | 'float32x2' | 'float32x4' | 'uint32' | 'uint32x2' | 'uint32x3';

function createViewFactory(graph: GPUCommandGraph) {
  let serial = 0;
  return <F extends Format>(format: F, length: number) =>
    createTransientView(graph, `wgsl-parse-${serial++}`, format, length);
}

function expectKernelsParse(nodes: readonly unknown[], label: string): void {
  const kernels = getGPUCommandNodeKernelSources(nodes);
  expect(kernels.length, label).toBeGreaterThan(0);
  for (const kernel of kernels) {
    expect(() => new WgslReflect(kernel.source), `${label} ${kernel.id}`).not.toThrow();
  }
}

it('GPUDelaunayTessellation symbolic super-vertex kernels parse', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = createViewFactory(graph);
  const delaunay = new GPUDelaunayTessellation({
    positions: view('float32x2', 8),
    output: {
      triangles: view('uint32x3', 17),
      status: {
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1),
        invalidCount: view('uint32', 1)
      },
      duplicateCount: view('uint32', 1)
    }
  });
  expectKernelsParse(delaunay.getCommandNodes(graph), 'delaunay');
});

it('GPUBufferSurface compaction and winding membership kernels parse', () => {
  for (const kind of ['polygons', 'lines'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = createViewFactory(graph);
    const status = () => ({
      count: view('uint32', 1),
      requiredCount: view('uint32', 1),
      overflow: view('uint32', 1),
      candidateOverflow: view('uint32', 1)
    });
    const sourceIds = view('uint32', 8);
    const surface = new GPUBufferSurface({
      geometry:
        kind === 'polygons'
          ? {
              kind,
              positions: view('float32x2', 5),
              featureOffsets: view('uint32', 2),
              polygonOffsets: view('uint32', 2),
              ringOffsets: view('uint32', 2)
            }
          : {kind, positions: view('float32x2', 5), lineOffsets: view('uint32', 2)},
      parameters: view('float32', 4),
      joinStyle: 'mitre',
      vertexTolerance: 1e-5,
      capacity: {intersections: 64, nodedSegments: 128},
      output: {
        geometry: {
          kind: 'polygons',
          positions: view('float32x2', 64),
          featureOffsets: view('uint32', 9),
          polygonOffsets: view('uint32', 9),
          ringOffsets: view('uint32', 9),
          sourceIds
        },
        sourceIds,
        status: status(),
        boundary: {
          endpoints: view('float32x4', 64),
          operandIds: view('uint32', 64),
          sourceFeatureIds: view('uint32', 64),
          status: status()
        }
      }
    } as never);
    expectKernelsParse(surface.getCommandNodes(graph), `buffer ${kind}`);
  }
});

it('GPUMakeValid source-parity filter kernels parse', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = createViewFactory(graph);
  const status = () => ({
    count: view('uint32', 1),
    requiredCount: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1)
  });
  const segments = 64;
  const rings = 8;
  const makeValid = new GPUMakeValid({
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 8),
      featureOffsets: view('uint32', 2),
      polygonOffsets: view('uint32', 2),
      ringOffsets: view('uint32', 3)
    },
    intersectionCapacity: 64,
    precision: {vertexTolerance: 1e-5, predicates: 'exact-or-uncertain', storage: 'float32'},
    output: {
      nodedSegments: {
        endpoints: view('float32x4', segments),
        sourceFeatureIds: view('uint32', segments),
        sourceRingIds: view('uint32', segments),
        sourceEdgeIds: view('uint32', segments),
        sourceStartParameters: view('float32', segments),
        sourceEndParameters: view('float32', segments),
        status: status()
      },
      directedEdges: {
        endpoints: view('float32x4', 2 * segments),
        sourceSegmentIds: view('uint32', 2 * segments),
        sourceFeatureIds: view('uint32', 2 * segments),
        twinIds: view('uint32', 2 * segments),
        ringIds: view('uint32', 2 * segments),
        flags: view('uint32', 2 * segments),
        status: status()
      },
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 2 * segments),
        featureOffsets: view('uint32', rings + 1),
        polygonOffsets: view('uint32', rings + 1),
        ringOffsets: view('uint32', rings + 1),
        sourceIds: view('uint32', rings)
      },
      diagnostics: {
        edgeClasses: view('uint32', segments),
        closedRingCount: view('uint32', 1),
        openChainCount: view('uint32', 1),
        cutEdgeCount: view('uint32', 1),
        dangleCount: view('uint32', 1),
        unusedEdgeCount: view('uint32', 1)
      },
      status: status()
    }
  } as never);
  expectKernelsParse(makeValid.getCommandNodes(graph), 'make-valid');
});

it('GPUZoneEvents exact tie-rule kernels parse', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = createViewFactory(graph);
  const zoneEvents = new GPUZoneEvents({
    positions: view('float32x2', 8),
    timestamps: view('float32', 8),
    trackOffsets: view('uint32', 3),
    edgeStarts: view('float32x2', 4),
    edgeEnds: view('float32x2', 4),
    edgeZones: view('uint32', 4),
    zoneCount: 2,
    candidateCapacity: 16,
    maxEventsPerTrack: 4,
    events: {
      output: {ids: view('uint32', 8), count: view('uint32', 1), overflow: view('uint32', 1)},
      eventTimes: view('float32', 8),
      eventZones: view('uint32', 8),
      eventTypes: view('uint32', 8),
      eventRows: view('uint32', 8)
    },
    dwellTimes: view('float32', 4),
    visitCounts: view('uint32', 4),
    trackEventCounts: view('uint32', 2)
  } as never);
  expectKernelsParse(zoneEvents.getCommandNodes(graph), 'zones');
});
