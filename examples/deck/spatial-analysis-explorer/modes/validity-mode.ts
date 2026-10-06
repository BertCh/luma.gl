// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Validity: `GPUGeometryValidity` checks the San Francisco ZIP polygons plus seven synthetic polygons
 * whose rings can be switched between valid and invalid (same vertex counts, so the switch is a buffer
 * write). The per-feature bitmask is read back and each outline is colored by its first failing bit;
 * hover a polygon for all its bits. The graph encodes only when an input changed. The orientation
 * convention is a compile-time option (the graph is rebuilt and the control says so).
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_GEOMETRY_VALIDITY_BIT,
  GPUGeometryValidity,
  type GPUGeometryValidityOrientation
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {SummaryReader} from './summary-reader';
import {findContainingFeature} from './raster-join-layers';

const INTERSECTION_CAPACITY = 4096;
const NO_VALUE = 0xffffffff;
/** Failing bits in display priority, each with a palette slot. */
const VALIDITY_CLASSES = [
  {bit: GPU_GEOMETRY_VALIDITY_BIT.selfIntersection, name: 'selfIntersection', color: [255, 70, 70]},
  {bit: GPU_GEOMETRY_VALIDITY_BIT.crossingRings, name: 'crossingRings', color: [255, 148, 72]},
  {
    bit: GPU_GEOMETRY_VALIDITY_BIT.holeOutsideShell,
    name: 'holeOutsideShell',
    color: [245, 220, 87]
  },
  {bit: GPU_GEOMETRY_VALIDITY_BIT.repeatedVertex, name: 'repeatedVertex', color: [87, 235, 168]},
  {bit: GPU_GEOMETRY_VALIDITY_BIT.shortRing, name: 'shortRing', color: [78, 201, 255]},
  {bit: GPU_GEOMETRY_VALIDITY_BIT.badOrientation, name: 'badOrientation', color: [189, 122, 255]},
  {bit: GPU_GEOMETRY_VALIDITY_BIT.nonFinite, name: 'nonFinite', color: [255, 105, 168]},
  {bit: GPU_GEOMETRY_VALIDITY_BIT.uncertain, name: 'uncertain', color: [255, 255, 255]}
] as const;
const ALL_BIT_NAMES = [
  ['nonFinite', GPU_GEOMETRY_VALIDITY_BIT.nonFinite],
  ['unclosedRing', GPU_GEOMETRY_VALIDITY_BIT.unclosedRing],
  ['shortRing', GPU_GEOMETRY_VALIDITY_BIT.shortRing],
  ['repeatedVertex', GPU_GEOMETRY_VALIDITY_BIT.repeatedVertex],
  ['selfIntersection', GPU_GEOMETRY_VALIDITY_BIT.selfIntersection],
  ['crossingRings', GPU_GEOMETRY_VALIDITY_BIT.crossingRings],
  ['holeOutsideShell', GPU_GEOMETRY_VALIDITY_BIT.holeOutsideShell],
  ['badOrientation', GPU_GEOMETRY_VALIDITY_BIT.badOrientation],
  ['uncertain', GPU_GEOMETRY_VALIDITY_BIT.uncertain]
] as const;

type Point = [number, number];

/** Synthetic polygons: the same vertex counts valid and invalid, so a toggle is a buffer write. */
const SYNTHETIC_FEATURES: readonly {name: string; valid: Point[][]; invalid: Point[][]}[] = [
  {
    name: 'bowtie',
    valid: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
      ]
    ],
    invalid: [
      [
        [0, 0],
        [1, 1],
        [1, 0],
        [0, 1]
      ]
    ]
  },
  {
    name: 'repeated vertex',
    valid: [
      [
        [0.5, 0],
        [1, 0.4],
        [0.8, 1],
        [0.2, 1],
        [0, 0.4]
      ]
    ],
    invalid: [
      [
        [0.5, 0],
        [0.5, 0],
        [0.8, 1],
        [0.2, 1],
        [0, 0.4]
      ]
    ]
  },
  {
    name: 'short ring',
    valid: [
      [
        [0, 0],
        [1, 0],
        [0.5, 1]
      ]
    ],
    invalid: [
      [
        [0, 0],
        [1, 0],
        [0, 0]
      ]
    ]
  },
  {
    name: 'hole outside shell',
    valid: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
      ],
      [
        [0.3, 0.3],
        [0.3, 0.7],
        [0.7, 0.7],
        [0.7, 0.3]
      ]
    ],
    invalid: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
      ],
      [
        [1.5, 0.3],
        [1.5, 0.7],
        [1.9, 0.7],
        [1.9, 0.3]
      ]
    ]
  },
  {
    name: 'ring crossing the shell',
    valid: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
      ],
      [
        [0.3, 0.3],
        [0.3, 0.7],
        [0.7, 0.7],
        [0.7, 0.3]
      ]
    ],
    invalid: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
      ],
      [
        [0.7, 0.3],
        [0.7, 0.7],
        [1.3, 0.7],
        [1.3, 0.3]
      ]
    ]
  },
  {
    name: 'clockwise shell',
    valid: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
      ]
    ],
    invalid: [
      [
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0]
      ]
    ]
  },
  {
    name: 'non-finite vertex',
    valid: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
      ]
    ],
    invalid: [
      [
        [0, 0],
        [1, 0],
        [Number.NaN, 1],
        [0, 1]
      ]
    ]
  }
];

type PolygonScene = {
  polygonPositions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  featureCount: number;
  zipCount: number;
  zipVertexCount: number;
  outlineSegments: Float32Array;
  outlineRows: Uint32Array;
  /** Writes the synthetic polygons' vertices and outline segments. */
  writeSynthetic: (invalid: boolean) => void;
  names: string[];
};

/** ZIP polygons followed by the synthetic features, laid out around the ZIP extent. */
function createPolygonScene(zips: {
  polygonPositions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  featureNames: readonly string[];
  outlineSegments: Float32Array;
  outlineFeatureRows: Uint32Array;
}): PolygonScene {
  const zipCount = zips.featureOffsets.length - 1;
  const zipVertexCount = zips.polygonPositions.length / 2;
  const ringStarts = Array.from(zips.ringOffsets);
  const polygonOffsets = Array.from(zips.polygonOffsets);
  const featureOffsets = Array.from(zips.featureOffsets);
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < zips.polygonPositions.length; index += 2) {
    minX = Math.min(minX, zips.polygonPositions[index]);
    maxX = Math.max(maxX, zips.polygonPositions[index]);
    minY = Math.min(minY, zips.polygonPositions[index + 1]);
    maxY = Math.max(maxY, zips.polygonPositions[index + 1]);
  }
  const span = Math.min(maxX - minX, maxY - minY);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const size = 0.07 * span;
  let vertexCount = zipVertexCount;
  const syntheticRingVertexCounts: number[][] = [];
  SYNTHETIC_FEATURES.forEach(feature => {
    const ringCounts: number[] = [];
    for (const ring of feature.valid) {
      vertexCount += ring.length;
      ringStarts.push(vertexCount);
      ringCounts.push(ring.length);
    }
    syntheticRingVertexCounts.push(ringCounts);
    polygonOffsets.push(ringStarts.length - 1);
    featureOffsets.push(polygonOffsets.length - 1);
  });
  const positions = new Float32Array(vertexCount * 2);
  positions.set(zips.polygonPositions);
  const syntheticSegmentCount = vertexCount - zipVertexCount;
  const zipSegmentCount = zips.outlineSegments.length / 4;
  const outlineSegments = new Float32Array((zipSegmentCount + syntheticSegmentCount) * 4);
  outlineSegments.set(zips.outlineSegments);
  const outlineRows = new Uint32Array(zipSegmentCount + syntheticSegmentCount);
  outlineRows.set(zips.outlineFeatureRows);
  let row = zipSegmentCount;
  SYNTHETIC_FEATURES.forEach((feature, index) => {
    for (const ring of feature.valid)
      for (let vertex = 0; vertex < ring.length; vertex++) outlineRows[row++] = zipCount + index;
  });
  const writeSynthetic = (invalid: boolean) => {
    let vertexCursor = zipVertexCount;
    let segmentCursor = zipSegmentCount;
    SYNTHETIC_FEATURES.forEach((feature, index) => {
      const angle = (index / SYNTHETIC_FEATURES.length) * Math.PI * 2 + 0.4;
      const originX = centerX + Math.cos(angle) * 0.3 * span - size;
      const originY = centerY + Math.sin(angle) * 0.3 * span - size;
      for (const ring of invalid ? feature.invalid : feature.valid) {
        const first = vertexCursor;
        ring.forEach(([x, y], vertex) => {
          positions[(first + vertex) * 2] = originX + x * size * 2;
          positions[(first + vertex) * 2 + 1] = originY + y * size * 2;
        });
        for (let vertex = 0; vertex < ring.length; vertex++) {
          const next = (vertex + 1) % ring.length;
          outlineSegments[segmentCursor * 4] = positions[(first + vertex) * 2];
          outlineSegments[segmentCursor * 4 + 1] = positions[(first + vertex) * 2 + 1];
          outlineSegments[segmentCursor * 4 + 2] = positions[(first + next) * 2];
          outlineSegments[segmentCursor * 4 + 3] = positions[(first + next) * 2 + 1];
          segmentCursor++;
        }
        vertexCursor += ring.length;
      }
    });
  };
  writeSynthetic(false);
  return {
    polygonPositions: positions,
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringStarts),
    featureCount: zipCount + SYNTHETIC_FEATURES.length,
    zipCount,
    zipVertexCount,
    outlineSegments,
    outlineRows,
    writeSynthetic,
    names: [...zips.featureNames, ...SYNTHETIC_FEATURES.map(feature => feature.name)]
  };
}

function describeMask(mask: number): string {
  const names = ALL_BIT_NAMES.filter(([, bit]) => mask & bit).map(([name]) => name);
  return names.length ? names.join(', ') : 'valid';
}

export const validityMode: SpatialAnalysisModeDefinition = {
  id: 'validity',
  title: 'Validity',
  contributors: ['GPUGeometryValidity'],
  description:
    'Per-polygon validity bits on San Francisco ZIP codes. Inject invalid rings and hover a polygon to see ' +
    'which bits the GPU set; outlines are colored by the first failing bit.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.7},

  async create(context) {
    const zips = await context.data.getSanFranciscoZipCodes();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'validity');
    const projection = new LocalMetricProjection(zips.origin);
    // ------------------------------------------------------------------ polygon validity
    const polygonScene = createPolygonScene(zips);
    const zipPositions = resources.createBuffer('zip-positions', polygonScene.polygonPositions);
    const zipFeatureOffsets = resources.createBuffer(
      'zip-feature-offsets',
      polygonScene.featureOffsets
    );
    const zipPolygonOffsets = resources.createBuffer(
      'zip-polygon-offsets',
      polygonScene.polygonOffsets
    );
    const zipRingOffsets = resources.createBuffer('zip-ring-offsets', polygonScene.ringOffsets);
    const outlineSegments = resources.createBuffer(
      'outline-segments',
      polygonScene.outlineSegments
    );
    const outlineRows = resources.createBuffer('outline-rows', polygonScene.outlineRows);
    const validityMask = resources.createBuffer('validity-mask', polygonScene.featureCount * 4);
    const validityOverflow = resources.createBuffer('validity-overflow', 4);
    const validityIntersections = resources.createBuffer('validity-intersections', 4);
    const failingClass = resources.createBuffer('failing-class', polygonScene.featureCount * 4);
    const buildValidityGraph = (orientation: GPUGeometryValidityOrientation) => {
      const graph = new GPUCommandGraph<void>(device, {id: `validity-polygons-${orientation}`});
      graph.add(
        new GPUGeometryValidity({
          id: 'zip-validity',
          polygons: {
            kind: 'polygons',
            positions: importGraphBuffer(
              graph,
              'zip-positions',
              zipPositions,
              'float32x2',
              polygonScene.polygonPositions.length / 2
            ),
            featureOffsets: importGraphBuffer(
              graph,
              'zip-feature-offsets',
              zipFeatureOffsets,
              'uint32',
              polygonScene.featureOffsets.length
            ),
            polygonOffsets: importGraphBuffer(
              graph,
              'zip-polygon-offsets',
              zipPolygonOffsets,
              'uint32',
              polygonScene.polygonOffsets.length
            ),
            ringOffsets: importGraphBuffer(
              graph,
              'zip-ring-offsets',
              zipRingOffsets,
              'uint32',
              polygonScene.ringOffsets.length
            )
          },
          mask: importGraphBuffer(
            graph,
            'validity-mask',
            validityMask,
            'uint32',
            polygonScene.featureCount
          ),
          intersectionCapacity: INTERSECTION_CAPACITY,
          overflow: importGraphBuffer(graph, 'validity-overflow', validityOverflow, 'uint32', 1),
          intersectionCount: importGraphBuffer(
            graph,
            'validity-intersections',
            validityIntersections,
            'uint32',
            1
          ),
          orientation
        })
      );
      return graph.compile();
    };
    let orientation: GPUGeometryValidityOrientation = 'ignore';
    let validityGraph = resources.track(buildValidityGraph(orientation));

    // ------------------------------------------------------------------ state and controls
    let injectInvalid = false;
    let polygonsDirty = true;
    let destroyed = false;
    let latestMask = new Uint32Array(polygonScene.featureCount);

    context.controls.addToggle({
      label: 'Inject invalid rings (7 synthetic polygons)',
      value: injectInvalid,
      onChange: value => {
        injectInvalid = value;
        polygonScene.writeSynthetic(injectInvalid);
        zipPositions.write(polygonScene.polygonPositions);
        outlineSegments.write(polygonScene.outlineSegments);
        polygonsDirty = true;
      }
    });
    context.controls.addSelect<GPUGeometryValidityOrientation>({
      label: 'Ring orientation convention (compile-time: rebuilds graph)',
      options: [
        {value: 'counter-clockwise-shell', label: 'counter-clockwise shell'},
        {value: 'clockwise-shell', label: 'clockwise shell'},
        {value: 'ignore', label: 'ignore'}
      ],
      value: orientation,
      onChange: value => {
        orientation = value;
        const previous = validityGraph;
        validityGraph = resources.track(buildValidityGraph(orientation));
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            if (!destroyed) resources.release(previous);
          })
        );
        polygonsDirty = true;
      }
    });
    context.controls.addLegend({
      title: 'ZIP outline: first failing bit (valid = thin gray)',
      entries: VALIDITY_CLASSES.map(item => ({color: item.color, label: item.name}))
    });
    const validReadout = context.controls.addReadout('Valid polygons', '...');
    const bitReadout = context.controls.addReadout('Failing bits', '...');
    const validityFlagReadout = context.controls.addReadout('Intersection list', '...');
    context.controls.addNote(
      'Hover a polygon for its bits. The graph encodes only when an input changed.'
    );
    context.controls.addReadout('Data', zips.attribution);

    const polygonReader = new SummaryReader(
      resources,
      'validity-polygons',
      [
        {buffer: validityOverflow, size: 4},
        {buffer: validityIntersections, size: 4},
        {buffer: validityMask, size: polygonScene.featureCount * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        latestMask = words.slice(2, 2 + polygonScene.featureCount);
        const classes = new Uint32Array(polygonScene.featureCount);
        const bitCounts = new Map<string, number>();
        let validCount = 0;
        latestMask.forEach((mask, row) => {
          const failing = VALIDITY_CLASSES.findIndex(item => mask & item.bit);
          classes[row] = failing < 0 ? NO_VALUE : failing;
          if (mask === 0) validCount++;
          for (const [name, bit] of ALL_BIT_NAMES) {
            if (mask & bit) bitCounts.set(name, (bitCounts.get(name) ?? 0) + 1);
          }
        });
        failingClass.write(classes);
        validReadout.setValue(
          `${validCount} of ${polygonScene.featureCount} (${polygonScene.zipCount} ZIPs + ${SYNTHETIC_FEATURES.length} synthetic)`
        );
        bitReadout.setValue(
          bitCounts.size
            ? [...bitCounts.entries()].map(([name, number]) => `${name} ${number}`).join(', ')
            : 'none'
        );
        validityFlagReadout.setValue(
          `${formatCount(words[1])} same-feature hits${words[0] ? ', OVERFLOW' : ''}`
        );
      }
    );
    resources.track({
      destroy: () => {
        polygonReader.stop();
      }
    });

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [validityGraph] as readonly CompiledGPUCommandGraph<never>[],
      encode(commandEncoder) {
        if (polygonsDirty) {
          polygonsDirty = false;
          validityGraph.encode(commandEncoder, {parameters: undefined});
          polygonReader.request(commandEncoder);
        } else {
          polygonReader.flush(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [zips.origin[0], zips.origin[1], 0];
        const layers: Layer[] = [];
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'validity-zip-outline',
            coordinateOrigin,
            segments: outlineSegments,
            instanceCount: polygonScene.outlineRows.length,
            color: [170, 185, 205, 190],
            widthPixels: 1.5
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'validity-zip-failing',
            coordinateOrigin,
            segments: outlineSegments,
            instanceCount: polygonScene.outlineRows.length,
            values: failingClass,
            valueFormat: 'uint32',
            valueIndices: outlineRows,
            colormap: 'category',
            palette: VALIDITY_CLASSES.map(item => [...item.color, 255] as const),
            noDataColor: [0, 0, 0, 0],
            widthPixels: 5
          })
        );
        return layers;
      },
      getTooltip(event) {
        if (!event.coordinate) return null;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const row = findContainingFeature(polygonScene, x, y);
        if (row < 0) return null;
        const name = polygonScene.names[row];
        const mask = latestMask[row] ?? 0;
        return `${name}\nmask ${mask} = ${describeMask(mask)}`;
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
