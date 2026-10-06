// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUOutlineGeometryParameterValues,
  getGPUOutlineGeometryVerticesPerInput,
  getGPURectangleClipParameterValues,
  getGPUShapeDescriptorsParameterValues,
  GPULabelPoint,
  GPUOutlineGeometry,
  GPURectangleClip,
  GPUShapeDescriptors,
  GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH,
  GPU_RECTANGLE_CLIP_PARAMETER_LENGTH,
  GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {PairSegmentLayer, PolygonFanLayer} from './geometry-layers';
import {TriangleListLayer} from './geometry-tools-layers';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';

const CIRCLE_SEGMENTS = 40;
/** Round-join resolution of the outline buffer (compile-time). */
const JOIN_SEGMENTS = 8;
const DESCRIPTOR_NAMES = ['polsbyPopper', 'schwartzberg', 'elongation', 'convexity'] as const;

type DescriptorChoice = (typeof DESCRIPTOR_NAMES)[number] | 'none';

const DESCRIPTOR_LABELS: Record<DescriptorChoice, string> = {
  polsbyPopper: 'Polsby-Popper compactness (1 = circle)',
  schwartzberg: 'Schwartzberg compactness (1 = circle)',
  elongation: 'Elongation (0 = round, 1 = a line)',
  convexity: 'Convexity (area / convex hull area)',
  none: 'None (outlines only)'
};

type Statistics = {minimum: number; mean: number; maximum: number};

function getStatistics(values: Float32Array): Statistics {
  let minimum = Infinity;
  let maximum = -Infinity;
  let sum = 0;
  let count = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sum += value;
    count++;
  }
  return {minimum, mean: count ? sum / count : NaN, maximum};
}

/**
 * Geometry utilities on the San Francisco ZIP code polygons: a live outline buffer
 * (`GPUOutlineGeometry`, render only), inscribed label anchors (`GPULabelPoint`), shape
 * descriptors that color the polygons (`GPUShapeDescriptors`) and a draggable clip rectangle
 * (`GPURectangleClip`). Every slider, select and drag is a buffer write; the three compiled graphs
 * encode only when their inputs changed.
 */
export const geometryToolsMode: SpatialAnalysisModeDefinition = {
  id: 'geometry-tools',
  title: 'Geometry tools',
  contributors: ['GPUOutlineGeometry', 'GPULabelPoint', 'GPUShapeDescriptors', 'GPURectangleClip'],
  description:
    'San Francisco ZIP codes colored by a shape descriptor, each with its inscribed label ' +
    'circle and long-axis glyph. Pick a descriptor, grow the outline buffer, then drag the ' +
    'rectangle (or click) to clip the polygons on the GPU.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.3},

  async create(context) {
    const zones = await context.data.getSanFranciscoZipCodes();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(zones.origin);
    const resources = new SpatialAnalysisResources(device, 'geometry-tools');
    const featureCount = zones.featureIds.length;
    const vertexCount = zones.polygonPositions.length / 2;
    const ringCount = zones.ringOffsets.length - 1;
    const edgeCount = zones.outlineSegments.length / 4;
    const coordinateOrigin: [number, number, number] = [zones.origin[0], zones.origin[1], 0];

    const featureRingOffsets = new Uint32Array(featureCount + 1);
    for (let feature = 0; feature <= featureCount; feature++) {
      featureRingOffsets[feature] = zones.polygonOffsets[zones.featureOffsets[feature]];
    }
    let minimumX = Infinity;
    let minimumY = Infinity;
    let maximumX = -Infinity;
    let maximumY = -Infinity;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const x = zones.polygonPositions[vertex * 2];
      const y = zones.polygonPositions[vertex * 2 + 1];
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);
    }

    // State.
    let descriptor: DescriptorChoice = 'polsbyPopper';
    let outlineDistance = 120;
    let showOutline = true;
    let showLabels = true;
    let showGlyphs = true;
    let showClip = true;
    let rectangle = {
      centerX: (minimumX + maximumX) / 2,
      centerY: (minimumY + maximumY) / 2,
      halfWidth: (maximumX - minimumX) * 0.18,
      halfHeight: (maximumY - minimumY) * 0.18
    };
    let outlineDirty = true;
    let clipDirty = true;
    let dragging = false;
    let dragOffset: [number, number] = [0, 0];
    let statistics: Record<(typeof DESCRIPTOR_NAMES)[number], Statistics> | null = null;
    let radii: Float32Array | null = null;
    let descriptorColumns: Record<(typeof DESCRIPTOR_NAMES)[number], Float32Array> | null = null;

    // Inputs.
    const positionsBuffer = resources.createBuffer('positions', zones.polygonPositions);
    const ringOffsetsBuffer = resources.createBuffer('ring-offsets', zones.ringOffsets);
    const featureRingsBuffer = resources.createBuffer('feature-rings', featureRingOffsets);
    const edgeSegments = resources.createBuffer('edge-segments', zones.outlineSegments);
    const edgeFeatureRows = resources.createBuffer('edge-feature-rows', zones.outlineFeatureRows);
    const outlineParameters = resources.createParameterBuffer(
      'outline-parameters',
      'float32',
      GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH
    );
    const shapeParameters = resources.createParameterBuffer(
      'shape-parameters',
      'float32',
      GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH,
      getGPUShapeDescriptorsParameterValues()
    );
    const clipParameters = resources.createParameterBuffer(
      'clip-parameters',
      'float32',
      GPU_RECTANGLE_CLIP_PARAMETER_LENGTH
    );

    // Outputs.
    const verticesPerInput = getGPUOutlineGeometryVerticesPerInput(JOIN_SEGMENTS);
    const outlineVertexCount = vertexCount * verticesPerInput;
    const outlineTriangles = resources.createBuffer('outline-triangles', outlineVertexCount * 8);
    const labelPoints = resources.createBuffer('label-points', featureCount * 8);
    const labelRadii = resources.createBuffer('label-radii', featureCount * 4);
    const labelDegenerate = resources.createBuffer('label-degenerate', featureCount * 4);
    const columns = Object.fromEntries(
      [...DESCRIPTOR_NAMES, 'orientation'].map(name => [
        name,
        resources.createBuffer(name, featureCount * 4)
      ])
    ) as Record<
      (typeof DESCRIPTOR_NAMES)[number] | 'orientation',
      ReturnType<typeof resources.createBuffer>
    >;
    const circleStarts = resources.createBuffer(
      'circle-starts',
      featureCount * CIRCLE_SEGMENTS * 8
    );
    const circleEnds = resources.createBuffer('circle-ends', featureCount * CIRCLE_SEGMENTS * 8);
    const glyphStarts = resources.createBuffer('glyph-starts', featureCount * 8);
    const glyphEnds = resources.createBuffer('glyph-ends', featureCount * 8);
    const clipCapacity = vertexCount * 2 + 64;
    const clipPositions = resources.createBuffer('clip-positions', clipCapacity * 8);
    const clipOffsets = resources.createBuffer('clip-offsets', (ringCount + 1) * 4);
    const clipCount = resources.createBuffer('clip-count', 4);
    const clipOverflow = resources.createBuffer('clip-overflow', 4);
    const clipStarts = resources.createBuffer('clip-starts', clipCapacity * 8);
    const clipEnds = resources.createBuffer('clip-ends', clipCapacity * 8);
    const rectangleSegments = resources.createBuffer('rectangle-segments', 4 * 16);

    // Shape graph (static): label points, descriptors, circle and glyph segments.
    const shapeGraph = new GPUCommandGraph<void>(device, {id: 'geometry-tools-shape'});
    {
      const positions = importGraphBuffer(
        shapeGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        vertexCount
      );
      const ringOffsets = importGraphBuffer(
        shapeGraph,
        'ring-offsets',
        ringOffsetsBuffer,
        'uint32',
        ringCount + 1
      );
      const featureRings = importGraphBuffer(
        shapeGraph,
        'feature-rings',
        featureRingsBuffer,
        'uint32',
        featureCount + 1
      );
      const points = importGraphBuffer(
        shapeGraph,
        'label-points',
        labelPoints,
        'float32x2',
        featureCount
      );
      const radiiView = importGraphBuffer(
        shapeGraph,
        'label-radii',
        labelRadii,
        'float32',
        featureCount
      );
      const degenerateView = importGraphBuffer(
        shapeGraph,
        'label-degenerate',
        labelDegenerate,
        'uint32',
        featureCount
      );
      shapeGraph.add(
        new GPULabelPoint({
          id: 'label-point',
          positions,
          ringOffsets,
          featureRingOffsets: featureRings,
          output: {points, distances: radiiView, degenerate: degenerateView}
        })
      );
      const columnViews = Object.fromEntries(
        Object.entries(columns).map(([name, buffer]) => [
          name,
          importGraphBuffer(shapeGraph, name, buffer, 'float32', featureCount)
        ])
      ) as Record<keyof typeof columns, GraphDataView<'float32'>>;
      shapeGraph.add(
        new GPUShapeDescriptors({
          id: 'shape-descriptors',
          positions,
          ringOffsets,
          featureRingOffsets: featureRings,
          holeRule: 'first-ring-exterior',
          parameters: shapeParameters.importToGraph(shapeGraph),
          output: {
            polsbyPopper: columnViews.polsbyPopper,
            schwartzberg: columnViews.schwartzberg,
            elongation: columnViews.elongation,
            convexity: columnViews.convexity,
            orientation: columnViews.orientation
          }
        })
      );
      addKernelPass(shapeGraph, {
        id: 'inscribed-circles',
        invocationCount: featureCount * CIRCLE_SEGMENTS,
        bindings: [
          {name: 'points', view: points, type: 'f32', access: 'read'},
          {name: 'radii', view: radiiView, type: 'f32', access: 'read'},
          {name: 'degenerate', view: degenerateView, type: 'u32', access: 'read'},
          {
            name: 'starts',
            view: importGraphBuffer(
              shapeGraph,
              'circle-starts',
              circleStarts,
              'float32',
              featureCount * CIRCLE_SEGMENTS * 2
            ),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'ends',
            view: importGraphBuffer(
              shapeGraph,
              'circle-ends',
              circleEnds,
              'float32',
              featureCount * CIRCLE_SEGMENTS * 2
            ),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const SEGMENTS: u32 = ${CIRCLE_SEGMENTS}u;`,
        body: /* wgsl */ `
  let feature = index / SEGMENTS;
  let step = index % SEGMENTS;
  let center = vec2<f32>(points[pointsOffset + feature * 2u], points[pointsOffset + feature * 2u + 1u]);
  let radius = radii[radiiOffset + feature];
  let zero = f32(index) * 0.0;
  var start = vec2<f32>(zero / zero);
  var end = start;
  if (degenerate[degenerateOffset + feature] == 0u && radius == radius) {
    let angle0 = 6.2831853 * f32(step) / f32(SEGMENTS);
    let angle1 = 6.2831853 * f32(step + 1u) / f32(SEGMENTS);
    start = center + radius * vec2<f32>(cos(angle0), sin(angle0));
    end = center + radius * vec2<f32>(cos(angle1), sin(angle1));
  }
  starts[startsOffset + index * 2u] = start.x;
  starts[startsOffset + index * 2u + 1u] = start.y;
  ends[endsOffset + index * 2u] = end.x;
  ends[endsOffset + index * 2u + 1u] = end.y;`
      });
      addKernelPass(shapeGraph, {
        id: 'orientation-glyphs',
        invocationCount: featureCount,
        bindings: [
          {name: 'points', view: points, type: 'f32', access: 'read'},
          {name: 'orientation', view: columnViews.orientation, type: 'f32', access: 'read'},
          {name: 'elongation', view: columnViews.elongation, type: 'f32', access: 'read'},
          {
            name: 'starts',
            view: importGraphBuffer(
              shapeGraph,
              'glyph-starts',
              glyphStarts,
              'float32',
              featureCount * 2
            ),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'ends',
            view: importGraphBuffer(
              shapeGraph,
              'glyph-ends',
              glyphEnds,
              'float32',
              featureCount * 2
            ),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: /* wgsl */ `
  let center = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  let angle = orientation[orientationOffset + index];
  let halfLength = 120.0 + 900.0 * elongation[elongationOffset + index];
  let direction = vec2<f32>(cos(angle), sin(angle)) * halfLength;
  let start = center - direction;
  let end = center + direction;
  starts[startsOffset + index * 2u] = start.x;
  starts[startsOffset + index * 2u + 1u] = start.y;
  ends[endsOffset + index * 2u] = end.x;
  ends[endsOffset + index * 2u + 1u] = end.y;`
      });
    }
    const compiledShape = resources.track(shapeGraph.compile());

    // Outline graph: re-encoded when the distance slider moves.
    const outlineGraph = new GPUCommandGraph<void>(device, {id: 'geometry-tools-outline'});
    outlineGraph.add(
      new GPUOutlineGeometry({
        id: 'outline',
        positions: importGraphBuffer(
          outlineGraph,
          'positions',
          positionsBuffer,
          'float32x2',
          vertexCount
        ),
        geometryType: 'rings',
        pathOffsets: importGraphBuffer(
          outlineGraph,
          'ring-offsets',
          ringOffsetsBuffer,
          'uint32',
          ringCount + 1
        ),
        joinSegments: JOIN_SEGMENTS,
        parameters: outlineParameters.importToGraph(outlineGraph),
        output: {
          positions: importGraphBuffer(
            outlineGraph,
            'outline-triangles',
            outlineTriangles,
            'float32x2',
            outlineVertexCount
          )
        }
      })
    );
    const compiledOutline = resources.track(outlineGraph.compile());

    // Clip graph: re-encoded when the rectangle moves.
    const clipGraph = new GPUCommandGraph<void>(device, {id: 'geometry-tools-clip'});
    {
      const clippedPositions = importGraphBuffer(
        clipGraph,
        'clip-positions',
        clipPositions,
        'float32x2',
        clipCapacity
      );
      const clippedOffsets = importGraphBuffer(
        clipGraph,
        'clip-offsets',
        clipOffsets,
        'uint32',
        ringCount + 1
      );
      const countView = importGraphBuffer(clipGraph, 'clip-count', clipCount, 'uint32', 1);
      clipGraph.add(
        new GPURectangleClip({
          id: 'rectangle-clip',
          positions: importGraphBuffer(
            clipGraph,
            'positions',
            positionsBuffer,
            'float32x2',
            vertexCount
          ),
          geometryType: 'polygons',
          pathOffsets: importGraphBuffer(
            clipGraph,
            'ring-offsets',
            ringOffsetsBuffer,
            'uint32',
            ringCount + 1
          ),
          parameters: clipParameters.importToGraph(clipGraph),
          output: {
            positions: clippedPositions,
            pathOffsets: clippedOffsets,
            count: countView,
            overflow: importGraphBuffer(clipGraph, 'clip-overflow', clipOverflow, 'uint32', 1)
          }
        })
      );
      // Rings come out open: emit one segment per vertex, the last one closing back to the start.
      addKernelPass(clipGraph, {
        id: 'close-clipped-rings',
        invocationCount: clipCapacity,
        bindings: [
          {name: 'positions', view: clippedPositions, type: 'f32', access: 'read'},
          {name: 'offsets', view: clippedOffsets, type: 'u32', access: 'read'},
          {name: 'count', view: countView, type: 'u32', access: 'read'},
          {
            name: 'starts',
            view: importGraphBuffer(
              clipGraph,
              'clip-starts',
              clipStarts,
              'float32',
              clipCapacity * 2
            ),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'ends',
            view: importGraphBuffer(clipGraph, 'clip-ends', clipEnds, 'float32', clipCapacity * 2),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const OFFSET_COUNT: u32 = ${ringCount + 1}u;`,
        body: /* wgsl */ `
  let total = count[countOffset];
  let zero = f32(index) * 0.0;
  var start = vec2<f32>(zero / zero);
  var end = start;
  if (index < total) {
    var low = 0u;
    var high = OFFSET_COUNT;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (offsets[offsetsOffset + middle] <= index) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let path = max(low, 1u) - 1u;
    let first = offsets[offsetsOffset + path];
    var last = total;
    if (path + 1u < OFFSET_COUNT) {
      last = min(offsets[offsetsOffset + path + 1u], total);
    }
    if (last - first >= 2u) {
      var next = index + 1u;
      if (next >= last) {
        next = first;
      }
      start = vec2<f32>(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
      end = vec2<f32>(positions[positionsOffset + next * 2u], positions[positionsOffset + next * 2u + 1u]);
    }
  }
  starts[startsOffset + index * 2u] = start.x;
  starts[startsOffset + index * 2u + 1u] = start.y;
  ends[endsOffset + index * 2u] = end.x;
  ends[endsOffset + index * 2u + 1u] = end.y;`
      });
    }
    const compiledClip = resources.track(clipGraph.compile());

    // Parameter writes.
    const writeOutline = () => {
      outlineParameters.write(getGPUOutlineGeometryParameterValues({distance: outlineDistance}));
      outlineDirty = true;
    };
    const writeRectangle = () => {
      const {centerX, centerY, halfWidth, halfHeight} = rectangle;
      const left = centerX - halfWidth;
      const right = centerX + halfWidth;
      const bottom = centerY - halfHeight;
      const top = centerY + halfHeight;
      clipParameters.write(
        getGPURectangleClipParameterValues({minX: left, minY: bottom, maxX: right, maxY: top})
      );
      rectangleSegments.write(
        Float32Array.of(
          left,
          bottom,
          right,
          bottom,
          right,
          bottom,
          right,
          top,
          right,
          top,
          left,
          top,
          left,
          top,
          left,
          bottom
        )
      );
      clipDirty = true;
    };
    writeOutline();
    writeRectangle();

    // ---- Controls ----
    context.controls.addSelect<DescriptorChoice>({
      label: 'Color polygons by (GPUShapeDescriptors column)',
      options: (Object.keys(DESCRIPTOR_LABELS) as DescriptorChoice[]).map(value => ({
        value,
        label: DESCRIPTOR_LABELS[value]
      })),
      value: descriptor,
      onChange: value => {
        descriptor = value;
        context.updateLayers();
        updateReadouts();
      }
    });
    context.controls.addSlider({
      label: 'Outline buffer distance (per-frame, render only)',
      min: 0,
      max: 600,
      step: 10,
      value: outlineDistance,
      format: value => `${value} m`,
      onChange: value => {
        outlineDistance = value;
        writeOutline();
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Clip rectangle size',
      min: 5,
      max: 45,
      step: 1,
      value: 18,
      format: value => `${value}% of the extent`,
      onChange: value => {
        rectangle = {
          ...rectangle,
          halfWidth: ((maximumX - minimumX) * value) / 100,
          halfHeight: ((maximumY - minimumY) * value) / 100
        };
        writeRectangle();
        context.updateLayers();
      }
    });
    for (const [label, getValue, setValue] of [
      ['Outline buffer', () => showOutline, (value: boolean) => (showOutline = value)],
      [
        'Label anchors and inscribed circles',
        () => showLabels,
        (value: boolean) => (showLabels = value)
      ],
      [
        'Long-axis glyphs (elongation, orientation)',
        () => showGlyphs,
        (value: boolean) => (showGlyphs = value)
      ],
      ['Clip rectangle and clipped rings', () => showClip, (value: boolean) => (showClip = value)]
    ] as const) {
      context.controls.addToggle({
        label,
        value: getValue(),
        onChange: value => {
          setValue(value);
          context.updateLayers();
        }
      });
    }
    context.controls.addLegend({
      title: 'Selected descriptor (viridis; elongation inferno)',
      gradient: {
        colors: [
          [68, 1, 84],
          [59, 82, 139],
          [33, 145, 140],
          [94, 201, 98],
          [253, 231, 37]
        ],
        minimumLabel: '0',
        maximumLabel: '1'
      }
    });
    context.controls.addNote(
      'Drag inside the rectangle (or click the map) to move the clip. White circles are the ' +
        'largest inscribed circles around each label anchor; glyphs show the long axis, longer ' +
        'for more elongated ZIP codes.'
    );
    context.controls.addReadout(
      'ZIP code polygons',
      `${formatCount(featureCount)} (${formatCount(vertexCount)} vertices)`
    );
    const outlineReadout = context.controls.addReadout(
      'Outline triangles',
      formatCount(outlineVertexCount / 3)
    );
    const radiusReadout = context.controls.addReadout('Inscribed radius mean / max');
    const degenerateReadout = context.controls.addReadout('Label points degenerate');
    const statisticsReadout = context.controls.addReadout('Selected: min / mean / max');
    const clipReadout = context.controls.addReadout('Clipped vertices');
    const clipOverflowReadout = context.controls.addReadout('Clip overflow');
    context.controls.addReadout('Data', zones.attribution);

    function updateReadouts() {
      if (!statistics) return;
      if (descriptor === 'none') {
        statisticsReadout.setValue('n/a');
        return;
      }
      const {minimum, mean, maximum} = statistics[descriptor];
      statisticsReadout.setValue(
        `${minimum.toFixed(2)} / ${mean.toFixed(2)} / ${maximum.toFixed(2)}`
      );
    }

    // ---- Readbacks ----
    const columnList = DESCRIPTOR_NAMES.map(name => columns[name]);
    const shapeReader = new SummaryReader(
      resources,
      'geometry-tools-shape',
      [
        ...columnList.map(buffer => ({buffer, size: featureCount * 4})),
        {buffer: labelRadii, size: featureCount * 4},
        {buffer: labelDegenerate, size: featureCount * 4}
      ],
      bytes => {
        const floats = new Float32Array(bytes);
        const words = new Uint32Array(bytes);
        descriptorColumns = {} as NonNullable<typeof descriptorColumns>;
        statistics = {} as NonNullable<typeof statistics>;
        DESCRIPTOR_NAMES.forEach((name, index) => {
          const values = floats.slice(index * featureCount, (index + 1) * featureCount);
          descriptorColumns![name] = values;
          statistics![name] = getStatistics(values);
        });
        radii = floats.slice(4 * featureCount, 5 * featureCount);
        let degenerateCount = 0;
        for (let feature = 0; feature < featureCount; feature++) {
          if (words[5 * featureCount + feature]) degenerateCount++;
        }
        const radiusStatistics = getStatistics(radii);
        radiusReadout.setValue(
          `${radiusStatistics.mean.toFixed(0)} m / ${radiusStatistics.maximum.toFixed(0)} m`
        );
        degenerateReadout.setValue(`${degenerateCount} of ${featureCount}`);
        updateReadouts();
      }
    );
    const clipReader = new SummaryReader(
      resources,
      'geometry-tools-clip',
      [
        {buffer: clipCount, size: 4},
        {buffer: clipOverflow, size: 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        clipReadout.setValue(`${formatCount(words[0])} of ${formatCount(clipCapacity)}`);
        clipOverflowReadout.setValue(words[1] ? 'yes' : 'no');
      }
    );
    outlineReadout.setValue(formatCount(outlineVertexCount / 3));

    // ---- Hit testing ----
    const isInsideRectangle = (x: number, y: number) =>
      Math.abs(x - rectangle.centerX) <= rectangle.halfWidth &&
      Math.abs(y - rectangle.centerY) <= rectangle.halfHeight;
    const findFeature = (x: number, y: number): number => {
      for (let feature = 0; feature < featureCount; feature++) {
        let inside = false;
        for (
          let ring = featureRingOffsets[feature];
          ring < featureRingOffsets[feature + 1];
          ring++
        ) {
          const first = zones.ringOffsets[ring];
          const last = zones.ringOffsets[ring + 1];
          for (let current = first, previous = last - 1; current < last; previous = current++) {
            const xc = zones.polygonPositions[current * 2];
            const yc = zones.polygonPositions[current * 2 + 1];
            const xp = zones.polygonPositions[previous * 2];
            const yp = zones.polygonPositions[previous * 2 + 1];
            if (yc > y !== yp > y && x < ((xp - xc) * (y - yc)) / (yp - yc) + xc) {
              inside = !inside;
            }
          }
        }
        if (inside) return feature;
      }
      return -1;
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiledShape, compiledOutline, compiledClip],
      encode(commandEncoder, frame) {
        if (frame.frameIndex < 2) {
          compiledShape.encode(commandEncoder, {parameters: undefined});
          shapeReader.request(commandEncoder);
        }
        if (outlineDirty || frame.frameIndex < 2) {
          compiledOutline.encode(commandEncoder, {parameters: undefined});
          outlineDirty = false;
        }
        if (clipDirty || frame.frameIndex < 2) {
          compiledClip.encode(commandEncoder, {parameters: undefined});
          clipReader.request(commandEncoder);
          clipDirty = false;
        }
        shapeReader.flush(commandEncoder);
        clipReader.flush(commandEncoder);
      },
      getLayers() {
        const layers: Layer[] = [];
        if (showOutline && outlineDistance > 0) {
          layers.push(
            new TriangleListLayer({
              id: 'geometry-tools-outline-buffer',
              coordinateOrigin,
              positions: outlineTriangles,
              triangleCount: outlineVertexCount / 3,
              color: [36, 96, 130, 255],
              opacity: 0.55
            })
          );
        }
        if (descriptor !== 'none') {
          layers.push(
            new PolygonFanLayer({
              id: 'geometry-tools-fill',
              coordinateOrigin,
              segments: edgeSegments,
              featureRows: edgeFeatureRows,
              centroids: labelPoints,
              instanceCount: edgeCount,
              values: columns[descriptor],
              colormap: descriptor === 'elongation' ? 'inferno' : 'viridis',
              valueRange: [0, 1],
              color: [255, 255, 255, 215]
            })
          );
        }
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'geometry-tools-edges',
            coordinateOrigin,
            segments: edgeSegments,
            instanceCount: edgeCount,
            widthPixels: 1,
            color: [10, 16, 30, 230]
          })
        );
        if (showLabels) {
          layers.push(
            new PairSegmentLayer({
              id: 'geometry-tools-circles',
              coordinateOrigin,
              starts: circleStarts,
              ends: circleEnds,
              instanceCount: featureCount * CIRCLE_SEGMENTS,
              widthPixels: 1.4,
              color: [255, 255, 255, 235]
            }),
            new SpatialAnalysisPointLayer({
              id: 'geometry-tools-anchors',
              coordinateOrigin,
              positions: labelPoints,
              instanceCount: featureCount,
              radiusPixels: 3.5,
              color: [255, 120, 70, 255]
            })
          );
        }
        if (showGlyphs) {
          layers.push(
            new PairSegmentLayer({
              id: 'geometry-tools-glyphs',
              coordinateOrigin,
              starts: glyphStarts,
              ends: glyphEnds,
              instanceCount: featureCount,
              widthPixels: 2.4,
              color: [255, 120, 70, 230]
            })
          );
        }
        if (showClip) {
          layers.push(
            new PairSegmentLayer({
              id: 'geometry-tools-clipped',
              coordinateOrigin,
              starts: clipStarts,
              ends: clipEnds,
              instanceCount: clipCapacity,
              widthPixels: 3,
              color: [255, 70, 190, 255]
            }),
            new SpatialAnalysisSegmentLayer({
              id: 'geometry-tools-rectangle',
              coordinateOrigin,
              segments: rectangleSegments,
              instanceCount: 4,
              widthPixels: 2,
              color: [255, 255, 255, 255]
            })
          );
        }
        return layers;
      },
      onClick(event) {
        if (!event.coordinate || !showClip) return false;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        rectangle = {...rectangle, centerX: x, centerY: y};
        writeRectangle();
        return true;
      },
      onDragStart(event) {
        if (!event.coordinate || !showClip) return false;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        if (!isInsideRectangle(x, y)) return false;
        dragging = true;
        dragOffset = [x - rectangle.centerX, y - rectangle.centerY];
        context.setMapDragEnabled(false);
        return true;
      },
      onDrag(event) {
        if (!dragging || !event.coordinate) return;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        rectangle = {...rectangle, centerX: x - dragOffset[0], centerY: y - dragOffset[1]};
        writeRectangle();
      },
      onDragEnd() {
        dragging = false;
        context.setMapDragEnabled(true);
      },
      getTooltip(event) {
        if (!event.coordinate || !descriptorColumns || !radii) return null;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const feature = findFeature(x, y);
        if (feature < 0) return null;
        const values = descriptorColumns;
        return (
          `ZIP ${zones.featureNames[feature]}\n` +
          `Polsby-Popper ${values.polsbyPopper[feature].toFixed(2)}, ` +
          `Schwartzberg ${values.schwartzberg[feature].toFixed(2)}\n` +
          `Elongation ${values.elongation[feature].toFixed(2)}, ` +
          `convexity ${values.convexity[feature].toFixed(2)}\n` +
          `Inscribed radius ${radii[feature].toFixed(0)} m`
        );
      },
      destroy() {
        if (dragging) context.setMapDragEnabled(true);
        shapeReader.stop();
        clipReader.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
