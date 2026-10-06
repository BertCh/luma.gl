// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Polygon join: `GPUPointInPolygonJoin` assigns every San Francisco bike-parking point to the ZIP
 * code polygon that contains it. The points drift on small circles and are rewritten every frame,
 * so the join re-runs on the GPU each frame with no recompile. Each point is colored with the
 * same categorical palette as its polygon's outline, which makes a wrong join visible at a glance.
 *
 * `featureIds` is deliberately not passed: the outputs are dense feature rows, which index the
 * categorical palette and the per-feature `featureCounts` directly. ZIP codes are looked up on the
 * CPU only for the top-5 readout.
 *
 * `spatialSort` is a compile-time option: the toggle rebuilds the graph over the same buffers, and
 * "Measure spatialSort on vs off" times a temporary graph of the other setting outside the frame.
 *
 * "Prepared ZIP index" (compile-time: rebuilds the graph) hands the join a `GPUSpatialJoinPrepared`
 * handle over the ZIP polygons. Its bounds and BVH are built on the first encoding and reused while
 * the points move every frame; a readout counts the builds against the encoded frames, and the contrast
 * toggle invalidates the handle every frame to show the cost of rebuilding.
 */

import type {Layer} from '@deck.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_SPATIAL_JOIN_NO_FEATURE,
  GPUPointInPolygonJoin,
  GPUSpatialJoinPrepared
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {formatCompiledGraphTiming, formatSpeedup, measureCompiledGraph} from './vector-timing';

/** Peak circular drift of each point, in meters. */
const DRIFT_RADIUS_METERS = 150;
/** Frames between featureCounts readbacks. */
const READBACK_INTERVAL = 20;
/** Frames after creation before the automatic spatialSort measurement runs. */
const AUTO_MEASURE_FRAME = 40;
const CATEGORY_COLORS = [
  [78, 201, 255],
  [255, 148, 72],
  [189, 122, 255],
  [87, 235, 168],
  [255, 105, 168],
  [245, 220, 87],
  [107, 158, 255],
  [255, 92, 92]
] as const;

export const polygonJoinMode: SpatialAnalysisModeDefinition = {
  id: 'polygon-join',
  title: 'Polygon join',
  contributors: ['GPUPointInPolygonJoin', 'GPUSpatialJoinPrepared'],
  description:
    'Bike-parking points are joined to ZIP-code polygons on the GPU every frame while they drift. ' +
    'Each point takes the color of its containing polygon outline; gray points fall in no polygon.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 12.2},

  async create(context) {
    const [parking, zips] = await Promise.all([
      context.data.getSanFranciscoBikeParking(),
      context.data.getSanFranciscoZipCodes()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'polygon-join');
    const pointCount = parking.positions.length / 2;
    const featureCount = zips.featureOffsets.length - 1;
    const segmentCount = zips.outlineSegments.length / 4;

    const positionsBuffer = resources.createBuffer('positions', parking.positions);
    const polygonPositions = resources.createBuffer('polygon-positions', zips.polygonPositions);
    const featureOffsets = resources.createBuffer('feature-offsets', zips.featureOffsets);
    const polygonOffsets = resources.createBuffer('polygon-offsets', zips.polygonOffsets);
    const ringOffsets = resources.createBuffer('ring-offsets', zips.ringOffsets);
    const pointFeatures = resources.createBuffer('point-features', pointCount * 4);
    const featureCounts = resources.createBuffer('feature-counts', featureCount * 4);
    const overflow = resources.createBuffer('overflow', 4);
    const candidateCount = resources.createBuffer('candidate-count', 4);
    const outlineSegments = resources.createBuffer('outline-segments', zips.outlineSegments);
    const outlineFeatureRows = resources.createBuffer(
      'outline-feature-rows',
      zips.outlineFeatureRows
    );

    /** A compiled graph with the prepared handle it uses, when any. */
    type BuiltGraph = {
      compiled: CompiledGPUCommandGraph<void>;
      prepared: GPUSpatialJoinPrepared | null;
    };
    const buildGraph = (spatialSort: boolean, usePrepared: boolean): BuiltGraph => {
      const graph = new GPUCommandGraph<void>(device, {
        id: `polygon-join-${spatialSort ? 'sorted' : 'unsorted'}${usePrepared ? '-prepared' : ''}`
      });
      const polygons = {
        kind: 'polygons' as const,
        positions: importGraphBuffer(
          graph,
          'polygon-positions',
          polygonPositions,
          'float32x2',
          zips.polygonPositions.length / 2
        ),
        featureOffsets: importGraphBuffer(
          graph,
          'feature-offsets',
          featureOffsets,
          'uint32',
          zips.featureOffsets.length
        ),
        polygonOffsets: importGraphBuffer(
          graph,
          'polygon-offsets',
          polygonOffsets,
          'uint32',
          zips.polygonOffsets.length
        ),
        ringOffsets: importGraphBuffer(
          graph,
          'ring-offsets',
          ringOffsets,
          'uint32',
          zips.ringOffsets.length
        )
      };
      // The ZIP polygons never change, so their bounds and BVH are built once and reused.
      const prepared = usePrepared
        ? new GPUSpatialJoinPrepared({id: 'zip-prepared', geometry: polygons, spatialSort})
        : null;
      if (prepared) graph.add(prepared);
      graph.add(
        new GPUPointInPolygonJoin({
          id: 'polygon-join',
          points: importGraphBuffer(graph, 'points', positionsBuffer, 'float32x2', pointCount),
          polygonPositions: polygons.positions,
          featureOffsets: polygons.featureOffsets,
          polygonOffsets: polygons.polygonOffsets,
          ringOffsets: polygons.ringOffsets,
          candidateCapacity: Math.max(1024, pointCount * 4),
          // With a prepared handle the sort order belongs to the handle.
          ...(prepared ? {prepared} : {spatialSort}),
          pointFeatureIds: importGraphBuffer(
            graph,
            'point-features',
            pointFeatures,
            'uint32',
            pointCount
          ),
          featureCounts: importGraphBuffer(
            graph,
            'feature-counts',
            featureCounts,
            'uint32',
            featureCount
          ),
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
          candidateCount: importGraphBuffer(graph, 'candidate-count', candidateCount, 'uint32', 1)
        })
      );
      return {compiled: graph.compile(), prepared};
    };
    /** Tracks a built graph so it is destroyed before the handle that its graph imports. */
    const trackBuilt = (built: BuiltGraph): BuiltGraph => {
      if (built.prepared) {
        const {prepared} = built;
        resources.track({destroy: () => prepared.destroy()});
      }
      resources.track(built.compiled);
      return built;
    };
    const releaseBuilt = (built: BuiltGraph) => {
      resources.release(built.compiled);
      built.prepared?.destroy();
    };
    let spatialSort = false;
    let usePrepared = false;
    let invalidateEveryFrame = false;
    let built = trackBuilt(buildGraph(spatialSort, usePrepared));
    let compiled = built.compiled;

    const summaryWords = featureCount + 2;
    const readbackRing = new GPUReadbackRing(device, {
      id: 'polygon-join-readback',
      byteLength: summaryWords * 4
    });
    resources.track({destroy: () => readbackRing.destroy()});

    // Deterministic per-point drift phases and the animated positions scratch array.
    const phases = new Float32Array(pointCount);
    for (let index = 0; index < pointCount; index++) {
      phases[index] = (((index * 2654435761) >>> 0) / 4294967296) * Math.PI * 2;
    }
    const animated = new Float32Array(pointCount * 2);

    let animate = true;
    let destroyed = false;
    let measuring = false;
    let measureAgain = false;
    let readbackPending = false;
    let countRangeMaximum = Math.max(1, Math.ceil((pointCount / featureCount) * 3));

    const replaceGraph = () => {
      const previous = built;
      built = trackBuilt(buildGraph(spatialSort, usePrepared));
      compiled = built.compiled;
      builtFrames = 0;
      // Deck still draws with buffers only; the old graph is freed once no frame can encode it.
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          if (!destroyed) releaseBuilt(previous);
        })
      );
    };
    let builtFrames = 0;

    context.controls.addToggle({
      label: 'Animate points (per-frame join)',
      value: animate,
      onChange: value => {
        animate = value;
        if (!animate) positionsBuffer.write(parking.positions);
      }
    });
    context.controls.addToggle({
      label: 'Prepared ZIP index (compile-time: rebuilds graph)',
      value: usePrepared,
      onChange: value => {
        usePrepared = value;
        replaceGraph();
        if (measuring) measureAgain = true;
      }
    });
    context.controls.addToggle({
      label: 'Invalidate the prepared index every frame (contrast)',
      value: invalidateEveryFrame,
      onChange: value => {
        invalidateEveryFrame = value;
      }
    });
    context.controls.addToggle({
      label: 'spatialSort (compile-time: rebuilds graph)',
      value: spatialSort,
      onChange: value => {
        spatialSort = value;
        replaceGraph();
        if (measuring) measureAgain = true;
      }
    });
    context.controls.addButton({
      label: 'Measure spatialSort on vs off',
      onClick: () => void measureSpatialSort()
    });
    context.controls.addLegend({
      title: 'Point and thin outline: containing ZIP (palette cycles); gray = no polygon',
      entries: [
        ...CATEGORY_COLORS.slice(0, 4).map((color, index) => ({
          color,
          label: `ZIP row ${index} (mod 8)`
        })),
        {color: [150, 150, 150], label: 'Outside every polygon'}
      ]
    });
    context.controls.addLegend({
      title: 'Thick outline: points per ZIP (GPU featureCounts)',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '0',
        maximumLabel: 'max'
      }
    });
    context.controls.addReadout('Points', formatCount(pointCount));
    context.controls.addReadout('Polygons (ZIP codes)', formatCount(featureCount));
    const overflowReadout = context.controls.addReadout('Overflow', '...');
    const candidateReadout = context.controls.addReadout('BVH candidates', '...');
    const topReadout = context.controls.addReadout('Top ZIPs', '...');
    const sortedReadout = context.controls.addReadout('spatialSort on', '...');
    const unsortedReadout = context.controls.addReadout('spatialSort off', '...');
    const speedupReadout = context.controls.addReadout('Speedup', '...');
    const preparedReadout = context.controls.addReadout('ZIP index builds', 'not prepared');
    context.controls.addNote(
      'Timed outside the frame: GPU timestamps when the device has timestamp-query, otherwise wall clock / 8 repetitions (upper bound). ' +
        'spatialSort pays off for thousands of incoherent features; 25 ZIPs gain nothing.'
    );
    context.controls.addReadout('Data', `${parking.attribution}; ${zips.attribution}`);

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: featureCounts,
        destinationBuffer: ticket.buffer,
        size: featureCount * 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: overflow,
        destinationBuffer: ticket.buffer,
        destinationOffset: featureCount * 4,
        size: 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: candidateCount,
        destinationBuffer: ticket.buffer,
        destinationOffset: featureCount * 4 + 4,
        size: 4
      });
      ticket.markEncoded({byteLength: summaryWords * 4});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, summaryWords);
        const ranked = Array.from({length: featureCount}, (_, row) => row)
          .sort((a, b) => words[b] - words[a])
          .slice(0, 5);
        topReadout.setValue(
          ranked
            .map(row => `${zips.featureNames[row] ?? zips.featureIds[row]} (${words[row]})`)
            .join(', ')
        );
        overflowReadout.setValue(words[featureCount] ? 'YES' : 'no');
        candidateReadout.setValue(formatCount(words[featureCount + 1]));
        const maximum = Math.max(1, words[ranked[0]]);
        if (maximum !== countRangeMaximum) {
          countRangeMaximum = maximum;
          context.updateLayers();
        }
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    let encodedFrames = 0;
    let autoMeasured = false;
    /** Times the active graph and a temporary graph of the other spatialSort setting. */
    const measureSpatialSort = async () => {
      if (measuring || destroyed) return;
      measuring = true;
      measureAgain = false;
      sortedReadout.setValue('measuring...');
      unsortedReadout.setValue('measuring...');
      speedupReadout.setValue('...');
      let temporary: CompiledGPUCommandGraph<void> | null = null;
      let temporaryBuilt: BuiltGraph | null = null;
      try {
        const activeSort = spatialSort;
        const active = compiled;
        temporaryBuilt = buildGraph(!activeSort, usePrepared);
        temporary = temporaryBuilt.compiled;
        const options = {parameters: undefined, completionBuffer: overflow, signal: context.signal};
        const activeTiming = await measureCompiledGraph(device, active, options);
        const otherTiming = await measureCompiledGraph(device, temporary, options);
        if (destroyed) return;
        const sorted = activeSort ? activeTiming : otherTiming;
        const unsorted = activeSort ? otherTiming : activeTiming;
        sortedReadout.setValue(formatCompiledGraphTiming(sorted));
        unsortedReadout.setValue(formatCompiledGraphTiming(unsorted));
        speedupReadout.setValue(formatSpeedup(unsorted.milliseconds, sorted.milliseconds));
      } catch {
        if (!destroyed) {
          sortedReadout.setValue('interrupted');
          unsortedReadout.setValue('interrupted');
          measureAgain = true;
        }
      } finally {
        temporary?.destroy();
        temporaryBuilt?.prepared?.destroy();
        measuring = false;
        if (measureAgain && !destroyed) void measureSpatialSort();
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        encodedFrames++;
        if (!autoMeasured && encodedFrames >= AUTO_MEASURE_FRAME) {
          autoMeasured = true;
          void measureSpatialSort();
        }
        if (animate) {
          const angularSpeed = 1.2;
          for (let index = 0; index < pointCount; index++) {
            const angle = phases[index] + frame.timeSeconds * angularSpeed;
            animated[index * 2] =
              parking.positions[index * 2] + Math.cos(angle) * DRIFT_RADIUS_METERS;
            animated[index * 2 + 1] =
              parking.positions[index * 2 + 1] + Math.sin(angle) * DRIFT_RADIUS_METERS;
          }
          positionsBuffer.write(animated);
        }
        if (invalidateEveryFrame) built.prepared?.invalidate();
        compiled.encode(commandEncoder, {parameters: undefined});
        builtFrames++;
        if (built.prepared && frame.frameIndex % READBACK_INTERVAL === 0) {
          preparedReadout.setValue(
            `${built.prepared.encodedBuildCount} in ${formatCount(builtFrames)} frames` +
              (invalidateEveryFrame ? ' (rebuilding every frame)' : ' (reused)')
          );
        } else if (!built.prepared && frame.frameIndex % READBACK_INTERVAL === 0) {
          preparedReadout.setValue('not prepared (bounds and BVH rebuilt every frame)');
        }
        if (!readbackPending && frame.frameIndex % READBACK_INTERVAL === 0) {
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [zips.origin[0], zips.origin[1], 0];
        const layers: Layer[] = [
          new SpatialAnalysisSegmentLayer({
            id: 'polygon-join-count-outline',
            coordinateOrigin,
            segments: outlineSegments,
            instanceCount: segmentCount,
            values: featureCounts,
            valueFormat: 'uint32',
            valueIndices: outlineFeatureRows,
            colormap: 'viridis',
            valueRange: [0, countRangeMaximum],
            widthPixels: 9,
            opacity: 1
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'polygon-join-outline',
            coordinateOrigin,
            segments: outlineSegments,
            instanceCount: segmentCount,
            values: outlineFeatureRows,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: CATEGORY_COLORS.map(color => [...color, 255] as const),
            widthPixels: 2
          }),
          new SpatialAnalysisPointLayer({
            id: 'polygon-join-points',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            values: pointFeatures,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: CATEGORY_COLORS.map(color => [...color, 255] as const),
            noDataValue: GPU_SPATIAL_JOIN_NO_FEATURE,
            noDataColor: [150, 150, 150, 255],
            radiusPixels: 4
          })
        ];
        return layers;
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
