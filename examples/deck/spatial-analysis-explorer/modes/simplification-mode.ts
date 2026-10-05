// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Douglas-Peucker simplification of the New York taxi trips on the GPU. Two compiled graphs share
 * one importance column: the importance graph (`GPULineSimplification` with `computeImportance`)
 * is encoded exactly once, because importance depends only on the geometry; the selection graph
 * (`computeImportance: false`) re-runs only the keep mask, compaction and publish nodes whenever
 * the tolerance changes. The tolerance is a parameter-buffer write, set either by a log-scale
 * slider in meters or automatically as `k` pixels times the viewport's meters per pixel. Kept
 * segments are drawn from the compact kept IDs by a bespoke layer over the faint original trips.
 */

import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPULineSimplificationParameterValues,
  GPULineSimplification,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {KeptSegmentLayer} from './simplification-layers';

/** Compile-time cap on Douglas-Peucker rounds (five graph nodes each). */
const MAXIMUM_ROUNDS = 64;
/** Frames between summary readbacks when nothing changed. */
const READBACK_INTERVAL_FRAMES = 30;
/** Slider range as log10 of the tolerance in meters (0.1 m to 1 km). */
const LOG_TOLERANCE_RANGE = [-1, 3] as const;
const DEFAULT_LOG_TOLERANCE = 1;
const DEFAULT_AUTO_PIXELS = 2;
const WEB_MERCATOR_CIRCUMFERENCE_METERS = 40075016.686;
const DECK_TILE_SIZE = 512;

/** Meters covered by one CSS pixel at a viewport's center latitude. */
function getMetersPerPixel(zoom: number, latitude: number): number {
  return (
    (WEB_MERCATOR_CIRCUMFERENCE_METERS * Math.cos((latitude * Math.PI) / 180)) /
    (DECK_TILE_SIZE * 2 ** zoom)
  );
}

function formatMeters(meters: number): string {
  return meters < 10 ? `${meters.toFixed(2)} m` : `${meters.toFixed(0)} m`;
}

export const simplificationMode: SpatialAnalysisModeDefinition = {
  id: 'simplification',
  title: 'Simplify',
  contributors: ['GPULineSimplification'],
  description:
    'Douglas-Peucker simplification of New York taxi trips. Importance is computed once on the ' +
    'GPU; the tolerance (meters, or automatically k pixels at the current zoom) is a per-frame ' +
    'parameter that re-selects the kept vertices without recompiling. Orange is the simplified ' +
    'line over the faint originals.',
  initialViewState: {longitude: -73.985, latitude: 40.74, zoom: 12.6},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'simplification');

    const tripCount = trips.vendors.length;
    const vertexCount = trips.vertexPositions.length / 2;

    // Original drawn rows: one per consecutive vertex pair inside a trip.
    let segmentCount = 0;
    for (let trip = 0; trip < tripCount; trip++) {
      segmentCount += Math.max(0, trips.tripOffsets[trip + 1] - trips.tripOffsets[trip] - 1);
    }
    const segments = new Float32Array(segmentCount * 4);
    const vertexLines = new Uint32Array(vertexCount);
    let row = 0;
    for (let trip = 0; trip < tripCount; trip++) {
      const first = trips.tripOffsets[trip];
      const last = trips.tripOffsets[trip + 1] - 1;
      for (let vertex = first; vertex <= last; vertex++) vertexLines[vertex] = trip;
      for (let vertex = first; vertex < last; vertex++, row++) {
        segments.set(trips.vertexPositions.subarray(vertex * 2, vertex * 2 + 4), row * 4);
      }
    }

    const positionsBuffer = resources.createBuffer('positions', trips.vertexPositions);
    const offsetsBuffer = resources.createBuffer('track-offsets', trips.tripOffsets);
    const vertexLinesBuffer = resources.createBuffer('vertex-lines', vertexLines);
    const segmentsBuffer = resources.createBuffer('segments', segments);
    const importanceBuffer = resources.createBuffer('importance', vertexCount * 4);
    const convergedBuffer = resources.createBuffer('converged', 4);
    const roundCountBuffer = resources.createBuffer('round-count', 4);
    const keptIdsBuffer = resources.createBuffer('kept-ids', vertexCount * 4);
    const keptCountBuffer = resources.createBuffer('kept-count', 4);
    const keptOverflowBuffer = resources.createBuffer('kept-overflow', 4);
    const keptTotalBuffer = resources.createBuffer('kept-total', 4);
    const parameterBuffer = resources.createParameterBuffer(
      'tolerance',
      'float32',
      GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH
    );
    // The contributor has no drawIndirect writer: the kept count is copied into the record each time.
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'simplification-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const SUMMARY_WORDS = 5;
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'simplification-summary', byteLength: SUMMARY_WORDS * 4})
    );

    const importView = (graph: GPUCommandGraph<void>) => ({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
      trackOffsets: importGraphBuffer(
        graph,
        'track-offsets',
        offsetsBuffer,
        'uint32',
        tripCount + 1
      ),
      importance: importGraphBuffer(graph, 'importance', importanceBuffer, 'float32', vertexCount)
    });

    // Graph 1: importance, encoded once.
    const importanceGraph = new GPUCommandGraph<void>(device, {id: 'simplification-importance'});
    importanceGraph.add(
      new GPULineSimplification({
        id: 'simplification-importance',
        ...importView(importanceGraph),
        maximumRounds: MAXIMUM_ROUNDS,
        status: {
          converged: importGraphBuffer(importanceGraph, 'converged', convergedBuffer, 'uint32', 1),
          roundCount: importGraphBuffer(
            importanceGraph,
            'round-count',
            roundCountBuffer,
            'uint32',
            1
          )
        }
      })
    );
    const compiledImportance: CompiledGPUCommandGraph<void> = resources.track(
      importanceGraph.compile()
    );

    // Graph 2: tolerance selection, re-encoded when the tolerance changes.
    const selectionGraph = new GPUCommandGraph<void>(device, {id: 'simplification-selection'});
    selectionGraph.add(
      new GPULineSimplification({
        id: 'simplification-selection',
        ...importView(selectionGraph),
        computeImportance: false,
        parameters: parameterBuffer.importToGraph(selectionGraph),
        selection: {
          output: {
            ids: importGraphBuffer(
              selectionGraph,
              'kept-ids',
              keptIdsBuffer,
              'uint32',
              vertexCount
            ),
            count: importGraphBuffer(selectionGraph, 'kept-count', keptCountBuffer, 'uint32', 1),
            overflow: importGraphBuffer(
              selectionGraph,
              'kept-overflow',
              keptOverflowBuffer,
              'uint32',
              1
            ),
            totalCount: importGraphBuffer(
              selectionGraph,
              'kept-total',
              keptTotalBuffer,
              'uint32',
              1
            )
          }
        }
      })
    );
    const compiledSelection: CompiledGPUCommandGraph<void> = resources.track(
      selectionGraph.compile()
    );

    let logTolerance = DEFAULT_LOG_TOLERANCE;
    let automatic = true;
    let autoPixels = DEFAULT_AUTO_PIXELS;
    let showOriginal = true;
    let appliedTolerance = Number.NaN;
    let currentTolerance = 0;
    let importanceEncoded = false;
    let readbackRequested = true;
    let readbackPending = false;
    let destroyed = false;
    const parameterValues = new Float32Array(GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH);

    const toleranceReadout = context.controls.addReadout('Tolerance');
    const keptReadout = context.controls.addReadout('Kept vertices');
    const ratioReadout = context.controls.addReadout('Kept ratio');
    const convergedReadout = context.controls.addReadout('Importance converged');
    const roundsReadout = context.controls.addReadout('Rounds used');

    const manualSlider = context.controls.addSlider({
      label: 'Tolerance (meters, log scale; per-frame parameter)',
      min: LOG_TOLERANCE_RANGE[0],
      max: LOG_TOLERANCE_RANGE[1],
      step: 0.05,
      value: logTolerance,
      format: value => formatMeters(10 ** value),
      onChange: value => {
        logTolerance = value;
      }
    });
    context.controls.addToggle({
      label: 'Auto tolerance from zoom',
      value: automatic,
      onChange: value => {
        automatic = value;
        manualSlider.setDisabled(value);
      }
    });
    context.controls.addSlider({
      label: 'Auto tolerance (pixels at current zoom)',
      min: 0.25,
      max: 12,
      step: 0.25,
      value: autoPixels,
      format: value => `${value} px`,
      onChange: value => {
        autoPixels = value;
      }
    });
    context.controls.addToggle({
      label: 'Show original trips',
      value: showOriginal,
      onChange: value => {
        showOriginal = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Lines',
      entries: [
        {color: [255, 150, 40, 255], label: 'Simplified (kept vertices)'},
        {color: [120, 170, 255, 80], label: 'Original trips'}
      ]
    });
    context.controls.addNote(
      'The metric (segment distance) is compile-time and not exposed. Importance runs 64 gated ' +
        'rounds once; moving the tolerance only re-runs mask, compaction and publish.'
    );
    context.controls.addReadout(
      'Trips / vertices',
      `${formatCount(tripCount)} / ${formatCount(vertexCount)}`
    );
    context.controls.addReadout('Data', trips.attribution);
    manualSlider.setDisabled(automatic);

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const sources = [
        keptCountBuffer,
        keptOverflowBuffer,
        keptTotalBuffer,
        convergedBuffer,
        roundCountBuffer
      ];
      sources.forEach((sourceBuffer, index) => {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: index * 4,
          size: 4
        });
      });
      ticket.markEncoded({byteOffset: 0, byteLength: SUMMARY_WORDS * 4});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, SUMMARY_WORDS);
        const total = words[2];
        keptReadout.setValue(
          `${formatCount(words[0])} / ${formatCount(vertexCount)}${words[1] ? ' (overflow)' : ''}`
        );
        ratioReadout.setValue(`${((100 * total) / vertexCount).toFixed(1)}% kept`);
        convergedReadout.setValue(words[3] ? 'yes' : 'no (round cap, superset of DP)');
        roundsReadout.setValue(`${words[4]} of ${MAXIMUM_ROUNDS}`);
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiledImportance, compiledSelection],
      encode(commandEncoder, frame) {
        currentTolerance = automatic
          ? autoPixels *
            getMetersPerPixel(
              frame.viewport.zoom,
              frame.viewport.unproject([frame.viewport.width / 2, frame.viewport.height / 2])[1]
            )
          : 10 ** logTolerance;
        const tolerance = Math.fround(currentTolerance);
        // Importance depends only on the geometry, so it is encoded once. The selection depends
        // only on the tolerance, so it is re-encoded only when the tolerance changes.
        if (!importanceEncoded) {
          compiledImportance.encode(commandEncoder, {parameters: undefined});
          importanceEncoded = true;
          appliedTolerance = Number.NaN;
        }
        if (tolerance !== appliedTolerance) {
          appliedTolerance = tolerance;
          parameterBuffer.write(
            getGPULineSimplificationParameterValues({tolerance}, parameterValues)
          );
          compiledSelection.encode(commandEncoder, {parameters: undefined});
          // Instance-count word of the 16-byte draw record: the clamped kept count.
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: keptCountBuffer,
            sourceOffset: 0,
            destinationBuffer: drawCommands.buffer,
            destinationOffset: 4,
            size: 4
          });
          toleranceReadout.setValue(formatMeters(tolerance));
          readbackRequested = true;
        }
        if (
          !readbackPending &&
          (readbackRequested || frame.frameIndex % READBACK_INTERVAL_FRAMES === 0)
        ) {
          readbackRequested = false;
          void readSummary(commandEncoder);
        }
      },
      getLayers(): Layer[] {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        const layers: Layer[] = [];
        if (showOriginal) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'simplification-original',
              coordinateOrigin,
              segments: segmentsBuffer,
              instanceCount: segmentCount,
              widthPixels: 2.5,
              color: [120, 170, 255, 80]
            })
          );
        }
        layers.push(
          new KeptSegmentLayer({
            id: 'simplification-kept',
            coordinateOrigin,
            positions: positionsBuffer,
            keptIds: keptIdsBuffer,
            vertexLines: vertexLinesBuffer,
            keptCount: keptCountBuffer,
            drawCommands,
            widthPixels: 1.5,
            color: [255, 150, 40, 255]
          })
        );
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
