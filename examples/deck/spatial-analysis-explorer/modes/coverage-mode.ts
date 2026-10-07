// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * San Francisco ZIP codes as a polygon coverage.
 *
 * - `GPUCoverageSimplification` simplifies every arc shared by two polygons once, so neighbors keep
 *   identical vertices along their common boundary. The tolerance is a parameter-buffer write.
 *   It is topology-preserving: arc endpoints are fixed, a ring keeps at least three vertices, and a
 *   few rounds of crossing detection (`GPUSegmentIntersection`, exact predicates) restore the
 *   original vertex farthest from every simplified segment that crosses another. A toggle swaps in
 *   a second compiled graph without the repair, to show the crossings plain Douglas-Peucker makes.
 * - An independent `GPULineSimplification` of every ring (importance once, selection per tolerance)
 *   is the control: each polygon decides alone, so neighbors disagree about shared vertices and the
 *   shared boundary opens gaps and overlaps. A small CPU check over the two read-back keep masks
 *   counts the shared vertices one neighbor keeps and the other drops, and marks them.
 * - `GPUContiguityWeights` (rook) feeds `GPUMapColoring` in one graph. Polygon fills take their
 *   color from the GPU color IDs, so adjacent ZIP codes never share a color. The seed is a
 *   compile-time property of the coloring, so moving it rebuilds that one graph after a short
 *   debounce (the footer counts it); the tolerance never recompiles anything.
 *
 * Fills use a one-time CPU triangulation of the static input; every outline comes from GPU keep
 * masks and every color from the GPU color buffer.
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPULineSimplificationParameterValues,
  GPUContiguityWeights,
  GPUCoverageSimplification,
  GPULineSimplification,
  GPUMapColoring,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH,
  GPU_MAP_COLORING_UNCOLORED
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {RingOutlineLayer, triangulatePolygons, ZoneFillLayer} from './coverage-layers';
import {SummaryReader} from './summary-reader';

/** Slider range as log10 of the tolerance in meters (1 m to about 1.6 km). */
const LOG_TOLERANCE_RANGE = [0, 3.2] as const;
const DEFAULT_LOG_TOLERANCE = 1.9;
/** Neighbor slots per polygon in the contiguity capacity. */
const NEIGHBORS_PER_POLYGON = 24;
/** Douglas-Peucker round cap (compile-time); the SF rings have up to about 1,000 vertices. */
const MAXIMUM_ROUNDS = 256;
const MAXIMUM_SEED = 31;
/** Compile-time detect-and-repair rounds of the topology-preserving coverage graph. */
const TOPOLOGY_ROUNDS = 6;
const REBUILD_DEBOUNCE_MILLISECONDS = 200;
const FILL_PALETTE = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;
const COVERAGE_COLOR = [255, 255, 255, 255] as const;
const INDEPENDENT_COLOR = [255, 60, 60, 255] as const;
const GAP_MARKER_COLOR = [255, 230, 60, 255] as const;

type ViewId = 'coverage' | 'independent' | 'both';

const VIEW_OPTIONS: readonly {value: ViewId; label: string}[] = [
  {value: 'coverage', label: 'Coverage simplification (shared arcs)'},
  {value: 'independent', label: 'Independent per-polygon simplification'},
  {value: 'both', label: 'Both overlaid (white vs red)'}
];

function formatMeters(meters: number): string {
  return meters < 10 ? `${meters.toFixed(2)} m` : `${meters.toFixed(0)} m`;
}

export const coverageMode: SpatialAnalysisModeDefinition = {
  id: 'coverage',
  title: 'Coverage',
  contributors: [
    'GPUCoverageSimplification',
    'GPULineSimplification',
    'GPUContiguityWeights',
    'GPUMapColoring'
  ],
  description:
    'San Francisco ZIP codes as a coverage. Drag the tolerance: shared boundaries stay gap-free ' +
    '(white) while independent per-polygon simplification (red) opens gaps, marked by yellow dots. ' +
    'The coverage result is topology-preserving: arc ends stay fixed, rings keep three vertices, ' +
    'and simplified segments that would cross are repaired (readouts show crossings found, fixed ' +
    'and remaining; toggle "Preserve topology" to compare). ' +
    'Fills are a GPU map coloring of the rook contiguity graph; the seed picks another coloring.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.6},

  async create(context) {
    const zones = await context.data.getSanFranciscoZipCodes();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'coverage');
    const polygonCount = zones.polygonOffsets.length - 1;
    const ringCount = zones.ringOffsets.length - 1;
    const vertexCount = zones.polygonPositions.length / 2;
    const segmentCount = zones.outlineSegments.length / 4;
    const neighborCapacity = polygonCount * NEIGHBORS_PER_POLYGON;
    const coordinateOrigin: [number, number, number] = [zones.origin[0], zones.origin[1], 0];

    // Static CPU inputs: ring of every vertex, fill triangles, and the groups of identical vertices
    // that more than one ring owns (the places where simplification can open a gap).
    const vertexRings = new Uint32Array(vertexCount);
    for (let ring = 0; ring < ringCount; ring++) {
      vertexRings.fill(ring, zones.ringOffsets[ring], zones.ringOffsets[ring + 1]);
    }
    const sharedGroups: number[][] = [];
    {
      const byPosition = new Map<string, number[]>();
      for (let vertex = 0; vertex < vertexCount; vertex++) {
        const key = `${zones.polygonPositions[vertex * 2]},${zones.polygonPositions[vertex * 2 + 1]}`;
        const group = byPosition.get(key);
        if (group) group.push(vertex);
        else byPosition.set(key, [vertex]);
      }
      for (const group of byPosition.values()) {
        if (new Set(group.map(vertex => vertexRings[vertex])).size > 1) sharedGroups.push(group);
      }
    }
    const triangles = triangulatePolygons(
      zones.polygonPositions,
      zones.ringOffsets,
      zones.polygonOffsets
    );
    const triangleCount = triangles.owners.length;

    const positionsBuffer = resources.createBuffer('positions', zones.polygonPositions);
    const ringOffsetsBuffer = resources.createBuffer('ring-offsets', zones.ringOffsets);
    const polygonOffsetsBuffer = resources.createBuffer('polygon-offsets', zones.polygonOffsets);
    const vertexRingsBuffer = resources.createBuffer('vertex-rings', vertexRings);
    const triangleVertices = resources.createBuffer('triangle-vertices', triangles.corners);
    const triangleOwners = resources.createBuffer('triangle-owners', triangles.owners);
    const outlineBuffer = resources.createBuffer('outline', zones.outlineSegments);
    const markerBuffer = resources.createBuffer('gap-markers', Math.max(1, vertexCount) * 8);

    // Contiguity and coloring.
    const weightOffsets = resources.createBuffer('weight-offsets', (polygonCount + 1) * 4);
    const weightNeighbors = resources.createBuffer('weight-neighbors', neighborCapacity * 4);
    const weightValues = resources.createBuffer('weight-values', neighborCapacity * 4);
    const weightOverflow = resources.createBuffer('weight-overflow', 4);
    const colors = resources.createBuffer('colors', polygonCount * 4);
    const colorCount = resources.createBuffer('color-count', 4);
    const conflictCount = resources.createBuffer('conflict-count', 4);
    const coloringConverged = resources.createBuffer('coloring-converged', 4);
    const coloringRounds = resources.createBuffer('coloring-rounds', 4);

    // Coverage simplification outputs.
    const parameterBuffer = resources.createParameterBuffer(
      'tolerance',
      'float32',
      GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH
    );
    const coverageOutPositions = resources.createBuffer('coverage-out-positions', vertexCount * 8);
    const coverageOutRings = resources.createBuffer('coverage-out-rings', (ringCount + 1) * 4);
    const coverageKeepMask = resources.createBuffer('coverage-keep-mask', vertexCount * 4);
    const coverageOverflow = resources.createBuffer('coverage-overflow', 4);
    const coverageConverged = resources.createBuffer('coverage-converged', 4);
    const coverageTopology = resources.createBuffer('coverage-topology', 16);
    // Crossings of the input itself, from one repair-graph run at tolerance 0 (a copy of the stats).
    const baselineTopology = resources.createBuffer('baseline-topology', 16);

    // Independent per-ring simplification.
    const importance = resources.createBuffer('importance', vertexCount * 4);
    const importanceConverged = resources.createBuffer('importance-converged', 4);
    const importanceRounds = resources.createBuffer('importance-rounds', 4);
    const keptIds = resources.createBuffer('independent-kept-ids', vertexCount * 4);
    const keptCount = resources.createBuffer('independent-kept-count', 4);
    const keptOverflow = resources.createBuffer('independent-kept-overflow', 4);
    const independentKeepMask = resources.createBuffer('independent-keep-mask', vertexCount * 4);

    const importAll = <Format extends Parameters<typeof importGraphBuffer>[3]>(
      graph: GPUCommandGraph<void>,
      name: string,
      buffer: Parameters<typeof importGraphBuffer>[2],
      format: Format,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length);

    const buildColoringGraph = (seed: number): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {id: `coverage-coloring-${seed}`});
      const weights = {
        offsets: importAll(graph, 'weight-offsets', weightOffsets, 'uint32', polygonCount + 1),
        neighbors: importAll(
          graph,
          'weight-neighbors',
          weightNeighbors,
          'uint32',
          neighborCapacity
        ),
        weights: importAll(graph, 'weight-values', weightValues, 'float32', neighborCapacity)
      };
      graph.add(
        new GPUContiguityWeights({
          id: 'contiguity',
          criterion: 'rook',
          positions: importAll(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
          ringOffsets: importAll(graph, 'ring-offsets', ringOffsetsBuffer, 'uint32', ringCount + 1),
          polygonOffsets: importAll(
            graph,
            'polygon-offsets',
            polygonOffsetsBuffer,
            'uint32',
            polygonCount + 1
          ),
          weights,
          overflow: importAll(graph, 'weight-overflow', weightOverflow, 'uint32', 1)
        })
      );
      graph.add(
        new GPUMapColoring({
          id: 'coloring',
          weights,
          colors: importAll(graph, 'colors', colors, 'uint32', polygonCount),
          colorCount: importAll(graph, 'color-count', colorCount, 'uint32', 1),
          conflictCount: importAll(graph, 'conflict-count', conflictCount, 'uint32', 1),
          converged: importAll(graph, 'coloring-converged', coloringConverged, 'uint32', 1),
          roundCount: importAll(graph, 'coloring-rounds', coloringRounds, 'uint32', 1),
          seed
        })
      );
      return resources.track(graph.compile());
    };

    const buildCoverageGraph = (topologyRounds: number): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {
        id: `coverage-simplification-${topologyRounds}`
      });
      graph.add(
        new GPUCoverageSimplification({
          id: 'coverage',
          positions: importAll(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
          ringOffsets: importAll(graph, 'ring-offsets', ringOffsetsBuffer, 'uint32', ringCount + 1),
          polygonOffsets: importAll(
            graph,
            'polygon-offsets',
            polygonOffsetsBuffer,
            'uint32',
            polygonCount + 1
          ),
          parameters: parameterBuffer.importToGraph(graph),
          maximumRounds: MAXIMUM_ROUNDS,
          topologyRounds,
          converged: importAll(graph, 'coverage-converged', coverageConverged, 'uint32', 1),
          output: {
            positions: importAll(
              graph,
              'coverage-out-positions',
              coverageOutPositions,
              'float32x2',
              vertexCount
            ),
            ringOffsets: importAll(
              graph,
              'coverage-out-rings',
              coverageOutRings,
              'uint32',
              ringCount + 1
            ),
            keepMask: importAll(
              graph,
              'coverage-keep-mask',
              coverageKeepMask,
              'uint32',
              vertexCount
            ),
            overflow: importAll(graph, 'coverage-overflow', coverageOverflow, 'uint32', 1),
            topologyStats: importAll(graph, 'coverage-topology', coverageTopology, 'uint32', 4)
          }
        })
      );
      return resources.track(graph.compile());
    };

    const importanceView = (graph: GPUCommandGraph<void>) => ({
      positions: importAll(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
      trackOffsets: importAll(graph, 'ring-offsets', ringOffsetsBuffer, 'uint32', ringCount + 1),
      importance: importAll(graph, 'importance', importance, 'float32', vertexCount)
    });
    const importanceGraph = new GPUCommandGraph<void>(device, {id: 'coverage-importance'});
    importanceGraph.add(
      new GPULineSimplification({
        id: 'independent-importance',
        ...importanceView(importanceGraph),
        maximumRounds: MAXIMUM_ROUNDS,
        status: {
          converged: importAll(
            importanceGraph,
            'importance-converged',
            importanceConverged,
            'uint32',
            1
          ),
          roundCount: importAll(importanceGraph, 'importance-rounds', importanceRounds, 'uint32', 1)
        }
      })
    );
    const compiledImportance = resources.track(importanceGraph.compile());
    const selectionGraph = new GPUCommandGraph<void>(device, {id: 'coverage-independent'});
    selectionGraph.add(
      new GPULineSimplification({
        id: 'independent-selection',
        ...importanceView(selectionGraph),
        computeImportance: false,
        parameters: parameterBuffer.importToGraph(selectionGraph),
        selection: {
          output: {
            ids: importAll(selectionGraph, 'kept-ids', keptIds, 'uint32', vertexCount),
            count: importAll(selectionGraph, 'kept-count', keptCount, 'uint32', 1),
            overflow: importAll(selectionGraph, 'kept-overflow', keptOverflow, 'uint32', 1)
          },
          keepMask: importAll(
            selectionGraph,
            'independent-keep-mask',
            independentKeepMask,
            'uint32',
            vertexCount
          )
        }
      })
    );
    const compiledSelection = resources.track(selectionGraph.compile());
    const compiledRepaired = buildCoverageGraph(TOPOLOGY_ROUNDS);
    const compiledPlain = buildCoverageGraph(0);

    let seed = 0;
    let compiledColoring = buildColoringGraph(seed);
    const retiredColoring: CompiledGPUCommandGraph<void>[] = [];
    let coloringDirty = true;
    let importanceEncoded = false;
    let logTolerance = DEFAULT_LOG_TOLERANCE;
    let appliedTolerance = Number.NaN;
    let view: ViewId = 'both';
    let showFill = true;
    let showOriginal = true;
    let repairTopology = true;
    let baselineEncoded = false;
    let markerCount = 0;
    let destroyed = false;
    let rebuildTimer: ReturnType<typeof setTimeout> | undefined;
    const parameterValues = new Float32Array(GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH);

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSlider({
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
      label: `Preserve topology (${TOPOLOGY_ROUNDS} repair rounds; swaps compiled graphs)`,
      value: repairTopology,
      onChange: value => {
        repairTopology = value;
        appliedTolerance = Number.NaN;
      }
    });
    context.controls.addSelect<ViewId>({
      label: 'Outlines',
      options: VIEW_OPTIONS,
      value: view,
      onChange: value => {
        view = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Fill by map-coloring color ID',
      value: showFill,
      onChange: value => {
        showFill = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Original outline (faint)',
      value: showOriginal,
      onChange: value => {
        showOriginal = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Coloring seed (compile-time, rebuilds the coloring graph)',
      min: 0,
      max: MAXIMUM_SEED,
      step: 1,
      value: seed,
      format: value => `seed ${value}`,
      onChange: value => {
        seed = value;
        clearTimeout(rebuildTimer);
        rebuildTimer = setTimeout(() => {
          if (destroyed) return;
          const previous = compiledColoring;
          compiledColoring = buildColoringGraph(seed);
          coloringDirty = true;
          // Deck's frame encoder may still reference the old graph for a frame or two.
          retiredColoring.push(previous);
          setTimeout(() => {
            const index = retiredColoring.indexOf(previous);
            if (index >= 0) {
              retiredColoring.splice(index, 1);
              resources.release(previous);
            }
          }, 400);
        }, REBUILD_DEBOUNCE_MILLISECONDS);
      }
    });
    context.controls.addLegend({
      title: 'Outlines and gap markers',
      entries: [
        {color: COVERAGE_COLOR, label: 'Coverage simplification (no gaps)'},
        {color: INDEPENDENT_COLOR, label: 'Independent per-polygon simplification'},
        {color: GAP_MARKER_COLOR, label: 'Dot: shared vertex kept by one neighbor only'}
      ]
    });
    context.controls.addLegend({
      title: 'Map-coloring fill (color ID mod 8)',
      entries: FILL_PALETTE.map((color, index) => ({color, label: String(index)}))
    });
    context.controls.addNote(
      'Independent simplification decides each ring alone, so a vertex on a shared boundary can be ' +
        'kept by one polygon and dropped by its neighbor: that is where gaps and overlaps open. ' +
        'The coverage contributor shares the decision through point IDs, so its count stays 0. ' +
        'With "Preserve topology" on, simplified segments that would cross are found with exact ' +
        'predicates and the farthest original vertex is restored, rings keep three vertices, and ' +
        'nodes never move; turn it off to see the crossings and collapsed rings of plain ' +
        'Douglas-Peucker. The independent control never preserves topology.'
    );
    context.controls.addReadout(
      'Polygons / rings / vertices',
      `${formatCount(polygonCount)} / ${formatCount(ringCount)} / ${formatCount(vertexCount)}`
    );
    const toleranceReadout = context.controls.addReadout('Tolerance');
    const neighborsReadout = context.controls.addReadout('Rook adjacencies');
    const colorsReadout = context.controls.addReadout('Colors used');
    const conflictsReadout = context.controls.addReadout('Color conflicts');
    const coloringReadout = context.controls.addReadout('Coloring converged / rounds');
    const coverageKeptReadout = context.controls.addReadout('Coverage kept vertices');
    const independentKeptReadout = context.controls.addReadout('Independent kept vertices');
    const gapReadout = context.controls.addReadout('Gap vertices (coverage / independent)');
    const collapsedReadout = context.controls.addReadout('Collapsed rings (< 3 vertices)');
    const inputCrossingsReadout = context.controls.addReadout('Crossings in the input itself');
    const crossingsFoundReadout = context.controls.addReadout('Crossings found (before repair)');
    const crossingsRemainingReadout = context.controls.addReadout(
      'Crossings remaining / fixed (0 on a clean input)'
    );
    const restoredReadout = context.controls.addReadout('Vertices restored by repair');
    const convergenceReadout = context.controls.addReadout('Simplification converged');
    context.controls.addReadout('Data', zones.attribution);

    // --- Summary readback ----------------------------------------------------------------------
    const layoutWords = (() => {
      let words = 0;
      const next = (length: number) => {
        const start = words;
        words += length;
        return start;
      };
      return {
        colors: next(polygonCount),
        colorCount: next(1),
        conflicts: next(1),
        converged: next(1),
        rounds: next(1),
        offsets: next(polygonCount + 1),
        weightOverflow: next(1),
        coverageOverflow: next(1),
        coverageConverged: next(1),
        coverageTopology: next(4),
        baselineTopology: next(4),
        importanceConverged: next(1),
        keptOverflow: next(1),
        coverageMask: next(vertexCount),
        independentMask: next(vertexCount),
        total: words
      };
    })();
    const summary = new SummaryReader(
      resources,
      'coverage',
      [
        {buffer: colors, size: polygonCount * 4},
        {buffer: colorCount, size: 4},
        {buffer: conflictCount, size: 4},
        {buffer: coloringConverged, size: 4},
        {buffer: coloringRounds, size: 4},
        {buffer: weightOffsets, size: (polygonCount + 1) * 4},
        {buffer: weightOverflow, size: 4},
        {buffer: coverageOverflow, size: 4},
        {buffer: coverageConverged, size: 4},
        {buffer: coverageTopology, size: 16},
        {buffer: baselineTopology, size: 16},
        {buffer: importanceConverged, size: 4},
        {buffer: keptOverflow, size: 4},
        {buffer: coverageKeepMask, size: vertexCount * 4},
        {buffer: independentKeepMask, size: vertexCount * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        const at = (offset: number) => words[offset];
        const colorIds = words.subarray(layoutWords.colors, layoutWords.colors + polygonCount);
        const uncolored = colorIds.filter(value => value === GPU_MAP_COLORING_UNCOLORED).length;
        neighborsReadout.setValue(
          `${formatCount(at(layoutWords.offsets + polygonCount))} directed (${(
            at(layoutWords.offsets + polygonCount) / Math.max(polygonCount, 1)
          ).toFixed(1)} per polygon)${at(layoutWords.weightOverflow) ? ', OVERFLOW' : ''}`
        );
        colorsReadout.setValue(
          `${at(layoutWords.colorCount)}${uncolored ? ` (${uncolored} uncolored)` : ''}`
        );
        conflictsReadout.setValue(`${at(layoutWords.conflicts)} (seed ${seedShown})`);
        coloringReadout.setValue(
          `${at(layoutWords.converged) ? 'yes' : 'NO'} / ${at(layoutWords.rounds)}`
        );
        const coverageMask = words.subarray(
          layoutWords.coverageMask,
          layoutWords.coverageMask + vertexCount
        );
        const independentMask = words.subarray(
          layoutWords.independentMask,
          layoutWords.independentMask + vertexCount
        );
        const countKept = (mask: Uint32Array) => mask.reduce((sum, value) => sum + value, 0);
        const coverageKept = countKept(coverageMask);
        const independentKept = countKept(independentMask);
        coverageKeptReadout.setValue(
          `${formatCount(coverageKept)} / ${formatCount(vertexCount)} (${(
            (100 * coverageKept) / vertexCount
          ).toFixed(1)}%)${at(layoutWords.coverageOverflow) ? ', OVERFLOW' : ''}`
        );
        independentKeptReadout.setValue(
          `${formatCount(independentKept)} / ${formatCount(vertexCount)} (${(
            (100 * independentKept) / vertexCount
          ).toFixed(1)}%)${at(layoutWords.keptOverflow) ? ', OVERFLOW' : ''}`
        );
        const [found, remaining, restored, pairOverflow] = [0, 1, 2, 3].map(
          word => words[layoutWords.coverageTopology + word]
        );
        inputCrossingsReadout.setValue(
          `${formatCount(words[layoutWords.baselineTopology])} (overlaps the data already has; repair cannot remove them)`
        );
        if (repairTopology) {
          const overflowNote = pairOverflow ? ' (candidate list OVERFLOW: lower bounds)' : '';
          crossingsFoundReadout.setValue(`${formatCount(found)}${overflowNote}`);
          crossingsRemainingReadout.setValue(
            `${formatCount(remaining)} / ${formatCount(Math.max(found - remaining, 0))}`
          );
          restoredReadout.setValue(formatCount(restored));
        } else {
          crossingsFoundReadout.setValue('repair off');
          crossingsRemainingReadout.setValue('not checked');
          restoredReadout.setValue('0');
        }
        convergenceReadout.setValue(
          `coverage ${at(layoutWords.coverageConverged) ? 'yes' : 'NO'}, independent importance ${
            at(layoutWords.importanceConverged) ? 'yes' : 'NO (superset of DP)'
          }`
        );
        // Shared vertices that the rings of different polygons keep inconsistently.
        let coverageGaps = 0;
        const markers: number[] = [];
        for (const group of sharedGroups) {
          const coverageFirst = coverageMask[group[0]];
          if (group.some(vertex => coverageMask[vertex] !== coverageFirst)) coverageGaps++;
          const independentFirst = independentMask[group[0]];
          if (group.some(vertex => independentMask[vertex] !== independentFirst)) {
            markers.push(
              zones.polygonPositions[group[0] * 2],
              zones.polygonPositions[group[0] * 2 + 1]
            );
          }
        }
        const independentGaps = markers.length / 2;
        gapReadout.setValue(
          `${formatCount(coverageGaps)} / ${formatCount(independentGaps)} of ${formatCount(
            sharedGroups.length
          )} shared vertices`
        );
        if (markers.length > 0) markerBuffer.write(Float32Array.from(markers));
        if (independentGaps !== markerCount) {
          markerCount = independentGaps;
          context.updateLayers();
        }
        let coverageCollapsed = 0;
        let independentCollapsed = 0;
        for (let ring = 0; ring < ringCount; ring++) {
          let coverageRing = 0;
          let independentRing = 0;
          for (
            let vertex = zones.ringOffsets[ring];
            vertex < zones.ringOffsets[ring + 1];
            vertex++
          ) {
            coverageRing += coverageMask[vertex];
            independentRing += independentMask[vertex];
          }
          if (coverageRing < 3) coverageCollapsed++;
          if (independentRing < 3) independentCollapsed++;
        }
        collapsedReadout.setValue(
          `${coverageCollapsed} / ${independentCollapsed} (coverage / independent)`
        );
      }
    );
    let seedShown = seed;

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [
        compiledColoring,
        compiledRepaired,
        compiledPlain,
        compiledImportance,
        compiledSelection
      ],
      encode(commandEncoder, frame) {
        if (!importanceEncoded) {
          compiledImportance.encode(commandEncoder, {parameters: undefined});
          importanceEncoded = true;
        }
        if (coloringDirty) {
          compiledColoring.encode(commandEncoder, {parameters: undefined});
          coloringDirty = false;
          seedShown = seed;
          summary.markStale();
        }
        const tolerance = Math.fround(10 ** logTolerance);
        if (!baselineEncoded) {
          // One detection pass over the unsimplified geometry: the defects the input already has.
          baselineEncoded = true;
          parameterBuffer.write(
            getGPULineSimplificationParameterValues({tolerance: 0}, parameterValues)
          );
          compiledRepaired.encode(commandEncoder, {parameters: undefined});
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: coverageTopology,
            destinationBuffer: baselineTopology,
            size: 16
          });
        } else if (tolerance !== appliedTolerance) {
          appliedTolerance = tolerance;
          parameterBuffer.write(
            getGPULineSimplificationParameterValues({tolerance}, parameterValues)
          );
          (repairTopology ? compiledRepaired : compiledPlain).encode(commandEncoder, {
            parameters: undefined
          });
          compiledSelection.encode(commandEncoder, {parameters: undefined});
          toleranceReadout.setValue(formatMeters(tolerance));
          summary.markStale();
        }
        if (frame.frameIndex > 0) summary.flush(commandEncoder);
      },
      getLayers(): Layer[] {
        const layers: Layer[] = [];
        if (showFill) {
          layers.push(
            new ZoneFillLayer({
              id: 'coverage-fill',
              coordinateOrigin,
              triangleVertices,
              triangleOwners,
              zoneColors: colors,
              triangleCount,
              palette: FILL_PALETTE,
              fillOpacity: 0.55
            })
          );
        }
        if (showOriginal) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'coverage-original',
              coordinateOrigin,
              segments: outlineBuffer,
              instanceCount: segmentCount,
              widthPixels: 1,
              color: [255, 255, 255, 70]
            })
          );
        }
        const ringLayer = (
          id: string,
          keepMask: typeof coverageKeepMask,
          color: readonly number[],
          width: number
        ) =>
          new RingOutlineLayer({
            id,
            coordinateOrigin,
            vertices: positionsBuffer,
            keepMask,
            ringOffsets: ringOffsetsBuffer,
            vertexRings: vertexRingsBuffer,
            vertexCount,
            widthPixels: width,
            color: color as [number, number, number, number]
          });
        if (view === 'independent' || view === 'both') {
          layers.push(ringLayer('coverage-independent', independentKeepMask, INDEPENDENT_COLOR, 3));
        }
        if (view === 'coverage' || view === 'both') {
          layers.push(ringLayer('coverage-simplified', coverageKeepMask, COVERAGE_COLOR, 1.5));
        }
        if (view !== 'coverage' && markerCount > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'coverage-gap-markers',
              coordinateOrigin,
              positions: markerBuffer,
              instanceCount: markerCount,
              color: GAP_MARKER_COLOR,
              radiusPixels: 3.5
            })
          );
        }
        return layers;
      },
      destroy() {
        destroyed = true;
        clearTimeout(rebuildTimer);
        summary.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
