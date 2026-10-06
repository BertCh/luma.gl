// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Fleet dwell recipe scenes on the New York taxi trips: where do vehicles stop (stops variant) and
 * how long do tracks spend inside each zone (zone events variant). The zones are a synthetic mesh
 * of districts (the explorer has no polygon data for New York).
 */

import type {Layer} from '@deck.gl/core';
import {
  addFleetDwellRecipe,
  addFleetDwellZoneEventsRecipe,
  getGPUTrajectoryMetricsParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../spatial-analysis-layers';
import {formatCount} from '../spatial-analysis-resources';
import {formatCompact} from './classification-layers';
import {
  createDistrictPolygons,
  DirtyEncoder,
  getCoreBounds,
  getTripSegments,
  rasterizeFeatureRows,
  RecipeKit,
  SummaryReader,
  sliceSummary,
  type DistrictPolygons,
  type RecipeParameter,
  type RecipeSceneBuilder
} from './recipes-kit';

const FLEET_STOP_CAPACITY = 8192;
const FLEET_CANDIDATE_CAPACITY = 1048576;
const FLEET_COLUMNS = 6;
const FLEET_ROWS = 8;
const FLEET_RASTER_WIDTH = 200;
const FLEET_RASTER_HEIGHT = 240;
const DWELL_COLOR_SECONDS = 300;

function createFleetZones(
  pois: Float32Array,
  columns: number,
  rows: number,
  seed: number
): {polygons: DistrictPolygons; bounds: [number, number, number, number]} {
  const bounds = getCoreBounds(pois, 0.03);
  return {polygons: createDistrictPolygons(bounds, columns, rows, 0.3, seed), bounds};
}

/**
 * `addFleetDwellRecipe` (stops) and `addFleetDwellZoneEventsRecipe` (zone events) share a scene
 * because they answer the same question with different chains. `variant` is compile-time, so each
 * is its own select entry.
 */
export function createFleetDwellScene(variant: 'stops' | 'zone-events'): RecipeSceneBuilder {
  return async host => {
    const {context} = host;
    const [trips, pois] = await Promise.all([
      context.data.getNewYorkTrips(),
      context.data.getNewYorkPointsOfInterest()
    ]);
    context.signal.throwIfAborted();
    const {polygons, bounds} = createFleetZones(pois.positions, FLEET_COLUMNS, FLEET_ROWS, 7);
    const zoneCount = polygons.featureCount;
    const vertexCount = trips.vertexTimestamps.length;
    const tripCount = trips.tripOffsets.length - 1;

    let stopSpeed = 3;
    let stopDuration = 20;
    let colorByMean = false;
    let displayMaximum = 600;
    let stopCount = 0;

    const kit = new RecipeKit(context.device, `recipe-fleet-${variant}`);
    const positions = kit.input('positions', trips.vertexPositions, 'float32x2', vertexCount);
    const timestamps = kit.input('timestamps', trips.vertexTimestamps, 'float32', vertexCount);
    const trackOffsets = kit.input('track-offsets', trips.tripOffsets, 'uint32', tripCount + 1);
    const counts = kit.output('zone-counts', 'uint32', zoneCount);
    const sumValues = kit.output('zone-sums', 'float32', zoneCount);
    const means = kit.output('zone-means', 'float32', zoneCount);
    const maximums = kit.output('zone-maximums', 'float32', zoneCount);
    // The recipe tables are dense (one row per zone), so they are drawn directly: empty zones hold
    // 0 total dwell and NaN mean, which the raster layer treats as no data.
    const table = {
      counts: counts.view,
      sumValues: sumValues.view,
      means: means.view,
      maximums: maximums.view
    };

    const stopParameters = kit.parameter(
      'stop-parameters',
      'float32',
      getGPUTrajectoryMetricsParameterValues({
        stopSpeedThreshold: stopSpeed,
        stopMinimumDuration: stopDuration
      })
    );
    const stopCentroids = kit.output('stop-centroids', 'float32x2', FLEET_STOP_CAPACITY);
    const stopDurations = kit.output('stop-durations', 'float32', FLEET_STOP_CAPACITY);
    const stopTotal = kit.output('stop-count', 'uint32', 1);
    const stopOverflow = kit.output('stop-overflow', 'uint32', 1);
    const joinOverflow = kit.output('join-overflow', 'uint32', 1);
    const eventCount = kit.output('event-count', 'uint32', 1);
    const eventOverflow = kit.output('event-overflow', 'uint32', 1);
    const candidateCount = kit.output('candidate-count', 'uint32', 1);
    const candidateOverflow = kit.output('candidate-overflow', 'uint32', 1);
    const trackOverflow = kit.output('track-overflow', 'uint32', 1);
    const eventCapacityOverflow = kit.output('event-capacity-overflow', 'uint32', 1);

    let contributorCount = 0;
    if (variant === 'stops') {
      const recipe = addFleetDwellRecipe(kit.graph, {
        positions: positions.view,
        timestamps: timestamps.view,
        trackOffsets: trackOffsets.view,
        parameters: stopParameters.view,
        stopCapacity: FLEET_STOP_CAPACITY,
        zones: {
          polygonPositions: kit.input(
            'zones-positions',
            polygons.polygonPositions,
            'float32x2',
            polygons.polygonPositions.length / 2
          ).view,
          featureOffsets: kit.input(
            'zones-features',
            polygons.featureOffsets,
            'uint32',
            polygons.featureOffsets.length
          ).view,
          polygonOffsets: kit.input(
            'zones-polygons',
            polygons.polygonOffsets,
            'uint32',
            polygons.polygonOffsets.length
          ).view,
          ringOffsets: kit.input(
            'zones-rings',
            polygons.ringOffsets,
            'uint32',
            polygons.ringOffsets.length
          ).view,
          candidateCapacity: FLEET_STOP_CAPACITY * 4
        },
        stops: {
          count: stopTotal.view,
          overflow: stopOverflow.view,
          centroids: stopCentroids.view,
          durations: stopDurations.view
        },
        joinOverflow: joinOverflow.view,
        table
      });
      contributorCount = recipe.contributors.length;
    } else {
      const edgeCount = polygons.outlineSegments.length / 4;
      const edgeStarts = new Float32Array(edgeCount * 2);
      const edgeEnds = new Float32Array(edgeCount * 2);
      for (let edge = 0; edge < edgeCount; edge++) {
        edgeStarts.set(polygons.outlineSegments.subarray(edge * 4, edge * 4 + 2), edge * 2);
        edgeEnds.set(polygons.outlineSegments.subarray(edge * 4 + 2, edge * 4 + 4), edge * 2);
      }
      const recipe = addFleetDwellZoneEventsRecipe(kit.graph, {
        positions: positions.view,
        timestamps: timestamps.view,
        trackOffsets: trackOffsets.view,
        edgeStarts: kit.input('edge-starts', edgeStarts, 'float32x2', edgeCount).view,
        edgeEnds: kit.input('edge-ends', edgeEnds, 'float32x2', edgeCount).view,
        edgeZones: kit.input('edge-zones', polygons.outlineFeatureRows, 'uint32', edgeCount).view,
        zoneCount,
        candidateCapacity: FLEET_CANDIDATE_CAPACITY,
        maxEventsPerTrack: 64,
        events: {output: {count: eventCount.view, overflow: eventOverflow.view}},
        diagnostics: {
          candidateCount: candidateCount.view,
          candidateOverflow: candidateOverflow.view,
          trackOverflow: trackOverflow.view,
          eventOverflow: eventCapacityOverflow.view
        },
        table
      });
      contributorCount = recipe.contributors.length;
    }
    const compiled = kit.compile();

    const cellRows = kit.resources.createBuffer(
      'cell-rows',
      rasterizeFeatureRows(polygons, bounds, FLEET_RASTER_WIDTH, FLEET_RASTER_HEIGHT)
    );
    const outline = kit.resources.createBuffer('outline', polygons.outlineSegments);
    const {segments, segmentCount} = getTripSegments(trips.vertexPositions, trips.tripOffsets);
    const trails = kit.resources.createBuffer('trails', segments);

    const zoneBytes = zoneCount * 4;
    const summarySizes = [zoneBytes, zoneBytes, zoneBytes, 4, 4, 4, 4, 4, 4, 4, 4, 4];
    const reader = new SummaryReader(
      kit.resources,
      `fleet-${variant}`,
      [
        {buffer: counts.buffer, size: zoneBytes},
        {buffer: sumValues.buffer, size: zoneBytes},
        {buffer: maximums.buffer, size: zoneBytes},
        {buffer: stopTotal.buffer, size: 4},
        {buffer: stopOverflow.buffer, size: 4},
        {buffer: joinOverflow.buffer, size: 4},
        {buffer: eventCount.buffer, size: 4},
        {buffer: eventOverflow.buffer, size: 4},
        {buffer: candidateCount.buffer, size: 4},
        {buffer: candidateOverflow.buffer, size: 4},
        {buffer: trackOverflow.buffer, size: 4},
        {buffer: eventCapacityOverflow.buffer, size: 4}
      ],
      bytes => {
        const [
          countRows,
          sumRows,
          maximumRows,
          stops,
          stopFlag,
          joinFlag,
          events,
          eventFlag,
          candidates,
          candidateFlag,
          trackFlag,
          eventCapacityFlag
        ] = sliceSummary(bytes, summarySizes);
        let occupied = 0;
        let visits = 0;
        let totalDwell = 0;
        let longest = 0;
        let largestSum = 0;
        for (let row = 0; row < zoneCount; row++) {
          if (countRows.u32[row] === 0) {
            continue;
          }
          occupied++;
          visits += countRows.u32[row];
          totalDwell += sumRows.f32[row];
          longest = Math.max(longest, maximumRows.f32[row]);
          largestSum = Math.max(largestSum, sumRows.f32[row]);
        }
        const nextStopCount = Math.min(stops.u32[0], FLEET_STOP_CAPACITY);
        const nextMaximum = Math.max(60, colorByMean ? longest : largestSum);
        if (
          (variant === 'stops' && nextStopCount !== stopCount) ||
          Math.abs(nextMaximum - displayMaximum) > displayMaximum * 0.05
        ) {
          stopCount = nextStopCount;
          displayMaximum = nextMaximum;
          host.updateLayers();
        }
        host.setOutputs(
          variant === 'stops'
            ? [
                [
                  'Stops detected',
                  `${formatCount(stops.u32[0])} (capacity ${formatCount(FLEET_STOP_CAPACITY)})`
                ],
                ['Zones with stops', `${occupied} of ${zoneCount}`],
                ['Stops joined to zones', formatCount(visits)],
                [
                  'Total / longest dwell',
                  `${formatCompact(totalDwell)} s / ${formatCompact(longest)} s`
                ],
                ['Stop overflow', stopFlag.u32[0] ? 'YES' : 'no'],
                ['Join overflow', joinFlag.u32[0] ? 'YES' : 'no']
              ]
            : [
                ['Zone events', formatCount(events.u32[0])],
                ['Zones visited', `${occupied} of ${zoneCount}`],
                ['Track visits', formatCount(visits)],
                [
                  'Total / longest dwell',
                  `${formatCompact(totalDwell)} s / ${formatCompact(longest)} s`
                ],
                [
                  'Candidates needed / capacity',
                  `${formatCount(candidates.u32[0])} / ${formatCount(FLEET_CANDIDATE_CAPACITY)}`
                ],
                ['Candidate scratch overflow', candidateFlag.u32[0] ? 'YES' : 'no'],
                ['Per-track event bound overflow', trackFlag.u32[0] ? 'YES' : 'no'],
                ['Event list capacity overflow', eventCapacityFlag.u32[0] ? 'YES' : 'no'],
                ['Any overflow (combined flag)', eventFlag.u32[0] ? 'YES' : 'no']
              ]
        );
      }
    );
    const encoder = new DirtyEncoder(compiled, reader);

    const parameters: RecipeParameter[] = [];
    if (variant === 'stops') {
      parameters.push(
        {
          kind: 'slider',
          label: 'Stop speed threshold',
          minimum: 0.5,
          maximum: 8,
          step: 0.5,
          value: stopSpeed,
          format: value => `${value.toFixed(1)} m/s`,
          onChange: value => {
            stopSpeed = value;
            stopParameters.parameters.write(
              getGPUTrajectoryMetricsParameterValues({
                stopSpeedThreshold: stopSpeed,
                stopMinimumDuration: stopDuration
              })
            );
            encoder.markDirty();
          }
        },
        {
          kind: 'slider',
          label: 'Minimum stop duration',
          minimum: 5,
          maximum: 120,
          step: 5,
          value: stopDuration,
          format: value => `${value} s`,
          onChange: value => {
            stopDuration = value;
            stopParameters.parameters.write(
              getGPUTrajectoryMetricsParameterValues({
                stopSpeedThreshold: stopSpeed,
                stopMinimumDuration: stopDuration
              })
            );
            encoder.markDirty();
          }
        }
      );
    }
    parameters.push({
      kind: 'toggle',
      label: 'Color zones by mean dwell instead of total',
      value: colorByMean,
      onChange: value => {
        colorByMean = value;
        encoder.markDirty();
        host.updateLayers();
      }
    });

    const origin = trips.origin;
    return {
      compiled,
      contributorCount,
      chain:
        variant === 'stops'
          ? [
              'GPUTrajectoryMetrics (stops per trip)',
              'GPUPointInPolygonJoin (stop centroid to zone)',
              'stop-count mask adapter',
              'GPUGroupStatistics (dense, dwell per zone)'
            ]
          : [
              'GPUZoneEvents (enter and exit events)',
              'visited-cell key adapter',
              'GPUGroupStatistics (dense, dwell per zone)'
            ],
      parameters,
      legend:
        variant === 'stops'
          ? 'Zones colored by total (or mean) dwell time of the stops inside, dark to bright; dots are stops colored by duration.'
          : 'Zones colored by total (or mean) time the taxi tracks spend inside, dark to bright; no stop detection, any time inside counts.',
      dataNote:
        `${trips.attribution}; ${pois.attribution} for the zone area. Zones: synthetic ` +
        `${FLEET_COLUMNS} x ${FLEET_ROWS} mesh (no polygon data for New York)`,
      encode: commandEncoder => encoder.encode(commandEncoder),
      getLayers: (): Layer[] => {
        const layers: Layer[] = [
          new SpatialAnalysisRasterLayer({
            id: `recipe-fleet-${variant}-zones`,
            coordinateOrigin: [origin[0], origin[1], 0],
            gridSize: [FLEET_RASTER_WIDTH, FLEET_RASTER_HEIGHT],
            bounds,
            rowOrigin: 'south',
            values: colorByMean ? means.buffer : sumValues.buffer,
            valueFormat: 'float32',
            valueIndices: cellRows,
            colormap: 'inferno',
            valueRange: [0, displayMaximum],
            discardAtOrBelow: 0,
            noDataColor: [0, 0, 0, 0],
            sqrtScale: true,
            opacity: 0.75
          }),
          new SpatialAnalysisSegmentLayer({
            id: `recipe-fleet-${variant}-trails`,
            coordinateOrigin: [origin[0], origin[1], 0],
            segments: trails,
            instanceCount: segmentCount,
            color: [150, 170, 200, 40],
            widthPixels: 1
          }),
          new SpatialAnalysisSegmentLayer({
            id: `recipe-fleet-${variant}-outline`,
            coordinateOrigin: [origin[0], origin[1], 0],
            segments: outline,
            instanceCount: polygons.outlineSegments.length / 4,
            color: [235, 240, 250, 170],
            widthPixels: 1.4
          })
        ];
        if (variant === 'stops' && stopCount > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'recipe-fleet-stops',
              coordinateOrigin: [origin[0], origin[1], 0],
              positions: stopCentroids.buffer,
              instanceCount: stopCount,
              values: stopDurations.buffer,
              valueFormat: 'float32',
              colormap: 'viridis',
              valueRange: [0, DWELL_COLOR_SECONDS],
              radiusPixels: 3,
              opacity: 0.9
            })
          );
        }
        return layers;
      },
      destroy: () => {
        reader.stop();
        kit.resources.destroy();
      }
    };
  };
}
