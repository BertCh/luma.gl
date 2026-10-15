// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';
import {createGPULocalFrame} from '../../../analysis-core/contracts/local-frame';
import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';

registerLensScenario({
  id: 'gpu-spatial-analysis.fleet-dwell-recipe',
  entry: 'gpu-spatial-analysis',
  element: 'addFleetDwellRecipe',
  title: 'Vessel dwell per anchorage, Upper New York Bay',
  audience: 'analyst',
  rungs: ['result', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.aisTracks,
  load: async () => {
    const [
      {runS2Lens, uploadS2JoinPolygons},
      {runExample},
      {loadS3NewYorkHarborTracks, loadS3NewYorkHarborZones}
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./fleet-dwell-recipe.example'),
      import('./s3-lens-data')
    ]);
    const [tracks, zones] = await Promise.all([
      loadS3NewYorkHarborTracks(),
      loadS3NewYorkHarborZones()
    ]);
    return {
      run: device => {
        // One azimuthal-equidistant frame for tracks and zones, centred on the harbor.
        const frame = createGPULocalFrame({
          origin: [-74.03, 40.665],
          kind: 'azimuthal-equidistant',
          datum: 'EPSG:4326'
        });
        const fixCount = tracks.positionsLonLat.length / 2;
        const keySpaces: GPUComputationKeySpace[] = [
          {id: 'stop', label: 'stop slot', size: 160, indexSpace: '1d'},
          {id: 'fix', label: 'AIS fix', size: fixCount, indexSpace: '1d'},
          {
            id: 'track-offset',
            label: 'track offset',
            size: tracks.trackCount + 1,
            indexSpace: '1d'
          },
          {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.fleet-dwell-recipe',
          element: 'addFleetDwellRecipe',
          datasets: ['s3-ny-harbor-ais', 's3-ny-harbor-zones'],
          retain: [
            'stopTracks',
            'stopCentroids',
            'stopDurations',
            'stopCount',
            'stopOverflow',
            'stopZones',
            'zoneStopCounts',
            'zoneDwell',
            'zoneLongestDwell'
          ],
          keySpaces,
          choices: {stopSpeedThreshold: 0.5, stopMinimumDuration: 300},
          async build(graph, uploads) {
            const positions = uploads.view(
              'positions',
              frame.projectArray(tracks.positionsLonLat),
              'float32x2'
            );
            const timestamps = uploads.view('timestamps', tracks.timestamps, 'float32');
            const trackOffsets = uploads.view('trackOffsets', tracks.trackOffsets, 'uint32');
            // The zone feature → polygon → ring layout: the helper takes polygon → ring offsets
            // as its feature offsets because each anchorage is one polygon.
            const anchorageGeometry = uploadS2JoinPolygons(uploads, frame, 'anchorages', {
              vertices: Array.from(zones.positionsLonLat),
              featureOffsets: Array.from(zones.polygonOffsets),
              ringOffsets: Array.from(zones.ringOffsets)
            });
            keySpaces.push(...anchorageGeometry.keySpaces);
            const result = await runExample(graph, {
              positions,
              timestamps,
              trackOffsets,
              anchorages: anchorageGeometry.geometry,
              spatialContext: frame.spatialContext
            });
            uploads.own(result.parameters);
            const captures: Record<string, S2CaptureSpec> = {
              tracks: {
                view: positions,
                role: 'input',
                keySpace: 'fix',
                geometry: {
                  kind: 'lines',
                  offsetsCapture: 'trackOffsets',
                  coordinateModel: 'projected',
                  unit: 'm'
                }
              },
              trackOffsets: {view: trackOffsets, role: 'input', keySpace: 'track-offset'},
              timestamps: {view: timestamps, role: 'input', keySpace: 'fix'},
              ...anchorageGeometry.captures
            };
            captures['anchorages'].meaning = 'context';
            captures['resultAnchorages'] = {
              view: anchorageGeometry.display,
              role: 'output',
              keySpace: 'anchorages',
              geometry: {
                kind: 'polygons',
                offsetsCapture: 'anchoragesDisplayOffsets',
                valuesCapture: 'zoneDwell',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            const outputs: [string, {view: S2CaptureSpec['view']}, string][] = [
              ['zoneStopCounts', {view: result.table.counts}, 'anchorages'],
              ['zoneDwell', {view: result.table.sumValues}, 'anchorages'],
              ['zoneLongestDwell', {view: result.table.maximums}, 'anchorages'],
              ['stopTracks', {view: result.stops.ids}, 'stop'],
              ['stopCentroids', {view: result.stops.centroids}, 'stop'],
              ['stopDurations', {view: result.stops.durations}, 'stop'],
              ['stopZones', {view: result.stopZones}, 'stop'],
              ['stopCount', {view: result.stops.status.count}, 'status'],
              ['stopOverflow', {view: result.stops.status.overflow}, 'status']
            ];
            for (const [name, {view}, keySpace] of outputs) {
              captures[name] = {view, role: 'output', keySpace};
            }
            return captures;
          }
        });
      }
    };
  }
});
