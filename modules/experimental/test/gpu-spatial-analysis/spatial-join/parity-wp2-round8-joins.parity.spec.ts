// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUSpatialJoinPrepared,
  GPUSpatialJoinCandidates,
  GPUNearestFeatureJoin,
  GPUBufferSelection,
  GPUPointInPolygonJoin
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/fixtures/parity-wp2-round8-joins.json';

// Keep live radius updates in each graph, but bound serial readbacks per test.
// All 141 radii remain covered; the 120-second timeout and numeric bounds are unchanged.
for (const completeEntry of fixture.cases)
  for (let frameStart = 0; frameStart < completeEntry.frames.length; frameStart += 16) {
    const entry = {
      ...completeEntry,
      frames: completeEntry.frames.slice(frameStart, frameStart + 16)
    };
    for (const spatialSort of [false, true])
      for (const querySort of [false, true])
        for (const ties of ['lowest-id', 'all'] as const) {
          it(`showcase joins GEOS ${entry.name}/sort=${spatialSort}/querySort=${querySort}/ties=${ties}/frames=${frameStart}-${frameStart + entry.frames.length - 1}`, async () => {
            const device = await getWebGPUTestDevice();
            requireTestDevice(device);
            const rig = createParityGraph(device);
            const {input, output, graph} = rig;
            const rows = entry.queries.length / 2;
            const featuresCount = entry.distances[0].length;
            const queries = {
              kind: 'points' as const,
              positions: input('queries', entry.queries, 'float32x2').view
            };
            const positions = input('positions', entry.positions, 'float32x2').view;
            const features =
              entry.kind === 'points'
                ? {kind: 'points' as const, positions}
                : entry.kind === 'segments'
                  ? {
                      kind: 'segments' as const,
                      starts: input(
                        'starts',
                        entry.lineOffsets
                          .slice(0, -1)
                          .flatMap(index => entry.positions.slice(2 * index, 2 * index + 2)),
                        'float32x2'
                      ).view,
                      ends: input(
                        'ends',
                        entry.lineOffsets
                          .slice(1)
                          .flatMap(index => entry.positions.slice(2 * index - 2, 2 * index)),
                        'float32x2'
                      ).view
                    }
                  : entry.kind === 'lines'
                    ? {
                        kind: 'lines' as const,
                        positions,
                        lineOffsets: input('offsets', entry.lineOffsets, 'uint32').view
                      }
                    : {
                        kind: 'polygons' as const,
                        positions,
                        featureOffsets: input('features', entry.featureOffsets, 'uint32').view,
                        polygonOffsets: input('polygons', entry.polygonOffsets, 'uint32').view,
                        ringOffsets: input('rings', entry.ringOffsets, 'uint32').view
                      };
            const prepared =
              features.kind === 'polygons'
                ? new GPUSpatialJoinPrepared({geometry: features, spatialSort})
                : undefined;
            if (prepared) graph.add(prepared);
            const radius = input('radius', [0], 'float32');
            const neighborCapacity = ties === 'all' ? 16 : 8;
            const neighborIds = output('neighbor-ids', rows * neighborCapacity, 'uint32');
            const neighborDistances = output(
              'neighbor-distances',
              rows * neighborCapacity,
              'float32'
            );
            const neighborFeet = output('neighbor-feet', rows * neighborCapacity, 'float32x2');
            const neighborCounts = output('neighbor-counts', rows, 'uint32');
            const neighborOverflow = output('neighbor-overflow', 1, 'uint32');
            graph.add(
              new GPUNearestFeatureJoin({
                queries,
                features,
                prepared,
                k: 8,
                neighborCapacity,
                ties,
                radius: radius.view,
                spatialSort,
                querySort,
                output: {
                  neighborIds: neighborIds.view,
                  neighborDistances: neighborDistances.view,
                  neighborFootPoints: neighborFeet.view,
                  neighborCounts: neighborCounts.view,
                  status: {overflow: neighborOverflow.view}
                }
              })
            );
            const nearestIds = output('nearest-ids', rows, 'uint32');
            const nearestDistances = output('nearest-distances', rows, 'float32');
            const featureCounts = output('feature-counts', featuresCount, 'uint32');
            const nearestOverflow = output('nearest-overflow', 1, 'uint32');
            const mask = output('mask', rows, 'uint32');
            const selectedIds = output('selected-ids', rows, 'uint32');
            const selectedCount = output('selected-count', 1, 'uint32');
            const selectedOverflow = output('selected-overflow', 1, 'uint32');
            const drawCount = output('draw-count', 1, 'uint32');
            const bufferOverflow = output('buffer-overflow', 1, 'uint32');
            const bufferNearest = output('buffer-nearest', rows, 'uint32');
            const bufferDistances = output('buffer-distances', rows, 'float32');
            if (features.kind === 'points' || features.kind === 'segments') {
              graph.add(
                new GPUNearestFeatureJoin({
                  id: 'closest',
                  queries,
                  features,
                  radius: radius.view,
                  candidateCapacity: rows * featuresCount,
                  spatialSort,
                  querySort,
                  output: {
                    nearestFeatureIds: nearestIds.view,
                    nearestDistances: nearestDistances.view,
                    featureCounts: featureCounts.view,
                    status: {overflow: nearestOverflow.view}
                  }
                })
              );
              graph.add(
                new GPUBufferSelection({
                  points: queries,
                  features,
                  distance: radius.view,
                  candidateCapacity: rows * featuresCount,
                  spatialSort,
                  querySort,
                  output: {
                    outputMask: mask.view,
                    nearestFeatureIds: bufferNearest.view,
                    distances: bufferDistances.view,
                    drawInstanceCount: drawCount.view,
                    matches: {
                      ids: selectedIds.view,
                      status: {count: selectedCount.view, overflow: selectedOverflow.view}
                    },
                    status: {overflow: bufferOverflow.view}
                  }
                })
              );
            }
            const compiled = await graph.compileAsync();
            try {
              for (const [frameIndex, frame] of entry.frames.entries()) {
                radius.buffer.write(Float32Array.of(frame.radius));
                if (frameStart + frameIndex === 4) prepared?.invalidate();
                submitGraph(device, compiled, undefined);
                const label = `${entry.name}/sort=${spatialSort}/querySort=${querySort}/ties=${ties}/radius=${frame.radius}`;
                const compare = (
                  className: string,
                  metric: string,
                  actual: number[],
                  expected: number[],
                  absolute = 0
                ) =>
                  compareParity(
                    className,
                    label,
                    fixture.reference,
                    metric,
                    actual,
                    expected,
                    absolute
                  );
                const ordered = entry.distances.map(distances =>
                  distances
                    .map((distance, index) => ({distance, index}))
                    .filter(value => value.distance <= frame.radius)
                    .sort(
                      (first, second) =>
                        first.distance - second.distance || first.index - second.index
                    )
                );
                const selected = ordered.map(values =>
                  ties === 'all' && values.length > 8
                    ? values
                        .filter(value => value.distance <= values[7].distance)
                        .slice(0, neighborCapacity)
                    : values.slice(0, 8)
                );
                const ids = await readUint32(neighborIds.buffer, rows * neighborCapacity);
                const distances = await readFloat32(
                  neighborDistances.buffer,
                  rows * neighborCapacity
                );
                const feet = await readFloat32(neighborFeet.buffer, rows * neighborCapacity * 2);
                compare(
                  'GPUNearestFeatureJoin',
                  'all neighbor counts',
                  await readUint32(neighborCounts.buffer, rows),
                  selected.map(values => values.length)
                );
                compare(
                  'GPUNearestFeatureJoin',
                  'all neighbor IDs including padding',
                  ids,
                  selected.flatMap(values => [
                    ...values.map(value => value.index),
                    ...new Array(neighborCapacity - values.length).fill(0xffffffff)
                  ])
                );
                compare(
                  'GPUNearestFeatureJoin',
                  'every retained distance',
                  selected.flatMap((values, row) =>
                    values.map((_, slot) => distances[row * neighborCapacity + slot])
                  ),
                  selected.flatMap(values => values.map(value => value.distance)),
                  0.00390625
                ); // 16 ulps at max coordinate 4000.
                compare(
                  'GPUNearestFeatureJoin',
                  'every retained foot point',
                  selected.flatMap((values, row) =>
                    values.flatMap((_, slot) =>
                      feet.slice(
                        (row * neighborCapacity + slot) * 2,
                        (row * neighborCapacity + slot) * 2 + 2
                      )
                    )
                  ),
                  selected.flatMap((values, row) =>
                    values.flatMap(value => entry.feet[row][value.index])
                  ),
                  0.00390625
                );
                compare(
                  'GPUNearestFeatureJoin',
                  'neighbor overflow',
                  await readUint32(neighborOverflow.buffer, 1),
                  [0]
                );
                if (prepared)
                  compare(
                    'GPUSpatialJoinPrepared',
                    'build/reuse/invalidation with independently checked neighbor rows',
                    [prepared.encodedBuildCount],
                    // Only the first batch includes the invalidation at original frame 4.
                    [frameStart === 0 && frameIndex >= 4 ? 2 : 1]
                  );
                if (features.kind === 'points' || features.kind === 'segments') {
                  const expectedIds = ordered.map(values => values[0]?.index ?? 0xffffffff);
                  const expectedDistances = ordered.map(values => values[0]?.distance ?? -1);
                  const actualNearest = await readUint32(nearestIds.buffer, rows);
                  compare('GPUNearestFeatureJoin', 'closest IDs', actualNearest, expectedIds);
                  compare(
                    'GPUNearestFeatureJoin',
                    'closest distances and no-match sentinel',
                    await readFloat32(nearestDistances.buffer, rows),
                    expectedDistances,
                    0.00390625
                  );
                  compare(
                    'GPUNearestFeatureJoin',
                    'feature counts',
                    await readUint32(featureCounts.buffer, featuresCount),
                    Array.from(
                      {length: featuresCount},
                      (_, index) => expectedIds.filter(value => value === index).length
                    )
                  );
                  compare(
                    'GPUNearestFeatureJoin',
                    'closest overflow',
                    await readUint32(nearestOverflow.buffer, 1),
                    [0]
                  );
                  const expectedMask = ordered.map(values => Number(values.length > 0));
                  const expectedSelected = expectedMask.flatMap((value, index) =>
                    value ? [index] : []
                  );
                  compare(
                    'GPUBufferSelection',
                    'mask',
                    await readUint32(mask.buffer, rows),
                    expectedMask
                  );
                  compare(
                    'GPUBufferSelection',
                    'compact IDs',
                    (await readUint32(selectedIds.buffer, rows)).slice(0, expectedSelected.length),
                    expectedSelected
                  );
                  compare(
                    'GPUBufferSelection',
                    'count/indirect count/overflow',
                    [
                      ...(await readUint32(selectedCount.buffer, 1)),
                      ...(await readUint32(drawCount.buffer, 1)),
                      ...(await readUint32(selectedOverflow.buffer, 1)),
                      ...(await readUint32(bufferOverflow.buffer, 1))
                    ],
                    [expectedSelected.length, expectedSelected.length, 0, 0]
                  );
                  compare(
                    'GPUBufferSelection',
                    'nearest IDs',
                    await readUint32(bufferNearest.buffer, rows),
                    expectedIds
                  );
                  compare(
                    'GPUBufferSelection',
                    'nearest distances',
                    await readFloat32(bufferDistances.buffer, rows),
                    expectedDistances,
                    0.00390625
                  );
                }
              }
            } finally {
              compiled.destroy();
              prepared?.destroy();
              rig.destroy();
              // The cached test device is borrowed; only release this test's resources.
            }
          }, 120000);
        }
  }

// The candidate radius is compile-time. Cover all five select values in nearest-facility.
for (const entry of fixture.cases)
  for (const spatialSort of [false, true])
    for (const usePrepared of [false, true]) {
      it(`prepared candidates GEOS ${entry.kind}/sort=${spatialSort}/prepared=${usePrepared}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        for (const frame of entry.frames.filter(value =>
          [0, 250, 500, 1000, 2000].includes(value.radius)
        )) {
          const rig = createParityGraph(device);
          const positions = rig.input('positions', entry.positions, 'float32x2').view;
          // A two-vertex polyline has the same bounds as a segment; preserve source feature rows.
          const right =
            entry.kind === 'points'
              ? {kind: 'points' as const, positions}
              : entry.kind === 'polygons'
                ? {
                    kind: 'polygons' as const,
                    positions,
                    featureOffsets: rig.input('features', entry.featureOffsets, 'uint32').view,
                    polygonOffsets: rig.input('polygons', entry.polygonOffsets, 'uint32').view,
                    ringOffsets: rig.input('rings', entry.ringOffsets, 'uint32').view
                  }
                : {
                    kind: 'lines' as const,
                    positions,
                    lineOffsets: rig.input('offsets', entry.lineOffsets, 'uint32').view
                  };
          const prepared = usePrepared
            ? new GPUSpatialJoinPrepared({geometry: right, spatialSort})
            : undefined;
          if (prepared) rig.graph.add(prepared);
          const capacity = (entry.queries.length / 2) * entry.distances[0].length;
          const leftIds = rig.output('left', capacity, 'uint32');
          const rightIds = rig.output('right', capacity, 'uint32');
          const count = rig.output('count', 1, 'uint32');
          const total = rig.output('total', 1, 'uint32');
          const overflow = rig.output('overflow', 1, 'uint32');
          rig.graph.add(
            new GPUSpatialJoinCandidates({
              left: {
                kind: 'points',
                positions: rig.input('queries', entry.queries, 'float32x2').view
              },
              right,
              prepared,
              distance: frame.radius,
              output: {
                leftIds: leftIds.view,
                rightIds: rightIds.view,
                status: {count: count.view, requiredCount: total.view, overflow: overflow.view}
              }
            })
          );
          const compiled = await rig.graph.compileAsync();
          try {
            for (const encoding of [0, 1, 2]) {
              if (encoding === 2) prepared?.invalidate();
              submitGraph(device, compiled, undefined);
              const label = `${entry.kind}/sort=${spatialSort}/prepared=${usePrepared}/radius=${frame.radius}/encoding=${encoding}`;
              for (const className of usePrepared
                ? ['GPUSpatialJoinCandidates', 'GPUSpatialJoinPrepared']
                : ['GPUSpatialJoinCandidates']) {
                compareParity(
                  className,
                  label,
                  fixture.reference,
                  'count/required/overflow',
                  [
                    ...(await readUint32(count.buffer, 1)),
                    ...(await readUint32(total.buffer, 1)),
                    ...(await readUint32(overflow.buffer, 1))
                  ],
                  [frame.candidates.length, frame.candidates.length, 0]
                );
                compareParity(
                  className,
                  label,
                  fixture.reference,
                  'complete left rows',
                  (await readUint32(leftIds.buffer, capacity)).slice(0, frame.candidates.length),
                  frame.candidates.map(pair => pair[0])
                );
                compareParity(
                  className,
                  label,
                  fixture.reference,
                  'complete right rows',
                  (await readUint32(rightIds.buffer, capacity)).slice(0, frame.candidates.length),
                  frame.candidates.map(pair => pair[1])
                );
              }
            }
          } finally {
            compiled.destroy();
            prepared?.destroy();
            rig.destroy();
          }
        }
      }, 120000);
    }

for (const spatialSort of [false, true])
  for (const edgeSlabs of [false, true]) {
    it(`prepared polygon exact consumer sort=${spatialSort}/slabs=${edgeSlabs}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const entry = fixture.cases.find(value => value.kind === 'polygons')!;
      const rig = createParityGraph(device);
      const polygons = {
        kind: 'polygons' as const,
        positions: rig.input('positions', entry.positions, 'float32x2').view,
        featureOffsets: rig.input('features', entry.featureOffsets, 'uint32').view,
        polygonOffsets: rig.input('polygons', entry.polygonOffsets, 'uint32').view,
        ringOffsets: rig.input('rings', entry.ringOffsets, 'uint32').view
      };
      const prepared = new GPUSpatialJoinPrepared({geometry: polygons, spatialSort, edgeSlabs});
      rig.graph.add(prepared);
      const rows = entry.queries.length / 2;
      const points = {
        kind: 'points' as const,
        positions: rig.input('queries', entry.queries, 'float32x2').view
      };
      const ids = rig.output('ids', rows, 'uint32');
      const overflow = rig.output('overflow', 1, 'uint32');
      rig.graph.add(
        new GPUPointInPolygonJoin({
          points,
          polygons,
          prepared,
          candidateCapacity: rows * 16,
          boundaryRule: 'closed',
          output: {pointFeatureIds: ids.view, status: {overflow: overflow.view}}
        })
      );
      const compiled = await rig.graph.compileAsync();
      try {
        for (const encoding of [0, 1, 2]) {
          if (encoding === 2) prepared.invalidate();
          submitGraph(device, compiled, undefined);
          const label = `polygons/sort=${spatialSort}/slabs=${edgeSlabs}/encoding=${encoding}`;
          compareParity(
            'GPUSpatialJoinPrepared',
            label,
            fixture.reference,
            'every independently classified point',
            await readUint32(ids.buffer, rows),
            entry.distances.map(distances => {
              const index = distances.findIndex(distance => distance === 0);
              return index < 0 ? 0xffffffff : index;
            })
          );
          compareParity(
            'GPUSpatialJoinPrepared',
            label,
            fixture.reference,
            'exact consumer overflow',
            await readUint32(overflow.buffer, 1),
            [0]
          );
        }
      } finally {
        compiled.destroy();
        prepared.destroy();
        rig.destroy();
        // The cached test device belongs to the harness.
      }
    }, 120000);
  }
