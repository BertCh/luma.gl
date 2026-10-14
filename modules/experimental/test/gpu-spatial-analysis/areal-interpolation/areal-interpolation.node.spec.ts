// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUArealInterpolation,
  GPUPycnophylactic
} from '../../../src/gpu-spatial-analysis/areal-interpolation/index';
import {getSphericalRasterCellArea} from '../../../src/gpu-spatial-analysis/areal-interpolation/index';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  AREAL_NUMPY,
  PYCNOPHYLACTIC_NUMPY,
  SPHERICAL_CELL_AREA_NUMPY
} from './change-of-support-numpy-fixtures';
import {
  applyZoneAnchors,
  computeArealOracle,
  computePycnophylacticDetailedOracle,
  computeSphericalCellAreas
} from './areal-interpolation-oracle';

let serial = 0;

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'areal-nodes'});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {graph, view};
}

it('GPUArealInterpolation declares nodes and validates its props', () => {
  const {graph, view} = setup();
  const props = {
    sourceZones: view('uint32', 64),
    targetZones: view('uint32', 64),
    sourceCount: 4,
    targetCount: 3,
    weights: {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    },
    overflow: view('uint32', 1)
  };
  const nodes = new GPUArealInterpolation(props).getCommandNodes(graph);
  expect(nodes.length).toBeGreaterThan(10);
  expect(() => new GPUArealInterpolation({...props, targetCount: 4})).toThrow(/targetCount \+ 1/);
  expect(() => new GPUArealInterpolation({...props, targetZones: view('uint32', 63)})).toThrow(
    /same nonzero length/
  );
  expect(() => new GPUArealInterpolation({...props, alternateWeights: view('float32', 7)})).toThrow(
    /alternateWeights length/
  );
});

it('GPUPycnophylactic declares nodes per iteration and validates its props', () => {
  const {graph, view} = setup();
  const props = {
    width: 8,
    height: 4,
    zones: view('uint32', 32),
    zoneCount: 3,
    totals: view('float32', 3),
    iterations: 2,
    output: view('float32', 32)
  };
  const few = new GPUPycnophylactic(props).getCommandNodes(graph).length;
  const {graph: other, view: otherView} = setup();
  const many = new GPUPycnophylactic({
    ...props,
    zones: otherView('uint32', 32),
    totals: otherView('float32', 3),
    output: otherView('float32', 32),
    iterations: 4
  }).getCommandNodes(other).length;
  expect(many).toBeGreaterThan(few);
  expect(() => new GPUPycnophylactic({...props, iterations: -1})).toThrow(/iterations/);
  expect(() => new GPUPycnophylactic({...props, totals: view('float32', 2)})).toThrow(/totals/);
});

it('GPUSpatialLag accepts cross weights with sourceCount and columnCount', () => {
  const {graph, view} = setup();
  const weights = {
    offsets: view('uint32', 4),
    neighbors: view('uint32', 8),
    weights: view('float32', 8)
  };
  const props = {weights, sourceCount: 5, values: view('float32', 10), output: view('float32', 6)};
  expect(new GPUSpatialLag({...props, columnCount: 2}).getCommandNodes(graph)).toHaveLength(1);
  expect(() => new GPUSpatialLag(props)).toThrow(/sourceCount \* columnCount/);
  expect(() => new GPUSpatialLag({...props, columnCount: 2, mask: view('uint32', 3)})).toThrow(
    /square/
  );
});

const nanify = (values: readonly (number | null)[]) => values.map(value => value ?? Number.NaN);

it('getSphericalRasterCellArea matches the numeric integral of R^2 cos(lat)', () => {
  // numpy/scipy dblquad of R^2 cos(lat) over the cell (spherical_cell_area.py).
  for (const entry of SPHERICAL_CELL_AREA_NUMPY) {
    const area = getSphericalRasterCellArea(
      entry.originLatitude,
      entry.cellWidth,
      entry.cellHeight,
      entry.row
    );
    expect(Math.abs(area - entry.area) / entry.area).toBeLessThan(1e-9);
  }
});

it('the pycnophylactic oracle reproduces the numpy reimplementation', () => {
  const {fixture} = PYCNOPHYLACTIC_NUMPY;
  const base = {
    width: fixture.width,
    height: fixture.height,
    zones: fixture.zones,
    zoneCount: fixture.zoneCount,
    totals: fixture.totals,
    kernel: 'rook' as const
  };
  const expectMass = (actual: number[], expected: readonly number[]) => {
    expect(actual).toHaveLength(expected.length);
    for (const [cell, value] of expected.entries()) {
      expect(Math.abs(actual[cell] - value), `cell ${cell}`).toBeLessThanOrEqual(
        1e-9 * (1 + value)
      );
    }
  };
  expectMass(
    computePycnophylacticDetailedOracle({...base, iterations: 6}).mass,
    PYCNOPHYLACTIC_NUMPY.planar6.mass
  );
  const unanchored = computePycnophylacticDetailedOracle({...base, iterations: 6});
  expect(unanchored.invalidCount).toBe(PYCNOPHYLACTIC_NUMPY.planar6.invalidCount);
  const anchored = computePycnophylacticDetailedOracle({
    ...base,
    iterations: 6,
    zoneAnchorCells: fixture.anchors
  });
  expectMass(anchored.mass, PYCNOPHYLACTIC_NUMPY.planarAnchored6.mass);
  expect(anchored.invalidCount).toBe(0);
  // Anchoring keeps the mass of zones that have no raster cell.
  const total = (values: number[]) => values.reduce((sum, value) => sum + value, 0);
  expect(total(unanchored.mass)).toBeCloseTo(465, 9);
  expect(total(anchored.mass)).toBeCloseTo(507, 9);
  const converged = computePycnophylacticDetailedOracle({
    ...base,
    iterations: 12,
    zoneAnchorCells: fixture.anchors,
    tolerance: PYCNOPHYLACTIC_NUMPY.tolerance
  });
  expectMass(converged.mass, PYCNOPHYLACTIC_NUMPY.planarTolerance.mass);
  expect(converged.converged).toBe(PYCNOPHYLACTIC_NUMPY.planarTolerance.converged);
  expect(converged.convergedAfter).toBe(PYCNOPHYLACTIC_NUMPY.planarTolerance.flaggedAt);
  const limited = computePycnophylacticDetailedOracle({
    ...base,
    iterations: 3,
    zoneAnchorCells: fixture.anchors,
    tolerance: 1e-12
  });
  expect(limited.converged).toBe(false);
  expectMass(limited.mass, PYCNOPHYLACTIC_NUMPY.planarLimit.mass);
  const cellAreas = computeSphericalCellAreas(
    fixture.width,
    fixture.height,
    PYCNOPHYLACTIC_NUMPY.geographic.originLatitude,
    PYCNOPHYLACTIC_NUMPY.geographic.cellWidth,
    PYCNOPHYLACTIC_NUMPY.geographic.cellHeight
  );
  for (const [cell, value] of PYCNOPHYLACTIC_NUMPY.geographicAreas.entries()) {
    expect(Math.abs(cellAreas[cell] - value) / value).toBeLessThan(1e-12);
  }
  const geographic = computePycnophylacticDetailedOracle({
    ...base,
    iterations: 6,
    cellAreas,
    zoneAnchorCells: fixture.anchors
  });
  expectMass(geographic.mass, PYCNOPHYLACTIC_NUMPY.geographic6.mass);
  expect(total(geographic.mass)).toBeCloseTo(507, 6);
  expect(applyZoneAnchors(fixture.zones, fixture.zoneCount, fixture.anchors)).toEqual(
    PYCNOPHYLACTIC_NUMPY.planarAnchored6.zones
  );
});

it('the areal oracle reproduces the numpy denominators, diagnostics and anchors', () => {
  const {fixture} = AREAL_NUMPY;
  const base = {
    sourceZones: fixture.sourceZones,
    targetZones: fixture.targetZones,
    sourceCount: fixture.sourceCount,
    targetCount: fixture.targetCount
  };
  const cases = [
    ['zoneZone', {denominator: 'zone' as const}],
    ['overlapOverlap', {denominator: 'overlap' as const}],
    ['default', {}],
    ['defaultAnchored', {sourceAnchorCells: fixture.anchors}],
    ['geographicDefault', {cellWeights: AREAL_NUMPY.cellAreas}]
  ] as const;
  for (const [name, options] of cases) {
    const expected = AREAL_NUMPY[name];
    const actual = computeArealOracle({...base, ...options});
    const label = `areal case ${name}`;
    expect(actual.neighbors, label).toEqual(expected.pairs.map(([, source]) => source));
    for (const key of ['areas', 'extensive', 'intensive', 'sourceAreas'] as const) {
      expect(actual[key], `${label} ${key}`).toHaveLength(
        expected[key === 'areas' ? 'areas' : key].length
      );
      for (const [index, value] of expected[key].entries()) {
        expect(
          Math.abs(actual[key][index] - value),
          `${label} ${key}[${index}]`
        ).toBeLessThanOrEqual(1e-9 * (1 + Math.abs(value)));
      }
    }
    for (const [index, value] of nanify(expected.transferred).entries()) {
      if (Number.isNaN(value)) {
        expect(actual.sourceTransferredFractions[index]).toBeNaN();
      } else {
        expect(actual.sourceTransferredFractions[index]).toBeCloseTo(value, 12);
      }
    }
    expect(actual.invalidCount, label).toBe(expected.invalidCount);
  }
  // The default denominators differ per kind: intensive weights of a partly covered target
  // renormalize to the covered area.
  const zoneZone = computeArealOracle({...base, denominator: 'zone'});
  const byDefault = computeArealOracle(base);
  expect(byDefault.extensive).toEqual(zoneZone.extensive);
  expect(byDefault.intensive).not.toEqual(zoneZone.intensive);
});

it('GPUArealInterpolation resolves spatial context, denominators and diagnostics', () => {
  const {graph, view} = setup();
  const props = {
    sourceZones: view('uint32', 48),
    targetZones: view('uint32', 48),
    sourceCount: 5,
    targetCount: 3,
    weights: {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 12),
      weights: view('float32', 12)
    },
    overflow: view('uint32', 1)
  };
  const planar = new GPUArealInterpolation(props);
  expect(planar.spatialContext.coordinateSpace).toBe('planar');
  expect(planar.extensiveDenominator).toBe('zone');
  expect(planar.intensiveDenominator).toBe('overlap');
  const scalar = new GPUArealInterpolation({...props, denominator: 'zone'});
  expect(scalar.intensiveDenominator).toBe('zone');
  const perKind = new GPUArealInterpolation({...props, denominator: {extensive: 'overlap'}});
  expect([perKind.extensiveDenominator, perKind.intensiveDenominator]).toEqual([
    'overlap',
    'overlap'
  ]);
  expect(() => new GPUArealInterpolation({...props, denominator: 'union' as never})).toThrow(
    /denominator/
  );

  const longitudeLatitude = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle'
  } as const;
  const extent = view('float32', 4);
  expect(() => new GPUArealInterpolation({...props, spatialContext: longitudeLatitude})).toThrow(
    /extent and width/
  );
  expect(() => new GPUArealInterpolation({...props, extent, width: 8})).toThrow(
    /longitude-latitude spatialContext/
  );
  expect(
    () => new GPUArealInterpolation({...props, spatialContext: longitudeLatitude, extent, width: 7})
  ).toThrow(/dividing the cell count/);
  expect(
    () =>
      new GPUArealInterpolation({
        ...props,
        spatialContext: {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'}
      })
  ).toThrow(/does not support discrete-grid/);

  const baseline = new GPUArealInterpolation(props).getCommandNodes(graph).length;
  const richGraph = setup();
  const richProps = {
    ...props,
    sourceZones: richGraph.view('uint32', 48),
    targetZones: richGraph.view('uint32', 48),
    weights: {
      offsets: richGraph.view('uint32', 4),
      neighbors: richGraph.view('uint32', 12),
      weights: richGraph.view('float32', 12)
    },
    overflow: richGraph.view('uint32', 1),
    sourceAnchorCells: richGraph.view('uint32', 5),
    sourceAreas: richGraph.view('float32', 5),
    sourceTransferredFractions: richGraph.view('float32', 5),
    invalidCount: richGraph.view('uint32', 1),
    spatialContext: longitudeLatitude,
    extent: richGraph.view('float32', 4),
    width: 8
  };
  expect(
    new GPUArealInterpolation(richProps).getCommandNodes(richGraph.graph).length
  ).toBeGreaterThan(baseline + 5);
  expect(
    () => new GPUArealInterpolation({...richProps, sourceAreas: richGraph.view('float32', 4)})
  ).toThrow(/sourceAreas must have sourceCount rows/);
  expect(
    () => new GPUArealInterpolation({...richProps, sourceAnchorCells: richGraph.view('uint32', 4)})
  ).toThrow(/sourceAnchorCells/);
});

it('GPUPycnophylactic resolves spatial context, tolerance, anchors and status', () => {
  const {graph, view} = setup();
  const props = {
    width: 8,
    height: 6,
    zones: view('uint32', 48),
    zoneCount: 6,
    totals: view('float32', 6),
    iterations: 4,
    output: view('float32', 48)
  };
  expect(new GPUPycnophylactic(props).spatialContext.coordinateSpace).toBe('planar');
  const longitudeLatitude = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle'
  } as const;
  const extent = view('float32', 4);
  expect(() => new GPUPycnophylactic({...props, spatialContext: longitudeLatitude})).toThrow(
    /extent and width/
  );
  expect(() => new GPUPycnophylactic({...props, extent})).toThrow(/longitude-latitude/);
  expect(() => new GPUPycnophylactic({...props, converged: view('uint32', 1)})).toThrow(
    /need a tolerance/
  );
  expect(() => new GPUPycnophylactic({...props, zoneAnchorCells: view('uint32', 5)})).toThrow(
    /zoneCount rows/
  );
  expect(() => new GPUPycnophylactic({...props, density: view('float32', 47)})).toThrow(/density/);

  const baseline = new GPUPycnophylactic(props).getCommandNodes(graph).length;
  const rich = setup();
  const richProps = {
    ...props,
    zones: rich.view('uint32', 48),
    totals: rich.view('float32', 6),
    output: rich.view('float32', 48),
    spatialContext: longitudeLatitude,
    extent: rich.view('float32', 4),
    tolerance: rich.view('float32', 1),
    converged: rich.view('uint32', 1),
    iterationLimitReached: rich.view('uint32', 1),
    invalidCount: rich.view('uint32', 1),
    zoneAnchorCells: rich.view('uint32', 6),
    density: rich.view('float32', 48)
  };
  expect(new GPUPycnophylactic(richProps).getCommandNodes(rich.graph).length).toBeGreaterThan(
    baseline * 1.5
  );
});
