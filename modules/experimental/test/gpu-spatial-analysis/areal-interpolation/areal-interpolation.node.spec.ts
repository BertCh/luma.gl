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
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid/index';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
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
    grid: makeGPURasterMetadata({width: 8, height: 8}),
    sourceZones: view('uint32', 64),
    targetZones: view('uint32', 64),
    sourceZoneCount: 4,
    targetZoneCount: 3,
    output: {
      weights: {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      status: {overflow: view('uint32', 1)}
    }
  };
  const nodes = new GPUArealInterpolation(props).getCommandNodes(graph);
  expect(nodes.length).toBeGreaterThan(10);
  expect(() => new GPUArealInterpolation({...props, targetZoneCount: 4})).toThrow(
    /targetZoneCount \+ 1/
  );
  expect(() => new GPUArealInterpolation({...props, targetZones: view('uint32', 63)})).toThrow(
    /one row per grid cell/
  );
  expect(
    () =>
      new GPUArealInterpolation({
        ...props,
        output: {...props.output, alternateWeights: view('float32', 7)}
      })
  ).toThrow(/alternateWeights length/);
  expect(
    () =>
      new GPUArealInterpolation({
        ...props,
        output: {...props.output, categoryShares: view('float32', 6)}
      })
  ).toThrow(/categoryShares needs categories/);
  expect(
    () =>
      new GPUArealInterpolation({
        ...props,
        categories: {sourceCategories: view('uint32', 4), categoryCount: 2}
      })
  ).toThrow(/needs output.categoryShares/);
});

it('GPUPycnophylactic declares nodes per iteration and validates its props', () => {
  const {graph, view} = setup();
  const props = {
    grid: makeGPURasterMetadata({width: 8, height: 4}),
    zones: view('uint32', 32),
    zoneCount: 3,
    totals: view('float32', 3),
    maximumIterations: 2,
    output: {mass: view('float32', 32)}
  };
  const few = new GPUPycnophylactic(props).getCommandNodes(graph).length;
  const {graph: other, view: otherView} = setup();
  const many = new GPUPycnophylactic({
    ...props,
    zones: otherView('uint32', 32),
    totals: otherView('float32', 3),
    output: {mass: otherView('float32', 32)},
    maximumIterations: 4
  }).getCommandNodes(other).length;
  expect(many).toBeGreaterThan(few);
  expect(() => new GPUPycnophylactic({...props, maximumIterations: -1})).toThrow(
    /maximumIterations/
  );
  expect(() => new GPUPycnophylactic({...props, totals: view('float32', 2)})).toThrow(/totals/);
});

it('GPUSpatialLag accepts cross weights with the source count derived from values', () => {
  const {graph, view} = setup();
  const weights = {
    offsets: view('uint32', 4),
    neighbors: view('uint32', 8),
    weights: view('float32', 8)
  };
  const props = {weights, values: view('float32', 10), output: view('float32', 6)};
  // The load-balanced lag is a walk and a fixup node.
  expect(new GPUSpatialLag({...props, columnCount: 2}).getCommandNodes(graph)).toHaveLength(2);
  expect(() => new GPUSpatialLag(props)).toThrow(/rows \* columnCount/);
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
    grid: makeGPURasterMetadata({width: 8, height: 6}),
    sourceZones: view('uint32', 48),
    targetZones: view('uint32', 48),
    sourceZoneCount: 5,
    targetZoneCount: 3,
    output: {
      weights: {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 12),
        weights: view('float32', 12)
      },
      status: {overflow: view('uint32', 1)}
    }
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
  const parameters = view('float32', 4);
  expect(() => new GPUArealInterpolation({...props, spatialContext: longitudeLatitude})).toThrow(
    /need parameters/
  );
  expect(() => new GPUArealInterpolation({...props, parameters})).toThrow(
    /longitude-latitude spatialContext/
  );
  expect(
    () =>
      new GPUArealInterpolation({
        ...props,
        spatialContext: longitudeLatitude,
        parameters: view('float32', 3)
      })
  ).toThrow(/at least 4 values/);
  expect(
    () =>
      new GPUArealInterpolation({
        ...props,
        grid: makeGPURasterMetadata({width: 7, height: 7})
      })
  ).toThrow(/one row per grid cell/);
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
    output: {
      weights: {
        offsets: richGraph.view('uint32', 4),
        neighbors: richGraph.view('uint32', 12),
        weights: richGraph.view('float32', 12)
      },
      sourceAreas: richGraph.view('float32', 5),
      sourceTransferredFractions: richGraph.view('float32', 5),
      status: {overflow: richGraph.view('uint32', 1), invalidCount: richGraph.view('uint32', 1)}
    },
    sourceAnchorCells: richGraph.view('uint32', 5),
    spatialContext: longitudeLatitude,
    parameters: richGraph.view('float32', 4)
  };
  expect(
    new GPUArealInterpolation(richProps).getCommandNodes(richGraph.graph).length
  ).toBeGreaterThan(baseline + 5);
  expect(
    () =>
      new GPUArealInterpolation({
        ...richProps,
        output: {...richProps.output, sourceAreas: richGraph.view('float32', 4)}
      })
  ).toThrow(/sourceAreas must have sourceZoneCount rows/);
  expect(
    () => new GPUArealInterpolation({...richProps, sourceAnchorCells: richGraph.view('uint32', 4)})
  ).toThrow(/sourceAnchorCells/);
});

it('GPUPycnophylactic resolves spatial context, tolerance, anchors and status', () => {
  const {graph, view} = setup();
  const props = {
    grid: makeGPURasterMetadata({width: 8, height: 6}),
    zones: view('uint32', 48),
    zoneCount: 6,
    totals: view('float32', 6),
    maximumIterations: 4,
    output: {mass: view('float32', 48)}
  };
  expect(new GPUPycnophylactic(props).spatialContext.coordinateSpace).toBe('planar');
  const longitudeLatitude = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle'
  } as const;
  const parameters = view('float32', 4);
  expect(() => new GPUPycnophylactic({...props, spatialContext: longitudeLatitude})).toThrow(
    /need parameters/
  );
  expect(() => new GPUPycnophylactic({...props, parameters})).toThrow(/longitude-latitude/);
  expect(
    () =>
      new GPUPycnophylactic({
        ...props,
        output: {...props.output, status: {converged: view('uint32', 1)}}
      })
  ).toThrow(/need a tolerance/);
  expect(() => new GPUPycnophylactic({...props, zoneAnchorCells: view('uint32', 5)})).toThrow(
    /zoneCount rows/
  );
  expect(
    () => new GPUPycnophylactic({...props, output: {...props.output, density: view('float32', 47)}})
  ).toThrow(/density/);

  const baseline = new GPUPycnophylactic(props).getCommandNodes(graph).length;
  const rich = setup();
  const richProps = {
    ...props,
    zones: rich.view('uint32', 48),
    totals: rich.view('float32', 6),
    output: {
      mass: rich.view('float32', 48),
      density: rich.view('float32', 48),
      status: {
        converged: rich.view('uint32', 1),
        iterationCount: rich.view('uint32', 1),
        invalidCount: rich.view('uint32', 1)
      }
    },
    spatialContext: longitudeLatitude,
    parameters: rich.view('float32', 4),
    tolerance: rich.view('float32', 1),
    zoneAnchorCells: rich.view('uint32', 6)
  };
  expect(new GPUPycnophylactic(richProps).getCommandNodes(rich.graph).length).toBeGreaterThan(
    baseline * 1.5
  );
});

it('GPUPycnophylactic gates converged iterations (rule 13), copies nothing forward and keeps one pipeline per kernel (rule 12)', () => {
  const build = (iterations: number) => {
    const {graph, view} = setup();
    return new GPUPycnophylactic({
      id: 'pycno',
      grid: makeGPURasterMetadata({width: 8, height: 4}),
      zones: view('uint32', 32),
      zoneCount: 3,
      totals: view('float32', 3),
      maximumIterations: iterations,
      tolerance: view('float32', 1),
      output: {
        mass: view('float32', 32),
        status: {converged: view('uint32', 1), iterationCount: view('uint32', 1)}
      }
    }).getCommandNodes(graph) as readonly {id: string; condition?: {mode: string}}[];
  };
  expect(getGPUCommandNodePipelineKeys(build(4)).size).toBe(
    getGPUCommandNodePipelineKeys(build(3)).size
  );
  const nodes = build(3);
  for (const stage of ['smooth', 'restore-apply', 'scale-apply', 'change']) {
    expect(
      nodes.find(node => node.id === `pycno-iteration-1-${stage}`)?.condition?.mode,
      stage
    ).toBe('indirect');
  }
  expect(nodes.find(node => node.id === 'pycno-iteration-1-converged')?.condition).toBeUndefined();
  const ids = nodes.map(node => node.id);
  expect(ids.indexOf('pycno-settle')).toBeGreaterThan(ids.indexOf('pycno-iteration-2-converged'));
  for (const {id, source} of getGPUCommandNodeKernelSources(nodes)) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    // The converged-round copy of the whole surface is gone.
    expect(source, id).not.toMatch(
      /destination\[destinationOffset \+ index\] = source\[sourceOffset \+ index\]/
    );
  }
});

it('longitude/latitude cell areas mirror stored rows for a north-up placement and parse as WGSL', () => {
  const longitudeLatitude = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle'
  } as const;
  const arealSetup = setup();
  const areal = new GPUArealInterpolation({
    id: 'mirror-areal',
    grid: makeGPURasterMetadata({width: 8, height: 6}),
    sourceZones: arealSetup.view('uint32', 48),
    targetZones: arealSetup.view('uint32', 48),
    sourceZoneCount: 2,
    targetZoneCount: 2,
    spatialContext: longitudeLatitude,
    parameters: arealSetup.view('float32', 4),
    output: {
      weights: {
        offsets: arealSetup.view('uint32', 3),
        neighbors: arealSetup.view('uint32', 8),
        weights: arealSetup.view('float32', 8)
      },
      status: {overflow: arealSetup.view('uint32', 1)}
    }
  });
  const pycnophylacticSetup = setup();
  const pycnophylactic = new GPUPycnophylactic({
    id: 'mirror-pycno',
    grid: makeGPURasterMetadata({width: 8, height: 6}),
    zones: pycnophylacticSetup.view('uint32', 48),
    zoneCount: 2,
    totals: pycnophylacticSetup.view('float32', 2),
    maximumIterations: 1,
    spatialContext: longitudeLatitude,
    parameters: pycnophylacticSetup.view('float32', 4),
    output: {mass: pycnophylacticSetup.view('float32', 48)}
  });
  for (const [nodes, nodeId, functionName] of [
    [areal.getCommandNodes(arealSetup.graph), 'mirror-areal-area-weights', 'getPlacedCellArea'],
    [
      pycnophylactic.getCommandNodes(pycnophylacticSetup.graph),
      'mirror-pycno-cell-areas',
      'getPlacedCellArea'
    ]
  ] as const) {
    const kernel = getGPUCommandNodeKernelSources(nodes).find(({id}) => id === nodeId);
    expect(kernel, nodeId).toBeDefined();
    const source = kernel!.source;
    expect(() => new WgslReflect(source), nodeId).not.toThrow();
    // Stored rows are mirrored for a negative cell height and the area uses its magnitude.
    expect(source, nodeId).toContain('select(row, HEIGHT - 1u - row, signedCellHeight < 0.0)');
    expect(source, nodeId).toContain('abs(signedCellHeight)');
    expect(source, nodeId).toContain(`${functionName}(index / WIDTH)`);
  }
});
