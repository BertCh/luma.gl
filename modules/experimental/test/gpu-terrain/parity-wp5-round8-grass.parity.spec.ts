// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import pointLatticeReference from '../oracles/fixtures/parity-wp5-round10-point-horizon.json';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {GPUGeomorphons, getGPUGeomorphonsParameterValues} from '../../src/gpu-terrain/geomorphons';
import {
  GPUTerrainHorizon,
  getGPUTerrainHorizonParameterValues,
  GPUTerrainCastShadow,
  getGPUTerrainCastShadowParameterValues
} from '../../src/gpu-terrain/terrain-illumination';
import {
  GPUPointHorizonProfile,
  getGPUPointHorizonProfileParameterValues
} from '../../src/gpu-terrain/point-horizon';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {
  GEOMORPHON_FIXTURES,
  HORIZON_FIXTURES,
  SHADOW_FIXTURES,
  measureComparison,
  assertComparisons
} from './parity-wp5-round8-fixtures';

for (const fixture of GEOMORPHON_FIXTURES)
  for (const settings of fixture.cases) {
    it(`GPUGeomorphons GRASS ${fixture.input.name}/${settings.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {width, height, cellSize} = fixture.input;
      const count = width * height;
      const grid = makeGPURasterMetadata({width, height, cellSize: [cellSize, cellSize]});
      const graph = new GPUCommandGraph(device, {id: 'parity-wp5-r8-geomorphons'});
      const input = createInputBuffer(device, Float32Array.from(fixture.input.elevation));
      const forms = createOutputBuffer(device, count);
      const ternary = createOutputBuffer(device, count);
      const parameters = new GPUParameterBuffer(device, {
        id: 'settings',
        format: 'float32',
        length: 8,
        values: getGPUGeomorphonsParameterValues(settings, grid)
      });
      graph.add(
        new GPUGeomorphons({
          grid,
          elevation: {
            id: 'elevation',
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: importGraphBuffer(graph, 'elevation', input, 'float32', count)
            }
          },
          parameters: parameters.importToGraph(graph),
          searchRadius: settings.searchRadius,
          skipRadius: settings.skipRadius,
          comparison: settings.comparison,
          output: {
            forms: importGraphBuffer(graph, 'forms', forms, 'uint32', count),
            ternary: importGraphBuffer(graph, 'ternary', ternary, 'uint32', count)
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const errors = [];
        errors.push(
          measureComparison(
            'GPUGeomorphons',
            `${fixture.input.name}/${settings.name}`,
            'GRASS r.geomorphon unit grid + certified exact-ratio ties (composite adapter)',
            'forms',
            await readUint32(forms, count),
            settings.forms
          )
        );
        errors.push(
          measureComparison(
            'GPUGeomorphons',
            `${fixture.input.name}/${settings.name}`,
            'GRASS r.geomorphon unit grid + certified exact-ratio ties (composite adapter)',
            'ternary',
            await readUint32(ternary, count),
            settings.ternary
          )
        );
        assertComparisons(errors);
      } finally {
        compiled.destroy();
        parameters.destroy();
        input.destroy();
        forms.destroy();
        ternary.destroy();
      }
    });
  }
for (const fixture of HORIZON_FIXTURES)
  for (const settings of fixture.cases)
    for (const algorithm of ['march', 'sweep'] as const) {
      it(`GPUTerrainHorizon GRASS ${settings.name}/${algorithm}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const {width, height, cellSize} = fixture.input;
        const count = width * height;
        const grid = makeGPURasterMetadata({width, height, cellSize: [cellSize, cellSize]});
        const graph = new GPUCommandGraph(device, {id: 'parity-wp5-r8-horizon'});
        const input = createInputBuffer(device, Float32Array.from(fixture.input.elevation));
        const horizon = createOutputBuffer(device, count * 4);
        const skyViewFactor = createOutputBuffer(device, count);
        const parameters = new GPUParameterBuffer(device, {
          id: 'settings',
          format: 'float32',
          length: 8,
          values: getGPUTerrainHorizonParameterValues(settings, grid)
        });
        graph.add(
          new GPUTerrainHorizon({
            grid,
            elevation: {
              id: 'elevation',
              format: 'float32',
              storage: {
                kind: 'buffer',
                values: importGraphBuffer(graph, 'elevation', input, 'float32', count)
              }
            },
            parameters: parameters.importToGraph(graph),
            directionCount: 4,
            maximumRadius: settings.maximumRadius,
            algorithm,
            output: {
              horizon: importGraphBuffer(graph, 'horizon', horizon, 'float32', count * 4),
              skyViewFactor: importGraphBuffer(graph, 'skyview', skyViewFactor, 'float32', count)
            }
          })
        );
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const actual = (await readFloat32(horizon, count * 4)).map(value => Math.max(0, value));
          assertComparisons([
            measureComparison(
              'GPUTerrainHorizon',
              `${settings.name}/${algorithm}`,
              'GRASS r.horizon',
              'nonnegativeHorizonDegrees',
              actual,
              settings.horizon,
              'D'
            ),
            measureComparison(
              'GPUTerrainHorizon',
              `${settings.name}/${algorithm}`,
              'GRASS r.skyview',
              'skyViewFactor',
              await readFloat32(skyViewFactor, count),
              settings.skyViewFactor,
              'R'
            )
          ]);
        } finally {
          compiled.destroy();
          parameters.destroy();
          input.destroy();
          horizon.destroy();
          skyViewFactor.destroy();
        }
      });
    }
for (const fixture of HORIZON_FIXTURES)
  for (const settings of fixture.cases)
    for (const traversal of ['march', 'pyramid'] as const) {
      it(`GPUPointHorizonProfile SciPy lattice / GRASS audit ${settings.name}/${traversal}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const {width, height, cellSize} = fixture.input;
        const count = width * height;
        const grid = makeGPURasterMetadata({width, height, cellSize: [cellSize, cellSize]});
        const graph = new GPUCommandGraph(device, {id: 'parity-wp5-r8-point'});
        const input = createInputBuffer(device, Float32Array.from(fixture.input.elevation));
        const angles = createOutputBuffer(device, 4);
        const distances = createOutputBuffer(device, 4);
        const observers = createInputBuffer(device, Float32Array.of(16, 16, 0, 0));
        const parameters = new GPUParameterBuffer(device, {
          id: 'settings',
          format: 'float32',
          length: 8,
          values: getGPUPointHorizonProfileParameterValues(settings, grid)
        });
        graph.add(
          new GPUPointHorizonProfile({
            grid,
            elevation: {
              id: 'elevation',
              format: 'float32',
              storage: {
                kind: 'buffer',
                values: importGraphBuffer(graph, 'elevation', input, 'float32', count)
              }
            },
            parameters: parameters.importToGraph(graph),
            observers: importGraphBuffer(graph, 'observers', observers, 'float32x4', 1),
            azimuthCount: 4,
            minimumDistance: cellSize,
            maximumDistance: cellSize * settings.maximumRadius,
            cellSize,
            traversal,
            output: {
              skylineAngle: importGraphBuffer(graph, 'angles', angles, 'float32', 4),
              distance: importGraphBuffer(graph, 'distance', distances, 'float32', 4)
            }
          })
        );
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const actual = await readFloat32(angles, 4);
          const ordered = [...settings.point[0].horizons].sort(
            (first, second) => first.azimuth - second.azimuth
          );
          // GRASS does not expose the distance of negative horizons; project those to zero.
          const matchedReference = pointLatticeReference.cases.find(
            value => value.name === settings.name
          )!;
          const actualDistances = (await readFloat32(distances, 4)).map((value, index) =>
            actual[index] > 0 ? value : 0
          );
          // Keep the external GRASS diagnostic; its cell-centre sampler is a different model.
          measureComparison(
            'GPUPointHorizonProfile',
            `${settings.name}/${traversal}`,
            'GRASS r.horizon point JSON',
            'positiveHorizonDistanceMeters',
            actualDistances,
            ordered.map(value => value.distance),
            'D',
            {absolute: 1e-3, relative: 2e-6}
          );
          measureComparison(
            'GPUPointHorizonProfile',
            `${settings.name}/${traversal}`,
            'GRASS r.horizon point JSON',
            'nonnegativeHorizonDegrees',
            actual.map(value => Math.max(0, value)),
            ordered.map(value => value.angle),
            'D'
          );
          const errors = [
            measureComparison(
              'GPUPointHorizonProfile',
              `${settings.name}/${traversal}`,
              'composite SciPy interp1d on the public dyadic sample set',
              'nonnegativeHorizonDegrees',
              actual.map(value => Math.max(0, value)),
              matchedReference.horizon,
              'D'
            )
          ];
          // D: distances < 300 m; 8 ULP32 remains below the 1 mm floor.
          errors.push(
            measureComparison(
              'GPUPointHorizonProfile',
              `${settings.name}/${traversal}`,
              'composite SciPy interp1d on the public dyadic sample set',
              'positiveHorizonDistanceMeters',
              actualDistances,
              matchedReference.distance,
              'D',
              {absolute: 1e-3, relative: 2e-6}
            )
          );
          assertComparisons(errors);
        } finally {
          compiled.destroy();
          parameters.destroy();
          for (const buffer of [input, angles, distances, observers]) buffer.destroy();
        }
      });
    }
for (const fixture of SHADOW_FIXTURES)
  for (const settings of fixture.cases) {
    it(`GPUTerrainCastShadow GRASS ${settings.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {width, height, cellSize} = fixture.input;
      const count = width * height;
      const grid = makeGPURasterMetadata({width, height, cellSize: [cellSize, cellSize]});
      const graph = new GPUCommandGraph(device, {id: 'parity-wp5-r8-shadow'});
      const input = createInputBuffer(device, Float32Array.from(fixture.input.elevation));
      const visibility = createOutputBuffer(device, count);
      const parameters = new GPUParameterBuffer(device, {
        id: 'settings',
        format: 'float32',
        length: 16,
        values: getGPUTerrainCastShadowParameterValues({...settings, angularRadiusDegrees: 0}, grid)
      });
      graph.add(
        new GPUTerrainCastShadow({
          grid,
          elevation: {
            id: 'elevation',
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: importGraphBuffer(graph, 'elevation', input, 'float32', count)
            }
          },
          parameters: parameters.importToGraph(graph),
          output: {
            sunVisibility: importGraphBuffer(graph, 'visibility', visibility, 'float32', count)
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        assertComparisons([
          measureComparison(
            'GPUTerrainCastShadow',
            settings.name,
            'GRASS r.sunmask -z',
            'hardSunVisibility',
            await readFloat32(visibility, count),
            settings.sunVisibility
          )
        ]);
      } finally {
        compiled.destroy();
        parameters.destroy();
        input.destroy();
        visibility.destroy();
      }
    });
  }

// The addon was installed to an isolated /private/tmp prefix; its exact source SHA is in fixtures.
import alpsRuggedness from '../oracles/fixtures/parity-wp5-round8-alps-dem-ruggedness-addon.json';
import canyonRuggedness from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-ruggedness-addon.json';
import ridgeRuggedness from '../oracles/fixtures/parity-wp5-round8-ridge-ruggedness-addon.json';
import {decodeOracleFixture} from '../oracles/oracle-fixture';
import type {Fixture} from './parity-wp5-round8-fixtures';
import {
  GPUTerrainVectorRuggedness,
  getGPUTerrainVectorRuggednessParameterValues
} from '../../src/gpu-terrain/topographic-position';
type AddonCase = {name: string; radius: number; expected: number[]; validity: number[]};
for (const rawFixture of [alpsRuggedness, canyonRuggedness, ridgeRuggedness]) {
  const fixture = decodeOracleFixture<Fixture<AddonCase>>(rawFixture);
  for (const settings of fixture.cases)
    it(`GPUTerrainVectorRuggedness GRASS addon ${settings.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {width, height, cellSize} = fixture.input;
      const count = width * height;
      const grid = makeGPURasterMetadata({width, height, cellSize});
      const graph = new GPUCommandGraph(device, {id: 'parity-wp5-r8-addon'});
      const input = createInputBuffer(device, Float32Array.from(fixture.input.elevation));
      const output = createOutputBuffer(device, count);
      const validity = createOutputBuffer(device, count);
      const parameters = new GPUParameterBuffer(device, {
        id: 'settings',
        format: 'float32',
        length: 8,
        values: getGPUTerrainVectorRuggednessParameterValues({radius: settings.radius}, grid)
      });
      graph.add(
        new GPUTerrainVectorRuggedness({
          grid,
          elevation: {
            id: 'elevation',
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: importGraphBuffer(graph, 'elevation', input, 'float32', count)
            }
          },
          parameters: parameters.importToGraph(graph),
          maximumRadius: settings.radius,
          borderMode: 'nodata',
          output: {
            vectorRuggedness: importGraphBuffer(graph, 'output', output, 'float32', count),
            outputValidity: importGraphBuffer(graph, 'validity', validity, 'uint32', count)
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        assertComparisons([
          measureComparison(
            'GPUTerrainVectorRuggedness',
            settings.name,
            'GRASS r.vector.ruggedness exponent=0',
            'vectorRuggedness',
            await readFloat32(output, count),
            settings.expected,
            'R'
          ),
          measureComparison(
            'GPUTerrainVectorRuggedness',
            settings.name,
            'GRASS r.slope.aspect valid normal centres',
            'validity',
            await readUint32(validity, count),
            settings.validity
          )
        ]);
      } finally {
        compiled.destroy();
        parameters.destroy();
        input.destroy();
        output.destroy();
        validity.destroy();
      }
    });
}
