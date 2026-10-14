// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  createGPULocalFrame,
  type GPULocalFrameKind
} from '../../../src/analysis-core/contracts/local-frame';
import {getGPUSpatialContextCompatibility} from '../../../src/analysis-core/contracts/spatial-context';

// Oracle: <scratchpad>/fix/work/E/local-frame/oracle.py, pyproj 3.8.0 proj 9.8.1.
// azimuthal-equidistant: +proj=aeqd +R=6371008.8; lambert-equal-area: +proj=laea +ellps=WGS84.
const PYPROJ_CASES: {
  kind: GPULocalFrameKind;
  origin: [number, number];
  lngLat: [number, number];
  xy: [number, number];
}[] = [
  {
    kind: 'azimuthal-equidistant',
    origin: [8.5417, 47.3769],
    lngLat: [8.5517, 47.3969],
    xy: [752.6970749988335, 2223.949942610373]
  },
  {
    kind: 'azimuthal-equidistant',
    origin: [8.5417, 47.3769],
    lngLat: [7.241700000000001, 48.0769],
    xy: [-96578.83862835511, 78645.81976533646]
  },
  {
    kind: 'azimuthal-equidistant',
    origin: [8.5417, 47.3769],
    lngLat: [11.0417, 44.2769],
    xy: [199096.644419743, -341561.1731910821]
  },
  {
    kind: 'azimuthal-equidistant',
    origin: [8.5417, 47.3769],
    lngLat: [-21.4583, 72.3769],
    xy: [-1005365.6354857358, 3005025.8945158725]
  },
  {
    kind: 'lambert-equal-area',
    origin: [8.5417, 47.3769],
    lngLat: [8.5517, 47.3969],
    xy: [754.9097590130026, 2223.6155855374122]
  },
  {
    kind: 'lambert-equal-area',
    origin: [8.5417, 47.3769],
    lngLat: [7.241700000000001, 48.0769],
    xy: [-96866.3472921666, 78636.91697579775]
  },
  {
    kind: 'lambert-equal-area',
    origin: [8.5417, 47.3769],
    lngLat: [11.0417, 44.2769],
    xy: [199601.38449994507, -341372.05761121504]
  },
  {
    kind: 'lambert-equal-area',
    origin: [8.5417, 47.3769],
    lngLat: [-21.4583, 72.3769],
    xy: [-999754.9980393504, 2979049.3125485852]
  },
  {
    kind: 'azimuthal-equidistant',
    origin: [151.2093, -33.8688],
    lngLat: [151.2193, -33.8488],
    xy: [923.4866778309865, 2223.856700534863]
  },
  {
    kind: 'azimuthal-equidistant',
    origin: [151.2093, -33.8688],
    lngLat: [149.9093, -33.1688],
    xy: [-121000.22531142876, 77076.16691043862]
  },
  {
    kind: 'azimuthal-equidistant',
    origin: [151.2093, -33.8688],
    lngLat: [153.7093, -36.9688],
    xy: [222186.61028842014, -347479.19437867665]
  },
  {
    kind: 'azimuthal-equidistant',
    origin: [151.2093, -33.8688],
    lngLat: [121.20930000000001, -8.8688],
    xy: [-3380092.76853738, 2386812.8388891295]
  },
  {
    kind: 'lambert-equal-area',
    origin: [151.2093, -33.8688],
    lngLat: [151.2193, -33.8488],
    xy: [925.4811694227664, 2218.3523576391412]
  },
  {
    kind: 'lambert-equal-area',
    origin: [151.2093, -33.8688],
    lngLat: [149.9093, -33.1688],
    xy: [-121253.07765238198, 76880.01715449519]
  },
  {
    kind: 'lambert-equal-area',
    origin: [151.2093, -33.8688],
    lngLat: [153.7093, -36.9688],
    xy: [222678.6387427054, -346638.11255372164]
  },
  {
    kind: 'lambert-equal-area',
    origin: [151.2093, -33.8688],
    lngLat: [121.20930000000001, -8.8688],
    xy: [-3323522.9658019124, 2336010.479729497]
  },
  {
    kind: 'lambert-equal-area',
    origin: [0, 90],
    lngLat: [45, 80],
    xy: [788713.3037299615, -788713.3037299616]
  },
  {
    kind: 'lambert-equal-area',
    origin: [0, 90],
    lngLat: [-120, 70],
    xy: [-1924023.4274200066, 1110835.4437480867]
  },
  {
    kind: 'azimuthal-equidistant',
    origin: [179.5, -17],
    lngLat: [-179.2, -16.4],
    xy: [138673.84459493487, 66262.31973282012]
  }
];

describe('createGPULocalFrame', () => {
  it('matches pyproj aeqd and laea to a millimeter', () => {
    for (const {kind, origin, lngLat, xy} of PYPROJ_CASES) {
      const frame = createGPULocalFrame({origin, kind});
      const [x, y] = frame.project(lngLat[0], lngLat[1]);
      expect(Math.abs(x - xy[0]), `${kind} ${origin} -> ${lngLat} x`).toBeLessThan(1e-3);
      expect(Math.abs(y - xy[1]), `${kind} ${origin} -> ${lngLat} y`).toBeLessThan(1e-3);
      const [longitude, latitude] = frame.unproject(xy[0], xy[1]);
      const wrappedLongitudeError = ((longitude - lngLat[0] + 540) % 360) - 180;
      expect(Math.abs(wrappedLongitudeError)).toBeLessThan(1e-9);
      expect(Math.abs(latitude - lngLat[1])).toBeLessThan(1e-9);
    }
  });

  it('maps the origin to zero and keeps sub-millimeter float32 offsets near it', () => {
    const frame = createGPULocalFrame({origin: [151.2093, -33.8688]});
    expect(frame.project(151.2093, -33.8688)).toEqual([0, 0]);
    const projected = frame.projectArray([151.2093001, -33.8688, 0, 151.2093, -33.8687999, 1], 3);
    expect(projected.length).toBe(4);
    // pyproj aeqd +R=6371008.8 (oracle.py, near-origin line): 0.009232703869616882 m.
    expect(Math.abs(projected[0] - 0.009232703869616882)).toBeLessThan(1e-9);
    expect(Math.abs(projected[1])).toBeLessThan(1e-6);
  });

  it('carries a frame identity that makes different origins incompatible', () => {
    const first = createGPULocalFrame({origin: [8.5, 47.4]});
    const same = createGPULocalFrame({origin: [8.5, 47.4]});
    const other = createGPULocalFrame({origin: [8.6, 47.4]});
    const equalArea = createGPULocalFrame({origin: [8.5, 47.4], kind: 'lambert-equal-area'});
    expect(first.spatialContext.coordinateSpace).toBe('planar');
    expect(first.spatialContext.frame?.kind).toBe('local-azimuthal-equidistant');
    expect(getGPUSpatialContextCompatibility(first.spatialContext, same.spatialContext)).toEqual({
      compatible: true
    });
    expect(
      getGPUSpatialContextCompatibility(first.spatialContext, other.spatialContext).compatible
    ).toBe(false);
    expect(
      getGPUSpatialContextCompatibility(first.spatialContext, equalArea.spatialContext).compatible
    ).toBe(false);
  });

  it('returns NaN at the antipode and rejects invalid origins', () => {
    const frame = createGPULocalFrame({origin: [10, 20]});
    const [x, y] = frame.project(-170, -20);
    expect(Number.isNaN(x) && Number.isNaN(y)).toBe(true);
    expect(() => createGPULocalFrame({origin: [0, 91]})).toThrow();
  });
});

describe('createGPULocalFrame horizontal datum', () => {
  it('copies the declared datum without changing the frame identity', () => {
    const declared = createGPULocalFrame({origin: [10, 50], datum: 'EPSG:4258'});
    const undeclared = createGPULocalFrame({origin: [10, 50]});
    expect(declared.spatialContext.datum).toBe('EPSG:4258');
    expect(declared.spatialContext.frame?.id).toBe(undeclared.spatialContext.frame?.id);
    expect('datum' in undeclared.spatialContext).toBe(false);
  });

  it.each(['', '   '])('rejects an empty datum %j', datum => {
    expect(() => createGPULocalFrame({origin: [10, 50], datum})).toThrow(
      /datum must be a non-empty string/
    );
  });
});
