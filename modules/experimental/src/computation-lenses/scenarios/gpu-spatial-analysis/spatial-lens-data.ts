// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Compact on-disk Chicago geometry; deltas restart from the origin for each ring. */
export type ChicagoTractsJSON = {
  schema: string;
  source: string;
  licence: string;
  crs: string;
  count: number;
  quantum: number;
  origin: number[];
  ringOffsets: number[];
  polygonRingOffsets: number[];
  partFeature: number[];
  deltas: number[];
  columns: Record<string, (number | null)[]>;
};

/** Showcase-compatible typed arrays, in source GEOID order. Missing attributes remain NaN. */
export type ChicagoTracts = {
  count: number;
  vertices: Float32Array;
  ringOffsets: Uint32Array;
  polygonRingOffsets: Uint32Array;
  partFeature: Uint32Array;
  columns: Record<string, Float32Array>;
  dataset: {id: string; source: string; licence: string};
};

/** Decodes integer coordinates before converting to f32, preserving bit-equal shared vertices. */
export function decodeChicagoTracts(json: ChicagoTractsJSON): ChicagoTracts {
  const vertices = new Float32Array(json.deltas.length);
  for (let ring = 0; ring < json.ringOffsets.length - 1; ring++) {
    let longitude = 0;
    let latitude = 0;
    for (let vertex = json.ringOffsets[ring]; vertex < json.ringOffsets[ring + 1]; vertex++) {
      longitude += json.deltas[vertex * 2];
      latitude += json.deltas[vertex * 2 + 1];
      vertices[vertex * 2] = json.origin[0] + longitude * json.quantum;
      vertices[vertex * 2 + 1] = json.origin[1] + latitude * json.quantum;
    }
  }
  return {
    count: json.count,
    vertices,
    ringOffsets: new Uint32Array(json.ringOffsets),
    polygonRingOffsets: new Uint32Array(json.polygonRingOffsets),
    partFeature: new Uint32Array(json.partFeature),
    columns: Object.fromEntries(
      Object.entries(json.columns).map(([name, values]) => [
        name,
        Float32Array.from(values, value => value ?? NaN)
      ])
    ),
    dataset: {id: json.schema, source: json.source, licence: json.licence}
  };
}

/** Loads the docs-sized dataset only when a live scenario is activated. */
export async function loadChicagoTracts(): Promise<ChicagoTracts> {
  return decodeChicagoTracts((await import('../../data/s1-chicago-tracts.json')).default);
}
