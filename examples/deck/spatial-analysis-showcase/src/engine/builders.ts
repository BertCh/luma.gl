// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {LocalMetricProjection} from './projection';

/** Directed road network in COO form plus drawable polyline segments. All coordinates are meters. */
export type RoadNetwork = {
  /** `x, y` meters per node. Nodes are de-duplicated polyline vertices. */
  nodePositions: Float32Array;
  /** Directed edge sources. Every street segment produces both directions. */
  edgeSources: Uint32Array;
  /** Directed edge targets aligned with `edgeSources`. */
  edgeTargets: Uint32Array;
  /** Edge length in meters aligned with `edgeSources`. */
  edgeLengths: Float32Array;
  /** 1 when vehicles may use the edge (it does not run against a one-way street), otherwise 0. */
  edgeDrivable: Uint32Array;
  /** Undirected drawable segments `x0, y0, x1, y1` in meters (one per polyline segment). */
  segments: Float32Array;
  /** Road class per segment: 0 motorway/trunk, 1 primary/secondary, 2 tertiary, 3 other. */
  segmentClasses: Uint32Array;
  /** First endpoint node of each segment. */
  segmentNodes: Uint32Array;
};

/** Polygons in GeoArrow-style offsets, in meters. */
export type PolygonSet = {
  /** Flattened ring vertices `x, y` in meters. Rings close implicitly. */
  polygonPositions: Float32Array;
  /** `featureCount + 1` feature-to-polygon offsets. */
  featureOffsets: Uint32Array;
  /** `polygonCount + 1` polygon-to-ring offsets. */
  polygonOffsets: Uint32Array;
  /** `ringCount + 1` ring-to-vertex offsets. */
  ringOffsets: Uint32Array;
  /** Stable feature IDs. */
  featureIds: Uint32Array;
  /** Per-feature label. */
  featureNames: readonly string[];
  /** Ring outline segments `x0, y0, x1, y1` for drawing. */
  outlineSegments: Float32Array;
  /** Feature row per outline segment. */
  outlineFeatureRows: Uint32Array;
};

/** One elevation raster in meters. */
export type ElevationRaster = {
  width: number;
  height: number;
  /** Elevation meters, row-major, row 0 at the north edge. */
  elevation: Float32Array;
  /** `[minX, minY, maxX, maxY]` meters of the raster's outer cell edges. */
  bounds: readonly [number, number, number, number];
  /** `[cellWidth, cellHeight]` meters. */
  cellSize: readonly [number, number];
  /** `[minimum, maximum]` elevation. */
  elevationRange: readonly [number, number];
};

/** Maps an OSM-like feature class string to a road class 0..3. */
export function getRoadClass(featureClass: string): number {
  if (/motorway|trunk/.test(featureClass)) return 0;
  if (/primary|secondary/.test(featureClass)) return 1;
  if (/tertiary/.test(featureClass)) return 2;
  return 3;
}

/** De-duplicates polyline vertices into nodes and emits directed edges and drawable segments. */
export class RoadNetworkBuilder {
  private readonly nodeIndexes = new Map<string, number>();
  private readonly nodePositions: number[] = [];
  private readonly edgeSources: number[] = [];
  private readonly edgeTargets: number[] = [];
  private readonly edgeLengths: number[] = [];
  private readonly edgeDrivable: number[] = [];
  private readonly segments: number[] = [];
  private readonly segmentClasses: number[] = [];
  private readonly segmentNodes: number[] = [];

  addPolyline(points: {key: string; x: number; y: number}[], roadClass: number, oneway: string) {
    const nodes = points.map(point => this.getNode(point.key, point.x, point.y));
    for (let index = 0; index + 1 < nodes.length; index++) {
      const from = nodes[index];
      const to = nodes[index + 1];
      if (from === to) {
        continue;
      }
      const x0 = this.nodePositions[from * 2];
      const y0 = this.nodePositions[from * 2 + 1];
      const x1 = this.nodePositions[to * 2];
      const y1 = this.nodePositions[to * 2 + 1];
      const length = Math.hypot(x1 - x0, y1 - y0);
      // OSM `oneway`: B both directions, F forward only, T backward only.
      this.addEdge(from, to, length, oneway !== 'T');
      this.addEdge(to, from, length, oneway !== 'F');
      this.segments.push(x0, y0, x1, y1);
      this.segmentClasses.push(roadClass);
      this.segmentNodes.push(from);
    }
  }

  finish(): RoadNetwork {
    return {
      nodePositions: Float32Array.from(this.nodePositions),
      edgeSources: Uint32Array.from(this.edgeSources),
      edgeTargets: Uint32Array.from(this.edgeTargets),
      edgeLengths: Float32Array.from(this.edgeLengths),
      edgeDrivable: Uint32Array.from(this.edgeDrivable),
      segments: Float32Array.from(this.segments),
      segmentClasses: Uint32Array.from(this.segmentClasses),
      segmentNodes: Uint32Array.from(this.segmentNodes)
    };
  }

  private getNode(key: string, x: number, y: number): number {
    let node = this.nodeIndexes.get(key);
    if (node === undefined) {
      node = this.nodePositions.length / 2;
      this.nodePositions.push(x, y);
      this.nodeIndexes.set(key, node);
    }
    return node;
  }

  private addEdge(from: number, to: number, length: number, drivable: boolean): void {
    this.edgeSources.push(from);
    this.edgeTargets.push(to);
    this.edgeLengths.push(length);
    this.edgeDrivable.push(drivable ? 1 : 0);
  }
}

/** Builds a {@link PolygonSet} from features whose rings are already in meters. */
export function makePolygons(
  features: {id: number; name: string; rings: [number, number][][]}[]
): PolygonSet {
  const positions: number[] = [];
  const ringOffsets: number[] = [0];
  const polygonOffsets: number[] = [0];
  const featureOffsets: number[] = [0];
  const outlineSegments: number[] = [];
  const outlineFeatureRows: number[] = [];
  features.forEach((feature, featureRow) => {
    for (const ring of feature.rings) {
      // Drop an explicit closing vertex: rings close implicitly.
      const open =
        ring.length > 1 &&
        ring[0][0] === ring[ring.length - 1][0] &&
        ring[0][1] === ring[ring.length - 1][1]
          ? ring.slice(0, -1)
          : ring;
      for (const [x, y] of open) positions.push(x, y);
      ringOffsets.push(positions.length / 2);
      for (let index = 0; index < open.length; index++) {
        const [x0, y0] = open[index];
        const [x1, y1] = open[(index + 1) % open.length];
        outlineSegments.push(x0, y0, x1, y1);
        outlineFeatureRows.push(featureRow);
      }
    }
    // One polygon (shell plus holes) per feature.
    polygonOffsets.push(ringOffsets.length - 1);
    featureOffsets.push(polygonOffsets.length - 1);
  });
  return {
    polygonPositions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    featureIds: Uint32Array.from(features.map(feature => feature.id)),
    featureNames: features.map(feature => feature.name),
    outlineSegments: Float32Array.from(outlineSegments),
    outlineFeatureRows: Uint32Array.from(outlineFeatureRows)
  };
}

/**
 * Wraps an elevation grid with its planar-meter geometry.
 *
 * @param bounds `[west, south, east, north]` degrees of the raster's outer cell edges.
 */
export function makeElevationRaster(
  elevation: Float32Array,
  width: number,
  height: number,
  bounds: readonly [number, number, number, number],
  projection: LocalMetricProjection
): ElevationRaster {
  const [west, south, east, north] = bounds;
  const [minX, minY] = projection.project(west, south);
  const [maxX, maxY] = projection.project(east, north);
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of elevation) {
    if (Number.isFinite(value)) {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
  }
  return {
    width,
    height,
    elevation,
    bounds: [minX, minY, maxX, maxY],
    cellSize: [(maxX - minX) / width, (maxY - minY) / height],
    elevationRange: [minimum, maximum]
  };
}
