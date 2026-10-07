// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LngLat, MapHighlight} from '../cartography/types';
import {geodesicCircleRing, geodesicDestination, splitAtAntimeridian} from './annotation-geodesic';
import {
  type Projector,
  type ScreenPoint,
  boundsRing,
  circlePath,
  piecesToPath,
  pointsBounds,
  polylineMidpoint,
  projectCoordinate,
  projectCoordinates,
  projectPieces
} from './annotation-geometry';

/** The viewport fields the highlight layer reads. */
export type HighlightViewport = {width: number; height: number; project: Projector};

const SVG_NS = 'http://www.w3.org/2000/svg';
const DEFAULT_POINT_RADIUS = 6;
const CIRCLE_VERTEX_COUNT = 96;
const BOX_EDGE_STEPS = 12;
const MIN_PULSE_RADIUS = 8;
const MAX_PULSE_RADIUS = 60;

type HighlightNode = {
  highlight: MapHighlight;
  group: SVGGElement;
  halo: SVGPathElement;
  core: SVGPathElement;
  pulseGroup: SVGGElement | null;
  pulseRing: SVGCircleElement | null;
  lastPath: string;
  lastPulseTransform: string;
  lastPulseRadius: number;
};

/** Where a highlight's one-shot pulse starts: centre and starting radius, in CSS pixels. */
type Pulse = {x: number; y: number; radius: number};

/**
 * The topmost annotation layer: hover and linked-highlight geometry drawn as an achromatic ink
 * core over a ground-colour halo (or `--map-signal` with `tone: 'signal'`). Never culled, never
 * collides with labels.
 */
export class HighlightLayer {
  /** Full-size SVG, absolutely positioned; add it last so it paints above every other layer. */
  readonly element: SVGSVGElement;

  private signature = '[]';
  private nodes: HighlightNode[] = [];
  private width = -1;
  private height = -1;

  constructor() {
    this.element = document.createElementNS(SVG_NS, 'svg');
    this.element.setAttribute('class', 'map-annotation-highlights');
    this.element.setAttribute('aria-hidden', 'true');
  }

  /**
   * Replaces the highlighted geometry. Equal content is ignored, so a tooltip that calls this on
   * every pointer move neither rebuilds nodes nor restarts the pulse. Returns `true` when the
   * content changed (the caller then calls {@link update}).
   */
  setHighlights(highlights: readonly MapHighlight[]): boolean {
    const signature = JSON.stringify(highlights);
    if (signature === this.signature) return false;
    this.signature = signature;
    for (const node of this.nodes) node.group.remove();
    this.nodes = highlights.map(highlight => this.createNode(highlight));
    return true;
  }

  /** Re-projects every highlight for a new viewport. */
  update(viewport: HighlightViewport): void {
    if (viewport.width !== this.width || viewport.height !== this.height) {
      this.width = viewport.width;
      this.height = viewport.height;
      this.element.setAttribute('width', String(viewport.width));
      this.element.setAttribute('height', String(viewport.height));
      this.element.setAttribute('viewBox', `0 0 ${viewport.width} ${viewport.height}`);
    }
    for (const node of this.nodes) {
      const {path, pulse} = this.project(node.highlight, viewport.project);
      if (path !== node.lastPath) {
        node.lastPath = path;
        node.halo.setAttribute('d', path);
        node.core.setAttribute('d', path);
      }
      if (node.pulseGroup && node.pulseRing && pulse) {
        const transform = `translate(${pulse.x.toFixed(1)} ${pulse.y.toFixed(1)})`;
        if (transform !== node.lastPulseTransform) {
          node.lastPulseTransform = transform;
          node.pulseGroup.setAttribute('transform', transform);
        }
        if (pulse.radius !== node.lastPulseRadius) {
          node.lastPulseRadius = pulse.radius;
          node.pulseRing.setAttribute('r', pulse.radius.toFixed(1));
        }
      }
    }
  }

  /** Removes the layer. */
  destroy(): void {
    this.element.remove();
    this.nodes = [];
  }

  private createNode(highlight: MapHighlight): HighlightNode {
    const group = document.createElementNS(SVG_NS, 'g');
    group.setAttribute(
      'class',
      `map-annotation-highlight map-annotation-highlight-${highlight.tone ?? 'ink'}`
    );
    const halo = document.createElementNS(SVG_NS, 'path');
    halo.setAttribute('class', 'map-annotation-highlight-halo');
    const core = document.createElementNS(SVG_NS, 'path');
    core.setAttribute('class', 'map-annotation-highlight-core');
    group.append(halo, core);
    let pulseGroup: SVGGElement | null = null;
    let pulseRing: SVGCircleElement | null = null;
    if (highlight.pulse) {
      pulseGroup = document.createElementNS(SVG_NS, 'g');
      pulseRing = document.createElementNS(SVG_NS, 'circle');
      pulseRing.setAttribute('class', 'map-annotation-highlight-pulse');
      pulseGroup.append(pulseRing);
      group.append(pulseGroup);
    }
    this.element.append(group);
    return {
      highlight,
      group,
      halo,
      core,
      pulseGroup,
      pulseRing,
      lastPath: '',
      lastPulseTransform: '',
      lastPulseRadius: -1
    };
  }

  private project(
    highlight: MapHighlight,
    project: Projector
  ): {path: string; pulse: Pulse | null} {
    switch (highlight.kind) {
      case 'point': {
        const [x, y] = projectCoordinate(project, highlight.coordinate);
        const radius = highlight.radiusPixels ?? DEFAULT_POINT_RADIUS;
        return {path: circlePath(x, y, radius), pulse: {x, y, radius}};
      }
      case 'circle': {
        const split = splitAtAntimeridian(
          geodesicCircleRing(highlight.coordinate, highlight.radiusMeters, CIRCLE_VERTEX_COUNT),
          true
        );
        const [x, y] = projectCoordinate(project, highlight.coordinate);
        const [edgeX, edgeY] = projectCoordinate(
          project,
          geodesicDestination(highlight.coordinate, Math.PI / 2, highlight.radiusMeters)
        );
        return {
          path: piecesToPath(projectPieces(project, split.pieces), split.closed),
          pulse: {x, y, radius: clampPulse(Math.hypot(edgeX - x, edgeY - y))}
        };
      }
      case 'polygon': {
        const rings = highlight.rings.map(ring => projectCoordinates(project, ring));
        return {path: piecesToPath(rings, true), pulse: pulseAtBounds(rings[0] ?? [])};
      }
      case 'line': {
        const points = projectCoordinates(project, highlight.coordinates);
        const middle = polylineMidpoint(points);
        return {
          path: piecesToPath([points], false),
          pulse: middle ? {x: middle[0], y: middle[1], radius: 12} : null
        };
      }
      case 'box': {
        const ring: LngLat[] = boundsRing(highlight.bounds, BOX_EDGE_STEPS);
        const points = projectCoordinates(project, ring);
        return {path: piecesToPath([points], true), pulse: pulseAtBounds(points)};
      }
    }
  }
}

function clampPulse(radius: number): number {
  return Math.max(MIN_PULSE_RADIUS, Math.min(MAX_PULSE_RADIUS, radius));
}

function pulseAtBounds(points: readonly ScreenPoint[]): Pulse | null {
  const bounds = pointsBounds(points);
  if (!bounds) return null;
  return {
    x: bounds.x + bounds.width / 2,
    y: bounds.y + bounds.height / 2,
    radius: clampPulse(Math.max(bounds.width, bounds.height) / 2)
  };
}
