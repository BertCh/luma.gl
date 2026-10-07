// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './annotation-overlay.css';
import type {AnnotationTone, MapAnnotation, MapHighlight} from '../cartography/types';
import {
  formatDistance,
  geodesicCircleRing,
  geodesicDistanceMeters,
  localCircleRing,
  splitAtAntimeridian
} from './annotation-geodesic';
import {
  type Box,
  type ScreenPoint,
  boundsRing,
  piecesToPath,
  pointsBounds,
  polygonCentroid,
  polylineMidpoint,
  projectCoordinate,
  projectCoordinates,
  projectPieces,
  starPath
} from './annotation-geometry';
import {HighlightLayer} from './annotation-highlights';
import {
  EDGE_INSET,
  LABEL_GAP,
  type LabelCandidate,
  type LabelSize,
  chooseCandidate,
  directionalCandidates,
  offsetCandidate,
  offsetLeader,
  plainCandidate
} from './annotation-placement';

/** Minimal projection the overlay needs (the host adapts deck's Viewport to it). */
export type AnnotationViewport = {
  /** Map width in CSS pixels. */
  width: number;
  /** Map height in CSS pixels. */
  height: number;
  zoom: number;
  /** [longitude, latitude] -> [x, y] CSS pixels from the top-left of the map. */
  project: (coordinate: readonly [number, number]) => readonly [number, number];
};

/** A pre-placed rectangle (legend, cartouche, scale bar) that labels must not overlap. */
export type AnnotationObstacle = {x: number; y: number; width: number; height: number};

type Kind = MapAnnotation['kind'];

const SVG_NS = 'http://www.w3.org/2000/svg';
const RING_VERTEX_COUNT = 64;
const GEODESIC_RING_VERTEX_COUNT = 128;
const FRAME_EDGE_STEPS = 8;
const ARROW_HEAD_LENGTH = 10;
const ARROW_HEAD_HALF_WIDTH = 4.5;
const ARROW_CURVE = 0.12;
const CALLOUT_OFFSET_X = -16;
const CALLOUT_OFFSET_Y = -44;
const CALLOUT_POINTER_HEIGHT = 6;
const MARKER_SIZE = 18;
const DIMENSION_TICK = 8;
const BRACKET_OFFSET = 8;
const DEFAULT_STAR_RADIUS = 7;
const NOTE_DISTANCE = 36;
const NARROW_MAP_WIDTH = 600;
const NOTE_LIMIT = 3;
const NOTE_LIMIT_NARROW = 2;
const PLACE_LIMIT = 6;
const PLACE_LIMIT_NARROW = 4;

const DEFAULT_TONES: Record<Kind, AnnotationTone> = {
  point: 'ink',
  area: 'muted',
  water: 'water',
  landform: 'ink',
  note: 'ink',
  marker: 'ink',
  outline: 'accent',
  dimension: 'ink',
  bracket: 'ink',
  star: 'accent',
  line: 'ink',
  frame: 'muted',
  ring: 'accent',
  arrow: 'accent',
  callout: 'ink'
};

/** Kinds whose geometry is fixed to the map: never culled by collision or budget. */
const FIXED_KINDS: ReadonlySet<Kind> = new Set<Kind>([
  'ring',
  'arrow',
  'callout',
  'marker',
  'outline',
  'dimension',
  'bracket',
  'star',
  'line',
  'frame'
]);

/** One live annotation: its DOM nodes, cached measurement, and per-frame write caches. */
type Entry = {
  key: string;
  annotation: MapAnnotation;
  /** Style-affecting fields; a change rebuilds the DOM nodes, other changes update in place. */
  signature: string;
  order: number;
  /** Geometry group inside the SVG. */
  group: SVGGElement | null;
  /** Text node in the HTML layer (labels, plates, markers, callouts). */
  label: HTMLDivElement | null;
  /** Geometry shapes whose `d` is rewritten when it changes, by role. */
  shapes: Record<string, SVGElement>;
  shapeData: Record<string, string>;
  /** Measured label size, `null` until the element is attached and laid out. */
  size: LabelSize | null;
  groupVisible: boolean;
  labelVisible: boolean;
  lastTransform: string;
  lastGroupTransform: string;
  /** Serialised label content, so DOM text is rewritten only when it changed. */
  contentKey: string;
  /** Values computed once from the annotation (dimension length, outline centroid). */
  dimensionText: string;
  centroid: readonly [number, number] | null;
  /** Candidate index used in the last pass, -1 when hidden (hysteresis). */
  candidateIndex: number;
};

/** What a flexible (collision-placed) annotation offers to the placement pass. */
type FlexibleLayout = {
  candidates: LabelCandidate[];
  /** Feature position, where the SVG group is translated to. */
  featureX: number;
  featureY: number;
  /** Boxes reserved besides the label (the marker symbol). */
  reserve: Box[];
};

/** What a fixed annotation offers: candidate label boxes (first free wins) and reserved boxes. */
type FixedLayout = {candidates: LabelCandidate[]; reserve: Box[]};

type Part = {className: string; text: string};

/**
 * Screen-space map annotations: a full-size SVG layer for geometry (markers, leaders, rings,
 * arrows, outlines, lines) under an HTML layer for haloed text, and a topmost SVG layer for
 * hover highlights. Re-projects on every camera change, culls by zoom range, time and a label
 * budget, and places labels with priority-ordered greedy collision avoidance (NE, SE, NW, SW
 * candidates, hysteresis between frames, host-supplied obstacles and a 12 px edge inset).
 *
 * Put {@link AnnotationOverlay.element} inside the map stage above the canvas and call
 * {@link AnnotationOverlay.update} on every camera change.
 */
export class AnnotationOverlay {
  /** Absolutely positioned, inset 0, pointer-events none, overflow hidden. */
  readonly element: HTMLDivElement;

  private readonly svg: SVGSVGElement;
  private readonly textLayer: HTMLDivElement;
  private readonly highlightLayer = new HighlightLayer();
  private entries = new Map<string, Entry>();
  private ordered: Entry[] = [];
  private obstacles: Box[] = [];
  private time: number | null = null;
  private lastViewport: AnnotationViewport | null = null;
  private svgWidth = -1;
  private svgHeight = -1;
  private destroyed = false;

  /** Web fonts change glyph widths: forget every measurement and place again. */
  private readonly handleFontsChanged = (): void => {
    if (this.destroyed) return;
    for (const entry of this.ordered) if (entry.annotation.kind !== 'marker') entry.size = null;
    if (this.lastViewport) this.update(this.lastViewport);
  };

  constructor() {
    this.element = document.createElement('div');
    this.element.className = 'map-annotation-overlay';
    this.svg = document.createElementNS(SVG_NS, 'svg');
    this.svg.setAttribute('class', 'map-annotation-svg');
    this.svg.setAttribute('aria-hidden', 'true');
    this.textLayer = document.createElement('div');
    this.textLayer.className = 'map-annotation-text';
    this.element.append(this.svg, this.textLayer, this.highlightLayer.element);
    // Boxes measured before the web fonts load are wrong; measure again once they have.
    if (typeof document !== 'undefined' && document.fonts) {
      void document.fonts.ready.then(this.handleFontsChanged);
      document.fonts.addEventListener('loadingdone', this.handleFontsChanged);
    }
  }

  /**
   * Replaces the annotation list. Entries with an `id` keep their DOM nodes (and update in place
   * unless a style field changed); entries without one are keyed by their content. Cheap when
   * the list is equal.
   */
  setAnnotations(annotations: readonly MapAnnotation[]): void {
    const next = new Map<string, Entry>();
    const ordered: Entry[] = [];
    const occurrences = new Map<string, number>();
    annotations.forEach((annotation, order) => {
      const base = annotation.id !== undefined ? `id:${annotation.id}` : JSON.stringify(annotation);
      const count = occurrences.get(base) ?? 0;
      occurrences.set(base, count + 1);
      const key = count === 0 ? base : `${base}#${count}`;
      const signature = structureSignature(annotation);
      let entry = this.entries.get(key);
      if (entry && entry.signature !== signature) {
        this.removeEntry(entry);
        entry = undefined;
      }
      if (!entry) entry = this.createEntry(key, annotation, signature);
      else if (annotation.id !== undefined) this.assign(entry, annotation);
      entry.order = order;
      next.set(key, entry);
      ordered.push(entry);
    });
    for (const [key, entry] of this.entries) if (next.get(key) !== entry) this.removeEntry(entry);
    this.entries = next;
    this.ordered = ordered;
    if (this.lastViewport) this.update(this.lastViewport);
  }

  /**
   * Pre-placed boxes (map furniture, legend, cartouche) in CSS pixels relative to the overlay.
   * Labels never overlap them; fixed geometry labels avoid them when they can.
   */
  setObstacles(rects: readonly AnnotationObstacle[]): void {
    this.obstacles = rects.map(({x, y, width, height}) => ({x, y, width, height}));
    if (this.lastViewport) this.update(this.lastViewport);
  }

  /**
   * Sets the annotation time: annotations with a `timeRange` are shown only while
   * `start <= time < end`. `null` shows every annotation.
   */
  setTime(time: number | null): void {
    if (time === this.time) return;
    this.time = time;
    if (this.lastViewport) this.update(this.lastViewport);
  }

  /**
   * Sets the hover / linked highlight geometry, drawn above everything and never culled.
   * Equal content is ignored, so it is safe to call on every pointer move.
   */
  setHighlights(highlights: readonly MapHighlight[]): void {
    if (this.highlightLayer.setHighlights(highlights) && this.lastViewport) {
      this.highlightLayer.update(this.lastViewport);
    }
  }

  /**
   * Chooses the text halo: `'normal'` (2.5 px) or `'heavy'` (3 px, for labels over dense or
   * saturated data). Scenes may also set `--annotation-halo-width` themselves.
   */
  setHaloWeight(weight: 'normal' | 'heavy'): void {
    this.element.classList.toggle('is-halo-heavy', weight === 'heavy');
  }

  /**
   * Re-projects, culls (zoom, time, budget, collision), places labels and writes transforms.
   * Measures a label only the first time it is seen (or after fonts load).
   */
  update(viewport: AnnotationViewport): void {
    this.lastViewport = viewport;
    const {width, height, zoom} = viewport;
    if (width !== this.svgWidth || height !== this.svgHeight) {
      this.svgWidth = width;
      this.svgHeight = height;
      this.svg.setAttribute('width', String(width));
      this.svg.setAttribute('height', String(height));
      this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    }

    // Reads first (once per label), so there is at most one layout before the writes below.
    for (const entry of this.ordered) this.measure(entry);

    const placed: Box[] = [...this.obstacles];
    const edge: Box = {
      x: EDGE_INSET,
      y: EDGE_INSET,
      width: Math.max(0, width - 2 * EDGE_INSET),
      height: Math.max(0, height - 2 * EDGE_INSET)
    };

    const fixed: Entry[] = [];
    const flexible: {entry: Entry; previous: number}[] = [];
    for (const entry of this.ordered) {
      if (!this.isActive(entry, zoom)) {
        this.hide(entry);
        continue;
      }
      if (FIXED_KINDS.has(entry.annotation.kind)) fixed.push(entry);
      else flexible.push({entry, previous: entry.candidateIndex});
    }

    // Fixed geometry first: never culled, and its labels reserve space before movable labels.
    fixed.sort(
      (a, b) => (b.annotation.priority ?? 0) - (a.annotation.priority ?? 0) || a.order - b.order
    );
    for (const entry of fixed) this.placeFixed(entry, viewport, placed);

    // Movable labels, highest priority first; on equal priority the visible one keeps its place.
    flexible.sort(
      (a, b) =>
        (b.entry.annotation.priority ?? 0) - (a.entry.annotation.priority ?? 0) ||
        Number(b.previous >= 0) - Number(a.previous >= 0) ||
        a.entry.order - b.entry.order
    );
    const narrow = width < NARROW_MAP_WIDTH;
    let noteCount = 0;
    let placeCount = 0;
    const noteLimit = narrow ? NOTE_LIMIT_NARROW : NOTE_LIMIT;
    const placeLimit = narrow ? PLACE_LIMIT_NARROW : PLACE_LIMIT;
    for (const {entry, previous} of flexible) {
      const isNote = entry.annotation.kind === 'note';
      if (isNote ? noteCount >= noteLimit : placeCount >= placeLimit) {
        this.hide(entry);
        continue;
      }
      const layout = this.layoutFlexible(entry, viewport);
      const index = layout ? chooseCandidate(layout.candidates, previous, placed, edge) : -1;
      if (!layout || index < 0) {
        this.hide(entry);
        continue;
      }
      const candidate = layout.candidates[index];
      placed.push(candidate.box, ...layout.reserve);
      if (isNote) noteCount++;
      else placeCount++;
      entry.candidateIndex = index;
      this.writeGroup(
        entry,
        `translate(${layout.featureX.toFixed(2)} ${layout.featureY.toFixed(2)})`
      );
      if (candidate.leader !== undefined) {
        this.writeShape(entry, 'leaderHalo', candidate.leader);
        this.writeShape(entry, 'leader', candidate.leader);
      }
      this.setGroupVisible(entry, true);
      this.setLabelVisible(entry, true);
      this.writeLabel(entry, candidate.labelX, candidate.labelY, !candidate.exact);
    }

    this.highlightLayer.update(viewport);
  }

  /** Removes the overlay and its font listeners. */
  destroy(): void {
    this.destroyed = true;
    if (typeof document !== 'undefined' && document.fonts) {
      document.fonts.removeEventListener('loadingdone', this.handleFontsChanged);
    }
    this.highlightLayer.destroy();
    this.element.remove();
    this.entries.clear();
    this.ordered = [];
    this.lastViewport = null;
  }

  // -------------------------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------------------------

  private createEntry(key: string, annotation: MapAnnotation, signature: string): Entry {
    const entry: Entry = {
      key,
      annotation,
      signature,
      order: 0,
      group: null,
      label: null,
      shapes: {},
      shapeData: {},
      size: null,
      groupVisible: false,
      labelVisible: false,
      lastTransform: '',
      lastGroupTransform: '',
      contentKey: '',
      dimensionText: '',
      centroid: null,
      candidateIndex: -1
    };
    const tone = annotation.tone ?? defaultTone(annotation);
    const toneClass = `map-annotation-tone-${tone}`;

    switch (annotation.kind) {
      case 'point': {
        const group = this.createGroup(toneClass);
        const marker = annotation.marker ?? 'dot';
        if (annotation.offset) {
          const [dx, dy] = annotation.offset;
          const leader = offsetLeader(dx, dy);
          group.append(
            svgElement('path', {class: 'map-annotation-leader-halo', d: leader}),
            svgElement('path', {class: 'map-annotation-leader', d: leader})
          );
          if (marker === 'none') group.append(...leaderDot());
        }
        if (marker === 'dot') {
          group.append(
            svgElement('circle', {class: 'map-annotation-halo-fill', r: '3', 'stroke-width': '3'}),
            svgElement('circle', {class: 'map-annotation-ink-fill', r: '3'})
          );
        } else if (marker === 'ring') {
          group.append(
            svgElement('circle', {
              class: 'map-annotation-halo-stroke',
              r: '3.5',
              'stroke-width': '5'
            }),
            svgElement('circle', {
              class: 'map-annotation-ink-stroke',
              r: '3.5',
              'stroke-width': '2'
            })
          );
        }
        entry.group = group;
        entry.label = this.createLabel(
          `map-annotation-label map-annotation-point ${annotation.rank === 'context' ? 'is-context' : ''} ${toneClass}`
        );
        break;
      }
      case 'landform': {
        if (annotation.marker === 'peak') {
          const group = this.createGroup(toneClass);
          const triangle = 'M0 -5L5 3.5H-5Z';
          group.append(
            svgElement('path', {
              class: 'map-annotation-halo-fill',
              d: triangle,
              'stroke-width': '3',
              'stroke-linejoin': 'round'
            }),
            svgElement('path', {class: 'map-annotation-ink-fill', d: triangle})
          );
          entry.group = group;
        }
        entry.label = this.createLabel(
          `map-annotation-label map-annotation-landform size-${annotation.size ?? 'medium'} ${toneClass}`
        );
        break;
      }
      case 'area':
      case 'water': {
        entry.label = this.createLabel(
          `map-annotation-label map-annotation-${annotation.kind} size-${annotation.size ?? 'medium'} ${toneClass}`
        );
        break;
      }
      case 'note': {
        const group = this.createGroup(toneClass);
        group.append(
          svgElement('path', {class: 'map-annotation-leader-halo'}),
          svgElement('path', {class: 'map-annotation-leader'}),
          ...leaderDot()
        );
        entry.shapes = {
          leaderHalo: group.children[0] as SVGElement,
          leader: group.children[1] as SVGElement
        };
        entry.group = group;
        entry.label = this.createLabel(`map-annotation-label map-annotation-note ${toneClass}`);
        break;
      }
      case 'marker': {
        entry.label = this.createLabel(`map-annotation-label map-annotation-marker ${toneClass}`);
        entry.label.setAttribute('role', 'img');
        break;
      }
      case 'ring': {
        const group = this.createGroup(toneClass);
        const halo = svgElement('path', {
          class: 'map-annotation-halo-stroke',
          'stroke-width': '3.5'
        });
        const line = svgElement('path', {
          class: 'map-annotation-ink-stroke',
          'stroke-width': '1.75'
        });
        if (annotation.dashed) line.setAttribute('stroke-dasharray', '6 4');
        group.append(halo, line);
        entry.group = group;
        entry.shapes = {halo, line};
        if (annotation.text) entry.label = this.createGeometryLabel(toneClass);
        break;
      }
      case 'arrow': {
        const group = this.createGroup(toneClass);
        const haloLine = svgElement('path', {
          class: 'map-annotation-halo-stroke',
          'stroke-width': '4'
        });
        const haloHead = svgElement('path', {
          class: 'map-annotation-halo-fill',
          'stroke-width': '3'
        });
        const line = svgElement('path', {class: 'map-annotation-ink-stroke', 'stroke-width': '2'});
        const head = svgElement('path', {class: 'map-annotation-ink-fill'});
        group.append(haloLine, haloHead, line, head);
        entry.group = group;
        entry.shapes = {haloLine, line, haloHead, head};
        if (annotation.text) entry.label = this.createGeometryLabel(toneClass);
        break;
      }
      case 'outline': {
        const group = this.createGroup(toneClass);
        const halo = svgElement('path', {class: 'map-annotation-halo-stroke', 'stroke-width': '6'});
        const line = svgElement('path', {class: 'map-annotation-ink-stroke', 'stroke-width': '2'});
        if (annotation.dashed) line.setAttribute('stroke-dasharray', '6 4');
        group.append(halo, line);
        entry.group = group;
        entry.shapes = {halo, line};
        if (annotation.text) entry.label = this.createGeometryLabel(toneClass);
        break;
      }
      case 'dimension':
      case 'bracket': {
        const group = this.createGroup(toneClass);
        const halo = svgElement('path', {
          class: 'map-annotation-halo-stroke',
          'stroke-width': '3.5'
        });
        const line = svgElement('path', {
          class: 'map-annotation-ink-stroke',
          'stroke-width': '1.25'
        });
        group.append(halo, line);
        entry.group = group;
        entry.shapes = {halo, line};
        // A dimension always has a label (its length); a bracket only with text.
        if (annotation.kind === 'dimension' || annotation.text) {
          entry.label = this.createGeometryLabel(toneClass);
        }
        break;
      }
      case 'star': {
        const group = this.createGroup(toneClass);
        const radius = annotation.radiusPixels ?? DEFAULT_STAR_RADIUS;
        const path = starPath(radius);
        group.append(
          svgElement('path', {
            class: 'map-annotation-halo-fill',
            d: path,
            'stroke-width': '3',
            'stroke-linejoin': 'round'
          }),
          svgElement('path', {class: 'map-annotation-ink-fill', d: path})
        );
        entry.group = group;
        if (annotation.text) entry.label = this.createGeometryLabel(toneClass);
        break;
      }
      case 'line': {
        const group = this.createGroup(toneClass);
        const width = annotation.widthPixels ?? 1.5;
        const halo = svgElement('path', {
          class: 'map-annotation-halo-stroke',
          'stroke-width': String(width + 2.5)
        });
        const line = svgElement('path', {
          class: 'map-annotation-ink-stroke',
          'stroke-width': String(width)
        });
        if (annotation.dashed) line.setAttribute('stroke-dasharray', '6 4');
        group.append(halo, line);
        entry.group = group;
        entry.shapes = {halo, line};
        if (annotation.text) entry.label = this.createGeometryLabel(toneClass);
        break;
      }
      case 'frame': {
        const group = this.createGroup(toneClass);
        const halo = svgElement('path', {
          class: 'map-annotation-halo-stroke',
          'stroke-width': '3.5'
        });
        const line = svgElement('path', {class: 'map-annotation-frame-line'});
        group.append(halo, line);
        entry.group = group;
        entry.shapes = {halo, line};
        if (annotation.text) {
          entry.label = this.createLabel(`map-annotation-label map-annotation-frame ${toneClass}`);
        }
        break;
      }
      case 'callout': {
        entry.label = this.createLabel('map-annotation-callout');
        break;
      }
    }
    this.assign(entry, annotation);
    return entry;
  }

  /** Stores a (possibly updated) annotation and refreshes values derived from it. */
  private assign(entry: Entry, annotation: MapAnnotation): void {
    entry.annotation = annotation;
    if (annotation.kind === 'dimension') {
      entry.dimensionText =
        annotation.text ?? formatDistance(geodesicDistanceMeters(annotation.from, annotation.to));
    } else if (annotation.kind === 'outline' && annotation.rings.length > 0) {
      entry.centroid = polygonCentroid(annotation.rings[0]);
    }
    this.syncContent(entry);
  }

  /** Rewrites label text only when it changed; a change forces a re-measure on the next update. */
  private syncContent(entry: Entry): void {
    const label = entry.label;
    if (!label) return;
    const parts = contentParts(entry);
    const contentKey = JSON.stringify(parts);
    if (contentKey === entry.contentKey) return;
    entry.contentKey = contentKey;
    if (parts.length === 1 && parts[0].className === '') {
      label.textContent = parts[0].text;
    } else {
      label.replaceChildren(
        ...parts.map(part => {
          const node = document.createElement('div');
          if (part.className) node.className = part.className;
          node.textContent = part.text;
          return node;
        })
      );
    }
    if (entry.annotation.kind === 'marker') {
      const {number, text} = entry.annotation;
      label.setAttribute('aria-label', text ?? `Marker ${number}`);
    }
    entry.size = null;
  }

  private removeEntry(entry: Entry): void {
    entry.group?.remove();
    entry.label?.remove();
  }

  private createGroup(toneClass: string): SVGGElement {
    const group = document.createElementNS(SVG_NS, 'g');
    group.setAttribute('class', `map-annotation-item is-hidden ${toneClass}`);
    this.svg.append(group);
    return group;
  }

  private createLabel(className: string): HTMLDivElement {
    const label = document.createElement('div');
    label.className = `map-annotation-item is-hidden ${className}`;
    this.textLayer.append(label);
    return label;
  }

  private createGeometryLabel(toneClass: string): HTMLDivElement {
    return this.createLabel(`map-annotation-label map-annotation-geometry-label ${toneClass}`);
  }

  /** Measures a label once, as soon as it is attached and has a layout. */
  private measure(entry: Entry): void {
    if (!entry.label || entry.size) return;
    if (entry.annotation.kind === 'marker') {
      entry.size = {width: MARKER_SIZE, height: MARKER_SIZE};
      return;
    }
    const width = entry.label.offsetWidth;
    const height = entry.label.offsetHeight;
    if (width > 0 && height > 0) entry.size = {width, height};
  }

  private isActive(entry: Entry, zoom: number): boolean {
    const {minZoom, maxZoom, timeRange} = entry.annotation;
    if (minZoom !== undefined && zoom < minZoom) return false;
    if (maxZoom !== undefined && zoom >= maxZoom) return false;
    if (
      timeRange &&
      this.time !== null &&
      !(this.time >= timeRange[0] && this.time < timeRange[1])
    ) {
      return false;
    }
    return true;
  }

  // -------------------------------------------------------------------------------------------
  // Layout: fixed geometry
  // -------------------------------------------------------------------------------------------

  /** Draws fixed geometry, then places its label on the first free candidate (never culled). */
  private placeFixed(entry: Entry, viewport: AnnotationViewport, placed: Box[]): void {
    const layout = this.layoutFixed(entry, viewport);
    this.setGroupVisible(entry, true);
    placed.push(...layout.reserve);
    if (!entry.label || !entry.size || layout.candidates.length === 0) {
      if (entry.label) this.setLabelVisible(entry, false);
      entry.candidateIndex = 0;
      return;
    }
    let index = chooseCandidate(layout.candidates, entry.candidateIndex, placed, null);
    if (index < 0)
      index = Math.max(0, Math.min(entry.candidateIndex, layout.candidates.length - 1));
    entry.candidateIndex = index;
    const candidate = layout.candidates[index];
    placed.push(candidate.box);
    this.setLabelVisible(entry, true);
    this.writeLabel(entry, candidate.labelX, candidate.labelY, !candidate.exact);
  }

  private layoutFixed(entry: Entry, viewport: AnnotationViewport): FixedLayout {
    const annotation = entry.annotation;
    const size = entry.size;
    const project = viewport.project;
    const none: FixedLayout = {candidates: [], reserve: []};

    switch (annotation.kind) {
      case 'callout': {
        const [x, y] = projectCoordinate(project, annotation.coordinate);
        if (!size) return none;
        return {
          candidates: [
            {
              box: {
                x: x + CALLOUT_OFFSET_X,
                y: y + CALLOUT_OFFSET_Y,
                width: size.width,
                height: size.height + CALLOUT_POINTER_HEIGHT
              },
              labelX: x,
              labelY: y,
              exact: true
            }
          ],
          reserve: []
        };
      }
      case 'marker': {
        const [x, y] = projectCoordinate(project, annotation.coordinate);
        return {
          candidates: [
            plainCandidate(
              x - MARKER_SIZE / 2,
              y - MARKER_SIZE / 2,
              entry.size ?? {width: MARKER_SIZE, height: MARKER_SIZE}
            )
          ],
          reserve: []
        };
      }
      case 'ring': {
        const split = annotation.geodesic
          ? splitAtAntimeridian(
              geodesicCircleRing(
                annotation.coordinate,
                annotation.radiusMeters,
                GEODESIC_RING_VERTEX_COUNT
              ),
              true
            )
          : {
              pieces: [
                localCircleRing(annotation.coordinate, annotation.radiusMeters, RING_VERTEX_COUNT)
              ],
              closed: true
            };
        const pieces = projectPieces(project, split.pieces);
        const path = piecesToPath(pieces, split.closed);
        this.writeShape(entry, 'halo', path);
        this.writeShape(entry, 'line', path);
        if (!size) return none;
        let top: ScreenPoint | null = null;
        let bottom: ScreenPoint | null = null;
        for (const piece of pieces) {
          for (const point of piece) {
            if (!top || point[1] < top[1]) top = point;
            if (!bottom || point[1] > bottom[1]) bottom = point;
          }
        }
        if (!top || !bottom) return none;
        return {
          candidates: [
            plainCandidate(top[0] - size.width / 2, top[1] - size.height - 4, size),
            plainCandidate(bottom[0] - size.width / 2, bottom[1] + 4, size)
          ],
          reserve: []
        };
      }
      case 'arrow': {
        const [fromX, fromY] = projectCoordinate(project, annotation.from);
        const [toX, toY] = projectCoordinate(project, annotation.to);
        const dx = toX - fromX;
        const dy = toY - fromY;
        const length = Math.hypot(dx, dy) || 1;
        const ux = dx / length;
        const uy = dy / length;
        // Control point offset perpendicular to the chord for a gentle bow.
        const controlX = (fromX + toX) / 2 - uy * length * ARROW_CURVE;
        const controlY = (fromY + toY) / 2 + ux * length * ARROW_CURVE;
        // End tangent (control -> tip) orients the head.
        let tx = toX - controlX;
        let ty = toY - controlY;
        const tangentLength = Math.hypot(tx, ty) || 1;
        tx /= tangentLength;
        ty /= tangentLength;
        const baseX = toX - tx * ARROW_HEAD_LENGTH;
        const baseY = toY - ty * ARROW_HEAD_LENGTH;
        const curve = `M${fromX.toFixed(1)} ${fromY.toFixed(1)}Q${controlX.toFixed(1)} ${controlY.toFixed(1)} ${baseX.toFixed(1)} ${baseY.toFixed(1)}`;
        const head = `M${toX.toFixed(1)} ${toY.toFixed(1)}L${(baseX - ty * ARROW_HEAD_HALF_WIDTH).toFixed(1)} ${(baseY + tx * ARROW_HEAD_HALF_WIDTH).toFixed(1)}L${(baseX + ty * ARROW_HEAD_HALF_WIDTH).toFixed(1)} ${(baseY - tx * ARROW_HEAD_HALF_WIDTH).toFixed(1)}Z`;
        this.writeShape(entry, 'haloLine', curve);
        this.writeShape(entry, 'line', curve);
        this.writeShape(entry, 'haloHead', head);
        this.writeShape(entry, 'head', head);
        if (!size) return none;
        // Label sits behind the tail, clear of the line.
        const reach = 8 + Math.abs(ux) * (size.width / 2) + Math.abs(uy) * (size.height / 2);
        return {
          candidates: [
            plainCandidate(
              fromX - ux * reach - size.width / 2,
              fromY - uy * reach - size.height / 2,
              size
            )
          ],
          reserve: []
        };
      }
      case 'outline': {
        const rings = annotation.rings.map(ring => projectCoordinates(project, ring));
        const path = piecesToPath(rings, true);
        this.writeShape(entry, 'halo', path);
        this.writeShape(entry, 'line', path);
        const at = annotation.labelAt ?? entry.centroid;
        if (!size || !at) return none;
        const [x, y] = projectCoordinate(project, at);
        return {
          candidates: [plainCandidate(x - size.width / 2, y - size.height / 2, size)],
          reserve: []
        };
      }
      case 'dimension': {
        const [fromX, fromY] = projectCoordinate(project, annotation.from);
        const [toX, toY] = projectCoordinate(project, annotation.to);
        const length = Math.hypot(toX - fromX, toY - fromY) || 1;
        // Tick direction: perpendicular to the line, 8 px long in total.
        const nx = (-(toY - fromY) / length) * (DIMENSION_TICK / 2);
        const ny = ((toX - fromX) / length) * (DIMENSION_TICK / 2);
        const path =
          `M${fromX.toFixed(1)} ${fromY.toFixed(1)}L${toX.toFixed(1)} ${toY.toFixed(1)}` +
          `M${(fromX - nx).toFixed(1)} ${(fromY - ny).toFixed(1)}L${(fromX + nx).toFixed(1)} ${(fromY + ny).toFixed(1)}` +
          `M${(toX - nx).toFixed(1)} ${(toY - ny).toFixed(1)}L${(toX + nx).toFixed(1)} ${(toY + ny).toFixed(1)}`;
        this.writeShape(entry, 'halo', path);
        this.writeShape(entry, 'line', path);
        if (!size) return none;
        const middleX = (fromX + toX) / 2;
        const middleY = (fromY + toY) / 2;
        return {
          candidates: [
            plainCandidate(middleX - size.width / 2, middleY - size.height - 6, size),
            plainCandidate(middleX - size.width / 2, middleY + 6, size)
          ],
          reserve: []
        };
      }
      case 'bracket': {
        const [fromX, fromY] = projectCoordinate(project, annotation.from);
        const [toX, toY] = projectCoordinate(project, annotation.to);
        const length = Math.hypot(toX - fromX, toY - fromY) || 1;
        const ux = (toX - fromX) / length;
        const uy = (toY - fromY) / length;
        // Screen y points down, so the left of the direction of travel is (uy, -ux).
        const sign = annotation.side === 'right' ? -1 : 1;
        const nx = uy * sign;
        const ny = -ux * sign;
        const offsetX = nx * BRACKET_OFFSET;
        const offsetY = ny * BRACKET_OFFSET;
        const path =
          `M${fromX.toFixed(1)} ${fromY.toFixed(1)}L${(fromX + offsetX).toFixed(1)} ${(fromY + offsetY).toFixed(1)}` +
          `L${(toX + offsetX).toFixed(1)} ${(toY + offsetY).toFixed(1)}L${toX.toFixed(1)} ${toY.toFixed(1)}`;
        this.writeShape(entry, 'halo', path);
        this.writeShape(entry, 'line', path);
        if (!size) return none;
        const reach =
          BRACKET_OFFSET +
          LABEL_GAP +
          Math.abs(nx) * (size.width / 2) +
          Math.abs(ny) * (size.height / 2);
        const centerX = (fromX + toX) / 2 + nx * reach;
        const centerY = (fromY + toY) / 2 + ny * reach;
        return {
          candidates: [plainCandidate(centerX - size.width / 2, centerY - size.height / 2, size)],
          reserve: []
        };
      }
      case 'star': {
        const [x, y] = projectCoordinate(project, annotation.coordinate);
        this.writeGroup(entry, `translate(${x.toFixed(2)} ${y.toFixed(2)})`);
        const radius = annotation.radiusPixels ?? DEFAULT_STAR_RADIUS;
        const reserve = [{x: x - radius, y: y - radius, width: 2 * radius, height: 2 * radius}];
        if (!size) return {candidates: [], reserve};
        return {
          candidates: directionalCandidates('auto', x, y, radius + LABEL_GAP, size, false),
          reserve
        };
      }
      case 'line': {
        const points = projectCoordinates(project, annotation.coordinates);
        const path = piecesToPath([points], false);
        this.writeShape(entry, 'halo', path);
        this.writeShape(entry, 'line', path);
        const middle = polylineMidpoint(points);
        if (!size || !middle) return none;
        return {
          candidates: [
            plainCandidate(middle[0] - size.width / 2, middle[1] - size.height / 2, size)
          ],
          reserve: []
        };
      }
      case 'frame': {
        const ring = annotation.bounds
          ? boundsRing(annotation.bounds, FRAME_EDGE_STEPS)
          : (annotation.ring ?? []);
        const points = projectCoordinates(project, ring);
        const path = piecesToPath([points], true);
        this.writeShape(entry, 'halo', path);
        this.writeShape(entry, 'line', path);
        const bounds = pointsBounds(points);
        if (!size || !bounds) return none;
        // Top-left inside corner, nudged to stay on screen while the corner is panned away.
        const left = Math.min(
          Math.max(bounds.x + LABEL_GAP, EDGE_INSET),
          Math.max(bounds.x + LABEL_GAP, bounds.x + bounds.width - size.width - LABEL_GAP)
        );
        const top = Math.min(
          Math.max(bounds.y + LABEL_GAP, EDGE_INSET),
          Math.max(bounds.y + LABEL_GAP, bounds.y + bounds.height - size.height - LABEL_GAP)
        );
        return {candidates: [plainCandidate(left, top, size)], reserve: []};
      }
      default:
        return none;
    }
  }

  // -------------------------------------------------------------------------------------------
  // Layout: movable labels
  // -------------------------------------------------------------------------------------------

  /** Candidate label boxes of a point, landform, note, area or water annotation. */
  private layoutFlexible(entry: Entry, viewport: AnnotationViewport): FlexibleLayout | null {
    const annotation = entry.annotation;
    const size = entry.size;
    if (!size) return null;
    if (!('coordinate' in annotation)) return null;
    const [x, y] = projectCoordinate(viewport.project, annotation.coordinate);

    switch (annotation.kind) {
      case 'area':
      case 'water': {
        // Centred on the coordinate, nudged to stay inside the inset; off-screen coordinates cull.
        if (x < 0 || y < 0 || x > viewport.width || y > viewport.height) return null;
        const left = clamp(
          x - size.width / 2,
          EDGE_INSET,
          viewport.width - EDGE_INSET - size.width
        );
        const top = clamp(
          y - size.height / 2,
          EDGE_INSET,
          viewport.height - EDGE_INSET - size.height
        );
        return {
          candidates: [plainCandidate(left, top, size)],
          featureX: x,
          featureY: y,
          reserve: []
        };
      }
      case 'point': {
        const marker = annotation.marker ?? 'dot';
        const radius = marker === 'dot' ? 4.5 : marker === 'ring' ? 5 : 0;
        const reserve = radius > 0 ? [featureBox(x, y, radius)] : [];
        if (annotation.offset) {
          const [dx, dy] = annotation.offset;
          return {
            candidates: [offsetCandidate(x, y, dx, dy, size)],
            featureX: x,
            featureY: y,
            reserve
          };
        }
        return {
          candidates: directionalCandidates(
            annotation.anchor,
            x,
            y,
            radius + LABEL_GAP,
            size,
            false
          ),
          featureX: x,
          featureY: y,
          reserve
        };
      }
      case 'landform': {
        const radius = annotation.marker === 'peak' ? 5 : 0;
        return {
          candidates: directionalCandidates(
            annotation.anchor,
            x,
            y,
            radius + LABEL_GAP,
            size,
            false
          ),
          featureX: x,
          featureY: y,
          reserve: radius > 0 ? [featureBox(x, y, radius)] : []
        };
      }
      case 'note':
        return {
          candidates: directionalCandidates(
            annotation.anchor,
            x,
            y,
            annotation.distance ?? NOTE_DISTANCE,
            size,
            true
          ),
          featureX: x,
          featureY: y,
          reserve: [featureBox(x, y, 3)]
        };
      default:
        return null;
    }
  }

  // -------------------------------------------------------------------------------------------
  // Writes
  // -------------------------------------------------------------------------------------------

  private writeLabel(entry: Entry, x: number, y: number, round = true): void {
    if (!entry.label) return;
    const value = round
      ? `translate(${Math.round(x)}px, ${Math.round(y)}px)`
      : `translate(${x.toFixed(2)}px, ${y.toFixed(2)}px)`;
    if (value !== entry.lastTransform) {
      entry.lastTransform = value;
      entry.label.style.transform = value;
    }
  }

  private writeGroup(entry: Entry, value: string): void {
    if (entry.group && value !== entry.lastGroupTransform) {
      entry.lastGroupTransform = value;
      entry.group.setAttribute('transform', value);
    }
  }

  private writeShape(entry: Entry, role: string, path: string): void {
    const shape = entry.shapes[role];
    if (shape && entry.shapeData[role] !== path) {
      entry.shapeData[role] = path;
      shape.setAttribute('d', path);
    }
  }

  private setGroupVisible(entry: Entry, visible: boolean): void {
    if (entry.groupVisible === visible) return;
    entry.groupVisible = visible;
    entry.group?.classList.toggle('is-hidden', !visible);
  }

  private setLabelVisible(entry: Entry, visible: boolean): void {
    if (entry.labelVisible === visible) return;
    entry.labelVisible = visible;
    entry.label?.classList.toggle('is-hidden', !visible);
  }

  private hide(entry: Entry): void {
    entry.candidateIndex = -1;
    this.setGroupVisible(entry, false);
    this.setLabelVisible(entry, false);
  }
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

function svgElement(name: string, attributes: Record<string, string>): SVGElement {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

/** The 3 px ink dot that ends a leader at its feature, with a halo underlay. */
function leaderDot(): SVGElement[] {
  return [
    svgElement('circle', {class: 'map-annotation-halo-fill', r: '1.5', 'stroke-width': '2.5'}),
    svgElement('circle', {class: 'map-annotation-ink-fill', r: '1.5'})
  ];
}

function featureBox(x: number, y: number, radius: number): Box {
  return {x: x - radius, y: y - radius, width: 2 * radius, height: 2 * radius};
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function defaultTone(annotation: MapAnnotation): AnnotationTone {
  if (annotation.kind === 'point' && annotation.rank === 'context') return 'muted';
  return DEFAULT_TONES[annotation.kind];
}

/** Fields whose change needs new DOM nodes (everything else updates in place). */
function structureSignature(annotation: MapAnnotation): string {
  const fields = annotation as unknown as Record<string, unknown>;
  return JSON.stringify([
    annotation.kind,
    annotation.tone,
    fields.marker,
    fields.dashed,
    fields.size,
    fields.rank,
    fields.radiusPixels,
    fields.widthPixels,
    fields.offset,
    Boolean(fields.text),
    Boolean(fields.detail)
  ]);
}

/** The label's text parts: one plain part, or named parts styled by class. */
function contentParts(entry: Entry): Part[] {
  const annotation = entry.annotation;
  switch (annotation.kind) {
    case 'point':
      return annotation.detail
        ? [
            {className: 'map-annotation-name', text: annotation.text},
            {className: 'map-annotation-detail', text: annotation.detail}
          ]
        : [{className: '', text: annotation.text}];
    case 'landform': {
      const parts: Part[] = [{className: 'map-annotation-name', text: annotation.text}];
      if (annotation.elevationMeters !== undefined) {
        parts.push({
          className: 'map-annotation-elevation',
          text: `${Math.round(annotation.elevationMeters).toLocaleString('en-US')} m`
        });
      }
      return parts;
    }
    case 'note': {
      const parts: Part[] = [{className: 'map-annotation-note-title', text: annotation.title}];
      if (annotation.text)
        parts.push({className: 'map-annotation-note-body', text: annotation.text});
      return parts;
    }
    case 'marker':
      return [{className: 'map-annotation-marker-numeral', text: String(annotation.number)}];
    case 'dimension':
      return [{className: '', text: entry.dimensionText}];
    case 'area':
    case 'water':
    case 'ring':
    case 'arrow':
    case 'outline':
    case 'bracket':
    case 'star':
    case 'line':
    case 'frame':
    case 'callout':
      return [{className: '', text: annotation.text ?? ''}];
  }
}
