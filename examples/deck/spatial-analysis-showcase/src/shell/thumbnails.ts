// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSeededRandom} from '../engine/projection';

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';
const WIDTH = 240;
const HEIGHT = 110;

type Draw = (random: () => number, add: Add) => void;
type Add = <Tag extends keyof SVGElementTagNameMap>(
  tag: Tag,
  attributes: Record<string, string | number>
) => SVGElementTagNameMap[Tag];

const chart = (index: number) => `var(--chart-${index})`;

/** FNV-1a hash of a string, for a per-scene seed. */
function hashString(text: string): number {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

const MOTIFS: Record<string, Draw> = {
  points: (random, add) => {
    for (let cluster = 0; cluster < 3; cluster++) {
      const cx = 40 + random() * 160;
      const cy = 25 + random() * 60;
      add('circle', {cx, cy, r: 28, fill: chart(1), 'fill-opacity': 0.1});
      add('circle', {cx, cy, r: 16, fill: chart(1), 'fill-opacity': 0.14});
      for (let index = 0; index < 22; index++) {
        const angle = random() * Math.PI * 2;
        const radius = Math.abs(random() + random() - 1) * 26;
        add('circle', {
          cx: cx + Math.cos(angle) * radius,
          cy: cy + Math.sin(angle) * radius * 0.8,
          r: 1.6,
          fill: chart(1)
        });
      }
    }
  },
  joins: (random, add) => {
    const jitter = () => random() * 8;
    add('polygon', {
      points: `30,${30 + jitter()} 120,20 135,${80 + jitter()} 40,90`,
      fill: chart(1),
      'fill-opacity': 0.22,
      stroke: chart(1),
      'stroke-width': 1.5
    });
    add('polygon', {
      points: `105,28 205,${35 + jitter()} 195,92 100,${75 + jitter()}`,
      fill: chart(2),
      'fill-opacity': 0.22,
      stroke: chart(2),
      'stroke-width': 1.5
    });
    for (let index = 0; index < 26; index++) {
      const x = 28 + random() * 180;
      const y = 22 + random() * 70;
      const inside = x > 105 && x < 135;
      add('circle', {
        cx: x,
        cy: y,
        r: inside ? 2.6 : 1.8,
        fill: inside ? chart(3) : 'var(--muted)'
      });
    }
  },
  geometry: (random, add) => {
    const points: [number, number][] = [];
    for (let index = 0; index < 14; index++) {
      const angle = (index / 14) * Math.PI * 2;
      const radius = 34 + random() * 14;
      points.push([120 + Math.cos(angle) * radius * 1.6, 55 + Math.sin(angle) * radius]);
    }
    add('polygon', {
      points: points.map(point => point.join(',')).join(' '),
      fill: chart(1),
      'fill-opacity': 0.16,
      stroke: chart(1),
      'stroke-width': 1.5
    });
    add('polygon', {
      points: points
        .filter((_, index) => index % 3 === 0)
        .map(point => point.join(','))
        .join(' '),
      fill: 'none',
      stroke: chart(2),
      'stroke-width': 1.5,
      'stroke-dasharray': '4 3'
    });
    for (const [x, y] of points) add('circle', {cx: x, cy: y, r: 2.2, fill: chart(1)});
  },
  weights: (random, add) => {
    const columns = 7;
    const rows = 4;
    const nodes: [number, number][] = [];
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        nodes.push([28 + column * 31 + random() * 6, 22 + row * 21 + random() * 6]);
      }
    }
    const center = nodes[1 * columns + 3];
    nodes.forEach(([x, y], index) => {
      const near = Math.hypot(x - center[0], y - center[1]) < 40 && index !== 1 * columns + 3;
      if (near) {
        add('line', {
          x1: center[0],
          y1: center[1],
          x2: x,
          y2: y,
          stroke: chart(2),
          'stroke-width': 1.4
        });
      }
    });
    nodes.forEach(([x, y], index) => {
      const isCenter = index === 1 * columns + 3;
      const near = Math.hypot(x - center[0], y - center[1]) < 40;
      add('circle', {
        cx: x,
        cy: y,
        r: isCenter ? 5 : 3,
        fill: isCenter ? chart(2) : near ? chart(1) : 'var(--muted)',
        'fill-opacity': isCenter || near ? 1 : 0.5
      });
    });
  },
  statistics: (random, add) => {
    const bars = 14;
    for (let index = 0; index < bars; index++) {
      const value = Math.exp(-(((index - 6.5 + random() * 1.4) / 3.6) ** 2));
      const height = 8 + value * 70;
      add('rect', {
        x: 22 + index * 14.5,
        y: 96 - height,
        width: 12,
        height,
        rx: 2,
        fill: chart(index < 4 ? 1 : index < 9 ? 3 : 2),
        'fill-opacity': 0.85
      });
    }
    for (const x of [78, 134, 178]) {
      add('line', {
        x1: x,
        y1: 14,
        x2: x,
        y2: 98,
        stroke: 'var(--text)',
        'stroke-opacity': 0.4,
        'stroke-dasharray': '3 3'
      });
    }
  },
  regression: (random, add) => {
    const slope = 0.35 + random() * 0.3;
    add('line', {
      x1: 20,
      y1: 90,
      x2: 220,
      y2: 90 - slope * 200 * 0.45,
      stroke: chart(2),
      'stroke-width': 2
    });
    for (let index = 0; index < 34; index++) {
      const x = 24 + random() * 190;
      const y = 90 - (x - 20) * slope * 0.45 + (random() - 0.5) * 30;
      add('circle', {cx: x, cy: y, r: 2.2, fill: chart(1)});
    }
  },
  interpolation: (random, add) => {
    const samples = Array.from({length: 6}, () => ({
      x: 25 + random() * 190,
      y: 18 + random() * 76,
      value: random()
    }));
    const size = 10;
    for (let x = 14; x < 226; x += size) {
      for (let y = 8; y < 102; y += size) {
        let weight = 0;
        let total = 0;
        for (const sample of samples) {
          const w = 1 / (1 + (Math.hypot(sample.x - x, sample.y - y) / 24) ** 2);
          weight += w * sample.value;
          total += w;
        }
        add('rect', {
          x,
          y,
          width: size,
          height: size,
          fill: chart(1),
          'fill-opacity': (0.08 + 0.8 * (weight / total)).toFixed(2)
        });
      }
    }
    for (const sample of samples) {
      add('circle', {
        cx: sample.x,
        cy: sample.y,
        r: 3,
        fill: 'var(--surface)',
        stroke: chart(2),
        'stroke-width': 1.8
      });
    }
  },
  cells: (random, add) => {
    const radius = 13;
    const height = Math.sqrt(3) * radius;
    for (let column = 0; column < 11; column++) {
      for (let row = 0; row < 5; row++) {
        const cx = 20 + column * radius * 1.5;
        const cy = 18 + row * height + (column % 2 ? height / 2 : 0);
        const points = Array.from({length: 6}, (_, corner) => {
          const angle = (Math.PI / 3) * corner;
          return `${(cx + Math.cos(angle) * (radius - 1)).toFixed(1)},${(cy + Math.sin(angle) * (radius - 1)).toFixed(1)}`;
        }).join(' ');
        add('polygon', {
          points,
          fill: chart(1),
          'fill-opacity': (0.05 + random() ** 2 * 0.75).toFixed(2),
          stroke: chart(1),
          'stroke-opacity': 0.4,
          'stroke-width': 0.8
        });
      }
    }
  },
  networks: (random, add) => {
    const nodes = Array.from({length: 16}, (_, index) => ({
      x: 22 + (index % 4) * 62 + random() * 22,
      y: 16 + Math.floor(index / 4) * 25 + random() * 10
    }));
    const edges: [number, number][] = [];
    nodes.forEach((_, index) => {
      if (index % 4 < 3) edges.push([index, index + 1]);
      if (index < 12) edges.push([index, index + 4]);
    });
    for (const [a, b] of edges) {
      add('line', {
        x1: nodes[a].x,
        y1: nodes[a].y,
        x2: nodes[b].x,
        y2: nodes[b].y,
        stroke: 'var(--muted)',
        'stroke-opacity': 0.5
      });
    }
    const path = [0, 1, 5, 6, 10, 11, 15];
    add('polyline', {
      points: path.map(index => `${nodes[index].x},${nodes[index].y}`).join(' '),
      fill: 'none',
      stroke: chart(2),
      'stroke-width': 2.6,
      'stroke-linejoin': 'round'
    });
    nodes.forEach((node, index) =>
      add('circle', {
        cx: node.x,
        cy: node.y,
        r: path.includes(index) ? 3.6 : 2.4,
        fill: path.includes(index) ? chart(2) : 'var(--muted)'
      })
    );
  },
  flows: (random, add) => {
    const hubs = Array.from({length: 6}, () => ({x: 26 + random() * 188, y: 22 + random() * 66}));
    for (let index = 0; index < 11; index++) {
      const a = hubs[Math.floor(random() * hubs.length)];
      const b = hubs[Math.floor(random() * hubs.length)];
      if (a === b) continue;
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2 - Math.hypot(a.x - b.x, a.y - b.y) * 0.3;
      add('path', {
        d: `M${a.x} ${a.y}Q${mx} ${my} ${b.x} ${b.y}`,
        fill: 'none',
        stroke: chart(1 + (index % 3)),
        'stroke-width': 1 + random() * 2.4,
        'stroke-opacity': 0.7,
        'stroke-linecap': 'round'
      });
    }
    for (const hub of hubs)
      add('circle', {cx: hub.x, cy: hub.y, r: 3.4, fill: 'var(--text)', 'fill-opacity': 0.75});
  },
  movement: (random, add) => {
    for (let track = 0; track < 5; track++) {
      let x = 18 + random() * 40;
      let y = 20 + track * 17;
      let heading = (random() - 0.5) * 0.8;
      let d = `M${x} ${y}`;
      for (let step = 0; step < 16; step++) {
        heading += (random() - 0.5) * 0.9;
        x += 11 * Math.cos(heading * 0.6);
        y += 11 * Math.sin(heading);
        y = Math.min(100, Math.max(10, y));
        d += `L${x.toFixed(1)} ${y.toFixed(1)}`;
      }
      add('path', {
        d,
        fill: 'none',
        stroke: chart(1 + (track % 4)),
        'stroke-width': 1.6,
        'stroke-opacity': 0.8,
        'stroke-linejoin': 'round'
      });
      add('circle', {cx: x, cy: y, r: 3, fill: chart(1 + (track % 4))});
    }
  },
  time: (random, add) => {
    for (let slice = 0; slice < 4; slice++) {
      const top = 10 + slice * 24;
      let d = '';
      for (let step = 0; step <= 30; step++) {
        const y =
          top + 18 - (Math.sin(step / 3.2 + slice) * 0.5 + 0.5) * 12 * (0.6 + random() * 0.4);
        d += `${step ? 'L' : 'M'}${20 + step * 6.6} ${y.toFixed(1)}`;
      }
      add('path', {
        d,
        fill: 'none',
        stroke: chart(1 + (slice % 3)),
        'stroke-width': 1.6,
        'stroke-linejoin': 'round'
      });
      add('line', {x1: 20, x2: 218, y1: top + 20, y2: top + 20, stroke: 'var(--border)'});
    }
    add('line', {x1: 150, x2: 150, y1: 8, y2: 104, stroke: chart(2), 'stroke-width': 1.5});
  },
  terrain: (random, add) => {
    const cx = 100 + random() * 40;
    const cy = 45 + random() * 20;
    for (let ring = 7; ring >= 1; ring--) {
      add('ellipse', {
        cx: cx + (7 - ring) * 1.5,
        cy: cy - (7 - ring) * 0.8,
        rx: ring * 15 + random() * 5,
        ry: ring * 7.5 + random() * 3,
        fill: chart(1),
        'fill-opacity': (0.05 + (7 - ring) * 0.045).toFixed(2),
        stroke: chart(1),
        'stroke-opacity': 0.6,
        'stroke-width': 0.9
      });
    }
    add('polygon', {
      points: `${cx - 4},${cy + 2} ${cx},${cy - 7} ${cx + 4},${cy + 2}`,
      fill: chart(2)
    });
  },
  hydrology: (random, add) => {
    const branch = (
      x: number,
      y: number,
      angle: number,
      length: number,
      width: number,
      depth: number
    ) => {
      if (depth === 0 || length < 6) return;
      const x2 = x + Math.cos(angle) * length;
      const y2 = y + Math.sin(angle) * length;
      add('line', {
        x1: x,
        y1: y,
        x2,
        y2,
        stroke: chart(1),
        'stroke-width': width,
        'stroke-linecap': 'round'
      });
      branch(x2, y2, angle - 0.5 - random() * 0.4, length * 0.75, width * 0.72, depth - 1);
      branch(x2, y2, angle + 0.5 + random() * 0.4, length * 0.72, width * 0.72, depth - 1);
    };
    branch(120, 104, -Math.PI / 2, 26, 4.2, 5);
  },
  raster: (random, add) => {
    const size = 12;
    for (let column = 0; column < 18; column++) {
      for (let row = 0; row < 8; row++) {
        const value = 0.5 + 0.5 * Math.sin(column / 3 + random() * 0.6) * Math.cos(row / 2.2);
        add('rect', {
          x: 12 + column * size,
          y: 8 + row * size,
          width: size,
          height: size,
          fill: chart(value > 0.55 ? 3 : 1),
          'fill-opacity': (0.12 + value * 0.7).toFixed(2)
        });
      }
    }
  },
  dataframe: (random, add) => {
    for (let row = 0; row < 6; row++) {
      const y = 12 + row * 15;
      add('rect', {
        x: 20,
        y,
        width: 200,
        height: 11,
        rx: 2,
        fill: row === 0 ? chart(1) : 'var(--muted)',
        'fill-opacity': row === 0 ? 0.3 : 0.1
      });
      if (row > 0) {
        add('rect', {
          x: 26,
          y: y + 3,
          width: 30,
          height: 5,
          rx: 1.5,
          fill: 'var(--muted)',
          'fill-opacity': 0.5
        });
        add('rect', {
          x: 70,
          y: y + 3,
          width: 20 + random() * 120,
          height: 5,
          rx: 1.5,
          fill: chart(1 + (row % 3))
        });
      }
    }
  }
};

/**
 * A generated card thumbnail: a chapter motif drawn with the theme's chart colors and varied by a
 * seed derived from the scene id, so cards in one chapter look related but not identical.
 */
export function createThumbnail(chapterId: string, sceneId: string): SVGSVGElement {
  const root = document.createElementNS(SVG_NAMESPACE, 'svg');
  root.setAttribute('viewBox', `0 0 ${WIDTH} ${HEIGHT}`);
  root.setAttribute('preserveAspectRatio', 'xMidYMid slice');
  root.setAttribute('class', 'thumb-art');
  root.setAttribute('aria-hidden', 'true');
  const add: Add = (tag, attributes) => {
    const element = document.createElementNS(SVG_NAMESPACE, tag);
    for (const [name, value] of Object.entries(attributes))
      element.setAttribute(name, String(value));
    root.append(element);
    return element;
  };
  const random = createSeededRandom(hashString(`${chapterId}/${sceneId}`));
  (MOTIFS[chapterId] ?? MOTIFS['points'])(random, add);
  return root;
}
