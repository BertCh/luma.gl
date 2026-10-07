// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {TopologicalOptions} from './topological-relations.compute';

const COMBO_LABELS: Record<TopologicalOptions['combo'], {left: string; right: string}> = {
  'tracts-areas': {left: 'tracts', right: 'community areas'},
  'roads-tracts': {left: 'road edges', right: 'tracts'},
  'tracts-self': {left: 'tracts', right: 'neighbouring tracts'}
};

export default defineScene<TopologicalOptions>({
  id: 'topological-relations',
  title: 'Topological relations between layers',
  chapter: 'joins',
  order: 4,
  summary:
    'Exact OGC predicates on the GPU: which tracts nest in community areas, which straddle a boundary, which roads cross tracts, and queen and rook contiguity from DE-9IM patterns. Anti joins, engines and relate matrices included.',
  contributors: ['GPUSpatialPredicateJoin', 'GPUSpatialJoinPrepared'],
  datasets: [
    {id: 'chicago-tracts', role: 'left and right polygons'},
    {id: 'chicago-community-areas', role: 'right polygons'},
    {id: 'chicago-roads', role: 'left lines'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 10},

  options: [
    {
      kind: 'select',
      id: 'combo',
      label: 'Layers',
      group: 'Join',
      apply: 'compile',
      default: 'tracts-areas',
      help: 'Which left features are tested against which right features. Geometry kinds are compile-time: polygons against polygons, lines against polygons, and a polygon self join.',
      options: [
        {value: 'tracts-areas', label: 'Tracts vs community areas (polygon / polygon)'},
        {value: 'roads-tracts', label: 'Roads vs tracts (line / polygon)'},
        {
          value: 'tracts-self',
          label: 'Tracts vs tracts (self join, contiguity)',
          help: 'excludeSameRow keeps a tract from matching itself and the join writes a cross-weights CSR.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'predicate',
      label: 'Predicate',
      group: 'Join',
      apply: 'compile',
      default: 'within',
      help: 'The OGC predicate evaluated as predicate(left, right). It selects the kernel, so changing it rebuilds; the pattern and the distance below do not.',
      options: [
        {
          value: 'intersects',
          label: 'intersects',
          help: 'The closed geometries share at least one point.'
        },
        {
          value: 'within',
          label: 'within',
          help: 'Every point of left is in right and the interiors intersect.'
        },
        {
          value: 'coveredBy',
          label: 'coveredBy',
          help: 'Like within, but boundary contact is allowed.'
        },
        {
          value: 'contains',
          label: 'contains',
          help: 'The reverse of within: right is inside left.'
        },
        {value: 'covers', label: 'covers', help: 'The reverse of coveredBy.'},
        {
          value: 'containsProperly',
          label: 'containsProperly',
          help: 'Right lies in the interior of left with no boundary contact.'
        },
        {
          value: 'touches',
          label: 'touches',
          help: 'Shared boundary points but no shared interior. Queen contiguity for polygons.'
        },
        {
          value: 'overlaps',
          label: 'overlaps',
          help: 'Same dimension, interiors intersect, neither contains the other.'
        },
        {
          value: 'crosses',
          label: 'crosses',
          help: 'Interiors intersect in a lower dimension: a line through a polygon.'
        },
        {value: 'equals', label: 'equals', help: 'The same point set, whatever the vertex order.'},
        {
          value: 'dwithin',
          label: 'dwithin (distance)',
          help: 'The minimum planar distance is at most the per-frame distance below.'
        },
        {
          value: 'relate',
          label: 'relate (DE-9IM pattern)',
          help: 'Any DE-9IM pattern, chosen below. Per-frame, so presets never recompile.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'pattern',
      label: 'DE-9IM pattern',
      group: 'Join',
      apply: 'param',
      default: 'rook',
      disabledWhen: state => state.predicate !== 'relate',
      help: 'A pattern of nine cells (interior, boundary, exterior of left against right) from T, F, *, 0, 1, 2. The patterns are written to a uint32 parameter buffer; an any-of list holds up to four.',
      options: [
        {value: 'rook', label: 'F***1****  shared edge (rook contiguity)'},
        {value: 'queen', label: 'any shared boundary point (queen)'},
        {value: 'corner', label: 'F***0****  corner contact only'},
        {value: 'interiors', label: 'T********  interiors intersect'},
        {value: 'straddle', label: 'T*T***T**  overlap on both sides'},
        {value: 'inside', label: 'T*F**F***  completely inside'},
        {value: 'crossing', label: 'T*T******  through and out'}
      ]
    },
    {
      kind: 'slider',
      id: 'distance',
      label: 'dwithin distance',
      group: 'Join',
      apply: 'param',
      min: 0,
      max: 500,
      step: 5,
      default: 25,
      unit: 'm',
      disabledWhen: state => state.predicate !== 'dwithin',
      help: 'Per-frame float. The candidate buffer is sized for 500 m. Zero distance equals intersects.'
    },
    {
      kind: 'select',
      id: 'engine',
      label: 'Engine',
      group: 'Join',
      apply: 'compile',
      default: 'auto',
      disabledWhen: state =>
        !['intersects', 'contains', 'within', 'dwithin'].includes(state.predicate),
      help: 'Fast is a short-circuiting kernel (best for small features); relate is the DE-9IM engine (3 to 15 times faster for features with about 32 or more vertices); auto picks by vertex counts. Other predicates always use relate.',
      options: [
        {value: 'auto', label: 'Auto'},
        {value: 'fast', label: 'Fast kernel'},
        {value: 'relate', label: 'Relate engine'}
      ]
    },
    {
      kind: 'toggle',
      id: 'matrix',
      label: 'Output DE-9IM matrices',
      group: 'Join',
      apply: 'compile',
      default: false,
      disabledWhen: state => state.predicate === 'dwithin',
      help: 'Also writes the nine-cell matrix of every matched pair. Requesting it makes every predicate run on the relate engine.'
    },
    {
      kind: 'select',
      id: 'show',
      label: 'Colour left features by',
      group: 'Display',
      apply: 'param',
      default: 'status',
      help: 'The join runs both ways every time: the inner join (matches) and the anti join (left features with no match). This choice only decides which result is drawn.',
      options: [
        {value: 'status', label: 'Has a match (inner join)'},
        {value: 'anti', label: 'Has no match (anti join)'},
        {value: 'count', label: 'Number of matches'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Count ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.show !== 'count',
      help: 'Ramp for the number of matches.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (colour-blind optimised)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showRight',
      label: 'Outline the right features',
      group: 'Display',
      apply: 'param',
      default: true,
      disabledWhen: state => state.combo !== 'tracts-areas',
      help: 'Community area boundaries, drawn from the polygon buffer the join reads.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Fill opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'Lower it to read street names under the tracts.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time fast vs relate engine',
      group: 'Compare',
      help: 'Builds the current predicate with both engines and times them outside the frame.'
    }
  ],

  readouts: [
    {id: 'left', label: 'Left features'},
    {id: 'right', label: 'Right features'},
    {
      id: 'pairs',
      label: 'Matched pairs',
      help: 'Pairs sorted by (left, right), read back once after the map settles.'
    },
    {
      id: 'anti',
      label: 'Anti join',
      help: 'Left features with no match. It should be exactly the complement of the matched left features.'
    },
    {
      id: 'candidates',
      label: 'Candidates vs exact',
      help: 'Bounding-box candidate pairs from the BVH and how many pass the exact predicate.'
    },
    {
      id: 'flags',
      label: 'Overflow and uncertainty',
      help: 'Uncertain pairs could not be decided exactly; 0 means every decision is certified.'
    },
    {id: 'engine', label: 'Engine path'},
    {
      id: 'matrices',
      label: 'Most common matrices',
      help: 'DE-9IM strings of the matched pairs, such as 212101212 for polygons that overlap.'
    },
    {
      id: 'selection',
      label: 'Selected feature',
      help: 'Click a left feature to list its matches and their matrices.'
    },
    {id: 'contiguity', label: 'Contiguity weights'},
    {id: 'parity', label: 'Reference (libpysal)'},
    {
      id: 'indexBuilds',
      label: 'Right-index builds',
      help: 'GPUSpatialJoinPrepared.encodedBuildCount against the number of join runs.'
    },
    {id: 'timing', label: 'Engine timing'}
  ],

  legends: state => {
    const labels = COMBO_LABELS[state.combo];
    const legends: LegendSpec[] = [];
    if (state.show === 'count') {
      legends.push({
        kind: 'ramp',
        title: `Matches per ${labels.left.replace(/s$/, '')}`,
        ramp: state.ramp,
        extent: [0, 8],
        unit: 'matches',
        format: value => Math.round(value).toString(),
        labels: ['none', '8 or more']
      });
    } else {
      legends.push({
        kind: 'categories',
        title: `${labels.left} (${state.predicate === 'relate' ? 'relate' : state.predicate})`,
        entries: [
          {
            color: [255, 150, 40, 235],
            label: state.show === 'anti' ? 'No match (anti join)' : 'Has a match'
          },
          {color: [120, 135, 160, 160], label: state.show === 'anti' ? 'Has a match' : 'No match'},
          {color: [40, 200, 255, 255], label: 'Selected (click)'},
          ...(state.combo === 'tracts-self'
            ? [{color: [90, 220, 120, 235] as const, label: 'Matches of the selection'}]
            : [])
        ]
      });
    }
    if (state.combo !== 'tracts-self') {
      legends.push({
        kind: 'categories',
        title: 'Right features',
        entries: [
          {color: [150, 80, 0, 220], label: labels.right},
          {color: [60, 230, 255, 255], label: 'Matches of the selected feature'}
        ]
      });
    }
    return legends;
  },

  snippet: state => {
    const left =
      state.combo === 'roads-tracts'
        ? "{kind: 'lines', positions, lineOffsets}"
        : "{kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets}";
    const right = state.combo === 'tracts-areas' ? 'communityAreas' : 'tracts';
    return `import {
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin,
  packGPUSpatialRelatePattern,
  formatGPUSpatialRelate
} from '@luma.gl/experimental/gpu-spatial-analysis';

const prepared = new GPUSpatialJoinPrepared({geometry: ${right}});
graph.add(prepared);                         // the right-hand BVH, built once
graph.add(
  new GPUSpatialPredicateJoin({
    left: ${left},
    right: ${right},
    predicate: '${state.predicate}',${
      state.predicate === 'relate'
        ? `\n    pattern: patternBuffer.importToGraph(graph), // per frame; packGPUSpatialRelatePattern(${JSON.stringify(state.pattern === 'queen' ? ['FT*******', 'F**T*****', 'F***T****'] : state.pattern === 'rook' ? ['F***1****'] : '...')}, 4)`
        : state.predicate === 'dwithin'
          ? '\n    distance: distanceBuffer.importToGraph(graph), // per frame float32'
          : ''
    }${['intersects', 'contains', 'within', 'dwithin'].includes(state.predicate) ? `\n    engine: '${state.engine}',` : ''}${state.combo === 'tracts-self' ? '\n    excludeSameRow: true,' : ''}
    candidateCapacity,
    prepared,
    pairs: {leftIds, rightIds, count, overflow, totalCount},${state.matrix && state.predicate !== 'dwithin' ? '\n    relate: matrices,                         // formatGPUSpatialRelate(word)' : ''}${state.combo === 'tracts-self' ? '\n    weights: {offsets, neighbors, weights},   // cross weights, CSR' : ''}
    uncertainCount, candidateCount
  })
);
graph.add(new GPUSpatialPredicateJoin({... how: 'anti', unmatched: {ids, count, overflow}}));
const compiled = graph.compile();            // once
compiled.encode(commandEncoder, {parameters: undefined});`;
  },

  about: {
    what: "`GPUSpatialPredicateJoin` joins two feature sets by an exact OGC predicate: all nine point, line and polygon combinations, evaluated with GEOS-compatible DE-9IM semantics. A BVH over the right side (prepared once with `GPUSpatialJoinPrepared`) yields bounding-box candidates, exact orientation tests decide them, and the result is a sorted pair table. `how: 'anti'` returns the left features that match nothing.",
    why: 'Overlay questions are topological: does this parcel lie within a zone, which roads cross a river, which counties touch. Exact predicates also catch data problems: a boundary that almost, but not quite, coincides makes a "within" fail where the eye sees nesting.',
    howToRead:
      'Orange features have a match under the predicate, gray ones do not; switch to the anti join to colour the opposite set. Click a left feature: its matches are outlined in cyan and the panel lists them with their DE-9IM matrices. The matrix reads row by row: interior, boundary, exterior of left against right.'
  },

  create: async ctx =>
    (await import('./topological-relations.compute')).createTopologicalRelations(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['combo', 'predicate', 'show'],
      readouts: ['pairs', 'candidates'],
      title: 'Do census tracts nest inside community areas?',
      body: 'Chicago has **791 census tracts** and **77 community areas**. Statistics are published for both, so analysts often assume each tract sits inside one area. **`GPUSpatialPredicateJoin`** tests that exactly: for every tract and every area it asks whether the tract is `within` the area, using the OGC definition and exact orientation tests. **Layers**, **Predicate** and **Colour left features by** below change the question.\n\nOrange tracts have a match: they are within one community area. Gray tracts are not within any, even where they look nested. Read **Candidates vs exact**: the BVH proposes bounding-box pairs and the exact predicate keeps a fraction of them.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10},
      options: {combo: 'tracts-areas', predicate: 'within', show: 'status'},
      highlight: {readout: 'pairs'}
    },
    {
      id: 'anti-join',
      controls: ['show'],
      readouts: ['anti', 'selection'],
      title: 'The anti join: tracts that nest nowhere',
      body: 'The same join also runs in **anti** mode and returns the left features with **no match**. It is the complement of the matched tracts, and the **Anti join** readout checks that on the CPU. Set **Colour left features by** to *Has no match (anti join)* to see the tracts that are not within any community area.\n\nCensus tracts and the city’s community areas come from different sources and the area boundaries are simplified, so boundaries that follow the same street need not coincide exactly and an exact `within` can fail on small slivers. Click a gray tract: its matches (if any) are outlined in cyan.',
      options: {show: 'anti'}
    },
    {
      id: 'straddlers',
      controls: ['predicate', 'matrix'],
      readouts: ['matrices', 'selection', 'engine'],
      title: 'overlaps finds the straddlers',
      body: 'Set **Predicate** to *overlaps*. A tract overlaps an area when their interiors intersect, neither contains the other and both are polygons: the tract straddles the area boundary. A tract can overlap two areas.\n\nTurn on **Output DE-9IM matrices** and click a straddling tract. The matrix of an overlapping pair has interior and exterior intersections of dimension 2 in both directions; **Most common matrices** lists the patterns across the whole join. Requesting the matrix forces the relate engine.',
      options: {predicate: 'overlaps', show: 'status', matrix: true}
    },
    {
      id: 'queen-contiguity',
      controls: ['combo', 'predicate', 'show'],
      readouts: ['contiguity', 'parity'],
      title: 'Queen contiguity is the touches predicate',
      body: 'Set **Layers** to *Tracts vs tracts (self join, contiguity)* and **Predicate** to *touches*. Touching (shared boundary points, no shared interior) is exactly queen contiguity, the neighbour definition behind spatial weights. `excludeSameRow` keeps a tract from matching itself, and the join also writes a cross-weights CSR.\n\nSet **Colour left features by** to *Number of matches* to see each tract’s degree. The readout reports the mean number of neighbours and the islands; libpysal on the same tracts gives 6.6 neighbours and one island (the O’Hare tract). Click a tract to see its neighbours in green.',
      options: {combo: 'tracts-self', predicate: 'touches', show: 'count', matrix: false}
    },
    {
      id: 'rook-from-a-pattern',
      controls: ['predicate', 'pattern'],
      readouts: ['contiguity', 'parity'],
      title: 'Rook contiguity from a DE-9IM pattern',
      body: 'Rook contiguity requires a shared **edge**, not just a corner: boundaries must meet in a line. No named predicate says that, but a DE-9IM pattern does: **`F***1****`** means the boundaries intersect in a one-dimensional set. Choose the *relate (DE-9IM pattern)* **Predicate** and the rook preset of **DE-9IM pattern**.\n\nThe pattern lives in a uint32 parameter buffer, so cycling through the presets (rook, queen, corner only) rewrites eight words and never rebuilds the graph. Expect about 4.7 neighbours per tract, one fewer neighbour than queen on average, with the same island.',
      options: {predicate: 'relate', pattern: 'rook', show: 'count'}
    },
    {
      id: 'roads-cross-tracts',
      controls: ['predicate', 'show'],
      readouts: ['pairs', 'engine'],
      title: 'Roads against tracts: crosses and within',
      body: 'Now test the road edges (motorway to tertiary, one direction each) against the tracts. A road **crosses** a tract when its interior passes through the tract and extends outside; it is **within** when the whole edge lies in one tract. Tract boundaries often follow streets, so many roads touch boundaries rather than cross them.\n\nTry intersects, within and crosses under **Predicate**, and the anti join under **Colour left features by**. The line/polygon engine is the DE-9IM relate engine; the fast kernel is available for intersects, contains, within and dwithin.',
      camera: {longitude: -87.66, latitude: 41.87, zoom: 11.2},
      options: {combo: 'roads-tracts', predicate: 'crosses', show: 'status', matrix: false}
    },
    {
      id: 'distance-and-limits',
      controls: ['predicate', 'distance', 'measure'],
      readouts: ['timing', 'flags'],
      title: 'A distance instead of topology, and the limits',
      body: '**dwithin** adds a tolerance: the minimum planar distance between the geometries is at most the per-frame distance. Slide **dwithin distance**: the distance is one float, so no rebuild. It is the right tool when "touches" is too strict for boundaries that nearly coincide.\n\nPress **Time fast vs relate engine** to compare the two kernels. **Limits:** coordinates are float32 meters around the city centre, polygons are assumed valid (holes inside shells, no self-crossings) and a candidate buffer that is too small sets the overflow flag. Try overlaps on the self join, or covers on tracts against areas (**Layers**).',
      options: {combo: 'roads-tracts', predicate: 'dwithin', distance: 30, show: 'status'}
    }
  ]
});
