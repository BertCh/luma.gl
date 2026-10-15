// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';
import {CELLS_FORMULAS} from './cells-formulas';

/**
 * Spatial computation-pack entries for the `cells` family, keyed by exported name. Every entry
 * must use `family: 'cells'`.
 */
export const CELLS_OPERATORS: SpatialFamilyOperators = {
  GPUCellAggregation: {
    family: 'cells',
    descriptor: {scope: 'zonal', reduction: 'sum'},
    variables: {
      cells: 'category',
      counts: 'count',
      sums: 'measure',
      sumValues: 'measure',
      minimums: 'score',
      maximums: 'score'
    },
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'cell', label: 'Occupied cell and its accepted rows', indexSpace: '1d'}],
    representation: {from: 'event', to: 'field'},
    guards: ['conservation', 'count-vs-rate'],
    choices: [
      {stage: 'support', name: 'family', alternatives: ['quadbin', 'h3'], sweepable: false},
      {stage: 'support', name: 'resolution', sweepable: true},
      {stage: 'operator', name: 'sumScale', sweepable: true}
    ],
    formula: CELLS_FORMULAS['aggregation'],
    metadata: {
      conservingReductions: 'sum,count',
      nonConservingReductions: 'mean,min,max',
      conservationDomain:
        'Accepted rows after masking, key validation and finite-value filtering; no capacity loss or integer overflow; sums conserve quantized, saturated values.',
      valueKinds:
        'sums and sumValues describe extensive inputs; minimums and maximums retain the input scale and are not additive',
      pointKeyingFamily: 'quadbin',
      exactness: 'integer counts and signed 64-bit fixed-point sums'
    }
  },
  GPUCellRollup: {
    family: 'cells',
    descriptor: {scope: 'block', scopeSecondary: 'zonal', reduction: 'sum'},
    variables: {
      cells: 'category',
      counts: 'count',
      sums: 'measure',
      sumValues: 'measure',
      minimums: 'score',
      maximums: 'score'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'cell', label: 'Parent cell and its child table rows', indexSpace: '1d'}],
    guards: ['conservation', 'count-vs-rate'],
    choices: [
      {stage: 'support', name: 'family', alternatives: ['quadbin', 'h3'], sweepable: false},
      {stage: 'support', name: 'sourceResolution', sweepable: true},
      {stage: 'support', name: 'resolution', sweepable: true},
      {
        stage: 'operator',
        name: 'sumScale',
        label: 'Scale already used by source sums',
        sweepable: false
      }
    ],
    formula: CELLS_FORMULAS['rollup'],
    metadata: {
      conservingReductions: 'sum,count',
      nonConservingReductions: 'mean,min,max',
      conservationDomain:
        'Valid source table at sourceResolution; no source or output capacity loss or integer overflow; sums use the source fixed-point scale.',
      valueKinds: 'sums and sumValues describe extensive inputs; extremes are non-additive',
      exactness: 'integer counts and signed 64-bit fixed-point sums'
    }
  },
  GPUCellPyramid: {
    family: 'cells',
    descriptor: {scope: 'block', scopeSecondary: 'zonal', reduction: 'sum'},
    variables: {
      'levels.cells': 'category',
      'levels.counts': 'count',
      'levels.sums': 'measure',
      'levels.sumValues': 'measure',
      'levels.minimums': 'score',
      'levels.maximums': 'score',
      levelCounts: 'count'
    },
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'cell', label: 'Cell within a pyramid level', indexSpace: 'batch'}],
    representation: {from: 'event', to: 'field'},
    guards: ['conservation', 'count-vs-rate'],
    choices: [
      {stage: 'support', name: 'family', alternatives: ['quadbin', 'h3'], sweepable: false},
      {
        stage: 'support',
        name: 'resolutions',
        label: 'Strictly decreasing resolutions, finest first',
        sweepable: true
      },
      {stage: 'operator', name: 'sumScale', sweepable: true}
    ],
    formula: CELLS_FORMULAS['pyramid'],
    metadata: {
      conservingReductions: 'sum,count',
      nonConservingReductions: 'mean,min,max',
      conservationDomain:
        'Compare totals between individual levels, never sum across levels; accepted input rows, no capacity loss at any level or integer overflow; quantized, saturated sums.',
      valueKinds: 'level sums describe extensive inputs; extremes are non-additive',
      pointKeyingFamily: 'quadbin',
      exactness: 'integer counts and signed 64-bit fixed-point sums'
    }
  },
  GPUCellLevelSelection: {
    family: 'cells',
    descriptor: {scope: 'local', reduction: 'select'},
    variables: {'status.count': 'count', firstRow: 'rank', drawArguments: 'count'},
    stages: ['support', 'presentation'],
    datumKinds: [{kind: 'cell', label: 'Selected level cell range', indexSpace: 'batch'}],
    guards: [],
    choices: [{stage: 'support', name: 'activeLevel', sweepable: true}],
    formula: CELLS_FORMULAS['selection'],
    metadata: {drawArgumentWrites: 'instanceCount and firstInstance only'}
  },
  GPUPointToCell: {
    family: 'cells',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {cells: 'category', validity: 'category'},
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'feature', label: 'Input point and its cell key', indexSpace: '1d'}],
    representation: {from: 'event', to: 'field'},
    guards: [],
    choices: [
      {
        stage: 'support',
        name: 'family',
        alternatives: ['quadbin', 'h3', 'quadkey', 'geohash', 's2'],
        sweepable: false
      },
      {stage: 'support', name: 'resolution', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Family-specific Mercator quantization, bit interleaving, S2 Hilbert traversal and H3 face/IJK digit construction produce keys procedurally; the kernel rung shows the selected encoder.'
    },
    metadata: {
      coordinateModel: 'geographic',
      metric: 'none',
      a5ForwardIndexing: false,
      exactness:
        'integer keying for quadbin, quadkey and geohash; float32 sphere math for H3 and S2'
    }
  },
  GPUCellGeometry: {
    family: 'cells',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {centers: 'coordinate', boundaries: 'coordinate', vertexCounts: 'count'},
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'cell', label: 'Cell center and boundary vertices', indexSpace: '1d'}],
    guards: [],
    choices: [
      {
        stage: 'support',
        name: 'family',
        alternatives: ['quadbin', 'h3', 'quadkey', 'geohash', 's2', 'a5'],
        sweepable: false
      },
      {
        stage: 'representation',
        name: 'maximumVertexCount',
        label: 'Boundary storage stride; smaller strides truncate',
        sweepable: false
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Family-specific key decoding constructs centers and boundary vertices, including H3 face-crossing insertions and A5 shapes; the kernel rung exposes these branches and stride truncation.'
    },
    metadata: {
      coordinateModel: 'geographic',
      metric: 'none',
      exactness: 'float32',
      edgeModels: 'great-circle: h3,s2,a5; linear: quadbin,quadkey,geohash',
      a5BoundaryVertices: '3 at resolution 1, otherwise 5'
    }
  },
  GPUCellTopology: {
    family: 'cells',
    // Neighbors are generated from one key, without reading neighboring input rows.
    descriptor: {scope: 'local', scopeSecondary: 'block', reduction: 'none'},
    variables: {cells: 'category', distances: 'count', counts: 'count'},
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'cell', label: 'Input cell and fixed fan-out row', indexSpace: '1d'}],
    guards: [],
    choices: [
      {stage: 'support', name: 'family', alternatives: ['quadbin', 'h3'], sweepable: false},
      {stage: 'support', name: 'operation.resolution', sweepable: true},
      {stage: 'support', name: 'operation.inputResolution', sweepable: true},
      {
        stage: 'neighbourhood',
        name: 'operation.type',
        alternatives: ['disk', 'ring', 'parent', 'children'],
        sweepable: false
      },
      {stage: 'neighbourhood', name: 'operation.k', unit: 'grid steps', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Bounded H3 breadth-first neighbor expansion, wrapped Quadbin enumeration and hierarchy digit traversal produce sorted fan-out rows; the kernel rung shows the operation-specific construction.'
    },
    metadata: {
      exactness: 'integer',
      distanceModel: 'H3 grid steps; Quadbin wrapped-column Chebyshev steps',
      conditionalChoices:
        'operation.k: disk/ring; operation.resolution: parent/children; operation.inputResolution: children'
    }
  },
  GPUCellCompaction: {
    family: 'cells',
    descriptor: {scope: 'global', scopeSecondary: 'block', reduction: 'scan-sort'},
    variables: {cells: 'category', 'status.count': 'count', 'status.droppedCount': 'count'},
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'cell', label: 'Cell in the compacted or expanded set', indexSpace: '1d'}],
    guards: [],
    choices: [
      {stage: 'support', name: 'family', alternatives: ['quadbin', 'h3'], sweepable: false},
      {stage: 'support', name: 'operation.resolution', sweepable: true},
      {stage: 'support', name: 'operation.minimumResolution', sweepable: true},
      {stage: 'support', name: 'operation.maximumDepth', sweepable: true},
      {
        stage: 'operator',
        name: 'operation.type',
        alternatives: ['compact', 'uncompact'],
        sweepable: false
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Sorting and deduplication precede bounded ancestor-completeness searches for compact, while uncompact counts, scans and enumerates descendants; the graph rung shows the passes rather than a closed-form output.'
    },
    metadata: {
      exactness: 'integer',
      conditionalChoices: 'minimumResolution: compact; maximumDepth: uncompact',
      duplicatePolicy: 'compact deduplicates; uncompact preserves duplicate or overlapping inputs'
    }
  },
  GPUCellGridPath: {
    family: 'cells',
    descriptor: {scope: 'pairwise', reduction: 'none'},
    variables: {distances: 'count', cells: 'category', counts: 'count'},
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'pair', label: 'Origin/destination cell pair', indexSpace: '1d'}],
    guards: [],
    choices: [
      {stage: 'support', name: 'family', alternatives: ['h3'], sweepable: false},
      {
        stage: 'operator',
        name: 'maximumPathLength',
        label: 'Stored path capacity',
        unit: 'cells',
        sweepable: false
      }
    ],
    formula: CELLS_FORMULAS['gridPath'],
    metadata: {
      exactness: 'integer',
      distanceModel: 'H3 local IJK grid distance',
      pathLimitations:
        'Integer rounding ties may differ from h3-js; pentagon traversal may fail while retaining a valid distance.'
    }
  },
  GPUCellMeasures: {
    family: 'cells',
    // One input key supplies the decoded boundary; area/perimeter sum that cell's edges.
    descriptor: {scope: 'local', scopeSecondary: 'zonal', reduction: 'sum'},
    variables: {
      areas: 'measure',
      perimeters: 'measure',
      edgeLengths: 'measure',
      edgeCounts: 'count',
      pentagons: 'category'
    },
    stages: ['support', 'operator'],
    datumKinds: [
      {kind: 'cell', label: 'H3 cell and its spherical boundary measures', indexSpace: '1d'}
    ],
    guards: [],
    choices: [
      {stage: 'support', name: 'family', alternatives: ['h3'], sweepable: false},
      {stage: 'operator', name: 'units', alternatives: ['meters', 'kilometers'], sweepable: false},
      {
        stage: 'operator',
        name: 'maximumEdgeCount',
        label: 'Stored edge-length stride; totals use every edge',
        sweepable: false
      }
    ],
    formula: CELLS_FORMULAS['measures'],
    metadata: {
      metric: 'geodesic',
      earthModel: 'sphere',
      sphere: 'WGS84 authalic',
      exactness: 'float32',
      fallbackAccuracy:
        'Pentagon and face-crossing boundary-vector measures are unreliable beyond resolution 5.'
    }
  },
  GPUCellCover: {
    family: 'cells',
    descriptor: {scope: 'zonal', scopeSecondary: 'block', reduction: 'select'},
    variables: {
      featureIds: 'category',
      cells: 'category',
      core: 'category',
      'status.count': 'count'
    },
    stages: ['representation', 'support', 'operator'],
    datumKinds: [
      {kind: 'feature', label: 'Polygon feature and its covering cells', indexSpace: '1d'}
    ],
    representation: {from: 'object', to: 'field'},
    guards: [],
    choices: [
      {stage: 'support', name: 'family', alternatives: ['quadbin', 'h3'], sweepable: false},
      {stage: 'support', name: 'resolution', sweepable: true},
      {
        stage: 'operator',
        name: 'containment',
        alternatives: ['center', 'full', 'intersects'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'candidateEnumeration',
        alternatives: ['bounds', 'scanline', 'auto'],
        sweepable: false
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Candidate lattice enumeration, exact containment tests, prefix scans and bounded emission construct the cover procedurally; the graph rung also exposes candidate truncation and conservative core classification.'
    },
    metadata: {
      coordinateModel: 'geographic',
      metric: 'none',
      edgeModel: 'linear',
      h3Containment: 'center only',
      quadbinContainment: 'center,full,intersects',
      predicates: 'exact orientation with Simulation-of-Simplicity ties on float32 coordinates'
    }
  },
  GPUCellSetOutline: {
    family: 'cells',
    descriptor: {scope: 'focal', scopeSecondary: 'global', reduction: 'select'},
    variables: {
      rows: 'category',
      cells: 'category',
      edgeIndices: 'rank',
      endpoints: 'coordinate',
      groups: 'category'
    },
    stages: ['representation', 'support', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'segment', label: 'Exposed cell boundary edge', indexSpace: '1d'}],
    representation: {from: 'field', to: 'network'},
    guards: ['coordinate-model'],
    choices: [
      {stage: 'support', name: 'family', alternatives: ['h3', 'quadbin'], sweepable: false},
      {stage: 'representation', name: 'rings.vertexTolerance', unit: 'degrees', sweepable: true},
      {stage: 'representation', name: 'rings.normalizeWinding', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Neighbor lookup and group comparison classify exposed edges, scans pack their segments and optional ring assembly chains them; the graph rung shows the topology and geometric H3 neighbor selection.'
    },
    metadata: {
      exactness: 'float32 endpoints',
      coordinateModel: 'geographic',
      edgeModels: 'great-circle: h3; linear: quadbin',
      coordinateGuardDomain:
        'Local degree-space H3 midpoint distances and optional ring vertex matching',
      inputOrder: 'sorted distinct cell keys'
    }
  },
  GPUSegmentRingAssembly: {
    family: 'cells',
    descriptor: {scope: 'global', reduction: 'scan-sort'},
    variables: {
      positions: 'coordinate',
      ringOffsets: 'rank',
      ringAreas: 'measure',
      ringIsHole: 'category',
      ringShells: 'category',
      ringGroups: 'category',
      segmentRings: 'category',
      segmentVertices: 'rank',
      segmentFlags: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'ring', label: 'Assembled ring and source segments', indexSpace: 'csr'}],
    representation: {from: 'network', to: 'object'},
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'vertexTolerance', unit: 'position units', sweepable: true},
      {
        stage: 'representation',
        name: 'interiorSide',
        alternatives: ['left', 'right'],
        sweepable: false
      },
      {stage: 'representation', name: 'normalizeWinding', sweepable: false},
      {stage: 'operator', name: 'pairedOpposites', sweepable: false},
      {stage: 'operator', name: 'cancelOpposingSegments', sweepable: false},
      {stage: 'operator', name: 'splitTouchingRings', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Quantized matching, deterministic turn selection, bounded pointer jumping, touching-ring splitting and shell assignment assemble rings procedurally; the graph and iterations rungs expose the passes.'
    },
    metadata: {
      exactness: 'float32 coordinates with exact orientation predicates',
      ringAreasMeaning:
        'Signed planar shoelace measure; longitude/latitude degrees squared are orientation-only, not geographic area.',
      iterationSchedule: 'Fixed logarithmic pointer-jumping rounds, not a convergence solver'
    }
  },
  GPULineNoding: {
    family: 'cells',
    descriptor: {scope: 'pairwise', scopeSecondary: 'global', reduction: 'scan-sort'},
    variables: {
      endpoints: 'coordinate',
      sourceFeatureIds: 'category',
      sourceRingIds: 'category',
      sourceEdgeIds: 'category',
      sourceStartParameters: 'rate',
      sourceEndParameters: 'rate'
    },
    stages: ['representation', 'operator'],
    datumKinds: [
      {kind: 'segment', label: 'Noded segment with source edge provenance', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'operator',
        name: 'intersectionCapacity',
        label: 'Intersection event capacity',
        sweepable: false
      },
      {stage: 'operator', name: 'spatialSort', sweepable: false},
      {stage: 'operator', name: 'leafCapacity', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Exact-or-uncertain pair intersections feed line splitting, segment scans and source-provenance reconstruction; the graph rung exposes the procedural noding stages and uncertainty counters.'
    },
    metadata: {
      planarOnly: true,
      exactness: 'float32 storage with exact-or-uncertain predicates',
      coordinateGuardDomain:
        'Euclidean source-edge reconstruction and normalized segment parameters',
      precisionProp: false
    }
  },
  GPUPolygonize: {
    family: 'cells',
    descriptor: {scope: 'global', scopeSecondary: 'pairwise', reduction: 'select'},
    variables: {
      'polygons.positions': 'coordinate',
      'polygons.sourceIds': 'category',
      'directedEdges.endpoints': 'coordinate',
      'directedEdges.ringIds': 'category',
      'diagnostics.edgeClasses': 'category',
      'status.count': 'count'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'ring', label: 'Bounded face ring and directed edges', indexSpace: 'csr'}],
    representation: {from: 'network', to: 'object'},
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'representation',
        name: 'precision.vertexTolerance',
        unit: 'position units',
        sweepable: true
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Noding, twin half-edge expansion and deterministic left-face ring traversal construct all bounded faces and diagnostics; the graph rung shows the assembly rather than a closed-form polygon.'
    },
    metadata: {
      planarOnly: true,
      exactness: 'float32 storage with exact-or-uncertain predicates',
      faceSelection: 'All bounded faces, including a polygon filling each enclosed hole'
    }
  },
  GPUMakeValid: {
    family: 'cells',
    descriptor: {scope: 'zonal', scopeSecondary: 'pairwise', reduction: 'select'},
    variables: {
      'polygons.positions': 'coordinate',
      'polygons.sourceIds': 'category',
      'diagnostics.edgeClasses': 'category',
      'status.count': 'count'
    },
    stages: ['representation', 'operator'],
    datumKinds: [
      {kind: 'feature', label: 'Source polygon and its repaired faces', indexSpace: 'csr'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'representation',
        name: 'precision.vertexTolerance',
        unit: 'position units',
        sweepable: true
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Each feature is noded and polygonized, then exact symbolic probes select faces by source-ring parity and scans repack them; the graph rung exposes repair, uncertainty and collapsed remnants.'
    },
    metadata: {
      planarOnly: true,
      exactness: 'float32 storage with exact-or-uncertain predicates',
      fillRule: 'even-odd',
      uncertainFacePolicy: 'keep'
    }
  },
  GPUCellTableCompare: {
    family: 'cells',
    descriptor: {scope: 'pairwise', scopeSecondary: 'global', reduction: 'scan-sort'},
    variables: {
      cells: 'category',
      presence: 'category',
      mask: 'category',
      before: 'measure',
      after: 'measure',
      delta: 'measure',
      ratio: 'rate',
      percentChange: 'rate',
      zScore: 'score'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'cell', label: 'Union cell with before/after values', indexSpace: '1d'}],
    guards: ['count-vs-rate'],
    choices: [
      {stage: 'operator', name: 'measure', alternatives: ['count', 'sum'], sweepable: false},
      {
        stage: 'operator',
        name: 'sumScale',
        label: 'Scale already used by both tables',
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'zScore',
        alternatives: ['poisson', 'standardized'],
        sweepable: false
      }
    ],
    formula: CELLS_FORMULAS['compare'],
    metadata: {
      inputSupport: 'Both tables must have the same grid family and resolution',
      valueKinds:
        'before, after and delta are counts for measure=count, extensive measures for measure=sum',
      standardizedPopulation: 'Retained union rows after output-capacity truncation',
      exactness: '64-bit integer difference before float32 decoding; fixed-order float32 statistics'
    }
  }
};
