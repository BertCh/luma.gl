// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';
import {GEOMETRY_FORMULAS} from './geometry-formulas';

/**
 * Spatial computation-pack entries for the `geometry` family, keyed by exported name. Every entry
 * must use `family: 'geometry'`.
 */
export const GEOMETRY_OPERATORS: SpatialFamilyOperators = {
  GPUOutlineGeometry: {
    family: 'geometry',
    descriptor: {scope: 'focal', reduction: 'none'},
    variables: {positions: 'coordinate'},
    stages: ['representation', 'operator'],
    datumKinds: [
      {kind: 'vertex', label: 'Input vertex and its outline triangles', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'operator', name: 'distance', unit: 'context length units', sweepable: true},
      {stage: 'representation', name: 'joinSegments', sweepable: true},
      {
        stage: 'representation',
        name: 'geometryType',
        alternatives: ['points', 'lines', 'polygons'],
        sweepable: false
      }
    ],
    formula: GEOMETRY_FORMULAS['GPUOutlineGeometry'],
    metadata: {
      renderOnly: true,
      longitudeLatitudeModel: 'local-equirectangular',
      exactness: 'float32'
    }
  },
  GPUOffsetCurve: {
    family: 'geometry',
    descriptor: {scope: 'focal', reduction: 'none'},
    variables: {positions: 'coordinate', counts: 'count'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'vertex', label: 'Input vertex and offset corner', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'operator', name: 'distance', unit: 'position units', sweepable: true},
      {stage: 'operator', name: 'mitreLimit', sweepable: true},
      {
        stage: 'operator',
        name: 'joinStyle',
        alternatives: ['round', 'mitre', 'bevel'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'insideTurns',
        alternatives: ['intersection', 'buffer'],
        sweepable: false
      },
      {stage: 'representation', name: 'quadSegments', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Corner emission branches between endpoint normals, inside intersections or buffer detours, round arcs, bevels and clipped mitres; the kernel rung shows the branch and repeated padding rows.'
    },
    metadata: {planarOnly: true, renderOnly: true, exactness: 'float32'}
  },
  GPUVertexSnap: {
    family: 'geometry',
    descriptor: {scope: 'focal', scopeSecondary: 'pairwise', reduction: 'argmin'},
    variables: {positions: 'coordinate', referenceRows: 'category'},
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'vertex', label: 'Vertex and selected reference', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [{stage: 'neighbourhood', name: 'tolerance', unit: 'position units', sweepable: true}],
    formula: GEOMETRY_FORMULAS['GPUVertexSnap'],
    metadata: {planarOnly: true, exactness: 'float32'}
  },
  GPUPolygonOverlay: {
    family: 'geometry',
    descriptor: {scope: 'pairwise', scopeSecondary: 'global', reduction: 'select'},
    variables: {
      'geometry.positions': 'coordinate',
      'geometry.sourceIds': 'category',
      'boundary.endpoints': 'coordinate',
      'boundary.operandIds': 'category',
      'boundary.sourceFeatureIds': 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [
      {kind: 'segment', label: 'Selected lattice boundary segment', indexSpace: '1d'},
      {kind: 'ring', label: 'Assembled output ring', indexSpace: 'csr'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'operator',
        name: 'operation',
        alternatives: ['intersection', 'union', 'difference', 'symmetric-difference', 'dissolve'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'fillRule',
        alternatives: ['even-odd', 'positive', 'negative', 'nonzero'],
        sweepable: false
      },
      {stage: 'representation', name: 'vertexTolerance', unit: 'position units', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Snap-rounded noding, winding propagation, Boolean boundary selection and ring assembly construct topology procedurally; the graph rung exposes these stages.'
    },
    metadata: {planarOnly: true, exactness: 'snap-rounded-lattice'}
  },
  GPUBufferSurface: {
    family: 'geometry',
    descriptor: {scope: 'global', scopeSecondary: 'focal', reduction: 'select'},
    variables: {
      'geometry.positions': 'coordinate',
      'geometry.sourceIds': 'category',
      'boundary.endpoints': 'coordinate',
      'boundary.operandIds': 'category',
      'boundary.sourceFeatureIds': 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'ring', label: 'Dissolved buffer ring', indexSpace: 'csr'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'operator', name: 'distance', unit: 'position units', sweepable: true},
      {stage: 'operator', name: 'mitreLimit', sweepable: true},
      {
        stage: 'operator',
        name: 'joinStyle',
        alternatives: ['round', 'mitre', 'bevel'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'capStyle',
        alternatives: ['round', 'flat', 'square'],
        sweepable: false
      },
      {stage: 'representation', name: 'quadSegments', sweepable: true},
      {stage: 'representation', name: 'vertexTolerance', unit: 'position units', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Duplicate removal, ring orientation, two-sided offsets, caps and negative-winding dissolve construct a bounded polygon surface; the graph rung shows the topology stages rather than an ideal buffer formula.'
    },
    metadata: {planarOnly: true, exactness: 'float32'}
  },
  GPUAffineTransform: {
    family: 'geometry',
    descriptor: {scope: 'local', scopeSecondary: 'zonal', reduction: 'none'},
    variables: {positions: 'coordinate'},
    stages: ['representation'],
    datumKinds: [{kind: 'vertex', label: 'Transformed vertex', indexSpace: '1d'}],
    guards: [],
    choices: [
      {
        stage: 'representation',
        name: 'rotate',
        label: 'Rotation (degrees unless useRadians)',
        sweepable: true
      },
      {stage: 'representation', name: 'useRadians', alternatives: [false, true], sweepable: false},
      {stage: 'representation', name: 'scale', sweepable: true},
      {stage: 'representation', name: 'skew', sweepable: true},
      {stage: 'representation', name: 'translate', unit: 'position units', sweepable: true},
      {
        stage: 'representation',
        name: 'origin',
        label: 'Fixed coordinate, center or centroid',
        sweepable: false
      }
    ],
    formula: GEOMETRY_FORMULAS['GPUAffineTransform'],
    metadata: {planarOnly: true, exactness: 'float32'}
  },
  GPUGeometryOrientation: {
    family: 'geometry',
    descriptor: {scope: 'local', scopeSecondary: 'zonal', reduction: 'none'},
    variables: {positions: 'coordinate', reversedRings: 'category'},
    stages: ['representation'],
    datumKinds: [{kind: 'ring', label: 'Oriented ring or reversed path', indexSpace: 'csr'}],
    guards: [],
    choices: [
      {
        stage: 'representation',
        name: 'mode',
        alternatives: ['reverse', 'orient-polygons'],
        sweepable: false
      },
      {
        stage: 'representation',
        name: 'exteriorClockwise',
        alternatives: [false, true],
        sweepable: false
      }
    ],
    formula: GEOMETRY_FORMULAS['GPUGeometryOrientation'],
    metadata: {planarOnly: true, windingPredicate: 'exact-extreme-vertex'}
  },
  GPUGeometryCleanup: {
    family: 'geometry',
    descriptor: {scope: 'global', scopeSecondary: 'zonal', reduction: 'scan-sort'},
    variables: {positions: 'coordinate', lineOffsets: 'rank', ringOffsets: 'rank'},
    stages: ['representation'],
    datumKinds: [
      {kind: 'vertex', label: 'Snapped or retained vertex', indexSpace: '1d'},
      {kind: 'ring', label: 'Cleaned ring', indexSpace: 'csr'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'gridSize', unit: 'position units', sweepable: true},
      {stage: 'representation', name: 'tolerance', unit: 'position units', sweepable: true},
      {
        stage: 'representation',
        name: 'removeRepeatedPoints',
        alternatives: [true, false],
        sweepable: false
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Half-up grid snapping is followed by a previous-kept-vertex walk or equivalent exact-repeat flags, endpoint and ring-collapse rules, then scan compaction; the graph rung explains the changing layout.'
    },
    metadata: {planarOnly: true, exactness: 'float32'}
  },
  GPUMinimumBounds: {
    family: 'geometry',
    descriptor: {scope: 'zonal', reduction: 'argmin'},
    variables: {
      rectangleCorners: 'coordinate',
      // Packed outputs use the primary kind; component semantics are recorded below.
      rectangleSizes: 'measure',
      circles: 'coordinate',
      longestLines: 'coordinate',
      diameters: 'measure',
      bounds: 'coordinate',
      hullSizes: 'count'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'feature', label: 'Feature or label group', indexSpace: 'csr'}],
    guards: ['coordinate-model'],
    choices: [{stage: 'operator', name: 'maximumHullVertices', sweepable: true}],
    formula: GEOMETRY_FORMULAS['GPUMinimumBounds'],
    metadata: {
      planarOnly: true,
      rectangleRelativeTieTolerance: 1e-6,
      rectangleSizesComponents: 'width:measure,height:measure,angle:direction,area:measure',
      circlesComponents: 'centerX:coordinate,centerY:coordinate,radius:measure'
    }
  },
  GPUPolygonTriangulation: {
    family: 'geometry',
    descriptor: {scope: 'zonal', reduction: 'select'},
    variables: {
      indices: 'category',
      valid: 'category',
      triangleCount: 'count',
      triangleOffsets: 'rank'
    },
    stages: ['representation'],
    datumKinds: [{kind: 'triangle', label: 'Triangle in a polygon slice', indexSpace: 'csr'}],
    guards: [],
    choices: [
      {stage: 'representation', name: 'maximumWork', sweepable: true},
      {
        stage: 'representation',
        name: 'largePolygonMethod',
        alternatives: ['monotone', 'none'],
        sweepable: false
      },
      {stage: 'representation', name: 'monotoneMinimumVertices', sweepable: true},
      {stage: 'representation', name: 'monotoneMaximumVertices', sweepable: true},
      {
        stage: 'representation',
        name: 'useZOrderHash',
        alternatives: [true, false],
        sweepable: false
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Earcut removes ears and bridges holes, with a bounded monotone-decomposition and stack-triangulation fallback; the iterations and graph rungs explain triangle construction.'
    },
    metadata: {planarOnly: true}
  },
  GPUDelaunayTessellation: {
    family: 'geometry',
    descriptor: {scope: 'global', scopeSecondary: 'solver', reduction: 'solve'},
    variables: {triangles: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'triangle', label: 'Canonical Delaunay triangle', indexSpace: '1d'}],
    representation: {from: 'event', to: 'object'},
    guards: ['coordinate-model'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Seed insertion and Lawson edge flips iterate until every site is inserted and no illegal edge remains, or the bounded budget is exhausted; the iterations rung exposes convergence and exhaustion.'
    },
    metadata: {planarOnly: true, maximumIterationCount: 4096}
  },
  GPUVoronoiDiagram: {
    family: 'geometry',
    descriptor: {scope: 'global', scopeSecondary: 'pairwise', reduction: 'scan-sort'},
    variables: {segments: 'coordinate', siteIds: 'category'},
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'segment', label: 'Voronoi segment and its two sites', indexSpace: '1d'}],
    representation: {from: 'object', to: 'network'},
    guards: ['coordinate-model'],
    choices: [{stage: 'support', name: 'clipBounds', unit: 'position units', sweepable: true}],
    formula: {
      formula: 'none',
      reason:
        'Sorted Delaunay edge keys establish adjacency before circumcenter connections, clipped hull rays and compaction emit Voronoi segments; the graph rung shows this procedural construction.'
    },
    metadata: {planarOnly: true, exactness: 'float32'}
  },
  GPULabelPoint: {
    family: 'geometry',
    descriptor: {scope: 'zonal', reduction: 'argmin'},
    variables: {points: 'coordinate', distances: 'measure', degenerate: 'category'},
    stages: ['operator'],
    datumKinds: [
      {kind: 'feature', label: 'Polygon feature and label candidate', indexSpace: 'csr'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'operator', name: 'initialGridSize', sweepable: true},
      {stage: 'operator', name: 'refinementGridSize', sweepable: true},
      {stage: 'operator', name: 'refinementRounds', sweepable: true},
      {stage: 'operator', name: 'refinementCandidates', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Scanline seeds and bounded greedy grid refinement minimize negative signed boundary distance over visited candidates, not all interior points; the iterations rung shows which candidates were searched.'
    },
    metadata: {planarOnly: true, search: 'bounded-greedy', exactness: 'float32'}
  },
  GPUShapeDescriptors: {
    family: 'geometry',
    descriptor: {scope: 'zonal', reduction: 'sum'},
    variables: {
      areas: 'measure',
      perimeters: 'measure',
      polsbyPopper: 'rate',
      schwartzberg: 'rate',
      elongation: 'rate',
      orientation: 'direction',
      convexity: 'rate',
      clockwise: 'category',
      sliver: 'category'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'feature', label: 'Polygon feature and shape indices', indexSpace: 'csr'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'operator', name: 'sliverThreshold', sweepable: true},
      {
        stage: 'operator',
        name: 'convexityMethod',
        alternatives: ['auto', 'gift-wrapping', 'monotone-chain'],
        sweepable: false
      }
    ],
    formula: GEOMETRY_FORMULAS['GPUShapeDescriptors'],
    metadata: {planarOnly: true, exactness: 'float32'}
  },
  GPULineDensity: {
    family: 'geometry',
    descriptor: {scope: 'zonal', scopeSecondary: 'block', reduction: 'sum'},
    variables: {lengths: 'measure', densities: 'rate'},
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'cell', label: 'Grid cell', indexSpace: '2d'}],
    representation: {from: 'network', to: 'field'},
    guards: ['coordinate-model', 'conservation', 'count-vs-rate'],
    choices: [
      {stage: 'support', name: 'minX', unit: 'position units', sweepable: true},
      {stage: 'support', name: 'minY', unit: 'position units', sweepable: true},
      {stage: 'support', name: 'cellWidth', unit: 'position units', sweepable: true},
      {stage: 'support', name: 'cellHeight', unit: 'position units', sweepable: true},
      {stage: 'support', name: 'columns', sweepable: true},
      {stage: 'support', name: 'rows', sweepable: true}
    ],
    formula: GEOMETRY_FORMULAS['GPULineDensity'],
    metadata: {
      longitudeLatitudeLengthModel: 'great-circle-sphere',
      longitudeLatitudeAreaModel: 'authalic-sphere',
      conservationSupport: 'retained clipped pieces within grid',
      exactness: 'float32'
    }
  },
  GPULineLengthPerPolygon: {
    family: 'geometry',
    descriptor: {scope: 'zonal', scopeSecondary: 'pairwise', reduction: 'sum'},
    variables: {lengths: 'measure', weightedLengths: 'measure', segmentCounts: 'count'},
    stages: ['support', 'operator'],
    datumKinds: [
      {kind: 'feature', label: 'Polygon length summary', indexSpace: 'csr'},
      {kind: 'pair', label: 'Segment and polygon candidate', indexSpace: '1d'}
    ],
    representation: {from: 'network', to: 'object'},
    guards: ['coordinate-model', 'count-vs-rate'],
    choices: [
      {stage: 'operator', name: 'maximumCrossingCount', sweepable: true},
      {stage: 'operator', name: 'pairCapacity', sweepable: true}
    ],
    formula: GEOMETRY_FORMULAS['GPULineLengthPerPolygon'],
    metadata: {
      longitudeLatitudeLengthModel: 'great-circle-sphere',
      clippingModel: 'linear-coordinate-edges',
      exactness: 'float32'
    }
  },
  GPUGridGenerator: {
    family: 'geometry',
    descriptor: {scope: 'block', scopeSecondary: 'pairwise', reduction: 'none'},
    variables: {
      positions: 'coordinate',
      centers: 'coordinate',
      corners: 'coordinate',
      intersects: 'category',
      cornerIntersects: 'category'
    },
    stages: ['representation', 'support'],
    datumKinds: [
      {kind: 'cell', label: 'Generated grid cell', indexSpace: '2d'},
      {kind: 'vertex', label: 'Unique lattice corner', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'support',
        name: 'gridType',
        alternatives: ['square', 'hex', 'triangle', 'point'],
        sweepable: false
      },
      {
        stage: 'support',
        name: 'hexOrientation',
        alternatives: ['pointy', 'flat'],
        sweepable: false
      },
      {stage: 'support', name: 'columns', sweepable: true},
      {stage: 'support', name: 'rows', sweepable: true},
      {stage: 'support', name: 'minX', unit: 'position units', sweepable: true},
      {stage: 'support', name: 'minY', unit: 'position units', sweepable: true},
      {stage: 'support', name: 'cellWidth', unit: 'position units', sweepable: true},
      {stage: 'support', name: 'cellHeight', unit: 'position units', sweepable: true}
    ],
    formula: GEOMETRY_FORMULAS['GPUGridGenerator'],
    metadata: {planarOnly: true, exactness: 'float32'}
  },
  GPUShapeGenerator: {
    family: 'geometry',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {positions: 'coordinate', offsets: 'rank', vertexCount: 'count'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'feature', label: 'Generated shape ring', indexSpace: 'csr'}],
    representation: {from: 'event', to: 'object'},
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'representation',
        name: 'shape',
        alternatives: ['circle', 'sector', 'ellipse'],
        sweepable: false
      },
      {
        stage: 'representation',
        name: 'ellipseSpacing',
        alternatives: ['parameter', 'arc-length'],
        sweepable: false
      },
      {stage: 'representation', name: 'segmentCount', sweepable: true},
      {stage: 'representation', name: 'maximumSegments', sweepable: true},
      {stage: 'operator', name: 'radiusScale', sweepable: true}
    ],
    formula: GEOMETRY_FORMULAS['GPUShapeGenerator'],
    metadata: {longitudeLatitudeModel: 'spherical-destination', exactness: 'float32'}
  },
  GPUHilbertKeys: {
    family: 'geometry',
    descriptor: {scope: 'local', scopeSecondary: 'global', reduction: 'scan-sort'},
    variables: {keys: 'rank', sortedRows: 'category', sortedKeys: 'rank'},
    stages: ['representation', 'support'],
    datumKinds: [{kind: 'feature', label: 'Point or feature-box center', indexSpace: '1d'}],
    guards: [],
    choices: [
      {stage: 'support', name: 'order', sweepable: true},
      {stage: 'support', name: 'bounds', unit: 'position units', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Clamped coordinate quantization feeds a bitwise parallel-prefix Hilbert transform and optional stable radix sort, with an invalid-key sentinel; the kernel and graph rungs show the discrete ordering algorithm.'
    },
    metadata: {metric: 'none', maximumOrder: 16}
  },
  GPULonLatPolygonPreparation: {
    family: 'geometry',
    descriptor: {scope: 'local', scopeSecondary: 'global', reduction: 'scan-sort'},
    variables: {
      positions: 'coordinate',
      featureOffsets: 'rank',
      polygonOffsets: 'rank',
      ringOffsets: 'rank',
      ringFlags: 'category'
    },
    stages: ['representation'],
    datumKinds: [
      {kind: 'feature', label: 'Prepared polygon feature', indexSpace: 'csr'},
      {kind: 'ring', label: 'Classified input ring', indexSpace: 'csr'}
    ],
    guards: [],
    choices: [
      {stage: 'representation', name: 'maximumSegmentDegrees', unit: 'degrees', sweepable: true},
      {stage: 'representation', name: 'maximumPieceCount', sweepable: true},
      {
        stage: 'representation',
        name: 'antimeridian',
        alternatives: ['split', 'reject'],
        sweepable: false
      },
      {
        stage: 'representation',
        name: 'poleRings',
        alternatives: ['cap', 'reject'],
        sweepable: false
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Densification, ring classification, heuristic nearer-pole capping, seam clipping and scan-based publication preserve doubled topology; the graph rung explains rejection and approximation diagnostics.'
    },
    metadata: {
      coordinateSpace: 'longitude-latitude',
      outputEdgeModel: 'linear',
      poleCap: 'nearer-pole-heuristic'
    }
  },
  GPURectangleClip: {
    family: 'geometry',
    descriptor: {scope: 'local', reduction: 'select'},
    variables: {positions: 'coordinate', lineOffsets: 'rank', sourceLineIndices: 'category'},
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'line', label: 'Clipped line or polygon ring', indexSpace: 'csr'}],
    guards: [],
    choices: [
      {stage: 'support', name: 'minX', unit: 'position units', sweepable: true},
      {stage: 'support', name: 'minY', unit: 'position units', sweepable: true},
      {stage: 'support', name: 'maxX', unit: 'position units', sweepable: true},
      {stage: 'support', name: 'maxY', unit: 'position units', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Liang-Barsky segment clipping rejoins surviving line pieces, while polygons use four Sutherland-Hodgman count-scan-emit stages; the graph rung shows layout changes and retained boundary bridges.'
    },
    metadata: {planarOnly: true, exactness: 'float32'}
  }
};
