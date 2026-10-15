// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';
import {LINES_FORMULAS} from './lines-formulas';

/**
 * Spatial computation-pack entries for the `lines` family, keyed by exported name. Every entry
 * must use `family: 'lines'`.
 */
export const LINES_OPERATORS: SpatialFamilyOperators = {
  GPULineSegmentize: {
    family: 'lines',
    descriptor: {scope: 'focal', scopeSecondary: 'zonal', reduction: 'none'},
    variables: {
      positions: 'coordinate',
      measures: 'measure',
      sourceRows: 'category',
      sourceLineIndices: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'segment', label: 'Source segment', indexSpace: 'csr'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'spatialContext', sweepable: false},
      {
        stage: 'operator',
        name: 'maximumSegmentLength',
        unit: 'context length unit',
        sweepable: true
      },
      {stage: 'operator', name: 'maximumPieceCount', sweepable: true},
      {stage: 'operator', name: 'closedLines', alternatives: [false, true], sweepable: false}
    ],
    metadata: {
      exactness: 'float32',
      metricModels: 'planar or great-circle sphere',
      formulaScope: 'segment subdivision count'
    },
    formula: LINES_FORMULAS.segmentize
  },
  GPUGreatCircleArcs: {
    family: 'lines',
    descriptor: {scope: 'pairwise', reduction: 'none'},
    variables: {
      positions: 'coordinate',
      measures: 'measure',
      sourceRows: 'category',
      sourceLineIndices: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'pair', label: 'Origin and destination pair', indexSpace: '1d'}],
    guards: [],
    choices: [
      {stage: 'representation', name: 'spatialContext', sweepable: false},
      {
        stage: 'operator',
        name: 'maximumSegmentLength',
        unit: 'context length unit',
        sweepable: true
      },
      {stage: 'operator', name: 'minimumSegments', sweepable: true},
      {stage: 'operator', name: 'maximumSegmentCount', sweepable: true}
    ],
    metadata: {metric: 'geodesic', earthModel: 'sphere', exactness: 'float32'},
    formula: LINES_FORMULAS.arcs
  },
  GPULineSmooth: {
    family: 'lines',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {positions: 'coordinate'},
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'vertex', label: 'Chaikin vertex', indexSpace: 'csr'}],
    guards: [],
    choices: [
      {stage: 'representation', name: 'closed', alternatives: [false, true], sweepable: false},
      {stage: 'neighbourhood', name: 'iterationCount', sweepable: true},
      {stage: 'operator', name: 'ratio', sweepable: true}
    ],
    metadata: {planarOnly: true, algorithm: 'Chaikin corner cutting'},
    formula: LINES_FORMULAS.smooth
  },
  GPULineChunk: {
    family: 'lines',
    descriptor: {scope: 'zonal', reduction: 'scan-sort'},
    variables: {
      positions: 'coordinate',
      measures: 'measure',
      sourceRows: 'category',
      sourceLineIndices: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'line', label: 'Source line and output pieces', indexSpace: 'csr'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'spatialContext', sweepable: false},
      {stage: 'operator', name: 'mode', alternatives: ['chunk', 'substring'], sweepable: false},
      {stage: 'operator', name: 'chunkLength', unit: 'context length unit', sweepable: true},
      {stage: 'operator', name: 'startMeasure', unit: 'context length unit', sweepable: true},
      {stage: 'operator', name: 'endMeasure', unit: 'context length unit', sweepable: true}
    ],
    metadata: {exactness: 'float32', metricModels: 'planar or great-circle sphere'},
    formula: {
      formula: 'none',
      reason:
        'Piece construction scans cumulative lengths and piece counts, searches measure boundaries, and emits bounded interpolated vertex runs; the graph rung shows these procedural stages.'
    }
  },
  GPUGeometryMeasures: {
    family: 'lines',
    descriptor: {scope: 'zonal', reduction: 'sum'},
    variables: {
      lengths: 'measure',
      areas: 'measure',
      centroids: 'coordinate',
      bounds: 'coordinate',
      vertexCounts: 'count',
      extremeVertices: 'category',
      'groupOutput.featureCounts': 'count'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'feature', label: 'Measured feature', indexSpace: 'csr'}],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'operator',
        name: 'geometryType',
        alternatives: ['points', 'lines', 'polygons'],
        sweepable: false
      },
      {stage: 'operator', name: 'spatialContext', sweepable: false},
      {
        stage: 'operator',
        name: 'ellipsoidalEdgeModel',
        alternatives: ['coordinate-linear', 'geodesic'],
        sweepable: false
      }
    ],
    metadata: {
      exactness: 'float32',
      metricModels: 'planar, sphere, or context ellipsoid (WGS84 default)',
      formulaScope: 'length and perimeter component'
    },
    formula: LINES_FORMULAS.measures
  },
  GPUGeodesicPairs: {
    family: 'lines',
    descriptor: {scope: 'pairwise', reduction: 'none'},
    variables: {
      distances: 'measure',
      initialBearings: 'direction',
      finalBearings: 'direction',
      midpoints: 'coordinate',
      'status.converged': 'category'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'pair', label: 'Origin and target pair', indexSpace: '1d'}],
    guards: [],
    choices: [
      {stage: 'operator', name: 'spatialContext', sweepable: false},
      {stage: 'operator', name: 'maximumIterations', sweepable: true}
    ],
    metadata: {
      metricModels: 'great-circle, rhumb, or ellipsoidal',
      earthModels: 'sphere or context ellipsoid (WGS84 default)',
      exactness: 'float32',
      midpointModel: 'sphere, including ellipsoidal mode'
    },
    formula: {
      formula: 'none',
      reason:
        'The model-dependent inverse uses capped Vincenty iterations with a spherical fallback, or spherical central-angle/rhumb kernels; no single closed form covers these outputs, and the kernel rung shows the branches and convergence.'
    }
  },
  GPUGeodesicDestination: {
    family: 'lines',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {
      destinations: 'coordinate',
      finalBearings: 'direction',
      'status.converged': 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'feature', label: 'Origin, bearing and distance row', indexSpace: '1d'}],
    guards: [],
    choices: [
      {stage: 'representation', name: 'spatialContext', sweepable: false},
      {stage: 'operator', name: 'maximumIterations', sweepable: true}
    ],
    metadata: {
      metricModels: 'great-circle, rhumb, or ellipsoidal',
      earthModels: 'sphere or context ellipsoid (WGS84 default)',
      exactness: 'float32'
    },
    formula: {
      formula: 'none',
      reason:
        'The ellipsoidal direct solution iterates Vincenty to a cap, while sphere and pole-reflecting rhumb modes use different kernels; the kernel rung shows these branches rather than a closed form valid for every mode.'
    }
  },
  GPULocalFrameProjection: {
    family: 'lines',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {output: 'coordinate', validity: 'category'},
    stages: ['representation'],
    datumKinds: [{kind: 'vertex', label: 'Projected coordinate row', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'frame', sweepable: false},
      {stage: 'representation', name: 'bounds', sweepable: false},
      {stage: 'representation', name: 'tolerance', unit: 'meters', sweepable: true}
    ],
    metadata: {
      outputUnits: 'meters',
      algorithm: 'adaptive polynomial patches',
      formulaScope: 'evaluation within a valid patch'
    },
    formula: LINES_FORMULAS.projection
  },
  GPULinearReferencing: {
    family: 'lines',
    descriptor: {scope: 'pairwise', scopeSecondary: 'zonal', reduction: 'argmin'},
    variables: {
      lineIndices: 'category',
      segmentIndices: 'rank',
      fractions: 'rate',
      footPoints: 'coordinate',
      distances: 'measure',
      measures: 'measure',
      normalizedMeasures: 'rate',
      sides: 'category',
      signedOffsets: 'measure',
      vertexMeasures: 'measure'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'pair', label: 'Query point and candidate segment', indexSpace: '2d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'neighbourhood', name: 'radius', unit: 'context length unit', sweepable: true},
      {stage: 'neighbourhood', name: 'candidateCapacity', sweepable: true},
      {stage: 'operator', name: 'spatialContext', sweepable: false}
    ],
    metadata: {
      exactness: 'float32',
      metricModels: 'planar or closest great-circle arc',
      formulaScope: 'matched along-line measures'
    },
    formula: LINES_FORMULAS.referencing
  },
  GPULineLocate: {
    family: 'lines',
    descriptor: {scope: 'zonal', scopeSecondary: 'focal', reduction: 'select'},
    variables: {
      positions: 'coordinate',
      segmentIndices: 'rank',
      tangents: 'direction',
      angles: 'direction',
      statuses: 'category',
      vertexMeasures: 'measure'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'feature', label: 'Event located on a line', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'spatialContext', sweepable: false},
      {
        stage: 'operator',
        name: 'measureMode',
        alternatives: ['distance', 'fraction'],
        sweepable: false
      },
      {stage: 'operator', name: 'measureScale', sweepable: true},
      {stage: 'operator', name: 'measureOffset', sweepable: true}
    ],
    metadata: {
      exactness: 'float32',
      formulaScope: 'planar location after measure search; spherical branch described in ascii'
    },
    formula: LINES_FORMULAS.locate
  },
  GPULineSplit: {
    family: 'lines',
    descriptor: {scope: 'pairwise', scopeSecondary: 'global', reduction: 'scan-sort'},
    variables: {
      'pieces.geometry.positions': 'coordinate',
      'pieces.sourceIds': 'category',
      'pieces.vertexSourceEdges': 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'segment', label: 'Intersecting source segment', indexSpace: 'csr'}],
    guards: [],
    choices: [{stage: 'operator', name: 'intersectionCapacity', sweepable: true}],
    metadata: {planarOnly: true},
    formula: {
      formula: 'none',
      reason:
        'Exact segment intersections become sorted, deduplicated split events and bounded vertex runs; this procedural noding pipeline is shown by the graph rung.'
    }
  },
  GPULineMerge: {
    family: 'lines',
    descriptor: {scope: 'global', reduction: 'scan-sort'},
    variables: {
      positions: 'coordinate',
      chainOffsets: 'rank',
      lineChains: 'category',
      lineOrders: 'rank',
      lineReversed: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'line', label: 'Line within a maximal chain', indexSpace: 'csr'}],
    guards: [],
    choices: [{stage: 'operator', name: 'directed', alternatives: [false, true], sweepable: false}],
    metadata: {planarOnly: true, endpointMatch: 'exact float32 coordinates, signed zeros equal'},
    formula: {
      formula: 'none',
      reason:
        'Endpoint sorting and degree-two adjacency feed fixed pointer-jumping rounds to rank paths and cycles; the graph and iterations rungs show chain assembly rather than a closed form.'
    }
  },
  GPULineClipByPolygon: {
    family: 'lines',
    descriptor: {scope: 'pairwise', scopeSecondary: 'global', reduction: 'select'},
    variables: {'pieces.geometry.positions': 'coordinate', 'pieces.sourceIds': 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'pair', label: 'Line and clip polygon', indexSpace: '2d'}],
    guards: [],
    choices: [
      {stage: 'operator', name: 'mode', alternatives: ['inside', 'outside'], sweepable: false},
      {stage: 'operator', name: 'intersectionCapacity', sweepable: true},
      {stage: 'operator', name: 'candidateCapacity', sweepable: true}
    ],
    metadata: {planarOnly: true, boundaryRule: 'boundary belongs to inside'},
    formula: {
      formula: 'none',
      reason:
        'The graph intersects segments, sorts and deduplicates crossings, classifies sub-piece midpoints and merges selected runs; this procedural clipping pipeline has no single closed-form output layout.'
    }
  },
  GPUSharedPaths: {
    family: 'lines',
    descriptor: {scope: 'pairwise', reduction: 'select'},
    variables: {
      'runs.positions': 'coordinate',
      'runs.leftLineIds': 'category',
      'runs.rightLineIds': 'category',
      'runs.forward': 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'pair', label: 'Left and right line', indexSpace: '2d'}],
    guards: [],
    choices: [{stage: 'operator', name: 'intersectionCapacity', sweepable: true}],
    metadata: {planarOnly: true},
    formula: {
      formula: 'none',
      reason:
        'Exact collinear overlaps are oriented, tagged by relative direction and chained into shared runs; the graph rung shows this procedural intersection and chain walk.'
    }
  },
  GPULineSimplification: {
    family: 'lines',
    descriptor: {scope: 'zonal', scopeSecondary: 'focal', reduction: 'select'},
    variables: {
      importance: 'measure',
      ids: 'category',
      outputMask: 'category',
      lineCounts: 'count',
      lineStarts: 'rank'
    },
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'vertex', label: 'Candidate retained vertex', indexSpace: 'csr'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'spatialContext', sweepable: false},
      {stage: 'neighbourhood', name: 'neighborhoodRadius', sweepable: true},
      {
        stage: 'operator',
        name: 'method',
        alternatives: ['douglas-peucker', 'visvalingam'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'metric',
        alternatives: ['segment', 'time-ratio'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'tolerance',
        label: 'Distance tolerance (effective area for Visvalingam)',
        sweepable: true
      },
      {stage: 'operator', name: 'maximumIterations', sweepable: true},
      {stage: 'operator', name: 'computeImportance', alternatives: [true, false], sweepable: false}
    ],
    metadata: {exactness: 'float32', visvalingamPlanarOnly: true, visvalingamApproximate: true},
    formula: {
      formula: 'none',
      reason:
        'Douglas-Peucker recursively splits farthest vertices, while approximate Visvalingam removes neighborhood minima and propagates effective areas; the iterations rung explains importance before threshold selection and compaction.'
    }
  },
  GPUCoverageSimplification: {
    family: 'lines',
    descriptor: {scope: 'global', scopeSecondary: 'zonal', reduction: 'select'},
    variables: {
      positions: 'coordinate',
      ringOffsets: 'rank',
      outputMask: 'category',
      topologyStatistics: 'count'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'ring', label: 'Coverage ring sharing owned arcs', indexSpace: 'csr'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'snapTolerance', unit: 'position units', sweepable: true},
      {stage: 'operator', name: 'tolerance', unit: 'position units', sweepable: true},
      {stage: 'operator', name: 'simplifyBoundary', alternatives: [true, false], sweepable: false},
      {stage: 'operator', name: 'maximumIterations', sweepable: true},
      {stage: 'operator', name: 'topologyIterationCount', sweepable: true},
      {stage: 'operator', name: 'topologyPairCapacity', sweepable: true}
    ],
    metadata: {
      planarOnly: true,
      algorithm: 'shared-arc Douglas-Peucker with bounded topology repair'
    },
    formula: {
      formula: 'none',
      reason:
        'Shared arcs are owned and simplified once, then junction and ring-size constraints and bounded intersection-repair rounds restore vertices; the graph and iterations rungs explain this topology-dependent selection.'
    }
  },
  GPUCoverageDissolve: {
    family: 'lines',
    descriptor: {scope: 'global', reduction: 'select'},
    variables: {
      positions: 'coordinate',
      ringOffsets: 'rank',
      'polygons.positions': 'coordinate',
      'polygons.sourceIds': 'category',
      'status.boundarySegmentCount': 'count'
    },
    stages: ['representation', 'operator'],
    datumKinds: [
      {kind: 'feature', label: 'Coverage feature and dissolve label', indexSpace: 'csr'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'snapTolerance', unit: 'position units', sweepable: true},
      {stage: 'operator', name: 'vertexTolerance', unit: 'position units', sweepable: true},
      {stage: 'operator', name: 'splitTouchingRings', alternatives: [false, true], sweepable: false}
    ],
    metadata: {planarOnly: true, inputRequirement: 'valid polygon coverage'},
    formula: {
      formula: 'none',
      reason:
        'Snapped edge keys remove same-label internal boundaries, then oriented survivors undergo segment-ring assembly; the graph rung shows sorting, selection and procedural shell/hole assembly.'
    }
  },
  GPUTrajectoryMetrics: {
    family: 'lines',
    descriptor: {scope: 'zonal', scopeSecondary: 'focal', reduction: 'sum'},
    variables: {
      trackLengths: 'measure',
      trackDurations: 'measure',
      averageSpeeds: 'rate',
      maximumSpeeds: 'rate',
      stepSpeeds: 'rate',
      stepHeadings: 'direction',
      stepAccelerations: 'rate',
      trackStopCounts: 'count',
      'stops.centroids': 'coordinate',
      'stops.durations': 'measure'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [
      {kind: 'track', label: 'Track', indexSpace: 'csr'},
      {kind: 'fix', label: 'Step ending at a fix', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'neighbourhood',
        name: 'stopSpeedThreshold',
        unit: 'context length per timestamp unit',
        sweepable: true
      },
      {
        stage: 'neighbourhood',
        name: 'stopMinimumDuration',
        unit: 'timestamp units',
        sweepable: true
      },
      {stage: 'operator', name: 'spatialContext', sweepable: false}
    ],
    metadata: {
      formulaScope: 'track length and average speed',
      stopCentroidModel: 'arithmetic coordinate mean'
    },
    formula: LINES_FORMULAS.metrics
  },
  GPUTrajectoryPlayhead: {
    family: 'lines',
    descriptor: {scope: 'zonal', scopeSecondary: 'focal', reduction: 'select'},
    variables: {
      currentPositions: 'coordinate',
      currentElevations: 'coordinate',
      headings: 'direction',
      speeds: 'rate',
      trackStatus: 'category',
      segmentRows: 'category',
      segmentFractions: 'rate',
      'activeTracks.ids': 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'track', label: 'Track at the playhead', indexSpace: 'csr'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'spatialContext', sweepable: false},
      {stage: 'operator', name: 'playhead', unit: 'timestamp units', sweepable: true},
      {stage: 'operator', name: 'maximumGap', unit: 'timestamp units', sweepable: true}
    ],
    metadata: {
      duplicateTimeRule: 'latest row wins',
      formulaScope: 'position and speed after bracket search'
    },
    formula: LINES_FORMULAS.playhead
  },
  GPUTrajectoryResample: {
    family: 'lines',
    descriptor: {scope: 'zonal', scopeSecondary: 'focal', reduction: 'none'},
    variables: {samples: 'coordinate', sampleElevations: 'coordinate', sampleTimes: 'measure'},
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'fix', label: 'Track and sample column', indexSpace: '2d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'spatialContext', sweepable: false},
      {
        stage: 'support',
        name: 'spacing',
        alternatives: ['time', 'arc-length', 'clock'],
        sweepable: false
      },
      {stage: 'support', name: 'sampleCount', sweepable: true},
      {stage: 'support', name: 'start', unit: 'timestamp units', sweepable: true},
      {stage: 'support', name: 'step', unit: 'timestamp units', sweepable: true}
    ],
    metadata: {clockTimeOrigin: 'shared clock start', otherTimeOrigin: 'track start'},
    formula: LINES_FORMULAS.resample
  },
  GPUZoneEvents: {
    family: 'lines',
    descriptor: {scope: 'pairwise', scopeSecondary: 'zonal', reduction: 'scan-sort'},
    variables: {
      'events.ids': 'category',
      'events.eventZones': 'category',
      'events.eventTypes': 'category',
      'events.eventTimes': 'measure',
      'events.eventPositions': 'coordinate',
      dwellTimes: 'measure',
      visitCounts: 'count',
      trackEventCounts: 'count'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'pair', label: 'Track and polygon zone', indexSpace: '2d'}],
    guards: ['count-vs-rate'],
    choices: [
      {stage: 'operator', name: 'candidateCapacity', sweepable: true},
      {stage: 'operator', name: 'trackEventCapacity', sweepable: true}
    ],
    metadata: {
      planarOnly: true,
      timeOrigin: 'track start',
      boundaryRule: 'sample shifted by infinitesimal (-epsilon squared, -epsilon)'
    },
    formula: {
      formula: 'none',
      reason:
        'Exact crossings and initial parity are sorted into track-zone event runs, then segmented spans produce dwell and visit counts; the graph rung shows this boundary-sensitive procedural event construction.'
    }
  },
  GPUTrajectoryEncounters: {
    family: 'lines',
    descriptor: {scope: 'pairwise', reduction: 'scan-sort'},
    variables: {
      'pairs.ids': 'category',
      'pairs.partners': 'category',
      'pairs.minimumDistances': 'measure',
      'pairs.firstBuckets': 'rank',
      'pairs.bucketCounts': 'count',
      'pairs.firstTimes': 'measure'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'pair', label: 'Tracks sharing clock buckets', indexSpace: '2d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'support', name: 'bucketCount', sweepable: true},
      {
        stage: 'support',
        name: 'cellSize',
        unit: 'position units (degrees for geographic)',
        sweepable: true
      },
      {stage: 'support', name: 'bounds', sweepable: false},
      {stage: 'neighbourhood', name: 'distance', unit: 'context length unit', sweepable: true},
      {stage: 'operator', name: 'hitCapacity', sweepable: true},
      {stage: 'operator', name: 'spatialContext', sweepable: false}
    ],
    metadata: {
      timeModel: 'discrete shared clock',
      formulaScope: 'statistics over accepted bucket hits'
    },
    formula: LINES_FORMULAS.encounters
  },
  GPUTrajectoryClosestApproach: {
    family: 'lines',
    descriptor: {scope: 'pairwise', reduction: 'argmin'},
    variables: {
      'pairs.ids': 'category',
      'pairs.partners': 'category',
      'pairs.minimumDistances': 'measure',
      'pairs.closestTimes': 'measure',
      'pairs.firstTimes': 'measure',
      'pairs.durations': 'measure'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'pair', label: 'Overlapping trajectory segment pair', indexSpace: '2d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'support', name: 'cellSize', unit: 'position units', sweepable: true},
      {stage: 'support', name: 'timeCellSize', unit: 'seconds', sweepable: true},
      {stage: 'neighbourhood', name: 'distance', unit: 'context length unit', sweepable: true},
      {stage: 'operator', name: 'entryCapacity', sweepable: true},
      {stage: 'operator', name: 'hitCapacity', sweepable: true}
    ],
    metadata: {
      planarOnly: true,
      timeModel: 'continuous piecewise linear motion',
      formulaScope: 'closest approach within one common segment interval'
    },
    formula: LINES_FORMULAS.closestApproach
  },
  GPUTrackSimilarity: {
    family: 'lines',
    descriptor: {scope: 'pairwise', reduction: 'argmin'},
    variables: {
      hausdorff: 'measure',
      frechet: 'measure',
      maximumDistance: 'measure',
      dynamicTimeWarping: 'measure',
      frechetWithin: 'category',
      pairStatus: 'category'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'pair', label: 'Track or ring pair', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'support', name: 'densify', sweepable: true},
      {stage: 'operator', name: 'spatialContext', sweepable: false},
      {stage: 'operator', name: 'maximumFrechetVertexCount', sweepable: true},
      {stage: 'operator', name: 'frechetEpsilon', unit: 'context length unit', sweepable: true},
      {stage: 'operator', name: 'dynamicTimeWarpingBandRadius', sweepable: true}
    ],
    metadata: {
      exactness: 'float32',
      hausdorffMetric: 'densified vertex to original polyline segments',
      dynamicTimeWarpingCost: 'sum of point distances',
      geographicSegmentMetric:
        'haversine to local-equirectangular closest point on degree-linear segment'
    },
    formula: {
      formula: 'none',
      reason:
        'Frechet and banded DTW sweep monotone dynamic programs, the decision branch prunes reachable paths, and Hausdorff searches vertex-to-segment minima; the kernel and iterations rungs expose these distinct algorithms.'
    }
  },
  addFleetDwellZoneEventsRecipe: {
    family: 'lines',
    descriptor: {scope: 'zonal', scopeSecondary: 'pairwise', reduction: 'sum'},
    variables: {
      'table.keys': 'category',
      'table.counts': 'count',
      'table.sumValues': 'measure',
      'table.means': 'rate',
      'table.maximums': 'measure',
      dwellTimes: 'measure',
      visitCounts: 'count'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'feature', label: 'Zone with fleet dwell totals', indexSpace: '1d'}],
    guards: ['count-vs-rate'],
    choices: [
      {stage: 'operator', name: 'candidateCapacity', sweepable: true},
      {stage: 'operator', name: 'trackEventCapacity', sweepable: true},
      {stage: 'operator', name: 'eventCapacity', sweepable: true}
    ],
    metadata: {planarOnly: true, aggregate: 'dwell time and visiting-track counts per zone'},
    formula: {formula: 'none', reason: 'a recipe composes contributors; its graph rung shows them'}
  }
};
