// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';

/** Curated statements of the named kernel components; labels delimit model and validity domains. */
export const LINES_FORMULAS = {
  segmentize: {
    id: 's3.line-segmentize',
    derivation: 'curated',
    tex: String.raw`\term{pieces}{n}=\begin{cases}\min(\term{cap}{N},\max(1,\lceil\term{length}{L}/\term{limit}{h}\rceil))&\term{limit}{h}>0,\term{length}{L}>0\\1&\text{otherwise}\end{cases}`,
    ascii:
      'Per source segment: n = min(N, max(1, ceil(L / h))) for h > 0 and L > 0; otherwise n = 1. L is Euclidean length or sphere-radius times the central angle. Emit equally spaced linear or great-circle interpolants; preserve input endpoints (implicit closure for rings).',
    symbols: [
      {
        id: 'pieces',
        role: 'output',
        label: 'Subdivision count before layout scans',
        binding: 'wgsl:counts'
      },
      {id: 'cap', role: 'parameter', label: 'maximumPieceCount', binding: 'wgsl:MAXIMUM_PIECES'},
      {
        id: 'length',
        role: 'input',
        label: 'Segment length in the selected metric',
        binding: 'wgsl:segmentLength'
      },
      {id: 'limit', role: 'parameter', label: 'maximumSegmentLength', binding: 'wgsl:maximumLength'}
    ]
  },
  arcs: {
    id: 's3.great-circle-arcs',
    derivation: 'curated',
    tex: String.raw`\term{position}{p_k}=\term{interpolate}{I_{\mathrm{sphere}}}(\term{source}{a},\term{target}{b},\term{angle}{\theta},\term{index}{k}/\term{segments}{n}),\quad \term{measure}{m_k}=\term{radius}{R}\term{angle}{\theta}\term{index}{k}/\term{segments}{n}`,
    ascii:
      'p[k] = sphericalInterpolate(a, b, theta, k/n); m[k] = R*theta*k/n, k=0..n. theta is the Vincenty-form spherical central angle. n is the capped maximum of the requested length subdivision and minimumSegments (at least 1). Endpoints are copied exactly, with target longitude unwrapped from a; coincident arcs use the kernel fallback.',
    symbols: [
      {id: 'position', role: 'output', binding: 'wgsl:outputPositions'},
      {id: 'measure', role: 'output', binding: 'wgsl:outputMeasures'},
      {id: 'source', role: 'input', binding: 'wgsl:source'},
      {
        id: 'target',
        role: 'input',
        label: 'Target with continuous longitude',
        binding: 'wgsl:destination'
      },
      {id: 'angle', role: 'intermediate', binding: 'wgsl:angle'},
      {id: 'segments', role: 'intermediate', binding: 'wgsl:segments'},
      {id: 'interpolate', role: 'intermediate', binding: 'wgsl:geodesicInterpolate'},
      {id: 'radius', role: 'parameter', binding: 'wgsl:RADIUS'},
      {id: 'index', role: 'index'}
    ]
  },
  smooth: {
    id: 's3.line-smooth',
    derivation: 'curated',
    tex: String.raw`\term{first}{q}=(1-\term{ratio}{r})\term{start}{a}+\term{ratio}{r}\term{end}{b},\qquad \term{second}{s}=\term{ratio}{r}\term{start}{a}+(1-\term{ratio}{r})\term{end}{b}`,
    ascii:
      'Each Chaikin pass replaces edge (a,b) by q=(1-r)*a+r*b and s=r*a+(1-r)*b. Open endpoints are copied; closed rings ignore a repeated input endpoint and repeat the first output vertex. Degenerate short lines are copied.',
    symbols: [
      {id: 'first', role: 'output', binding: 'wgsl:destination'},
      {id: 'second', role: 'output', binding: 'wgsl:destination'},
      {id: 'start', role: 'input', binding: 'wgsl:start'},
      {id: 'end', role: 'input', binding: 'wgsl:end'},
      {id: 'ratio', role: 'parameter', binding: 'wgsl:ratio'}
    ]
  },
  measures: {
    id: 's3.geometry-measures',
    derivation: 'curated',
    tex: String.raw`\term{lengths}{L_f}=\sum_{\term{edge}{e}\in\term{edges}{E_f}}\term{distance}{d}(\term{positions}{p_e},\term{positions}{p_{e+1}})`,
    ascii:
      'Length/perimeter component: L[f] = sum over feature edges e of d(p[e],p[e+1]), using compensated f32 sums. Lines are open; polygon rings close implicitly, including hole perimeters; points have no edges. d is Euclidean, spherical central-angle distance, or capped Vincenty inverse with Lambert reduced-latitude fallback, as selected by spatialContext. Area and centroid use separate model-specific kernels.',
    symbols: [
      {id: 'lengths', role: 'output', binding: 'wgsl:lengths'},
      {id: 'positions', role: 'input', binding: 'wgsl:positions'},
      {
        id: 'edges',
        role: 'input',
        label: 'Feature edge ranges derived from offsets',
        binding: 'wgsl:featureOffsets'
      },
      {id: 'distance', role: 'intermediate', binding: 'wgsl:getEdgeLength'},
      {id: 'edge', role: 'index'}
    ]
  },
  projection: {
    id: 's3.local-frame-projection',
    derivation: 'curated',
    tex: String.raw`\term{projected}{p}=\term{origin}{o}+\sum_{\term{index}{i}+\term{index}{j}\le3}\term{coefficients}{c_{ij}}\term{normalized}{u}^{\term{index}{i}}\term{normalized}{v}^{\term{index}{j}}`,
    ascii:
      'For a valid selected patch, p = destinationOrigin + sum(i+j<=3) c[i,j]*u^i*v^j, with nonnegative i,j and normalized source coordinates (u,v). The CPU adaptively fits the frame projection to tolerance; lower degrees are zero-padded. The shader uses Horner evaluation (double-single for precise inputs); invalid or out-of-bounds rows output zero and validity 0.',
    symbols: [
      {id: 'projected', role: 'output', binding: 'wgsl:projected'},
      {
        id: 'origin',
        role: 'parameter',
        label: 'Patch destination offset in frame meters',
        binding: 'wgsl:destinationOffset'
      },
      {id: 'coefficients', role: 'parameter', binding: 'wgsl:projectionPlans'},
      {id: 'normalized', role: 'input', binding: 'wgsl:normalized'},
      {id: 'index', role: 'index'}
    ]
  },
  referencing: {
    id: 's3.linear-referencing',
    derivation: 'curated',
    tex: String.raw`\term{measure}{m}=\term{prefix}{M_s}+\term{fraction}{t}\sqrt{\term{length}{\ell_s^2}},\qquad \term{normalized}{\widehat m}=\begin{cases}\term{measure}{m}/\term{total}{L}&\term{total}{L}>0\\0&\text{otherwise}\end{cases}`,
    ascii:
      'For a matched planar segment s: measure = vertexMeasures[s] + fraction*sqrt(lengthSquared); normalizedMeasure = measure/lineLength when lineLength > 0, else 0. Segment selection minimizes distance within radius, ties to the lowest source row. Spherical mode adds along-angle times sphere radius, or copies the endpoint prefix for endpoint matches. Unmatched measures are NaN.',
    symbols: [
      {id: 'measure', role: 'output', binding: 'wgsl:measures'},
      {id: 'normalized', role: 'output', binding: 'wgsl:normalizedMeasures'},
      {id: 'prefix', role: 'input', binding: 'wgsl:vertexMeasures'},
      {id: 'fraction', role: 'input', binding: 'wgsl:fraction'},
      {
        id: 'length',
        role: 'input',
        label: 'Squared planar segment length',
        binding: 'wgsl:lengthSquared'
      },
      {id: 'total', role: 'input', binding: 'wgsl:lineLength'}
    ]
  },
  locate: {
    id: 's3.line-locate',
    derivation: 'curated',
    tex: String.raw`\term{fraction}{t}=\operatorname{clamp}\left(\frac{\term{measure}{m}-\term{prefix}{M_s}}{\term{span}{M_{s+1}-M_s}},0,1\right),\quad \term{position}{p}=(1-\term{fraction}{t})\term{start}{a}+\term{fraction}{t}\term{end}{b}+\term{offset}{h}(-\term{tangent}{v_y},\term{tangent}{v_x})`,
    ascii:
      'Planar location on the selected positive-length segment: t=clamp((m-M[s])/(M[s+1]-M[s]),0,1); p=(1-t)*a+t*b+h*(-v.y,v.x). m is the scaled/offset event measure (then multiplied by line length in fraction mode) clamped to the line. v is the unit tangent. Zero spans use t=0; single vertices use v=0; invalid lines give NaN. Spherical mode instead slerps and applies the lateral offset by geodesic destination.',
    symbols: [
      {id: 'fraction', role: 'intermediate', binding: 'wgsl:fraction'},
      {id: 'measure', role: 'input', binding: 'wgsl:measure'},
      {id: 'prefix', role: 'input', binding: 'wgsl:vertexMeasures'},
      {id: 'span', role: 'input', binding: 'wgsl:segmentMeasure'},
      {id: 'position', role: 'output', binding: 'wgsl:position'},
      {id: 'start', role: 'input', binding: 'wgsl:start'},
      {id: 'end', role: 'input', binding: 'wgsl:end'},
      {id: 'offset', role: 'input', binding: 'wgsl:eventOffsets'},
      {id: 'tangent', role: 'intermediate', binding: 'wgsl:tangent'}
    ]
  },
  metrics: {
    id: 's3.trajectory-metrics',
    derivation: 'curated',
    tex: String.raw`\term{length}{L}=\sum_{\term{index}{i}\in\term{steps}{S}}\term{distances}{d_i},\qquad \term{average}{\bar v}=\begin{cases}\term{length}{L}/\term{duration}{T}&\term{duration}{T}>0\\0&\text{otherwise}\end{cases}`,
    ascii:
      'Track length L=sum of step distances d[i]; average speed=L/T for positive track duration T, otherwise 0. Steps stay within track offsets; distances are Euclidean or great-circle in context units. Time differences are computed before f32 conversion (Int64 or double-single where supplied). Stop extraction is a separate slow-run scan and compaction.',
    symbols: [
      {id: 'length', role: 'output', binding: 'wgsl:trackLengths'},
      {id: 'average', role: 'output', binding: 'wgsl:averageSpeeds'},
      {id: 'distances', role: 'input', binding: 'wgsl:stepDistances'},
      {id: 'duration', role: 'input', binding: 'wgsl:trackDuration'},
      {id: 'steps', role: 'input', binding: 'wgsl:trackOffsets'},
      {id: 'index', role: 'index'}
    ]
  },
  playhead: {
    id: 's3.trajectory-playhead',
    derivation: 'curated',
    tex: String.raw`\term{position}{p}=\term{interpolate}{I}(\term{start}{a},\term{end}{b},\term{fraction}{t}),\quad \term{speed}{v}=\begin{cases}\term{length}{d}(\term{start}{a},\term{end}{b})/\term{duration}{\Delta t}&\term{status}{s}=\text{active},\term{duration}{\Delta t}>0\\0&\text{otherwise}\end{cases}`,
    ascii:
      'For a nonempty track, p=I(a,b,t); speed=d(a,b)/duration only when active with positive duration, otherwise 0. I is linear interpolation in planar mode or spherical interpolation with continuous longitude. Binary search picks the latest duplicate row; t is the clamped elapsed-time fraction, held at 0 in a gap and at the endpoint outside the span. Empty tracks output zeros.',
    symbols: [
      {id: 'position', role: 'output', binding: 'wgsl:currentPositions'},
      {id: 'speed', role: 'output', binding: 'wgsl:speeds'},
      {id: 'start', role: 'input', binding: 'wgsl:startPosition'},
      {id: 'end', role: 'input', binding: 'wgsl:endPosition'},
      {id: 'fraction', role: 'input', binding: 'wgsl:fraction'},
      {id: 'duration', role: 'input', binding: 'wgsl:duration'},
      {id: 'status', role: 'input', binding: 'wgsl:status'},
      {id: 'interpolate', role: 'intermediate', binding: 'wgsl:interpolatePosition'},
      {id: 'length', role: 'intermediate', binding: 'wgsl:getSegmentLength'}
    ]
  },
  resample: {
    id: 's3.trajectory-resample',
    derivation: 'curated',
    tex: String.raw`\term{progress}{q_k}=\begin{cases}\term{total}{Q}\term{index}{k}/(\term{count}{n}-1)&\text{time or arc length},\ \term{count}{n}>1\\\term{offset}{c}+\term{index}{k}\term{step}{h}&\text{clock}\end{cases},\quad \term{position}{p_k}=\term{interpolate}{I}(\term{start}{a},\term{end}{b},\term{fraction}{t})`,
    ascii:
      'Target progress q[k]=Q*k/(n-1) for time or arc-length spacing with n>1 (0 for n=1); clock mode uses clockOffset+k*clockStep. Locate the bracketing progress rows and interpolate p[k]=I(a,b,t), linearly or spherically. Empty non-clock tracks output zero; absent clock samples are NaN. Clock sampleTimes are k*clockStep relative to clock start, while other sampleTimes are relative to track start.',
    symbols: [
      {id: 'progress', role: 'intermediate', binding: 'wgsl:targetProgress'},
      {id: 'total', role: 'input', binding: 'wgsl:total'},
      {id: 'count', role: 'parameter', binding: 'wgsl:SAMPLE_COUNT'},
      {id: 'offset', role: 'parameter', binding: 'wgsl:getClockOffset'},
      {id: 'step', role: 'parameter', binding: 'wgsl:clockStep'},
      {id: 'position', role: 'output', binding: 'wgsl:samples'},
      {id: 'start', role: 'input', binding: 'wgsl:startPosition'},
      {id: 'end', role: 'input', binding: 'wgsl:endPosition'},
      {id: 'fraction', role: 'input', binding: 'wgsl:fraction'},
      {id: 'interpolate', role: 'intermediate', binding: 'wgsl:interpolatePosition'},
      {id: 'index', role: 'index'}
    ]
  },
  encounters: {
    id: 's3.trajectory-encounters',
    derivation: 'curated',
    tex: String.raw`\term{count}{C}=|\term{hits}{H}|,\quad \term{minimum}{d_{\min}}=\min_{\term{index}{k}\in\term{hits}{H}}\term{distances}{d_k},\quad \term{first}{k_0}=\min\term{hits}{H}`,
    ascii:
      'For each emitted track pair, H is the nonempty set of shared clock buckets whose valid samples are within min(distance, maximumDistance). bucketCounts=|H|; minimumDistances=min over H of d[k]; firstBuckets=min H. Distances are Euclidean or great-circle. These are discrete hits only, and scratch overflow may make H incomplete.',
    symbols: [
      {id: 'count', role: 'output', binding: 'wgsl:bucketCounts'},
      {id: 'minimum', role: 'output', binding: 'wgsl:minimumDistances'},
      {id: 'first', role: 'output', binding: 'wgsl:firstBuckets'},
      {
        id: 'hits',
        role: 'input',
        label: 'Sorted bucket run for one pair',
        binding: 'wgsl:sortedBuckets'
      },
      {id: 'distances', role: 'input', binding: 'wgsl:sortedDistances'},
      {id: 'index', role: 'index'}
    ]
  },
  closestApproach: {
    id: 's3.trajectory-closest-approach',
    derivation: 'curated',
    tex: String.raw`\term{closest}{\tau_*}=\begin{cases}\operatorname{clamp}(-\term{separation}{D}\cdot\term{velocity}{V}/\lVert\term{velocity}{V}\rVert^2,0,\term{span}{T})&\lVert\term{velocity}{V}\rVert>0\\0&\text{otherwise}\end{cases},\quad \term{nearest}{d_*}=\lVert\term{separation}{D}+\term{velocity}{V}\term{closest}{\tau_*}\rVert`,
    ascii:
      'On one segment pair common time interval of span T: closest=clamp(-dot(D,V)/dot(V,V),0,T) for nonzero relative velocity V, else 0; d*=length(D+V*closest). D is separation at the common start. Quadratic roots bound the interval within distance. Track-pair output takes the minimum over hit segment pairs, earliest time on ties, and sums their encounter durations; times are relative to the lower track start.',
    symbols: [
      {
        id: 'closest',
        role: 'output',
        label: 'Segment-pair closest time offset',
        binding: 'wgsl:closest'
      },
      {
        id: 'nearest',
        role: 'output',
        label: 'Segment-pair minimum distance, packed hit word 4',
        binding: 'wgsl:hits'
      },
      {id: 'separation', role: 'input', binding: 'wgsl:separationStart'},
      {id: 'velocity', role: 'input', binding: 'wgsl:velocity'},
      {id: 'span', role: 'input', binding: 'wgsl:span'}
    ]
  }
} satisfies Record<string, GPUComputationFormulaRef>;
