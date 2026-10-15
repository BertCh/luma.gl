// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';

/** Curated mathematical cores of geometry operators, bound to their actual WGSL identifiers. */
export const GEOMETRY_FORMULAS = {
  GPUOutlineGeometry: {
    id: 's3.outline-geometry',
    derivation: 'curated',
    tex: String.raw`\term{output}{p'}=\term{point}{p}+\term{distance}{d}\begin{cases}\term{direction}{u}&\text{planar}\\\dfrac{180}{\pi\term{radius}{R}}\left(\dfrac{\term{direction}{u_x}}{\max(\cos(\pi\term{point}{p_y}/180),10^{-6})},\term{direction}{u_y}\right)&\text{longitude/latitude}\end{cases}`,
    ascii:
      'Each emitted offset vertex is p + d*u in planar coordinates, or p + (180*d/(pi*R))*(u.x/max(cos(pi*p.y/180),1e-6),u.y) in longitude/latitude. Disc directions are (cos(theta),sin(theta)); segment corners use signed local normals; centers use u=0. Distance d is clamped nonnegative. This is an equirectangular drawing approximation, not spherical destination.',
    symbols: [
      {
        id: 'output',
        role: 'output',
        label: 'Emitted triangle vertex',
        binding: 'wgsl:outputPositions'
      },
      {
        id: 'point',
        role: 'input',
        label: 'Endpoint or disc center in offsetPoint',
        binding: 'wgsl:p'
      },
      {
        id: 'distance',
        role: 'parameter',
        label: 'Nonnegative outline distance',
        binding: 'wgsl:distance'
      },
      {
        id: 'direction',
        role: 'intermediate',
        label: 'Disc direction or signed segment normal',
        binding: 'wgsl:direction'
      },
      {
        id: 'radius',
        role: 'constant',
        label: 'Context sphere radius in length units',
        binding: 'wgsl:RADIUS'
      }
    ]
  },
  GPUVertexSnap: {
    id: 's3.vertex-snap',
    derivation: 'curated',
    tex: String.raw`\begin{aligned}\term{reference-row}{j_*}&=\underset{\term{index}{j}\in\term{range}{J},\ \lVert\term{reference}{r_j}-\term{vertex}{p}\rVert^2<\term{tolerance}{t}^2}{\operatorname{argmin}}\left(\lVert\term{reference}{r_j}-\term{vertex}{p}\rVert^2,\term{index}{j}\right)\\\term{output}{p'}&=\begin{cases}\term{reference}{r_{j_*}}&\text{a candidate exists}\\\term{vertex}{p}&\text{otherwise}\end{cases}\end{aligned}`,
    ascii:
      'Among reference rows J strictly within squared tolerance t^2 of p, choose the lexicographic minimum (squared distance, reference row). Output that reference position, or retain p and report NO_REFERENCE when no candidate exists. J is the whole reference set or the matching feature range; t=max(parameters[0],0).',
    symbols: [
      {
        id: 'reference-row',
        role: 'output',
        label: 'Selected row, or NO_REFERENCE',
        binding: 'wgsl:outputReferenceRows'
      },
      {id: 'index', role: 'index', label: 'Reference row'},
      {
        id: 'range',
        role: 'intermediate',
        label: 'Rows in [referenceStart, referenceEnd)',
        binding: 'wgsl:referenceStart'
      },
      {
        id: 'reference',
        role: 'input',
        label: 'Reference positions',
        binding: 'wgsl:referencePositions'
      },
      {id: 'vertex', role: 'input', label: 'Input vertex', binding: 'wgsl:vertex'},
      {id: 'tolerance', role: 'parameter', label: 'Snap tolerance', binding: 'wgsl:tolerance'},
      {id: 'output', role: 'output', label: 'Snapped position', binding: 'wgsl:outputPositions'}
    ]
  },
  GPUAffineTransform: {
    id: 's3.affine-transform',
    derivation: 'curated',
    tex: String.raw`\term{output}{p'}=\begin{pmatrix}\term{a}{a}&\term{b}{b}\\\term{d}{d}&\term{e}{e}\end{pmatrix}(\term{point}{p}-\term{origin}{o})+\term{origin}{o}+\term{translation}{t}`,
    ascii:
      "p' = [[a,b],[d,e]]*(p-o)+o+t. Global parameters compose scale, skew, rotation and translation about the input's fixed, center or centroid origin. Per-feature transforms use their six coefficients with o=0 and t=(xoff,yoff).",
    symbols: [
      {id: 'output', role: 'output', label: 'Transformed vertex', binding: 'wgsl:outputPositions'},
      {id: 'point', role: 'input', label: 'Input vertex', binding: 'wgsl:point'},
      {id: 'a', role: 'parameter', label: 'Matrix coefficient a', binding: 'wgsl:a'},
      {id: 'b', role: 'parameter', label: 'Matrix coefficient b', binding: 'wgsl:b'},
      {id: 'd', role: 'parameter', label: 'Matrix coefficient d', binding: 'wgsl:d'},
      {id: 'e', role: 'parameter', label: 'Matrix coefficient e', binding: 'wgsl:e'},
      {id: 'origin', role: 'intermediate', label: 'Resolved origin', binding: 'wgsl:origin'},
      {
        id: 'translation',
        role: 'parameter',
        label: 'Translation vector',
        binding: 'wgsl:translation'
      }
    ]
  },
  GPUGeometryOrientation: {
    id: 's3.geometry-orientation',
    derivation: 'curated',
    tex: String.raw`\term{output}{p'_i}=\begin{cases}\term{positions}{p_{s+e-1-i}}&\term{flag}{f_r}=1,\ \term{start}{s}\le\term{index}{i}<\term{end}{e}\\\term{positions}{p_i}&\text{otherwise}\end{cases}`,
    ascii:
      "For a vertex i in ring [s,e), p'[i]=p[s+e-1-i] when the ring flag is 1, otherwise p'[i]=p[i]. Reverse mode flags every ring; orient-polygons flags a winding mismatch using exact orientation at the lexicographic extreme and nearest distinct neighbours, leaving zero or uncertain winding unchanged. Vertices outside rings pass through.",
    symbols: [
      {id: 'output', role: 'output', label: 'Oriented vertices', binding: 'wgsl:outputPositions'},
      {id: 'positions', role: 'input', label: 'Input vertices', binding: 'wgsl:positions'},
      {
        id: 'flag',
        role: 'intermediate',
        label: 'Exact winding decision / reversedRings',
        binding: 'wgsl:ringFlags'
      },
      {id: 'start', role: 'intermediate', label: 'Ring start', binding: 'wgsl:start'},
      {id: 'end', role: 'intermediate', label: 'Clamped ring end', binding: 'wgsl:end'},
      {id: 'index', role: 'index', label: 'Vertex row'}
    ]
  },
  GPUMinimumBounds: {
    id: 's3.minimum-bounds',
    derivation: 'curated',
    // The recurrence preserves the implementation's relative tie rule; a bare argmin would not.
    tex: String.raw`\begin{aligned}\term{area}{A_e}&=\left(\max_{\term{index}{j}}\term{points}{h_j}\cdot\term{direction}{u_e}-\min_{\term{index}{j}}\term{points}{h_j}\cdot\term{direction}{u_e}\right)\left(\max_{\term{index}{j}}\term{points}{h_j}\cdot\term{normal}{n_e}-\min_{\term{index}{j}}\term{points}{h_j}\cdot\term{normal}{n_e}\right)\\\term{best}{B_e}&=\begin{cases}\term{area}{A_e}&\term{area}{A_e}<(1-10^{-6})\term{best}{B_{e-1}}\\\term{best}{B_{e-1}}&\text{otherwise}\end{cases},\quad\term{best}{B_0}=\term{area}{A_0},\quad\term{output}{A_{\rm rect}}=\term{best}{B_{\rm last}}\end{aligned}`,
    ascii:
      'For each nonzero hull edge e, use its unit direction u and left normal n. Candidate area A[e]=(max dot(h,u)-min dot(h,u))*(max dot(h,n)-min dot(h,n)). Start B at the first candidate; replace it only when A[e] < (1-1e-6)*B. rectangleSizes.area is the last B. Hull points are anchor-relative; one-point/collinear rectangles have zero area, empty or overflowing hulls report NaN. This formula describes the rectangle selection; circles use shuffled Welzl with a recomputed covering radius.',
    symbols: [
      {
        id: 'area',
        role: 'intermediate',
        label: 'Hull-edge-aligned rectangle area',
        binding: 'wgsl:area'
      },
      {id: 'index', role: 'index', label: 'Hull vertex'},
      {
        id: 'points',
        role: 'input',
        label: 'Anchor-relative hull points',
        binding: 'wgsl:loadPoint'
      },
      {
        id: 'direction',
        role: 'intermediate',
        label: 'Unit hull-edge direction',
        binding: 'wgsl:direction'
      },
      {id: 'normal', role: 'intermediate', label: 'Left normal', binding: 'wgsl:normal'},
      {id: 'best', role: 'intermediate', label: 'Best area so far', binding: 'wgsl:bestArea'},
      {id: 'output', role: 'output', label: 'rectangleSizes area component', binding: 'wgsl:sizes'}
    ]
  },
  GPUShapeDescriptors: {
    id: 's3.shape-descriptors',
    derivation: 'curated',
    tex: String.raw`\begin{aligned}\term{compactness}{C}&=\dfrac{4\pi\term{area}{A}}{\term{perimeter}{P}^2},\quad\term{schwartzberg}{S}=\sqrt{\term{compactness}{C}},\quad\term{convexity}{V}=\min\left(\dfrac{\term{area}{A}}{\term{hull-area}{H}},1\right)\\\term{elongation}{E}&=1-\sqrt{\dfrac{\term{minor}{\lambda_2}}{\term{major}{\lambda_1}}},\quad\term{sliver}{F}=\mathbb{1}\left[\term{perimeter}{P}>0\land(\term{area}{A}\le0\lor\term{compactness}{C}<\term{threshold}{\tau})\right]\end{aligned}`,
    ascii:
      'Polsby-Popper C=4*pi*A/P^2; Schwartzberg S=sqrt(C); convexity V=min(A/H,1) for hull area H; elongation E=1-sqrt(lambda2/lambda1) with lambda2=max(mean-radius,0), lambda1=mean+radius from central moments. Sliver is P>0 and (A<=0 or C<threshold). C and S are NaN unless A>0 and P>0, V is NaN without positive hull area, E is NaN without a positive major eigenvalue. Shells contribute positive area and holes negative, regardless of winding.',
    symbols: [
      {id: 'compactness', role: 'output', label: 'Polsby-Popper', binding: 'wgsl:polsbyPopper'},
      {id: 'area', role: 'input', label: 'Measured polygon area', binding: 'wgsl:areas'},
      {
        id: 'perimeter',
        role: 'input',
        label: 'Measured perimeter including holes',
        binding: 'wgsl:perimeters'
      },
      {id: 'schwartzberg', role: 'output', label: 'Schwartzberg', binding: 'wgsl:schwartzberg'},
      {id: 'convexity', role: 'output', label: 'Convexity', binding: 'wgsl:convexities'},
      {id: 'hull-area', role: 'intermediate', label: 'Convex hull area', binding: 'wgsl:hullArea'},
      {id: 'elongation', role: 'output', label: 'Elongation', binding: 'wgsl:elongation'},
      {
        id: 'minor',
        role: 'intermediate',
        label: 'Nonnegative minor eigenvalue',
        binding: 'wgsl:minor'
      },
      {id: 'major', role: 'intermediate', label: 'Major eigenvalue', binding: 'wgsl:major'},
      {id: 'sliver', role: 'output', label: 'Sliver flag', binding: 'wgsl:sliver'},
      {
        id: 'threshold',
        role: 'parameter',
        label: 'Sliver threshold (parameters[0])',
        binding: 'wgsl:parameters'
      }
    ]
  },
  GPULineDensity: {
    id: 's3.line-density',
    derivation: 'curated',
    tex: String.raw`\begin{aligned}\term{lengths}{L_c}&=\sum_{\term{piece}{k}\in\term{support}{K_c}}\term{piece-length}{\ell_k},\qquad\term{densities}{D_c}=\dfrac{\term{lengths}{L_c}}{\term{area}{A_c}}\\\term{area}{A_c}&=\begin{cases}\term{width}{w}\term{height}{h}&\text{planar}\\\term{radius}{R_A}^2\dfrac{\pi\term{width}{w}}{180}\left[\sin\dfrac{\pi(\term{south}{s}+\term{height}{h})}{180}-\sin\dfrac{\pi\term{south}{s}}{180}\right]&\text{longitude/latitude}\end{cases}\end{aligned}`,
    ascii:
      'L[c]=sum of retained clipped-piece lengths in cell c; D[c]=L[c]/A[c]. Planar A=w*h. Longitude/latitude A=R_A^2*(pi*w/180)*(sin(pi*(south+h)/180)-sin(pi*south/180)), using the context authalic area radius. Piece length is (t1-t0)*length(b-a) in planar coordinates or context sphere radius times geodesicCentralAngle of the degree-linear clipped endpoints. Overflow omits pieces; sums use stable sorting and fixed-order reduction.',
    symbols: [
      {id: 'lengths', role: 'output', label: 'Cell lengths', binding: 'wgsl:lengths'},
      {id: 'piece', role: 'index', label: 'Clipped piece'},
      {
        id: 'support',
        role: 'intermediate',
        label: 'Retained records with this cell key',
        binding: 'wgsl:keys'
      },
      {
        id: 'piece-length',
        role: 'intermediate',
        label: 'Clipped-piece metric length',
        binding: 'wgsl:getPieceLength'
      },
      {id: 'densities', role: 'output', label: 'Length per area', binding: 'wgsl:densities'},
      {id: 'area', role: 'intermediate', label: 'Cell area', binding: 'wgsl:area'},
      {
        id: 'width',
        role: 'parameter',
        label: 'Cell width in position units',
        binding: 'wgsl:width'
      },
      {
        id: 'height',
        role: 'parameter',
        label: 'Cell height in position units',
        binding: 'wgsl:height'
      },
      {
        id: 'radius',
        role: 'constant',
        label: 'Authalic area radius in context units',
        binding: 'wgsl:AREA_RADIUS'
      },
      {
        id: 'south',
        role: 'intermediate',
        label: 'Southern latitude of cell in degrees',
        binding: 'wgsl:south'
      }
    ]
  },
  GPULineLengthPerPolygon: {
    id: 's3.line-length-per-polygon',
    derivation: 'curated',
    tex: String.raw`\term{lengths}{L_f}=\sum_{\term{index}{s}}\term{inside}{\ell_{sf}},\quad\term{weighted}{W_f}=\sum_{\term{index}{s}}\term{weights}{w_s}\term{inside}{\ell_{sf}},\quad\term{counts}{N_f}=\sum_{\term{index}{s}}\mathbb{1}[\term{inside}{\ell_{sf}}>0]`,
    ascii:
      'For polygon f, L[f]=sum_s insideLength[s,f], W[f]=sum_s pathWeight[s]*insideLength[s,f], N[f]=sum_s 1[insideLength[s,f]>0]. Inside length sums pieces between ring crossings whose midpoint passes even-odd containment. Planar pieces measure (t1-t0)*length(b-a); longitude/latitude pieces measure sphere radius times geodesicCentralAngle between their degree-linear endpoints. Weights are per path, repeated for each segment. Capacity overflow can undercount.',
    symbols: [
      // The two sums use separate invocations of the shared segmented sum kernel.
      {
        id: 'lengths',
        role: 'output',
        label: 'Length sum (unweighted sum dispatch)',
        binding: 'wgsl:segmentOutput'
      },
      {id: 'index', role: 'index', label: 'Input segment'},
      {
        id: 'inside',
        role: 'intermediate',
        label: 'Segment length inside polygon',
        binding: 'wgsl:insideLengths'
      },
      {
        id: 'weighted',
        role: 'output',
        label: 'Length sum (weighted sum dispatch)',
        binding: 'wgsl:segmentOutput'
      },
      {
        id: 'weights',
        role: 'input',
        label: 'Per-path weight copied to each segment',
        binding: 'wgsl:segmentWeights'
      },
      {id: 'counts', role: 'output', label: 'Contributing segment count', binding: 'wgsl:counts'}
    ]
  },
  GPUGridGenerator: {
    id: 's3.grid-generator',
    derivation: 'curated',
    // The vertex mean describes square, hex and triangle centers; it is explicitly the triangle kernel.
    tex: String.raw`\term{centers}{c}=\begin{cases}\term{origin}{o}+((\term{column}{i}+\tfrac12)\term{width}{w},(\term{row}{j}+\tfrac12)\term{height}{h})&\text{square or point}\\\dfrac{1}{\term{count}{V}}\sum_{\term{index}{k}=0}^{\term{count}{V}-1}\term{vertices}{v_k}&\text{hex or triangle}\end{cases}`,
    ascii:
      'Square and point centers are origin+((column+1/2)*cellWidth,(row+1/2)*cellHeight). Hex and triangle centers are the vertex mean of the generated cell (V=6 or 3). Hex kernels compute the equivalent center directly from width/sqrt(3), staggered rows or columns and orientation; triangles use (a+b+c)/3. Vertices and unique corners come from shared integer lattice expressions, preserving identical seam coordinates. This formula describes cell centers; extent masks use boundary-inclusive intersection predicates.',
    symbols: [
      {id: 'centers', role: 'output', label: 'Cell centers', binding: 'wgsl:centersOut'},
      {id: 'origin', role: 'parameter', label: 'Grid origin (minX,minY)', binding: 'wgsl:origin'},
      {id: 'column', role: 'index', label: 'Column'},
      {id: 'row', role: 'index', label: 'Row'},
      {id: 'width', role: 'parameter', label: 'Cell width', binding: 'wgsl:w'},
      {id: 'height', role: 'parameter', label: 'Cell height', binding: 'wgsl:h'},
      {id: 'count', role: 'constant', label: 'Vertices per polygon cell'},
      {id: 'index', role: 'index', label: 'Cell vertex'},
      {
        id: 'vertices',
        role: 'intermediate',
        label: 'Generated polygon vertices',
        binding: 'wgsl:positionsOut'
      }
    ]
  },
  GPUShapeGenerator: {
    id: 's3.shape-generator',
    derivation: 'curated',
    tex: String.raw`\begin{aligned}\term{latitude-out}{\phi'}&=\arcsin\!\left(\operatorname{clamp}\left(\sin\term{latitude}{\phi}\cos\term{angle}{\delta}+\cos\term{latitude}{\phi}\sin\term{angle}{\delta}\cos\term{bearing}{\beta},-1,1\right)\right),\quad\term{angle}{\delta}=\term{distance}{d}/\term{radius}{R}\\\term{output}{p'}&=\begin{cases}\term{center}{c}+\term{distance}{d}(\sin\term{bearing}{\beta},\cos\term{bearing}{\beta})&\text{planar}\\\left(\term{center}{c_x}+\dfrac{180}{\pi}\operatorname{atan2}\left(\sin\term{bearing}{\beta}\sin\term{angle}{\delta}\cos\term{latitude}{\phi},\cos\term{angle}{\delta}-\sin\term{latitude}{\phi}\sin\term{latitude-out}{\phi'}\right),\dfrac{180}{\pi}\term{latitude-out}{\phi'}\right)&\text{longitude/latitude}\end{cases}\end{aligned}`,
    ascii:
      'Given the sampled distance d and clockwise-from-north bearing beta, planar p=c+d*(sin(beta),cos(beta)). On a sphere, phi=pi*c.y/180, delta=d/R, phi2=asin(clamp(sin(phi)*cos(delta)+cos(phi)*sin(delta)*cos(beta),-1,1)), and p=(c.x+180/pi*atan2(sin(beta)*sin(delta)*cos(phi),cos(delta)-sin(phi)*sin(phi2)),180/pi*phi2). Circles sample negative full-turn bearings, sectors sample their clockwise sweep with center spokes, and ellipses derive d and beta from rotated scaled axes (optionally using the arc-length table). This is the destination core for finite centers; invalid centers emit NaN.',
    symbols: [
      {
        id: 'latitude-out',
        role: 'intermediate',
        label: 'Destination latitude in radians',
        binding: 'wgsl:latitude2'
      },
      {
        id: 'latitude',
        role: 'intermediate',
        label: 'Center latitude in radians',
        binding: 'wgsl:latitude'
      },
      {
        id: 'angle',
        role: 'intermediate',
        label: 'Angular distance',
        binding: 'wgsl:angularDistance'
      },
      {
        id: 'bearing',
        role: 'intermediate',
        label: 'Sampled bearing in radians',
        binding: 'wgsl:bearing'
      },
      {
        id: 'distance',
        role: 'intermediate',
        label: 'Sampled scaled radius or ellipse offset length',
        binding: 'wgsl:distance'
      },
      {
        id: 'radius',
        role: 'constant',
        label: 'Context sphere radius',
        binding: 'wgsl:EARTH_RADIUS'
      },
      {id: 'output', role: 'output', label: 'Generated ring vertex', binding: 'wgsl:positionsOut'},
      {
        id: 'center',
        role: 'input',
        label: 'Shape center in position units or degrees',
        binding: 'wgsl:center'
      }
    ]
  }
} as const satisfies Readonly<Record<string, GPUComputationFormulaRef>>;
