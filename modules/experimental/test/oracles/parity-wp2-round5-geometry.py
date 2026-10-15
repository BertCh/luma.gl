# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent GEOS geometry references, f32 inputs, full output arrays.

https://shapely.readthedocs.io/en/stable/reference/shapely.intersection.html
https://shapely.readthedocs.io/en/stable/reference/shapely.offset_curve.html
Lon/lat: all US county rings, implicit closure (remove explicit duplicate endpoint
only), no repair/simplification. np.unwrap(period=360) expresses short-way edges;
GEOS intersects each ring with the main/east world boxes. Fixed doubled topology,
hole roles and input winding are representation adapters, not GEOS union semantics.
No pole/ambiguous/invalid-ring repair claim. Synthetic seam shell/hole, multipart,
both seam directions and reject mode. Compare cyclic rings, preserving orientation.
Outline: first 128 complete roads from the existing hashed AEQD window. GEOS
Point.buffer(quad_segs=4) gives each 16-sided disc; offset_curve(+/-distance) gives
segment sides. Triangles follow the public fixed disc-fan/quad storage contract;
no local TS or WGSL oracle. Includes points, lines, closed rings, stationary and
singleton rows, and two widths on the same graph. Not a dissolved buffer API.
Rectangle: ALL 3105 segments of the 1890-road window, each as an independent line,
so GEOS does not union retracing. GEOS intersection supplies endpoints, ordered
by line_locate_point. Corner-only Point results omitted by public contract;
stationary input uses Point.intersects and duplicates its coordinate when kept.
Synthetic also tests self-crossing polylines and connected clipping. Each segment
is intersected separately; concatenate only through an original retained vertex,
preserving self-crossing multiplicity instead of GEOS whole-line noding/union.
G bound: 16 ulps at the largest input/output coordinate scale, rtol=0. This covers
short interpolation/normalization chains; no tolerance fitted to GPU results.
"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import shapely
from shapely import Polygon,LineString,Point,box
HERE=Path(__file__).resolve().parent
DATA=HERE.parents[3]/'examples/deck/spatial-analysis-showcase/public/data'
VERSIONS=dict(python=platform.python_version(),numpy=np.__version__,shapely=shapely.__version__,geos=shapely.geos_version_string)

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(topic,case,source,reference):
    path=HERE/f'fixtures/parity-wp2-{topic}.json'
    fixture=dict(generator=Path(__file__).name,versions=VERSIONS,source=source,reference=reference,
                 conventions=dict(inputPrecision='float32 before reference',units='degrees' if topic.startswith('lon-lat') else 'meters',crs='EPSG:4326 short-way linear edges' if topic.startswith('lon-lat') else 'local planar AEQD for roads; Cartesian for synthetic'),
                 tolerance=dict(profile='G',absolute='16 ulps at input/output coordinate scale; see generator derivation',relative=0),cases=[case])
    path.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n')
    assert path.stat().st_size<1500000,(path,path.stat().st_size)
    print(path.name,path.stat().st_size)
def bound(positions): return 16*float(np.spacing(np.float32(max(1,np.abs(positions).max()))))
def flatten(features):
    positions=[];rings=[0];polygons=[0];feature_offsets=[0]
    for feature in features:
        for polygon in feature:
            for ring in polygon:
                positions.extend(np.asarray(ring).reshape(-1).tolist());rings.append(len(positions)//2)
            polygons.append(len(rings)-1)
        feature_offsets.append(len(polygons)-1)
    return dict(positions=positions,ringOffsets=rings,polygonOffsets=polygons,featureOffsets=feature_offsets)
def open_ring(points):
    points=np.asarray(points,dtype=np.float32).astype(float)
    return points[:-1] if np.array_equal(points[0],points[-1]) else points

def prepare(name,features,reject=False):
    features=[[[open_ring(ring) for ring in polygon] for polygon in feature] for feature in features]
    source=flatten(features);expected=[];flags=[];invalid=0
    for feature in features:
        unwrapped=[];feature_crossing=False
        for polygon in feature:
            rings=[]
            for ring in polygon:
                crossing=bool(np.any(np.abs(np.diff(np.r_[ring[:,0],ring[0,0]]))>180))
                flags.append(int(crossing));feature_crossing|=crossing
                points=ring.copy();points[:,0]=np.unwrap(points[:,0],period=360)
                assert np.ptp(points[:,0])<180
                rings.append(points)
            shell_min=rings[0][:,0].min();centre=(shell_min+rings[0][:,0].max())/2
            for index,ring in enumerate(rings):
                if index: ring[:,0]+=360*np.floor((centre-ring[0,0])/360+.5)
                ring[:,0]-=360*np.floor((shell_min+180)/360)
            unwrapped.append(rings)
        invalid+=int(reject and feature_crossing)
        copies=[]
        for copy in range(2):
            for polygon in unwrapped:
                output_rings=[]
                for ring in polygon:
                    geometry=Polygon(ring)
                    assert geometry.is_valid
                    clipped=shapely.intersection(geometry,box(-180+360*copy,-90,180+360*copy,90))
                    if clipped.is_empty or (reject and feature_crossing): coordinates=[]
                    else:
                        assert clipped.geom_type=='Polygon'
                        clipped=shapely.orient_polygons(clipped,exterior_cw=not shapely.is_ccw(geometry.exterior))
                        coordinates=np.asarray(clipped.exterior.coords)[:-1].copy();coordinates[:,0]-=360*copy
                        coordinates=coordinates.tolist()
                    output_rings.append(coordinates)
                copies.append(output_rings)
        expected.append(copies)
    output=flatten(expected)
    return dict(name=name,**source,reject=reject,expected=output,ringFlags=flags,invalidCount=invalid,absolute=bound(source['positions']))

county=DATA/'us-counties';names=['vertices.bin','ringOffsets.bin','polygonRingOffsets.bin','countyPolygonOffsets.bin']
vertices=np.fromfile(county/names[0],'<f4').reshape(-1,2);ring_offsets=np.fromfile(county/names[1],'<u4');polygon_offsets=np.fromfile(county/names[2],'<u4');feature_offsets=np.fromfile(county/names[3],'<u4')
county_features=[[[vertices[ring_offsets[ring]:ring_offsets[ring+1]] for ring in range(polygon_offsets[polygon],polygon_offsets[polygon+1])] for polygon in range(feature_offsets[feature],feature_offsets[feature+1])] for feature in range(len(feature_offsets)-1)]
for start in range(0,len(county_features),256):
    stop=min(start+256,len(county_features))
    write(f'lon-lat-polygons-counties-{start//256:02}',prepare(f'counties-{start}-{stop}',county_features[start:stop]),dict(dataset='us-counties',hashes={name:digest(county/name) for name in names},sourceRange=[start,stop],fullCount=len(county_features)), 'shapely.intersection/orient_polygons; numpy.unwrap representation adapter')
seam=[[[[[177,-19],[-178,-19],[-178,-16],[177,-16]],[[-179,-18],[179,-18],[179,-17],[-179,-17]]]],[[[[-179,10],[178,11],[178,14],[-179,14]]],[[[20,20],[21,20],[21,21],[20,21]]]]]
for reject in [False,True]: write('lon-lat-polygons-'+('reject' if reject else 'seams'),prepare('seams-holes-multipart'+str(reject),seam,reject),dict(dataset='synthetic'),'shapely.intersection/orient_polygons; numpy.unwrap representation adapter')

road_path=HERE/'fixtures/parity-wp2-prepared-nearest.json';roads=json.loads(road_path.read_text())['cases'][0]
road_positions=np.array(roads['frames'][0]['positions']).reshape(-1,2);road_offsets=roads['lineOffsets'];lines=[road_positions[start:stop] for start,stop in zip(road_offsets[:-1],road_offsets[1:])]
road_source=dict(dataset='complete Chicago road window; WGS84 AEQD -87.63,41.88 meters; f32 1/16m snap',hashes={road_path.name:digest(road_path)},completeRoads=len(lines))

def outline(name,lines,mode):
    lines=[np.asarray(line,dtype=np.float32).astype(float) for line in lines]
    flat=np.concatenate(lines);offsets=np.r_[0,np.cumsum([len(line) for line in lines])].tolist();frames=[]
    for distance in [2.,25.]:
        expected=[]
        for line in lines:
            for index,center in enumerate(line):
                # GEOS starts due east and proceeds clockwise; reverse circular indices for CCW fan.
                disc=np.asarray(Point(center).buffer(distance,quad_segs=4).exterior.coords)[:-1]
                for segment in range(16): expected.extend([center,disc[-segment%16],disc[-(segment+1)%16]])
                endpoint=line[(index+1)%len(line)] if mode=='polygons' or (mode=='lines' and index+1<len(line)) else center
                if np.array_equal(endpoint,center): expected.extend([center]*6)
                else:
                    segment=LineString([center,endpoint]);left=list(shapely.offset_curve(segment,distance).coords);right=list(shapely.offset_curve(segment,-distance).coords)
                    expected.extend([left[0],right[0],right[-1],left[0],right[-1],left[-1]])
        frames.append(dict(distance=distance,positions=np.asarray(expected).reshape(-1).tolist()))
    return dict(name=name,geometryType=mode,positions=flat.reshape(-1).tolist(),lineOffsets=offsets,frames=frames,absolute=bound(np.r_[flat.reshape(-1),np.array(frames[-1]['positions'])]))
for start in range(0,128,32):
    write(f'outline-roads-{start//32:02}',outline(f'roads-{start}-{start+32}',lines[start:start+32],'lines'),dict(**road_source,selection=[start,start+32]),'shapely.Point.buffer + shapely.offset_curve; public triangle layout adapter')
for mode in ['points','lines','polygons']:
    write(f'outline-{mode}',outline(mode,[[[0,0],[10,0],[10,10],[0,10]],[[3,3]],[[20,20],[20,20],[23,25]]],mode),dict(dataset='synthetic'),'shapely.Point.buffer + shapely.offset_curve; public triangle layout adapter')

def clip(name,lines,rectangles):
    lines=[np.asarray(line,dtype=np.float32).astype(float) for line in lines];flat=np.concatenate(lines)
    frames=[]
    for rectangle in rectangles:
        region=box(*rectangle);expected=[];owners=[];offsets=[0]
        for owner,line in enumerate(lines):
            parts=[]; previous_end=None
            for index in range(len(line)-1):
                first,second=line[index:index+2]
                if np.array_equal(first,second):
                    coordinates=[first,second] if region.intersects(Point(first)) else []
                else:
                    segment=LineString([first,second]);intersection=segment.intersection(region)
                    coordinates=list(intersection.coords) if intersection.geom_type=='LineString' and intersection.length>0 else []
                    if coordinates and segment.project(Point(coordinates[0]))>segment.project(Point(coordinates[-1])): coordinates.reverse()
                if not coordinates:
                    previous_end=None
                    continue
                if previous_end is not None and np.array_equal(previous_end,first) and np.array_equal(coordinates[0],first):
                    parts[-1].append(coordinates[-1])
                else: parts.append(list(coordinates))
                previous_end=coordinates[-1]
            for part in parts:
                expected.extend(np.asarray(part).tolist());owners.append(owner);offsets.append(len(expected))
        frames.append(dict(rectangle=rectangle,positions=np.asarray(expected).reshape(-1).tolist(),lineOffsets=offsets,sourceLineIndices=owners))
    return dict(name=name,positions=flat.reshape(-1).tolist(),lineOffsets=np.r_[0,np.cumsum([len(line) for line in lines])].tolist(),frames=frames,absolute=bound(flat))
segments=[line[index:index+2] for line in lines for index in range(len(line)-1)]
write('rectangle-roads',clip('all-road-segments',segments,[[-1000.03125,-1000.03125,1000.03125,1000.03125],[-500.03125,0.03125,500.03125,2000.03125]]),road_source,'shapely.intersection + line_locate_point')
write('rectangle-edges',clip('boundaries-corner-stationary-connected',[[[-5,5],[5,5],[15,5],[15,15],[5,12],[-2,2]],[[-1,-1],[0,0]],[[0,1],[0,9]],[[4,4],[4,4]],[[20,20],[20,20]],[[1,1]],[[9,9],[1,1]]],[[0,0,10,10],[2,2,8,8]]),dict(dataset='synthetic'),'shapely.intersection + line_locate_point')
