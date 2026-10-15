// Run from the repository root: node scripts/lens-data/s2-extract.mjs
// Source datasets are read-only. No network access or dependencies are required.
import assert from 'node:assert/strict';
import {appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';

const SOURCE_DIRECTORY = 'examples/deck/spatial-analysis-showcase/public/data';
const OUTPUT_DIRECTORY = 'modules/experimental/src/computation-lenses/data';
const MAX_BYTES = 150_000;
const POLYGON_NOTE =
  'Largest part only, ranked by absolute exterior shoelace area minus holes, scaled by cos(mean latitude); holes of that part retained. Duplicate closing vertices removed; "ringsClosed": false. No simplification beyond coordinate rounding.';
const outputs = [];

function readDataset(datasetId) {
  const directory = join(SOURCE_DIRECTORY, datasetId);
  const manifest = JSON.parse(readFileSync(join(directory, 'manifest.json'), 'utf8'));
  assert.equal(manifest.id, datasetId);
  assert.equal(manifest.crs, 'EPSG:4326');
  const cache = new Map();
  return {
    manifest,
    readJson: filename => JSON.parse(readFileSync(join(directory, filename), 'utf8')),
    readCsv: filename => parseCsv(readFileSync(join(directory, filename), 'utf8')),
    column(name) {
      if (!cache.has(name)) {
        const descriptor = manifest.columns[name];
        assert(descriptor, `${datasetId}: missing column ${name}`);
        const readers = {
          uint8: [1, 'readUInt8'],
          uint16: [2, 'readUInt16LE'],
          uint32: [4, 'readUInt32LE'],
          float32: [4, 'readFloatLE'],
          float64: [8, 'readDoubleLE']
        };
        const [width, reader] = readers[descriptor.dtype] || [];
        assert(width, `${datasetId}.${name}: unsupported dtype ${descriptor.dtype}`);
        const buffer = readFileSync(join(directory, descriptor.file));
        const length = descriptor.length * descriptor.components;
        assert.equal(buffer.length, length * width, `${datasetId}.${name}: binary size mismatch`);
        cache.set(name, Array.from({length}, (_, index) => buffer[reader](index * width)));
      }
      return cache.get(name);
    }
  };
}

// Handles quoted commas, doubled quotes, CRLF and quoted newlines without dependencies.
function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let index = 0; index < text.length; index++) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        field += '"';
        index++;
      } else {
        quoted = !quoted;
      }
    } else if (!quoted && (character === ',' || character === '\n' || character === '\r')) {
      row.push(field);
      field = '';
      if (character !== ',') {
        rows.push(row);
        row = [];
        if (character === '\r' && text[index + 1] === '\n') index++;
      }
    } else {
      field += character;
    }
  }
  assert(!quoted, 'Unterminated CSV quoted field');
  if (field || row.length) rows.push([...row, field]);
  const headers = rows.shift();
  return rows.map(values => {
    assert.equal(values.length, headers.length, 'CSV row width mismatch');
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function roundNumber(value, decimals = 4) {
  return Number.isFinite(value) ? Number(value.toFixed(decimals)) : null;
}

function selectRows(dataset, predicate) {
  return Array.from({length: dataset.manifest.count}, (_, index) => index).filter(predicate);
}

function selectColumns(dataset, rows, names) {
  return Object.fromEntries(names.map(name => {
    const values = dataset.column(name);
    return [name, rows.map(index => roundNumber(values[index]))];
  }));
}

function selectPositions(values, rows, decimals = 5) {
  return rows.flatMap(index => [
    roundNumber(values[index * 2], decimals),
    roundNumber(values[index * 2 + 1], decimals)
  ]);
}

function flattenText(value) {
  return typeof value === 'string' ? value : JSON.stringify(value);
}

// Licences the source manifests do not state, from the upstream providers' terms.
const LICENCE_OVERRIDES = {
  'chicago-tracts':
    'Public domain: US Census Bureau cartographic boundaries and ACS, CDC/ATSDR SVI 2022, CDC PLACES; ' +
    'natureObs/birdObs counts derive from iNaturalist observations (CC0 / CC BY 4.0 / CC BY-NC 4.0, iNaturalist contributors)',
  'chicago-community-areas': 'City of Chicago Data Portal Terms of Use',
  'us-counties':
    'Public domain: US Census Bureau cb_2022 boundaries, CDC/ATSDR SVI 2022, USDA ERS / BLS LAUS unemployment',
  'ghcn-stations': 'Public domain (NOAA NCEI GHCN-Daily, US Government work)'
};

function makeMetadata(dataset, id, title, note) {
  const {manifest} = dataset;
  const properties = manifest.properties || {};
  const source = properties.source || properties.sources || properties.licenses ||
    properties.attribution || properties.description;
  assert(source, `${manifest.id}: no source description`);
  const licence = LICENCE_OVERRIDES[manifest.id] || manifest.licence || manifest.license ||
    properties.licence || properties.license || properties.licenses;
  return {
    id,
    title,
    source: `${manifest.id}: ${flattenText(source)}`,
    licence: licence ? flattenText(licence) :
      'Not specified in source manifest; consult upstream source terms.',
    crs: 'EPSG:4326',
    note
  };
}

function readPolygons(dataset) {
  const vertices = dataset.column('vertices');
  const ringOffsets = dataset.column('ringOffsets');
  const polygonRingOffsets = dataset.column('polygonRingOffsets');
  const features = Array.from({length: dataset.manifest.count}, () => []);
  let partFeature;
  if (dataset.manifest.columns.partFeature) {
    partFeature = dataset.column('partFeature');
  } else {
    const countyOffsets = dataset.column('countyPolygonOffsets');
    partFeature = [];
    for (let feature = 0; feature < features.length; feature++) {
      for (let part = countyOffsets[feature]; part < countyOffsets[feature + 1]; part++) {
        partFeature[part] = feature;
      }
    }
  }
  assert.equal(partFeature.length + 1, polygonRingOffsets.length);
  for (let part = 0; part < partFeature.length; part++) {
    const rings = [];
    for (let ring = polygonRingOffsets[part]; ring < polygonRingOffsets[part + 1]; ring++) {
      const points = [];
      for (let vertex = ringOffsets[ring]; vertex < ringOffsets[ring + 1]; vertex++) {
        points.push([vertices[vertex * 2], vertices[vertex * 2 + 1]]);
      }
      while (points.length > 1 && samePoint(points[0], points.at(-1))) points.pop();
      assert(points.length >= 3, `${dataset.manifest.id}: degenerate source ring`);
      rings.push(points);
    }
    features[partFeature[part]].push(rings);
  }
  assert(features.every(parts => parts.length > 0));
  return features;
}

function samePoint(first, second) {
  return first[0] === second[0] && first[1] === second[1];
}

function getRingArea(ring) {
  // Translate first to avoid cancellation with large absolute longitude/latitude.
  const [originLongitude, originLatitude] = ring[0];
  let twiceArea = 0;
  for (let index = 0; index < ring.length; index++) {
    const current = ring[index];
    const next = ring[(index + 1) % ring.length];
    twiceArea += (current[0] - originLongitude) * (next[1] - originLatitude) -
      (next[0] - originLongitude) * (current[1] - originLatitude);
  }
  return Math.abs(twiceArea) / 2;
}

function getPartArea(rings) {
  const meanLatitude = rings[0].reduce((sum, point) => sum + point[1], 0) / rings[0].length;
  return (getRingArea(rings[0]) - rings.slice(1).reduce((sum, ring) => sum + getRingArea(ring), 0)) *
    Math.cos(meanLatitude * Math.PI / 180);
}

function selectLargestPart(parts) {
  return parts.reduce((largest, part) => getPartArea(part) > getPartArea(largest) ? part : largest);
}

function makePolygons(features, rows, decimals = 5) {
  const polygons = {featureOffsets: [0], ringOffsets: [0], vertices: []};
  for (const feature of rows) {
    for (const ring of selectLargestPart(features[feature])) {
      const points = ring.map(point => point.map(value => roundNumber(value, decimals)));
      while (points.length > 1 && samePoint(points[0], points.at(-1))) points.pop();
      assert(points.length >= 3, 'Rounding collapsed a polygon ring');
      for (const point of points) polygons.vertices.push(...point);
      polygons.ringOffsets.push(polygons.vertices.length / 2);
    }
    polygons.featureOffsets.push(polygons.ringOffsets.length - 1);
  }
  return polygons;
}

function makeCsr(neighbors) {
  const result = {offsets: [0], neighbors: []};
  for (const adjacent of neighbors) {
    result.neighbors.push(...Array.from(adjacent).sort((first, second) => first - second));
    result.offsets.push(result.neighbors.length);
  }
  return result;
}

function makeQueen(features, rows, decimals) {
  const owners = new Map();
  const neighbors = rows.map(() => new Set());
  rows.forEach((feature, index) => {
    // Use all source parts for contiguity, even when only the largest is displayed.
    for (const part of features[feature]) {
      for (const ring of part) {
        for (const point of ring) {
          const key = point.map(value => roundNumber(value, decimals)).join(',');
          if (!owners.has(key)) owners.set(key, new Set());
          owners.get(key).add(index);
        }
      }
    }
  });
  for (const shared of owners.values()) {
    for (const first of shared) {
      for (const second of shared) {
        if (first !== second) neighbors[first].add(second);
      }
    }
  }
  return {queen: makeCsr(neighbors), islands: neighbors.filter(adjacent => adjacent.size === 0).length};
}

function containsPoint(part, point) {
  let inside = false;
  for (const ring of part) {
    for (let index = 0; index < ring.length; index++) {
      const first = ring[index];
      const second = ring[(index + 1) % ring.length];
      if (getOrientation(first, second, point) === 0 &&
        point[0] >= Math.min(first[0], second[0]) && point[0] <= Math.max(first[0], second[0]) &&
        point[1] >= Math.min(first[1], second[1]) && point[1] <= Math.max(first[1], second[1])) {
        return true;
      }
      if ((first[1] > point[1]) !== (second[1] > point[1]) &&
        point[0] < (second[0] - first[0]) * (point[1] - first[1]) /
          (second[1] - first[1]) + first[0]) inside = !inside;
    }
  }
  return inside;
}

function getOrientation(first, second, third) {
  return (second[0] - first[0]) * (third[1] - first[1]) -
    (second[1] - first[1]) * (third[0] - first[0]);
}

function inspectRings(polygons) {
  const selfIntersecting = [];
  let selfIntersectingRings = 0;
  let repeatedVertexRings = 0;
  for (let feature = 0; feature < polygons.featureOffsets.length - 1; feature++) {
    let featureIntersects = false;
    for (let ring = polygons.featureOffsets[feature]; ring < polygons.featureOffsets[feature + 1]; ring++) {
      const points = [];
      for (let vertex = polygons.ringOffsets[ring]; vertex < polygons.ringOffsets[ring + 1]; vertex++) {
        points.push(polygons.vertices.slice(vertex * 2, vertex * 2 + 2));
      }
      if (new Set(points.map(point => point.join(','))).size < points.length) repeatedVertexRings++;
      let intersects = false;
      for (let first = 0; first < points.length; first++) {
        for (let second = first + 1; second < points.length; second++) {
          if (second === first + 1 || (first === 0 && second === points.length - 1)) continue;
          const firstStart = points[first];
          const firstEnd = points[(first + 1) % points.length];
          const secondStart = points[second];
          const secondEnd = points[(second + 1) % points.length];
          // Strict signs detect proper crossings only: touches and collinear overlaps do not count.
          if (getOrientation(firstStart, firstEnd, secondStart) *
              getOrientation(firstStart, firstEnd, secondEnd) < 0 &&
            getOrientation(secondStart, secondEnd, firstStart) *
              getOrientation(secondStart, secondEnd, firstEnd) < 0) intersects = true;
        }
      }
      if (intersects) {
        selfIntersectingRings++;
        featureIntersects = true;
      }
    }
    if (featureIntersects) selfIntersecting.push(feature);
  }
  return {selfIntersecting, selfIntersectingRings, repeatedVertexRings};
}

function getBytes(object) {
  return Buffer.byteLength(JSON.stringify(object));
}

function addOutput(object, rows, details = {}) {
  const bytes = getBytes(object);
  assert(bytes <= MAX_BYTES, `${object.id}: ${bytes} bytes exceeds ${MAX_BYTES}-byte cap`);
  outputs.push({object, rows, bytes, ...details});
}

const areas = readDataset('chicago-community-areas');
const requestedAreas = [
  [1, 'Rogers Park'], [2, 'West Ridge'], [3, 'Uptown'], [4, 'Lincoln Square'],
  [5, 'North Center'], [6, 'Lake View'], [7, 'Lincoln Park'], [77, 'Edgewater'],
  [13, 'North Park'], [14, 'Albany Park'], [16, 'Irving Park'], [21, 'Avondale'],
  [22, 'Logan Square']
];
const areaAdjustments = [];
const areaRows = requestedAreas.map(([requestedId, name]) => {
  const index = areas.manifest.names.indexOf(name);
  assert(index >= 0, `Community area missing from manifest: ${name}`);
  assert.equal(areas.column('areaId')[index], index + 1);
  if (index + 1 !== requestedId) areaAdjustments.push(`${name}: ${requestedId} -> ${index + 1}`);
  return index;
}).sort((first, second) => first - second);
assert.equal(new Set(areaRows).size, 13);
const areaIds = new Set(areaRows.map(index => index + 1));
const areaFeatures = readPolygons(areas);
const studyParts = areaRows.flatMap(index => areaFeatures[index]);
const isInStudyArea = point => studyParts.some(part => containsPoint(part, point));
const studyNote = `Chicago North Side: ${areaRows.map(index => `${index + 1} ${areas.manifest.names[index]}`).join(', ')}. `;
process.stdout.write(`Community-area list adjustments: ${areaAdjustments.join('; ') || 'none'}\n`);

const tracts = readDataset('chicago-tracts');
const tractRows = selectRows(tracts, index => areaIds.has(tracts.column('communityArea')[index]));
const tractFeatures = readPolygons(tracts);
const tractIndices = new Map(tractRows.map((sourceIndex, subsetIndex) => [sourceIndex, subsetIndex]));
const tractQueen = makeQueen(tractFeatures, tractRows, 5);
addOutput({
  ...makeMetadata(tracts, 's2-chicago-north-tracts', 'Chicago North Side census tracts',
    `${studyNote}Selected by source communityArea. ${POLYGON_NOTE} Coordinates rounded to 5 decimals. Queen uses all source parts, shared vertices rounded to 5 decimals; islands: ${tractQueen.islands}.`),
  geoid: tractRows.map(index => tracts.manifest.geoid[index]),
  ...selectColumns(tracts, tractRows, [
    'communityArea', 'population', 'households', 'housingUnits', 'medianHouseholdIncome',
    'nhWhite', 'nhBlack', 'hispanic', 'nhAsian', 'nhOther', 'noVehicle', 'natureObs2023',
    'birdObs2023', 'areaKm2'
  ]),
  polygons: makePolygons(tractFeatures, tractRows),
  queen: tractQueen.queen
}, tractRows.length, {islands: tractQueen.islands});

addOutput({
  ...makeMetadata(areas, 's2-chicago-north-areas', 'Chicago North Side community areas',
    `${studyNote}${POLYGON_NOTE} Coordinates rounded to 5 decimals. ID/name list adjustments: ${areaAdjustments.join('; ') || 'none'}.`),
  ...selectColumns(areas, areaRows, ['areaId', 'areaKm2']),
  name: areaRows.map(index => areas.manifest.names[index]),
  polygons: makePolygons(areaFeatures, areaRows)
}, areaRows.length);

const nature = readDataset('chicago-nature');
const natureTimes = nature.column('timestamp');
const natureCandidates = selectRows(nature, index => areaIds.has(nature.column('communityArea')[index]))
  .sort((first, second) => natureTimes[first] - natureTimes[second] || first - second);
const natureStep = Math.ceil(natureCandidates.length / 2500) || 1;
const natureRows = natureCandidates.filter((_, index) => index % natureStep === 0);
assert(natureRows.length <= 2500);
addOutput({
  ...makeMetadata(nature, 's2-chicago-north-nature', 'Chicago North Side iNaturalist observations',
    `${studyNote}Kept ${natureRows.length} of ${natureCandidates.length} observations, every ${natureStep}-th row starting at zero after sorting by time (source index breaks ties). day is zero-based day of 2023; hour is local wall-clock hour, without timezone conversion. tract indexes s2-chicago-north-tracts, or -1 outside that subset. Coordinates rounded to 5 decimals.`),
  positions: selectPositions(nature.column('position'), natureRows),
  day: natureRows.map(index => Math.floor(natureTimes[index] / 86400)),
  hour: natureRows.map(index => Math.floor((natureTimes[index] % 86400) / 3600)),
  ...selectColumns(nature, natureRows, ['category', 'researchGrade']),
  categories: nature.manifest.columns.category.categories,
  tract: natureRows.map(index => tractIndices.get(nature.column('tract')[index]) ?? -1)
}, natureRows.length);

const facilities = readDataset('chicago-facilities');
const facilityNames = facilities.readCsv(facilities.manifest.namesCsv);
assert.equal(facilityNames.length, facilities.manifest.count);
const facilityPositions = facilities.column('position');
const facilityRows = selectRows(facilities, index =>
  isInStudyArea(facilityPositions.slice(index * 2, index * 2 + 2)));
addOutput({
  ...makeMetadata(facilities, 's2-chicago-north-facilities', 'Chicago North Side public facilities',
    `${studyNote}Even-odd point-in-polygon selection on full, unrounded source community-area parts and holes; boundary points included. Coordinates rounded to 5 decimals.`),
  positions: selectPositions(facilityPositions, facilityRows),
  ...selectColumns(facilities, facilityRows, ['category']),
  categories: facilities.manifest.columns.category.categories,
  name: facilityRows.map(index => facilityNames[index].name)
}, facilityRows.length);

const parks = readDataset('chicago-parks');
const parkFeatures = readPolygons(parks);
const parkCandidates = selectRows(parks, index => isInStudyArea(parkFeatures[index][0][0][0]));
function makeParks(rows) {
  const polygons = makePolygons(parkFeatures, rows);
  const inspection = inspectRings(polygons);
  return {
    object: {
      ...makeMetadata(parks, 's2-chicago-north-parks', 'Chicago North Side parks and green space',
        `${studyNote}Selection uses each feature's first source vertex inside full source community-area polygons (even-odd, boundary included). Kept ${rows.length} of ${parkCandidates.length} qualifying features${rows.length < parkCandidates.length ? ', largest by source areaKm2 to fit 150000 bytes (source index breaks ties)' : ', all fit'}. ${POLYGON_NOTE} Coordinates rounded to 5 decimals. Checks on every kept rounded ring: ${inspection.selfIntersectingRings} self-intersecting rings in ${inspection.selfIntersecting.length} features (proper crossings only); ${inspection.repeatedVertexRings} rings with repeated non-closing vertices. areaKm2 describes the full source feature; osmId may repeat across OSM ways and relations.`),
      name: rows.map(index => parks.manifest.names[parks.column('name')[index]] || null),
      kind: rows.map(index => parks.manifest.columns.kind.categories[parks.column('kind')[index]]),
      ...selectColumns(parks, rows, ['areaKm2', 'osmId']),
      polygons,
      selfIntersecting: inspection.selfIntersecting
    },
    inspection
  };
}
let parkRows = parkCandidates;
let parkOutput = makeParks(parkRows);
if (getBytes(parkOutput.object) > MAX_BYTES) {
  parkRows = [...parkCandidates].sort((first, second) =>
    parks.column('areaKm2')[second] - parks.column('areaKm2')[first] || first - second);
  do {
    parkRows.pop();
    assert(parkRows.length > 0, 'No park subset fits the byte cap');
    parkOutput = makeParks(parkRows);
  } while (getBytes(parkOutput.object) > MAX_BYTES);
}
addOutput(parkOutput.object, parkRows.length, {
  selfIntersectingRings: parkOutput.inspection.selfIntersectingRings,
  selfIntersectingFeatures: parkOutput.inspection.selfIntersecting.length,
  repeatedVertexRings: parkOutput.inspection.repeatedVertexRings
});

const roads = readDataset('chicago-roads');
const roadNodes = roads.column('nodes');
const edgeSources = roads.column('edgeSource');
const edgeTargets = roads.column('edgeTarget');
const edgeReverses = roads.column('edgeReverse');
const edgePathOffsets = roads.column('edgePathOffsets');
const edgeVertices = roads.column('edgeVertices');
function selectRoads(halfWidth) {
  const bbox = [-87.65 - halfWidth, 41.93 - halfWidth, -87.65 + halfWidth, 41.93 + halfWidth]
    .map(value => roundNumber(value, 5));
  const insideNodes = new Set(Array.from({length: roads.manifest.nodeCount}, (_, index) => index)
    .filter(index => isInBbox(roadNodes.slice(index * 2, index * 2 + 2), bbox)));
  const rows = selectRows(roads, index => {
    const source = edgeSources[index];
    const target = edgeTargets[index];
    if (!insideNodes.has(source) || !insideNodes.has(target)) return false;
    const reverse = edgeReverses[index];
    if (reverse === 0xffffffff) return true;
    assert.equal(edgeSources[reverse], target);
    assert.equal(edgeTargets[reverse], source);
    // A self-loop cannot satisfy source < target; none are retained from two-way pairs.
    return source < target;
  });
  const nodeRows = [...new Set(rows.flatMap(index => [edgeSources[index], edgeTargets[index]]))]
    .sort((first, second) => first - second);
  const nodeIndices = new Map(nodeRows.map((sourceIndex, subsetIndex) => [sourceIndex, subsetIndex]));
  const lines = {offsets: [0], vertices: []};
  for (const index of rows) {
    const start = edgePathOffsets[index];
    const end = edgePathOffsets[index + 1];
    assert(end - start >= 2);
    assert.deepEqual(edgeVertices.slice(start * 2, start * 2 + 2),
      roadNodes.slice(edgeSources[index] * 2, edgeSources[index] * 2 + 2));
    assert.deepEqual(edgeVertices.slice((end - 1) * 2, end * 2),
      roadNodes.slice(edgeTargets[index] * 2, edgeTargets[index] * 2 + 2));
    for (let vertex = start * 2; vertex < end * 2; vertex++) {
      lines.vertices.push(roundNumber(edgeVertices[vertex], 5));
    }
    lines.offsets.push(lines.vertices.length / 2);
  }
  return {
    rows,
    object: {
      ...makeMetadata(roads, 's2-chicago-north-roads', 'Lincoln Park and Lake View street network',
        `Both endpoints inside bbox ${JSON.stringify(bbox)} (west,south,east,north). Start with half-width 0.015 degrees; widen by 0.005 degrees if fewer than 600 retained edges; shrink by 0.001 degrees if needed for the byte cap. One-way edges retained; two-way pairs identified using edgeReverse retain source < target. Source edge order preserved; nodes renumbered in ascending original node index. edgePathOffsets are vertex indices and source polylines already include both end nodes. Coordinates rounded to 5 decimals. sourceNode holds edge source indices to avoid colliding with source provenance metadata; lengthM is source edgeLength rounded to 2 decimals.`),
      lines,
      ...selectColumns(roads, rows, ['edgeClass']),
      classes: roads.manifest.columns.edgeClass.categories,
      lengthM: rows.map(index => roundNumber(roads.column('edgeLength')[index], 2)),
      sourceNode: rows.map(index => nodeIndices.get(edgeSources[index])),
      target: rows.map(index => nodeIndices.get(edgeTargets[index])),
      nodePositions: selectPositions(roadNodes, nodeRows)
    }
  };
}

function isInBbox(point, bbox) {
  return point[0] >= bbox[0] && point[0] <= bbox[2] &&
    point[1] >= bbox[1] && point[1] <= bbox[3];
}

let roadHalfWidth = 0.015;
let roadOutput = selectRoads(roadHalfWidth);
if (roadOutput.rows.length < 600) {
  roadHalfWidth += 0.005;
  roadOutput = selectRoads(roadHalfWidth);
}
while (getBytes(roadOutput.object) > MAX_BYTES) {
  roadHalfWidth = roundNumber(roadHalfWidth - 0.001, 3);
  assert(roadHalfWidth > 0, 'No road bbox fits the byte cap');
  roadOutput = selectRoads(roadHalfWidth);
}
addOutput(roadOutput.object, roadOutput.rows.length, {nodes: roadOutput.object.nodePositions.length / 2});

function getHaversineDistance(first, second) {
  const radians = Math.PI / 180;
  const latitudeDifference = (second[1] - first[1]) * radians;
  const longitudeDifference = (second[0] - first[0]) * radians;
  const haversine = Math.sin(latitudeDifference / 2) ** 2 +
    Math.cos(first[1] * radians) * Math.cos(second[1] * radians) *
      Math.sin(longitudeDifference / 2) ** 2;
  return 2 * 6371008.8 * Math.asin(Math.sqrt(Math.min(1, haversine)));
}

function makeNearestNeighbors(centroids) {
  return makeCsr(centroids.map((point, index) => centroids
    .map((candidate, neighbor) => ({neighbor, distance: getHaversineDistance(point, candidate)}))
    .filter(candidate => candidate.neighbor !== index)
    .sort((first, second) => first.distance - second.distance || first.neighbor - second.neighbor)
    .slice(0, 6)
    .map(candidate => candidate.neighbor)));
}

const counties = readDataset('us-counties');
const mortality = readDataset('us-county-mortality');
assert.equal(counties.manifest.count, mortality.manifest.count);
assert(mortality.manifest.properties.alignedTo.includes('us-counties'));
const countyNames = counties.readJson('names.json');
for (const name of ['fips', 'name', 'state']) assert.equal(countyNames[name].length, counties.manifest.count);
const stateFips = new Set([17, 18, 39, 26, 55]);
const countyRows = selectRows(counties, index => stateFips.has(counties.column('stateFips')[index]));
for (const index of countyRows) {
  assert.equal(Number(countyNames.fips[index]), counties.column('fips')[index]);
  assert.equal(Number(countyNames.fips[index].slice(0, 2)), counties.column('stateFips')[index]);
}
const countyFeatures = readPolygons(counties);
const countyQueen = makeQueen(countyFeatures, countyRows, 4);
const countyCentroids = countyRows.map(index => counties.column('centroid').slice(index * 2, index * 2 + 2));
const unemploymentYears = Array.from({length: 16}, (_, index) => 2008 + index);
const countyNote =
  `All counties in IL (17), IN (18), OH (39), MI (26), WI (55), in source FIPS order. Queen uses all source parts, shared vertices rounded to 4 decimals; islands: ${countyQueen.islands}. knn6 is directed, using full-precision source centroids and haversine distance, ties by subset index; CSR neighbors sorted by index. Centroids rounded to 4 decimals. Unemployment values are percentages, rounded to 2 decimals. deaths counts motor-vehicle crash deaths in the county of crash (FARS), 2017-2023 total, not all-cause mortality; deaths2023 is the 2023 count; personYears is seven times 2020 Census residents, while population is ACS 2018-2022. ${POLYGON_NOTE}`;
const countyOutput = {
  ...makeMetadata(counties, 's2-midwest-counties', 'Midwest counties: population, unemployment and traffic deaths', countyNote),
  fips: countyRows.map(index => countyNames.fips[index]),
  name: countyRows.map(index => countyNames.name[index]),
  state: countyRows.map(index => countyNames.state[index]),
  centroid: countyCentroids.map(point => point.map(value => roundNumber(value, 4))),
  ...selectColumns(counties, countyRows, ['population', 'medianHouseholdIncome']),
  unemploymentRate: {
    years: unemploymentYears,
    values: countyRows.map(index => unemploymentYears.map(year =>
      roundNumber(counties.column(`unemploymentRate${year}`)[index], 2)))
  },
  ...selectColumns(mortality, countyRows, ['deaths', 'deaths2023', 'personYears']),
  queen: countyQueen.queen,
  knn6: makeNearestNeighbors(countyCentroids)
};
countyOutput.source += `; us-county-mortality: ${mortality.manifest.properties.sources}`;
countyOutput.licence =
  `${countyOutput.licence}; traffic deaths: NHTSA FARS 2017-2023, public domain`;
const countyPolygons = makePolygons(countyFeatures, countyRows, 4);
const countyWithPolygons = {
  ...countyOutput,
  note: `${countyNote} Polygon coordinates rounded to 4 decimals.`,
  polygons: countyPolygons
};
if (getBytes(countyWithPolygons) <= MAX_BYTES) {
  addOutput(countyWithPolygons, countyRows.length, {islands: countyQueen.islands});
} else {
  countyOutput.note += ' Display geometry is in s2-midwest-county-shapes.json, rounded to 3 decimals; queen remains computed at 4 decimals.';
  addOutput(countyOutput, countyRows.length, {islands: countyQueen.islands});
  addOutput({
    ...makeMetadata(counties, 's2-midwest-county-shapes', 'Midwest county display polygons',
      `All ${countyRows.length} counties in IL, IN, OH, MI, WI; FIPS order matches s2-midwest-counties.json. Split from attributes to satisfy the 150000-byte cap. ${POLYGON_NOTE} Coordinates rounded to 3 decimals for display; use the companion table's queen graph computed from full source geometry at 4 decimals.`),
    fips: countyOutput.fips,
    polygons: makePolygons(countyFeatures, countyRows, 3)
  }, countyRows.length);
}

const rain = readDataset('ghcn-stations');
const stationNames = rain.readCsv('stations.csv');
assert.equal(stationNames.length, rain.manifest.count);
const rainPositions = rain.column('position');
const rainCandidates = selectRows(rain, index => Number.isFinite(rain.column('prcp')[index]) &&
  isInBbox(rainPositions.slice(index * 2, index * 2 + 2), [-86, 32, -78, 37.5]));
const rainStep = Math.ceil(rainCandidates.length / 700) || 1;
const rainRows = rainCandidates.filter((_, index) => index % rainStep === 0);
assert(rainRows.length <= 700);
addOutput({
  ...makeMetadata(rain, 's2-helene-rain', 'Hurricane Helene daily rainfall stations',
    `${rain.manifest.columns.prcp.description}. ${rain.manifest.properties.event}. Stations inside lon [-86,-78], lat [32,37.5] with finite precipitation; kept ${rainRows.length} of ${rainCandidates.length}, every ${rainStep}-th row starting at zero in source order. Coordinates rounded to 5 decimals; weather values to 1 decimal. NaN becomes null. stationId holds source station IDs to preserve the top-level dataset id.`),
  positions: selectPositions(rainPositions, rainRows),
  prcpMm: rainRows.map(index => roundNumber(rain.column('prcp')[index], 1)),
  tmaxC: rainRows.map(index => roundNumber(rain.column('tmax')[index], 1)),
  elevationM: rainRows.map(index => roundNumber(rain.column('elevation')[index], 1)),
  stationId: rainRows.map(index => stationNames[index].ID),
  name: rainRows.map(index => stationNames[index].name),
  state: rainRows.map(index => stationNames[index].state)
}, rainRows.length);

// Validate every result before writing any output, so a size failure cannot leave a partial extraction.
const totalBytes = outputs.reduce((total, output) => total + output.bytes, 0);
assert(totalBytes <= 4_000_000, `S2 total ${totalBytes} exceeds 4 MB`);
mkdirSync(OUTPUT_DIRECTORY, {recursive: true});
for (const {object, rows, bytes, ...details} of outputs) {
  const filename = `${object.id}.json`;
  writeFileSync(join(OUTPUT_DIRECTORY, filename), JSON.stringify(object));
  process.stdout.write(`${filename}: ${rows} rows/features, ${bytes} bytes${Object.keys(details).length ? `, ${JSON.stringify(details)}` : ''}\n`);
}

// Re-read immediately before editing the shared README; append only missing rows, never rewrite it.
const readmePath = join(OUTPUT_DIRECTORY, 'README.md');
const readme = existsSync(readmePath) ? readFileSync(readmePath, 'utf8') : '';
const escapeCell = value => value.replaceAll('|', '\\|').replaceAll('\n', ' ');
// Other extractions may create the shared table first, with a different column schema.
const existingHeader = readme.split('\n').find(line => /^\|\s*file\s*\|/i.test(line));
const headerCells = existingHeader ? existingHeader.split('|').slice(1, -1)
  .map(cell => cell.trim().toLowerCase()) : ['file', 'dataset', 'licence', 'extracted by'];
const readmeRows = outputs.filter(({object}) => !readme.includes(`| ${object.id}.json |`))
  .map(({object, rows}) => {
    const cells = {
      file: `${object.id}.json`,
      dataset: object.title,
      source: object.source,
      licence: object.licence,
      license: object.licence,
      'extracted by': '`scripts/lens-data/s2-extract.mjs`',
      'what was extracted': `${rows} rows/features. ${object.note} Regenerate with \`node scripts/lens-data/s2-extract.mjs\`.`
    };
    return `| ${headerCells.map(header => {
      assert(cells[header], `Unsupported shared README column: ${header}`);
      return escapeCell(cells[header]);
    }).join(' | ')} |`;
  });
if (!existsSync(readmePath)) {
  appendFileSync(readmePath,
    'Docs-sized real open data for lens scenarios; each file ≤ 150 KB, total ≤ 4 MB; extracted by `scripts/lens-data/*`.\n\n' +
    '| File | Dataset | Licence | Extracted by |\n| --- | --- | --- | --- |\n');
}
if (readmeRows.length) {
  appendFileSync(readmePath, `${readme && !readme.endsWith('\n') ? '\n' : ''}${readmeRows.join('\n')}\n`);
}
process.stdout.write(`Total: ${outputs.length} JSON files, ${totalBytes} bytes\n`);
