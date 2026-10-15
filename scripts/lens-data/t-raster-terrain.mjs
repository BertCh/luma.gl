// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Run from any directory: node scripts/lens-data/t-raster-terrain.mjs.
import {appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {PNG} from 'pngjs';

const REPOSITORY = resolve(dirname(decodeURIComponent(new URL(import.meta.url).pathname)), '../..');
const SOURCE_DIRECTORY = resolve(REPOSITORY, 'examples/deck/spatial-analysis-showcase/public/data');
const OUTPUT_DIRECTORY = resolve(REPOSITORY, 'modules/experimental/src/computation-lenses/data');
const SCRIPT = 'scripts/lens-data/t-raster-terrain.mjs';
const WINDOW_START = 366;
const BLOCK_SIZE = 6;
const GRID_SIZE = 64;
const DEM_SOURCE =
  'AWS Terrain Tiles (Terrarium, USGS 3DEP ~10 m) resampled to 20 m by the showcase, block mean to 120 m';
const DEM_LICENCE =
  'Public domain (USGS 3DEP); terrain tiles by Mapzen/Tilezen on AWS Open Data';

function checkCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function readJson(dataset, file = 'manifest.json') {
  return JSON.parse(readFileSync(resolve(SOURCE_DIRECTORY, dataset, file), 'utf8'));
}

// Read all manifests before opening their datasets.
const fireManifest = readJson('dixie-fire');
const perimeterManifest = readJson('dixie-perimeter');
const windManifest = readJson('gfs-wind');
const sourceGrid = fireManifest.raster;
checkCondition(sourceGrid.width === 750 && sourceGrid.height === 750, 'Unexpected Dixie grid');
checkCondition(sourceGrid.cellSizeM === 20, 'Unexpected Dixie cell size');
const [minimumEasting, minimumNorthing, maximumEasting, maximumNorthing] = sourceGrid.boundsProjected;
const origin = [minimumEasting + WINDOW_START * 20, maximumNorthing - WINDOW_START * 20];
const windowBounds = [origin[0], origin[1] - 384 * 20, origin[0] + 384 * 20, origin[1]];

function readRaster(dataset, file, width, height) {
  const bytes = readFileSync(resolve(SOURCE_DIRECTORY, dataset, file));
  const raster = PNG.sync.read(bytes);
  checkCondition(raster.width === width && raster.height === height, `Unexpected dimensions: ${file}`);
  checkCondition(bytes[24] === 8, `Expected 8-bit PNG: ${file}`);
  console.log(`${dataset}/${file}: PNG colour type ${bytes[25]}, ${width} x ${height}`);
  return {raster, colourType: bytes[25]};
}

function roundNumber(value, places = 0) {
  return Number(value.toFixed(places));
}

function summarizeValues(values) {
  return {
    min: roundNumber(Math.min(...values), 3),
    max: roundNumber(Math.max(...values), 3),
    mean: roundNumber(values.reduce((sum, value) => sum + value, 0) / values.length, 3)
  };
}

function makeHeader(id, description, source, licence, unit) {
  return {id, description, source, licence, crs: 'EPSG:32610', width: GRID_SIZE,
    height: GRID_SIZE, cellSize: 120, origin, unit};
}

function aggregateBlocks(readValue, aggregateValues) {
  const values = [];
  for (let row = 0; row < GRID_SIZE; row++) {
    for (let column = 0; column < GRID_SIZE; column++) {
      const block = [];
      for (let rowOffset = 0; rowOffset < BLOCK_SIZE; rowOffset++) {
        for (let columnOffset = 0; columnOffset < BLOCK_SIZE; columnOffset++) {
          const sourceRow = WINDOW_START + row * BLOCK_SIZE + rowOffset;
          const sourceColumn = WINDOW_START + column * BLOCK_SIZE + columnOffset;
          block.push(readValue(sourceRow * sourceGrid.width + sourceColumn));
        }
      }
      values.push(aggregateValues(block));
    }
  }
  return values;
}

function calculateMean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

const {raster: terrainRaster} = readRaster('dixie-fire', 'dem.png', 750, 750);
function decodeElevation(index) {
  const offset = index * 4;
  return terrainRaster.data[offset] * 256 + terrainRaster.data[offset + 1] +
    terrainRaster.data[offset + 2] / 256 - 32768;
}
const dem = {...makeHeader('t-dixie-dem',
  'Greenville window W: 6 x 6 block mean of Terrarium elevations; rows north to south.',
  DEM_SOURCE, DEM_LICENCE, 'm'),
values: aggregateBlocks(decodeElevation, values => roundNumber(calculateMean(values), 1))};
console.log('DEM min/max/mean (m):', summarizeValues(dem.values));

const bands = {...makeHeader('t-dixie-bands',
  'Greenville window W: Sentinel-2 B08/B12, 6 x 6 means over non-zero pixels, rounded DN; rows north to south.',
  'Copernicus Sentinel-2 L2A (scenes S2B_10TFK_20210713, S2B_10TFK_20210921)',
  'Contains modified Copernicus Sentinel data 2021 (free and open, Copernicus licence)',
  'reflectance x 10000'), bands: {}, scale: 0.0001, noData: 0,
  dates: {before: '2021-07-13', after: '2021-09-21'}};
for (const [name, file] of Object.entries({nirBefore: 'B08_before.bin', swirBefore: 'B12_before.bin',
  nirAfter: 'B08_after.bin', swirAfter: 'B12_after.bin'})) {
  const bytes = readFileSync(resolve(SOURCE_DIRECTORY, 'dixie-fire', file));
  checkCondition(bytes.length === 750 * 750 * 2, `Unexpected band size: ${file}`);
  bands.bands[name] = aggregateBlocks(index => bytes.readUInt16LE(index * 2),
    values => Math.round(calculateMean(values.filter(value => value !== 0))));
  console.log(`${name} mean (DN, valid blocks):`, roundNumber(calculateMean(bands.bands[name].filter(Boolean)), 3));
}
let validBlockCount = 0;
let burnedBlockCount = 0;
for (let index = 0; index < GRID_SIZE ** 2; index++) {
  const [nirBefore, swirBefore, nirAfter, swirAfter] = Object.values(bands.bands).map(values => values[index]);
  if (![nirBefore, swirBefore, nirAfter, swirAfter].every(value => value > 0)) continue;
  validBlockCount++;
  const before = (nirBefore - swirBefore) / (nirBefore + swirBefore);
  const after = (nirAfter - swirAfter) / (nirAfter + swirAfter);
  if (before - after > 0.27) burnedBlockCount++;
}
console.log(`dNBR > 0.27: ${burnedBlockCount}/${validBlockCount} valid blocks = ${burnedBlockCount / validBlockCount}; fraction of all blocks = ${burnedBlockCount / GRID_SIZE ** 2}`);

const {raster: coverRaster, colourType} = readRaster('dixie-fire', 'worldcover.png', 750, 750);
checkCondition([0, 2, 4, 6].includes(colourType), 'WorldCover must use grayscale or RGB class codes');
const declaredClasses = fireManifest.rasters.worldcover.classes;
const sourceCodes = new Set();
for (let index = 0; index < 750 * 750; index++) sourceCodes.add(coverRaster.data[index * 4]);
checkCondition([...sourceCodes].every(code => Object.hasOwn(declaredClasses, code)), 'Unknown WorldCover code');
console.log('WorldCover source codes (all match manifest):', [...sourceCodes].sort((first, second) => first - second));
function calculateMode(values) {
  const counts = new Map();
  for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
  return [...counts].sort(([firstCode, firstCount], [secondCode, secondCount]) =>
    secondCount - firstCount || firstCode - secondCode)[0][0];
}
const coverValues = aggregateBlocks(index => coverRaster.data[index * 4], calculateMode);
const coverCounts = {};
for (const value of coverValues) coverCounts[value] = (coverCounts[value] || 0) + 1;
const landcover = {...makeHeader('t-dixie-landcover',
  'Greenville window W: WorldCover 6 x 6 block mode; ties select the smallest code; rows north to south.',
  'ESA WorldCover 2021 v200, showcase 20 m grid, block mode to 120 m',
  'ESA WorldCover 2021 v200 (c) ESA, CC BY 4.0', 'class code'), values: coverValues,
  classes: Object.fromEntries(Object.keys(coverCounts).map(code => [code, declaredClasses[code]]))};
console.log('Land-cover output codes/counts:', coverCounts);

// WGS84 transverse Mercator, USGS Snyder series through longitude^6 (northing)
// and longitude^5 (easting). Zone 10N: central meridian -123 degrees, k0 0.9996.
function projectCoordinate([longitude, latitude]) {
  const semiMajorAxis = 6378137;
  const flattening = 1 / 298.257223563;
  const eccentricitySquared = flattening * (2 - flattening);
  const secondEccentricitySquared = eccentricitySquared / (1 - eccentricitySquared);
  const latitudeRadians = latitude * Math.PI / 180;
  const sineLatitude = Math.sin(latitudeRadians);
  const cosineLatitude = Math.cos(latitudeRadians);
  const tangentLatitude = Math.tan(latitudeRadians);
  const primeVerticalRadius = semiMajorAxis / Math.sqrt(1 - eccentricitySquared * sineLatitude ** 2);
  const tangentSquared = tangentLatitude ** 2;
  const curvatureTerm = secondEccentricitySquared * cosineLatitude ** 2;
  const longitudeTerm = cosineLatitude * (longitude + 123) * Math.PI / 180;
  const meridianArc = semiMajorAxis * (
    (1 - eccentricitySquared / 4 - 3 * eccentricitySquared ** 2 / 64 - 5 * eccentricitySquared ** 3 / 256) * latitudeRadians -
    (3 * eccentricitySquared / 8 + 3 * eccentricitySquared ** 2 / 32 + 45 * eccentricitySquared ** 3 / 1024) * Math.sin(2 * latitudeRadians) +
    (15 * eccentricitySquared ** 2 / 256 + 45 * eccentricitySquared ** 3 / 1024) * Math.sin(4 * latitudeRadians) -
    35 * eccentricitySquared ** 3 / 3072 * Math.sin(6 * latitudeRadians));
  const easting = 500000 + 0.9996 * primeVerticalRadius * (longitudeTerm +
    (1 - tangentSquared + curvatureTerm) * longitudeTerm ** 3 / 6 +
    (5 - 18 * tangentSquared + tangentSquared ** 2 + 72 * curvatureTerm - 58 * secondEccentricitySquared) * longitudeTerm ** 5 / 120);
  const northing = 0.9996 * (meridianArc + primeVerticalRadius * tangentLatitude * (
    longitudeTerm ** 2 / 2 + (5 - tangentSquared + 9 * curvatureTerm + 4 * curvatureTerm ** 2) * longitudeTerm ** 4 / 24 +
    (61 - 58 * tangentSquared + tangentSquared ** 2 + 600 * curvatureTerm - 330 * secondEccentricitySquared) * longitudeTerm ** 6 / 720));
  return [easting, northing];
}

// The manifest bbox stores the inverse-projected SW and NE grid corners.
const [west, south, east, north] = fireManifest.bbox;
for (const [name, coordinate, expected] of [
  ['SW', [west, south], [minimumEasting, minimumNorthing]],
  ['NE', [east, north], [maximumEasting, maximumNorthing]]
]) {
  const projected = projectCoordinate(coordinate);
  const residual = projected.map((value, index) => value - expected[index]);
  console.log(`UTM ${name} bbox corner residual (m):`, residual.map(value => roundNumber(value, 6)));
  checkCondition(Math.hypot(...residual) <= 30, 'Projection residual exceeds 30 m');
}

function equalPoints(first, second) {
  return first[0] === second[0] && first[1] === second[1];
}

function openRing(ring) {
  const points = ring.filter((point, index) => index === 0 || !equalPoints(point, ring[index - 1]));
  if (points.length > 1 && equalPoints(points[0], points.at(-1))) points.pop();
  return points;
}

// Sutherland-Hodgman: retain original ring roles, including clipped holes.
function clipRing(ring) {
  let points = openRing(ring);
  for (const [axis, boundary, direction] of [
    [0, windowBounds[0], 1], [0, windowBounds[2], -1],
    [1, windowBounds[1], 1], [1, windowBounds[3], -1]
  ]) {
    const clipped = [];
    for (let index = 0; index < points.length; index++) {
      const previous = points[(index + points.length - 1) % points.length];
      const current = points[index];
      const previousInside = (previous[axis] - boundary) * direction >= 0;
      const currentInside = (current[axis] - boundary) * direction >= 0;
      if (previousInside !== currentInside) {
        const fraction = (boundary - previous[axis]) / (current[axis] - previous[axis]);
        const intersection = previous.map((value, dimension) => value + fraction * (current[dimension] - value));
        intersection[axis] = boundary;
        clipped.push(intersection);
      }
      if (currentInside) clipped.push(current);
    }
    points = clipped;
  }
  return openRing(points.map(point => point.map(Math.round)));
}

function squaredDistanceToSegment(point, start, end) {
  const deltaEasting = end[0] - start[0];
  const deltaNorthing = end[1] - start[1];
  const squaredLength = deltaEasting ** 2 + deltaNorthing ** 2;
  const fraction = squaredLength ? Math.max(0, Math.min(1,
    ((point[0] - start[0]) * deltaEasting + (point[1] - start[1]) * deltaNorthing) / squaredLength)) : 0;
  return (point[0] - start[0] - fraction * deltaEasting) ** 2 +
    (point[1] - start[1] - fraction * deltaNorthing) ** 2;
}

function simplifyLine(points, tolerance = 60) {
  if (points.length <= 2) return points;
  let greatestDistance = tolerance ** 2;
  let splitIndex = -1;
  for (let index = 1; index < points.length - 1; index++) {
    const distance = squaredDistanceToSegment(points[index], points[0], points.at(-1));
    if (distance > greatestDistance) {
      greatestDistance = distance;
      splitIndex = index;
    }
  }
  return splitIndex < 0 ? [points[0], points.at(-1)] : [
    ...simplifyLine(points.slice(0, splitIndex + 1), tolerance).slice(0, -1),
    ...simplifyLine(points.slice(splitIndex), tolerance)
  ];
}

function simplifyRing(points) {
  if (points.length < 3) return [];
  // Split the closed ring at a distant vertex to avoid a degenerate baseline.
  let splitIndex = 1;
  for (let index = 2; index < points.length; index++) {
    if (squaredDistanceToSegment(points[index], points[0], points[0]) >
      squaredDistanceToSegment(points[splitIndex], points[0], points[0])) splitIndex = index;
  }
  const simplified = [
    ...simplifyLine(points.slice(0, splitIndex + 1)).slice(0, -1),
    ...simplifyLine([...points.slice(splitIndex), points[0]])
  ];
  if (simplified.length < 4) return [];
  const doubledArea = simplified.slice(1).reduce((sum, point, index) =>
    sum + simplified[index][0] * point[1] - point[0] * simplified[index][1], 0);
  return doubledArea ? simplified : [];
}

function containsPoint(rings, easting, northing) {
  let inside = false;
  for (const ring of rings) {
    for (let index = 1; index < ring.length; index++) {
      const previous = ring[index - 1];
      const current = ring[index];
      if ((previous[1] > northing) !== (current[1] > northing) &&
        easting < previous[0] + (northing - previous[1]) * (current[0] - previous[0]) / (current[1] - previous[1])) inside = !inside;
    }
  }
  return inside;
}

const perimeterGeojson = readJson('dixie-perimeter', perimeterManifest.geometry.file);
const clippedRings = [];
const ringRoles = [];
for (const feature of perimeterGeojson.features) {
  const geometry = feature.geometry;
  checkCondition(['Polygon', 'MultiPolygon'].includes(geometry.type), 'Unexpected perimeter geometry');
  const polygons = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
  for (const polygon of polygons) {
    polygon.forEach((ring, ringIndex) => {
      const simplified = simplifyRing(clipRing(ring.map(projectCoordinate)));
      if (simplified.length >= 4) {
        clippedRings.push(simplified);
        ringRoles.push(ringIndex === 0);
      }
    });
  }
}
const vertexCount = clippedRings.reduce((sum, ring) => sum + ring.length, 0);
checkCondition(vertexCount <= 600, `Perimeter exceeds 600 vertices: ${vertexCount}`);
let coveredCellCount = 0;
for (let row = 0; row < GRID_SIZE; row++) {
  for (let column = 0; column < GRID_SIZE; column++) {
    if (containsPoint(clippedRings, origin[0] + (column + 0.5) * 120, origin[1] - (row + 0.5) * 120)) coveredCellCount++;
  }
}
const coverage = coveredCellCount / GRID_SIZE ** 2;
console.log(`Perimeter: ${clippedRings.length} rings, ${vertexCount} vertices; coverage ${coveredCellCount}/4096 = ${coverage}`);
if (coverage < 0.05 || coverage > 0.95) console.log('PERIMETER COVERAGE OUT OF RANGE');
const perimeter = {...makeHeader('t-dixie-perimeter',
  'Final 2021 Dixie Fire perimeter clipped to W, rounded to 1 m, Douglas-Peucker tolerance 60 m. inside=true marks outer rings, false holes; coverage uses even-odd cell-centre inclusion on the stored rings.',
  'NIFC final 2021 Dixie Fire perimeter (showcase dixie-perimeter, via poopdeck-wildfires)',
  'Public domain (US Government work, NIFC)', 'm'),
  rings: clippedRings.map(ring => ring.flat()), inside: ringRoles, coverage};

const windGrid = windManifest.raster;
const {raster: windRaster} = readRaster('gfs-wind', 'wind10m_00.png', windGrid.width, windGrid.height);
const windCellSize = windGrid.cellSizeDeg;
const firstLongitude = windGrid.bounds[0] + windCellSize / 2;
const firstLatitude = windGrid.bounds[3] - windCellSize / 2;
const windColumn = Math.round((-100 - firstLongitude) / windCellSize);
const windRow = Math.round((firstLatitude - 42) / windCellSize);
checkCondition(windColumn >= 0 && windRow >= 0 && windColumn + 64 <= windRaster.width && windRow + 48 <= windRaster.height, 'Wind crop out of bounds');
const wind = {...makeHeader('t-gfs-wind',
  'NOAA GFS 10 m wind, 64 x 48 crop starting at cell centre (-100, 42); values quantized to 8 bits by the showcase packaging; rows north to south.',
  'NOAA GFS 0.25° 10 m wind (packaged by the showcase as 8-bit RG)',
  'Public domain (NOAA)', 'm/s'), crs: 'EPSG:4326', height: 48, cellSize: windCellSize,
  origin: [firstLongitude + windColumn * windCellSize - windCellSize / 2,
    firstLatitude - windRow * windCellSize + windCellSize / 2],
  u: [], v: [], validTime: windManifest.properties.validStart};
checkCondition(wind.validTime === '2024-09-26T12:00:00Z', 'Unexpected wind valid time');
for (let row = 0; row < wind.height; row++) {
  for (let column = 0; column < wind.width; column++) {
    const offset = ((windRow + row) * windRaster.width + windColumn + column) * 4;
    for (const [component, channel] of [['u', 0], ['v', 1]]) {
      const [minimum, maximum] = windManifest.properties.wind10m[`${component}Range`];
      wind[component].push(roundNumber(minimum + windRaster.data[offset + channel] / 255 * (maximum - minimum), 2));
    }
  }
}
console.log(`Wind crop row ${windRow}, column ${windColumn}, edge origin ${wind.origin}`);
const windSpeeds = wind.u.map((value, index) => Math.hypot(value, wind.v[index]));
console.log('Wind speed min/max (m/s):', roundNumber(Math.min(...windSpeeds), 3), roundNumber(Math.max(...windSpeeds), 3));

const terrarium = {...makeHeader('t-dixie-terrarium',
  'Raw Terrarium RGB from the 32 x 32 showcase DEM block at row 520, column 570; alpha 255; no averaging; rows north to south.',
  DEM_SOURCE, DEM_LICENCE, 'm'), width: 32, height: 32, cellSize: 20,
  origin: [minimumEasting + 570 * 20, maximumNorthing - 520 * 20],
  rgba: [], encoding: 'terrarium', decode: 'elevation = R*256 + G + B/256 - 32768'};
const decodedElevations = [];
const nearbyDemValues = [];
for (let row = 520; row < 552; row++) {
  for (let column = 570; column < 602; column++) {
    const offset = (row * sourceGrid.width + column) * 4;
    terrarium.rgba.push(...terrainRaster.data.subarray(offset, offset + 3), 255);
    decodedElevations.push(decodeElevation(row * sourceGrid.width + column));
    nearbyDemValues.push(dem.values[Math.floor((row - WINDOW_START) / BLOCK_SIZE) * GRID_SIZE + Math.floor((column - WINDOW_START) / BLOCK_SIZE)]);
  }
}
console.log('Terrarium decoded min/max/mean (m):', summarizeValues(decodedElevations));
console.log('DEM cells overlapping Terrarium block min/max/mean (m):', summarizeValues(nearbyDemValues));
checkCondition(Math.abs(calculateMean(decodedElevations) - calculateMean(nearbyDemValues)) < 30, 'Terrarium and overlapping DEM means differ by more than 30 m');

const outputs = [dem, bands, landcover, perimeter, wind, terrarium];
const serializedOutputs = outputs.map(output => JSON.stringify(output));
let totalBytes = 0;
serializedOutputs.forEach((serialized, index) => {
  const byteCount = Buffer.byteLength(serialized);
  totalBytes += byteCount;
  console.log(`${outputs[index].id}.json: ${byteCount} bytes (${(byteCount / 1000).toFixed(3)} KB)`);
  checkCondition(byteCount <= 150000, 'File exceeds 150 KB');
});
console.log(`All six files: ${totalBytes} bytes (${(totalBytes / 1000).toFixed(3)} KB)`);
checkCondition(totalBytes <= 600000, 'Total exceeds 600 KB');
mkdirSync(OUTPUT_DIRECTORY, {recursive: true});
outputs.forEach((output, index) => writeFileSync(resolve(OUTPUT_DIRECTORY, `${output.id}.json`), serializedOutputs[index]));

// Append only; repeated extraction does not duplicate this stream's README rows.
const readmePath = resolve(OUTPUT_DIRECTORY, 'README.md');
if (!existsSync(readmePath)) {
  writeFileSync(readmePath, 'Docs-sized real open data for lens scenarios. Each file ≤150 KB.\n\n| File | Contents | Extracted from | Licence | Script |\n| --- | --- | --- | --- | --- |\n', {flag: 'wx'});
}
const existingReadme = readFileSync(readmePath, 'utf8');
const descriptions = ['64 x 64 DEM, 120 m block means', '64 x 64 before/after NIR and SWIR reflectance',
  '64 x 64 WorldCover block modes', 'Clipped UTM fire rings and cell-centre coverage',
  '64 x 48 decoded 10 m wind vectors', '32 x 32 raw Terrarium RGBA at 20 m'];
const datasetDirectories = ['dixie-fire', 'dixie-fire', 'dixie-fire', 'dixie-perimeter', 'gfs-wind', 'dixie-fire'];
const newRows = outputs.flatMap((output, index) => existingReadme.includes(`| ${output.id}.json |`) ? [] :
  [`| ${output.id}.json | ${descriptions[index]} | examples/deck/spatial-analysis-showcase/public/data/${datasetDirectories[index]}/ | ${output.licence} | ${SCRIPT} |`]);
if (newRows.length) appendFileSync(readmePath, `${existingReadme.endsWith('\n') ? '' : '\n'}${newRows.join('\n')}\n`);
console.log(`README: appended ${newRows.length} rows.`);
