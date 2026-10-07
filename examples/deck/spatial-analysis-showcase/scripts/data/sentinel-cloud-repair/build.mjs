import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {fromUrl} from 'geotiff';
import pngjs from 'pngjs';

const {PNG} = pngjs;
const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const outputDirectory = path.resolve(scriptDirectory, '../../../public/data/sentinel-cloud-repair');
fs.mkdirSync(outputDirectory, {recursive: true});

const places = [
  {
    id: 'oahu',
    title: 'Waimānalo, Oʻahu',
    zone: 4,
    transform: [10, 0, 600000, 0, -10, 2400000],
    window: [3070, 3768, 3582, 4280],
    targetDate: '2026-09-29',
    priorDate: '2026-02-03',
    targetId: 'S2C_4QFJ_20260929_0_L2A',
    priorId: 'S2B_4QFJ_20260203_0_L2A',
    target: 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/4/Q/FJ/2026/9/S2C_4QFJ_20260929_0_L2A/TCI.tif',
    prior: 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/4/Q/FJ/2026/2/S2B_4QFJ_20260203_0_L2A/TCI.tif',
    mask: 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/4/Q/FJ/2026/9/S2C_4QFJ_20260929_0_L2A/SCL.tif'
  },
  {
    id: 'venice',
    title: 'Venice Lagoon, Italy',
    zone: 32,
    transform: [10, 0, 699960, 0, -10, 5100000],
    window: [5960, 5968, 6472, 6480],
    targetDate: '2026-08-25',
    priorDate: '2026-08-24',
    targetId: 'S2C_32TQR_20260825_0_L2A',
    priorId: 'S2A_32TQR_20260824_0_L2A',
    target: 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/QR/2026/8/S2C_32TQR_20260825_0_L2A/TCI.tif',
    prior: 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/QR/2026/8/S2A_32TQR_20260824_0_L2A/TCI.tif',
    mask: 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/32/T/QR/2026/8/S2C_32TQR_20260825_0_L2A/SCL.tif'
  }
];

function utmToLngLat(easting, northing, zone) {
  const a = 6378137;
  const flattening = 1 / 298.257223563;
  const k0 = 0.9996;
  const e2 = flattening * (2 - flattening);
  const ep2 = e2 / (1 - e2);
  const e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
  const x = easting - 500000;
  const m = northing / k0;
  const mu = m / (a * (1 - e2 / 4 - (3 * e2 * e2) / 64 - (5 * e2 ** 3) / 256));
  const phi1 = mu + ((3 * e1) / 2 - (27 * e1 ** 3) / 32) * Math.sin(2 * mu) + ((21 * e1 ** 2) / 16 - (55 * e1 ** 4) / 32) * Math.sin(4 * mu) + ((151 * e1 ** 3) / 96) * Math.sin(6 * mu) + ((1097 * e1 ** 4) / 512) * Math.sin(8 * mu);
  const sin1 = Math.sin(phi1);
  const cos1 = Math.cos(phi1);
  const tan1 = Math.tan(phi1);
  const n1 = a / Math.sqrt(1 - e2 * sin1 * sin1);
  const t1 = tan1 * tan1;
  const c1 = ep2 * cos1 * cos1;
  const r1 = (a * (1 - e2)) / (1 - e2 * sin1 * sin1) ** 1.5;
  const d = x / (n1 * k0);
  const latitude = phi1 - ((n1 * tan1) / r1) * (d ** 2 / 2 - ((5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4) / 24 + ((61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6) / 720);
  const longitudeOffset = (d - ((1 + 2 * t1 + c1) * d ** 3) / 6 + ((5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5) / 120) / cos1;
  const centralMeridian = (zone - 1) * 6 - 180 + 3;
  return [centralMeridian + (longitudeOffset * 180) / Math.PI, (latitude * 180) / Math.PI];
}

async function readWindow(url, window) {
  const tiff = await fromUrl(url);
  const image = await tiff.getImage();
  return image.readRasters({window, interleave: true});
}

function writeRgb(file, values) {
  const png = new PNG({width: 512, height: 512});
  for (let pixel = 0; pixel < 512 * 512; pixel++) {
    png.data[pixel * 4] = values[pixel * 3];
    png.data[pixel * 4 + 1] = values[pixel * 3 + 1];
    png.data[pixel * 4 + 2] = values[pixel * 3 + 2];
    png.data[pixel * 4 + 3] = 255;
  }
  fs.writeFileSync(file, PNG.sync.write(png, {colorType: 6}));
}

function writeMask(file, scl) {
  const raw = new Uint8Array(512 * 512);
  const invalid = new Set([1, 3, 7, 8, 9, 10]);
  for (let y = 0; y < 512; y++) {
    for (let x = 0; x < 512; x++) raw[y * 512 + x] = invalid.has(scl[Math.floor(y / 2) * 256 + Math.floor(x / 2)]) ? 255 : 0;
  }
  // One SCL-cell halo captures mixed cloud edges and shadow fringes.
  const dilated = raw.slice();
  for (let y = 2; y < 510; y++) {
    for (let x = 2; x < 510; x++) {
      if (!raw[y * 512 + x]) continue;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) dilated[(y + dy) * 512 + x + dx] = 255;
    }
  }
  const png = new PNG({width: 512, height: 512});
  for (let pixel = 0; pixel < raw.length; pixel++) {
    png.data[pixel * 4] = dilated[pixel];
    png.data[pixel * 4 + 1] = dilated[pixel];
    png.data[pixel * 4 + 2] = dilated[pixel];
    png.data[pixel * 4 + 3] = 255;
  }
  fs.writeFileSync(file, PNG.sync.write(png, {colorType: 6}));
  return dilated.reduce((sum, value) => sum + (value ? 1 : 0), 0);
}

const metadata = [];
for (const place of places) {
  console.log(`Reading ${place.title} target…`);
  const target = await readWindow(place.target, place.window);
  console.log(`Reading ${place.title} prior…`);
  const prior = await readWindow(place.prior, place.window);
  const sclWindow = place.window.map(value => value / 2);
  console.log(`Reading ${place.title} SCL…`);
  const scl = await readWindow(place.mask, sclWindow);
  writeRgb(path.join(outputDirectory, `${place.id}-cloudy.png`), target);
  writeRgb(path.join(outputDirectory, `${place.id}-prior.png`), prior);
  const maskedPixels = writeMask(path.join(outputDirectory, `${place.id}-mask.png`), scl);
  const [x0, y0, x1, y1] = place.window;
  const [cellX, , originX, , cellY, originY] = place.transform;
  const westX = originX + x0 * cellX;
  const eastX = originX + x1 * cellX;
  const northY = originY + y0 * cellY;
  const southY = originY + y1 * cellY;
  const southwest = utmToLngLat(westX, southY, place.zone);
  const northeast = utmToLngLat(eastX, northY, place.zone);
  metadata.push({...place, bounds: [southwest[0], southwest[1], northeast[0], northeast[1]], maskedPixels, cloudPercent: maskedPixels / (512 * 512) * 100});
}
console.log(JSON.stringify(metadata.map(({id, title, bounds, maskedPixels, cloudPercent}) => ({id, title, bounds, maskedPixels, cloudPercent})), null, 2));
