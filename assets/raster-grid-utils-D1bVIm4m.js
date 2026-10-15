import{w as e}from"./resources-CwvhEOBe.js";import{l as t,o as n}from"./wgsl-common-BNRVzgd8.js";import{o as r,u as i}from"./raster-grid-D2bmCwkc.js";function a(t,n,r,i,a,o,s){try{return e(t,i,a,o,s)}catch(e){throw e instanceof Error&&/resource id ".*" is already in use/.test(e.message)?Error(`${n} "${r}": internal transient resource "${i}" (generated from the contributor id) collides with an existing graph resource; choose a different contributor id or rename the resource`,{cause:e}):e}}function o(e,t){i(e,t,{cellSizeIndex:0,northEdgeIndex:2})}function s(e){let i=e.cellSizeMode===`uniform`?`return cellSize;`:`let rowFraction = rowCoordinate / f32(GRID_HEIGHT);
  let edge = mix(settings[settingsOffset + 2u], settings[settingsOffset + 3u], rowFraction);
  ${r(e.cellSizeMode)}
  return cellSize * groundScale;`;return`
const GRID_WIDTH: u32 = ${e.width}u;
const GRID_HEIGHT: u32 = ${e.height}u;
const GRID_NONE: u32 = 0xffffffffu;

// Bit tests: shader compilers may fold 'value != value' away under fast-math assumptions.
${t}
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn getInfinity() -> f32 { var bits = 0x7f800000u; return bitcast<f32>(bits); }
${n}

fn getD8ColumnOffset(direction: u32) -> i32 {
  var offsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  return offsets[direction];
}
fn getD8RowOffset(direction: u32) -> i32 {
  var offsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  return offsets[direction];
}
fn getD8Code(direction: u32) -> u32 { return 1u << direction; }

// Returns the neighboring cell index in one D8 direction, or GRID_NONE outside the grid.
fn getD8Neighbor(cell: u32, direction: u32) -> u32 {
  let column = i32(cell % GRID_WIDTH) + getD8ColumnOffset(direction);
  let row = i32(cell / GRID_WIDTH) + getD8RowOffset(direction);
  if (column < 0 || row < 0 || column >= i32(GRID_WIDTH) || row >= i32(GRID_HEIGHT)) {
    return GRID_NONE;
  }
  return u32(row) * GRID_WIDTH + u32(column);
}

// Ground meters per cell at a continuous row coordinate (row index + 0.5 is a cell center).
fn getGroundCellSizeAtRowCoordinate(rowCoordinate: f32) -> vec2<f32> {
  let cellSize = vec2<f32>(settings[settingsOffset], settings[settingsOffset + 1u]);
  ${i}
}

// Ground length of one D8 move from centerRow, measured at the midpoint row (symmetric).
fn getD8Distance(direction: u32, centerRow: u32) -> f32 {
  let rowOffset = getD8RowOffset(direction);
  let neighborRow = i32(centerRow) + rowOffset;
  let ground = getGroundCellSizeAtRowCoordinate((f32(centerRow) + f32(neighborRow) + 1.0) * 0.5);
  if (rowOffset == 0) { return ground.x; }
  if (getD8ColumnOffset(direction) == 0) { return ground.y; }
  return sqrt(ground.x * ground.x + ground.y * ground.y);
}`}export{s as n,o as r,a as t};