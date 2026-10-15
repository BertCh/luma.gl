import{M as e,_ as t,h as n,j as r}from"./resources-CwvhEOBe.js";import{s as i}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{n as a,t as o}from"./philox-random-CHwyk0dd.js";function s(e,t,n){return`
const FIELD_WIDTH: u32 = ${t}u;
const FIELD_HEIGHT: u32 = ${n}u;

fn isNanFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
}

fn readFieldVelocity(column: u32, row: u32) -> vec2<f32> {
  let base = ${e}Offset + 2u * (row * FIELD_WIDTH + column);
  return vec2<f32>(${e}[base], ${e}[base + 1u]);
}

fn isNanVelocity(value: vec2<f32>) -> bool {
  return isNanFloat(value.x) || isNanFloat(value.y);
}

fn sampleField(position: vec2<f32>, extent: vec4<f32>) -> vec3<f32> {
  let local = (position - extent.xy) / extent.zw;
  if (isNanVelocity(local)) {
    return vec3<f32>(0.0);
  }
  if (local.x < 0.0 || local.y < 0.0 || local.x >= f32(FIELD_WIDTH) || local.y >= f32(FIELD_HEIGHT)) {
    return vec3<f32>(0.0);
  }
  let grid = clamp(
    local - vec2<f32>(0.5),
    vec2<f32>(0.0),
    vec2<f32>(f32(FIELD_WIDTH - 1u), f32(FIELD_HEIGHT - 1u))
  );
  let column0 = min(u32(grid.x), FIELD_WIDTH - 1u);
  let row0 = min(u32(grid.y), FIELD_HEIGHT - 1u);
  let column1 = min(column0 + 1u, FIELD_WIDTH - 1u);
  let row1 = min(row0 + 1u, FIELD_HEIGHT - 1u);
  let fraction = grid - vec2<f32>(f32(column0), f32(row0));
  let v00 = readFieldVelocity(column0, row0);
  let v10 = readFieldVelocity(column1, row0);
  let v01 = readFieldVelocity(column0, row1);
  let v11 = readFieldVelocity(column1, row1);
  if (isNanVelocity(v00) || isNanVelocity(v10) || isNanVelocity(v01) || isNanVelocity(v11)) {
    return vec3<f32>(0.0);
  }
  let bottom = v00 * (1.0 - fraction.x) + v10 * fraction.x;
  let top = v01 * (1.0 - fraction.x) + v11 * fraction.x;
  return vec3<f32>(bottom * (1.0 - fraction.y) + top * fraction.y, 1.0);
}
`}var c=o.particleAdvectionSpawn.id,l=o.particleAdvectionDrop.id;function u(e,t,n=new Float32Array(12)){if(n.length<12)throw Error(`Particle advection parameter target must hold 12 elements`);let[r,i,a,o]=e.fieldExtent,s=e.spawnBounds??[r,i,r+a*t[0],i+o*t[1]];return n.set([r,i,a,o,e.timeStep,e.speedScale??1,e.dropRate??0,e.minimumSpeed??0,s[0],s[1],s[2],s[3]]),n}function d(e,t=new Uint32Array(4)){if(t.length<4)throw Error(`Particle advection uint32 parameter target must hold 4 elements`);for(let[t,n]of[[`seed`,e.seed],[`frame`,e.frame],[`maximumAge`,e.maximumAge]])if(!Number.isInteger(n)||n<0||n>4294967295)throw Error(`Particle advection ${t} must be a uint32 integer`);return t[0]=e.seed,t[1]=e.frame,t[2]=e.maximumAge,t[3]=e.reset?1:0,t}var f=`GPUParticleAdvection`,p=class{id;particleCount;props;constructor(t){let i=t.id??`particle-advection`,{fieldWidth:a,fieldHeight:o,state:s}=t,c=t.output??{},{trails:l}=c;for(let[e,t]of[[`fieldWidth`,a],[`fieldHeight`,o]])if(!Number.isInteger(t)||t<1||t>65535)throw Error(`${i} ${e} must be an integer in [1, 65535]`);if(e(t.velocities,[`float32x2`],`${i} velocities`),t.velocities.length!==a*o)throw Error(`${i} velocities must have fieldWidth * fieldHeight rows`);if(e(t.parameters,[`float32`],`${i} parameters`),t.parameters.length<12)throw Error(`${i} parameters must contain 12 float32 rows`);if(r(t.uint32Parameters,`${i} uint32Parameters`),t.uint32Parameters.length<4)throw Error(`${i} uint32Parameters must contain 4 uint32 rows`);e(s.positions,[`float32x2`],`${i} state.positions`);let u=s.positions.length;if(u<1)throw Error(`${i} needs at least one particle`);r(s.ages,`${i} state.ages`),r(s.generations,`${i} state.generations`),c.previousPositions&&e(c.previousPositions,[`float32x2`],`${i} output.previousPositions`),c.speeds&&e(c.speeds,[`float32`],`${i} output.speeds`);for(let[e,t]of[[`state.ages`,s.ages],[`state.generations`,s.generations],[`output.previousPositions`,c.previousPositions],[`output.speeds`,c.speeds]])if(t&&t.length!==u)throw Error(`${i} ${e} must have one row per particle`);if(l){if(!Number.isInteger(l.length)||l.length<1)throw Error(`${i} output.trails.length must be a positive integer`);if(e(l.positions,[`float32x2`],`${i} output.trails.positions`),l.positions.length!==u*l.length)throw Error(`${i} output.trails.positions must have particleCount * trails.length rows`)}n(i,[s.positions,s.ages,s.generations,c.previousPositions,c.speeds,l?.positions],[t.velocities,t.parameters,t.uint32Parameters]),this.id=i,this.particleCount=u,this.props=t}getCommandNodes(e){let{id:n,props:r,particleCount:o}=this,{state:u}=r,d=r.output??{},{trails:p}=d;t(n,e,[r.velocities,r.parameters,r.uint32Parameters,u.positions,u.ages,u.generations,d.previousPositions,d.speeds,p?.positions]);let m=[{name:`velocities`,view:r.velocities,type:`f32`,access:`read`},{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},{name:`words`,view:r.uint32Parameters,type:`u32`,access:`read`},{name:`positions`,view:u.positions,type:`f32`,access:`read_write`},{name:`ages`,view:u.ages,type:`u32`,access:`read_write`},{name:`generations`,view:u.generations,type:`u32`,access:`read_write`}];d.previousPositions&&m.push({name:`previousPositions`,view:d.previousPositions,type:`f32`,access:`read_write`}),d.speeds&&m.push({name:`speeds`,view:d.speeds,type:`f32`,access:`read_write`});let h=[i(e,{id:`${n}-step`,operation:f,variant:`step`,bindings:m,invocationCount:o,declarations:`${a}
${s(`velocities`,r.fieldWidth,r.fieldHeight)}
fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }
fn getSpeed(sample: vec2<f32>, speedScale: f32) -> f32 { return length(sample) * abs(speedScale); }`,body:`
  let extent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let timeStep = readParameter(4u);
  let speedScale = readParameter(5u);
  let dropRate = readParameter(6u);
  let minimumSpeed = readParameter(7u);
  let spawnMin = vec2<f32>(readParameter(8u), readParameter(9u));
  let spawnMax = vec2<f32>(readParameter(10u), readParameter(11u));
  let seed = words[wordsOffset];
  let frame = words[wordsOffset + 1u];
  let maximumAge = words[wordsOffset + 2u];
  let reset = words[wordsOffset + 3u] != 0u;
  let step = timeStep * speedScale;
  let halfStep = 0.5 * step;
  var position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  var previous = position;
  var age = ages[agesOffset + index];
  var generation = generations[generationsOffset + index];
  var respawn = reset;
  var speed = 0.0;
  if (reset) {
    generation = 0u;
  } else {
    let k1 = sampleField(position, extent);
    let middle = position + halfStep * k1.xy;
    let k2 = sampleField(middle, extent);
    let next = position + step * k2.xy;
    let landing = sampleField(next, extent);
    let dropWord = philox4x32(vec4<u32>(index, frame, 0u, 0u), vec2<u32>(seed, ${l}u)).x;
    let expired = maximumAge != 0u && age + 1u >= maximumAge;
    let tooSlow = k2.z != 0.0 && getSpeed(k2.xy, speedScale) < minimumSpeed;
    if (k1.z == 0.0 || k2.z == 0.0 || landing.z == 0.0 || expired || tooSlow ||
        philoxUnitFloat(dropWord) < dropRate) {
      respawn = true;
      generation = generation + 1u;
    } else {
      position = next;
      age = age + 1u;
      speed = getSpeed(landing.xy, speedScale);
    }
  }
  if (respawn) {
    let random = philox4x32(vec4<u32>(index, generation, 0u, 0u), vec2<u32>(seed, ${c}u));
    position = spawnMin + vec2<f32>(philoxUnitFloat(random.x), philoxUnitFloat(random.y)) * (spawnMax - spawnMin);
    previous = position;
    age = 0u;
    if (reset) {
      age = min(u32(floor(philoxUnitFloat(random.z) * f32(maximumAge))), max(maximumAge, 1u) - 1u);
    }
    let landing = sampleField(position, extent);
    speed = select(0.0, getSpeed(landing.xy, speedScale), landing.z != 0.0);
  }
  positions[positionsOffset + 2u * index] = position.x;
  positions[positionsOffset + 2u * index + 1u] = position.y;
  ages[agesOffset + index] = age;
  generations[generationsOffset + index] = generation;
  ${d.previousPositions?`previousPositions[previousPositionsOffset + 2u * index] = previous.x;
  previousPositions[previousPositionsOffset + 2u * index + 1u] = previous.y;`:`_ = previous;`}
  ${d.speeds?`speeds[speedsOffset + index] = speed;`:`_ = speed;`}`})];return p&&h.push(i(e,{id:`${n}-trails`,operation:f,variant:`trails`,bindings:[{name:`words`,view:r.uint32Parameters,type:`u32`,access:`read`},{name:`positions`,view:u.positions,type:`f32`,access:`read`},{name:`ages`,view:u.ages,type:`u32`,access:`read`},{name:`trails`,view:p.positions,type:`f32`,access:`read_write`}],invocationCount:o,declarations:`const TRAIL_LENGTH: u32 = ${p.length}u;`,body:`
  let frame = words[wordsOffset + 1u];
  let reset = words[wordsOffset + 3u] != 0u;
  let x = positions[positionsOffset + 2u * index];
  let y = positions[positionsOffset + 2u * index + 1u];
  let base = trailsOffset + 2u * index * TRAIL_LENGTH;
  if (reset || ages[agesOffset + index] == 0u) {
    for (var slot = 0u; slot < TRAIL_LENGTH; slot = slot + 1u) {
      trails[base + 2u * slot] = x;
      trails[base + 2u * slot + 1u] = y;
    }
  } else {
    let slot = frame % TRAIL_LENGTH;
    trails[base + 2u * slot] = x;
    trails[base + 2u * slot + 1u] = y;
  }`})),h}};export{s as i,u as n,d as r,p as t};