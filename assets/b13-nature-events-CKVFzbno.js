import{_ as e,h as t,y as n}from"./resources-CwvhEOBe.js";import{n as r,s as i}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{r as a}from"./time-words-C-gLAZQw.js";var o=1440,s=1e9;function c(e,t=0,n=new Int32Array(2)){if(n.length<2)throw Error(`Calendar bucket parameter target must hold 2 elements`);if(!Number.isInteger(e)||Math.abs(e)>1440)throw Error(`Calendar bucket UTC offset must be an integer in [-${o}, ${o}] minutes`);if(!Number.isInteger(t)||t<0||t>6)throw Error(`Calendar bucket first day of week must be an integer in [0, 6]`);return n[0]=e,n[1]=t,n}var l=`GPUCalendarBuckets`,u=8,d=[{name:`year`,field:`year`,signed:!0},{name:`month`,field:`month`},{name:`dayOfMonth`,field:`day`},{name:`hour`,field:`hour`},{name:`minute`,field:`minute`},{name:`weekday`,field:`weekday`},{name:`dayOfYear`,field:`dayOfYear`},{name:`isoWeek`,field:`isoWeek`},{name:`isoWeekYear`,field:`isoWeekYear`,signed:!0},{name:`quarter`,field:`quarter`}],f=class{id;props;rowCount;constructor(e){this.id=e.id??`calendar-buckets`,this.props=e;let r=this.id,{output:i}=e;if(n(e.timestamps,[`uint32x2`],`${r} timestamps`),this.rowCount=e.timestamps.length,this.rowCount<1)throw Error(`${r} needs at least one row`);if(e.mask&&(n(e.mask,[`uint32`],`${r} mask`),e.mask.length!==this.rowCount))throw Error(`${r} mask length must equal timestamps length`);if(e.utcOffsets&&(n(e.utcOffsets,[`sint32`],`${r} utcOffsets`),e.utcOffsets.length!==this.rowCount))throw Error(`${r} utcOffsets length must equal timestamps length`);if(e.timeZone){if(e.utcOffsets)throw Error(`${r} timeZone and utcOffsets are mutually exclusive`);if(n(e.timeZone.transitions,[`uint32x2`],`${r} timeZone.transitions`),n(e.timeZone.offsets,[`sint32`],`${r} timeZone.offsets`),e.timeZone.transitions.length<1)throw Error(`${r} timeZone needs at least one transition`);if(e.timeZone.offsets.length!==e.timeZone.transitions.length+1)throw Error(`${r} timeZone.offsets length must be transitions length + 1`)}if(n(e.parameters,[`sint32`],`${r} parameters`),e.parameters.length<2)throw Error(`${r} parameters must hold 2 sint32 values`);let a=0;for(let e of d){let t=i[e.name];if(t&&(a++,n(t,[e.signed?`sint32`:`uint32`],`${r} output.${e.name}`),t.length<this.rowCount))throw Error(`${r} output.${e.name} must hold one row per timestamp`)}if(i.hourWeekdayCounts&&(a++,n(i.hourWeekdayCounts,[`uint32`],`${r} output.hourWeekdayCounts`),i.hourWeekdayCounts.length<168))throw Error(`${r} output.hourWeekdayCounts must hold 168 elements`);if(a===0)throw Error(`${r} needs at least one output`);t(r,this.getOutputViews(),this.getInputViews())}getCommandNodes(t){let{props:n,id:a,rowCount:o}=this,{output:s}=n;e(a,t,[...this.getInputViews(),...this.getOutputViews()]);let c=[],f=[];for(let e of d)s[e.name]&&f.push({name:e.name,type:e.signed?`i32`:`u32`,write:`${e.name}Out[${e.name}OutOffset + index] = fields.${e.field};`});s.hourWeekdayCounts&&(c.push(r(t,{id:`${a}-fill-matrix`,operation:l,view:s.hourWeekdayCounts,type:`u32`,value:`0u`,componentCount:168})),f.push({name:`hourWeekdayCounts`,type:`atomic<u32>`,write:`if (fields.valid) {
    atomicAdd(&hourWeekdayCountsOut[hourWeekdayCountsOutOffset + fields.weekday * 24u + fields.hour], 1u);
  }`}));let h=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),g=[h(`timestamps`,n.timestamps,`u32`),h(`params`,n.parameters,`i32`)];n.mask&&g.push(h(`rowMask`,n.mask,`u32`)),n.utcOffsets&&g.push(h(`rowOffsets`,n.utcOffsets,`i32`)),n.timeZone&&g.push(h(`zoneTransitions`,n.timeZone.transitions,`u32`),h(`zoneOffsets`,n.timeZone.offsets,`i32`));let _=u-g.length;for(let e=0,r=0;e<f.length;e+=_,r++){let u=f.slice(e,e+_);c.push(i(t,{id:`${a}-decode-${r}`,operation:l,variant:`decode`,bindings:[...g,...u.map(e=>({name:`${e.name}Out`,view:s[e.name],type:e.type,access:`read_write`}))],invocationCount:o,declarations:n.timeZone?`const TRANSITION_COUNT: u32 = ${n.timeZone.transitions.length}u;\n${p}${m}`:p,body:`let time = vec2<u32>(timestamps[timestampsOffset + 2u * index], timestamps[timestampsOffset + 2u * index + 1u]);
  let offsetMilliseconds = ${n.timeZone?`getZoneOffsetMilliseconds(time)`:`minutesToMilliseconds(${n.utcOffsets?`rowOffsets[rowOffsetsOffset + index]`:`params[paramsOffset]`})`};
  let firstDay = u32(((params[paramsOffset + 1u] % 7) + 7) % 7);
  var fields = decodeCalendar(time, offsetMilliseconds, firstDay);
  ${n.mask?`if (rowMask[rowMaskOffset + index] == 0u) {
    fields = invalidCalendar();
  }`:``}
  ${u.map(e=>e.write).join(`
  `)}`}))}return c}getInputViews(){let{props:e}=this;return[e.timestamps,e.mask,e.utcOffsets,e.timeZone?.transitions,e.timeZone?.offsets,e.parameters].filter(e=>!!e)}getOutputViews(){return Object.values(this.props.output).filter(e=>!!e)}},p=`
const MILLISECONDS_PER_DAY: u32 = 86400000u;
const MAXIMUM_DAYS: u32 = ${s}u;

struct CalendarFields {
  valid: bool,
  year: i32,
  month: u32,
  day: u32,
  hour: u32,
  minute: u32,
  weekday: u32,
  dayOfYear: u32,
  isoWeek: u32,
  isoWeekYear: i32,
  quarter: u32
}

fn invalidCalendar() -> CalendarFields {
  let invalidYear = bitcast<i32>(0x80000000u);
  return CalendarFields(false, invalidYear, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
    0xffffffffu, 0xffffffffu, 0xffffffffu, invalidYear, 0xffffffffu);
}

fn isLeapYear(year: i32) -> bool {
  return (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
}

// Hinnant civil_from_days: days since 1970-01-01 to (year, month, day).
fn civilFromDays(days: i32) -> vec3<i32> {
  let z = days + 719468;
  let era = select((z - 146096) / 146097, z / 146097, z >= 0);
  let dayOfEra = z - era * 146097;
  let yearOfEra = (dayOfEra - dayOfEra / 1460 + dayOfEra / 36524 - dayOfEra / 146096) / 365;
  let dayOfMarchYear = dayOfEra - (365 * yearOfEra + yearOfEra / 4 - yearOfEra / 100);
  let monthPrime = (5 * dayOfMarchYear + 2) / 153;
  let day = dayOfMarchYear - (153 * monthPrime + 2) / 5 + 1;
  let month = select(monthPrime - 9, monthPrime + 3, monthPrime < 10);
  let year = yearOfEra + era * 400 + select(0, 1, month <= 2);
  return vec3<i32>(year, month, day);
}

fn getDayOfYear(date: vec3<i32>) -> i32 {
  let adjustment = select(select(-2, -1, isLeapYear(date.x)), 0, date.y <= 2);
  return (367 * date.y - 362) / 12 + date.z + adjustment;
}

struct DivideResult {
  quotientLow: u32,
  quotientHigh: u32,
  remainder: u32
}

// Exact unsigned 64-bit division by a divisor below 2^28, one nibble at a time so that
// remainder * 16 never overflows 32 bits.
fn divideUnsigned64(value: vec2<u32>, divisor: u32) -> DivideResult {
  var remainder = value.y % divisor;
  var quotientLow = 0u;
  for (var nibble = 7i; nibble >= 0i; nibble = nibble - 1i) {
    let shift = u32(nibble) * 4u;
    let current = (remainder << 4u) | ((value.x >> shift) & 0xfu);
    quotientLow = quotientLow | ((current / divisor) << shift);
    remainder = current % divisor;
  }
  return DivideResult(quotientLow, value.y / divisor, remainder);
}

// Minutes to milliseconds; an offset beyond +-1440 minutes maps to a value the decode rejects.
fn minutesToMilliseconds(offsetMinutes: i32) -> i32 {
  return select(offsetMinutes * 60000, 0x7fffffff, offsetMinutes < -1440 || offsetMinutes > 1440);
}

fn decodeCalendar(time: vec2<u32>, offsetMilliseconds: i32, firstDay: u32) -> CalendarFields {
  let invalid = invalidCalendar();
  if (offsetMilliseconds < -86400000 || offsetMilliseconds > 86400000) {
    return invalid;
  }
  // Signed 64-bit local time = time + offset, with overflow detection.
  let offsetLow = bitcast<u32>(offsetMilliseconds);
  let offsetHigh = select(0u, 0xffffffffu, offsetMilliseconds < 0);
  let low = time.x + offsetLow;
  let high = time.y + offsetHigh + select(0u, 1u, low < time.x);
  let timeSign = time.y >> 31u;
  let sumSign = high >> 31u;
  if (timeSign == (offsetHigh >> 31u) && sumSign != timeSign) {
    return invalid;
  }
  let isNegative = sumSign != 0u;
  var magnitude = vec2<u32>(low, high);
  if (isNegative) {
    let negatedLow = ~low + 1u;
    magnitude = vec2<u32>(negatedLow, ~high + select(0u, 1u, negatedLow == 0u));
  }
  let division = divideUnsigned64(magnitude, MILLISECONDS_PER_DAY);
  if (division.quotientHigh != 0u || division.quotientLow > MAXIMUM_DAYS) {
    return invalid;
  }
  var days = i32(division.quotientLow);
  var millisecondsOfDay = division.remainder;
  if (isNegative) {
    // floor(-(q * D + r) / D) = -(q + 1) with D - r left over when r is not zero.
    if (millisecondsOfDay != 0u) {
      days = days + 1;
      millisecondsOfDay = MILLISECONDS_PER_DAY - millisecondsOfDay;
    }
    days = -days;
  }
  let date = civilFromDays(days);
  let isoWeekday = u32(((days + 3) % 7 + 7) % 7);
  let thursday = days - i32(isoWeekday) + 3;
  let thursdayDate = civilFromDays(thursday);
  var fields = invalid;
  fields.valid = true;
  fields.year = date.x;
  fields.month = u32(date.y);
  fields.day = u32(date.z);
  fields.hour = millisecondsOfDay / 3600000u;
  fields.minute = (millisecondsOfDay % 3600000u) / 60000u;
  fields.weekday = (isoWeekday + 7u - firstDay) % 7u;
  fields.dayOfYear = u32(getDayOfYear(date));
  fields.isoWeek = u32((getDayOfYear(thursdayDate) - 1) / 7 + 1);
  fields.isoWeekYear = thursdayDate.x;
  fields.quarter = u32((date.y - 1) / 3 + 1);
  return fields;
}
`,m=`
// Signed 64-bit a <= b on (low, high) words.
fn lessOrEqualInt64(a: vec2<u32>, b: vec2<u32>) -> bool {
  let aHigh = bitcast<i32>(a.y);
  let bHigh = bitcast<i32>(b.y);
  if (aHigh != bHigh) {
    return aHigh < bHigh;
  }
  return a.x <= b.x;
}

// Upper-bound binary search: the number of transitions at or before the instant, which indexes the
// offset in force.
fn getZoneOffsetMilliseconds(time: vec2<u32>) -> i32 {
  var low = 0u;
  var high = TRANSITION_COUNT;
  loop {
    if (low >= high) {
      break;
    }
    let middle = low + (high - low) / 2u;
    let transition = vec2<u32>(
      zoneTransitions[zoneTransitionsOffset + 2u * middle],
      zoneTransitions[zoneTransitionsOffset + 2u * middle + 1u]);
    if (lessOrEqualInt64(transition, time)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let seconds = zoneOffsets[zoneOffsetsOffset + low];
  // A table offset beyond +-86400 s maps to a value the decode rejects.
  return select(seconds * 1000, 0x7fffffff, seconds < -86400 || seconds > 86400);
}
`,h=Date.UTC(2023,0,1);function g(e){let t=e.defaultOrigin,n=e.column(`timestamp`),r=n.length,i=new Float32Array(r),o=new BigInt64Array(r);for(let e=0;e<r;e++)i[e]=n[e],o[e]=BigInt(h+n[e]*1e3);return{count:r,origin:t,positions:e.projectColumn(`position`,t),seconds:i,timeWords:a(o),category:e.column(`category`),categoryNames:e.categories(`category`)}}function _(e,t,n){let r=t===`all`?-1:Number(t),i=0;for(let t=0;t<e.count;t++){let a=r<0||e.category[t]===r?1:0;n[t]=a,i+=a}return i}export{c as i,g as n,f as r,_ as t};