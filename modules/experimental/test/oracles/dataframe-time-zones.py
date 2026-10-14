# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference time zone transitions and local calendar fields for GPUCalendarBuckets.

Writes fixtures/dataframe-time-zones.json next to this script (consumed by
test/gpu-dataframe/calendar-buckets/time-zone-reference.node.spec.ts and the 'time zone table'
cases of gpu-calendar-buckets.spec.ts). Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/dataframe-time-zones.py
Only the standard library (zoneinfo, datetime) is used. zoneinfo reads the system tz database
(or the `tzdata` package), whose release is recorded in the fixture; Node's Intl uses the ICU data
bundled with V8, which may be another release, so the node spec states what it asserts exactly.

Zones: America/New_York (US DST), Europe/London (UTC+0/+1, pre-1972 British Standard Time),
Australia/Lord_Howe (30-minute DST), Asia/Kathmandu (+05:45 and LMT +05:41:16), Africa/Casablanca
(Ramadan DST), Europe/Amsterdam (a link to Europe/Brussels since tzdata 2022b, so it no longer
has the old +00:19:32 Amsterdam mean time inside the range and covers only whole-minute pre-1940
offsets), Europe/Dublin (Dublin mean time -00:25:21 before 1916, an offset with seconds), and UTC
(no transition). Seconds-resolution offsets are therefore asserted on Asia/Kathmandu (+05:41:16
until 1920) and Europe/Dublin.

Transitions: zoneinfo has no public transition list, so the generator samples the UTC offset every
day over [1900-01-01, 2040-01-01] and bisects each change to the exact millisecond (zoneinfo
works in whole seconds, so the first millisecond of a transition is a whole second). One day
cannot skip a transition pair because the zones keep weeks between transitions; the generator
asserts every consecutive pair is at least 7 days apart.

Samples: for every transition t the instants t - 1 ms, t and t + 1 ms, plus the range ends and the
epoch. Each sample stores the local fields in the GPUCalendarBuckets vocabulary: year, month,
dayOfMonth, hour, minute, weekday (Monday = 0), dayOfYear, isoWeek, isoWeekYear, quarter and the
offset in seconds, computed with datetime from the zone-local wall time.
"""

import datetime
import json
import os
import sys
import zoneinfo

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'dataframe-time-zones.json')

ZONES = [
    'America/New_York',
    'Europe/London',
    'Australia/Lord_Howe',
    'Asia/Kathmandu',
    'Africa/Casablanca',
    'Europe/Amsterdam',
    'Europe/Dublin',
    'UTC',
]
EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
RANGE_START = int(datetime.datetime(1900, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000)
RANGE_END = int(datetime.datetime(2040, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000)
DAY = 86_400_000


def local_time(zone, milliseconds):
    # timedelta resolves microseconds; milliseconds are exact in it.
    instant = EPOCH + datetime.timedelta(milliseconds=milliseconds)
    return instant.astimezone(zone)


def offset_seconds(zone, milliseconds):
    return int(local_time(zone, milliseconds).utcoffset().total_seconds())


def find_transitions(zone):
    transitions = []
    offsets = [offset_seconds(zone, RANGE_START)]
    previous_instant = RANGE_START
    instant = RANGE_START
    while instant < RANGE_END:
        instant = min(instant + DAY, RANGE_END)
        offset = offset_seconds(zone, instant)
        if offset != offsets[-1]:
            low, high = previous_instant, instant
            while high - low > 1:
                middle = low + (high - low) // 2
                if offset_seconds(zone, middle) == offsets[-1]:
                    low = middle
                else:
                    high = middle
            transitions.append(high)
            offsets.append(offset_seconds(zone, high))
            assert offsets[-1] == offset, 'two transitions inside one sampling day'
        previous_instant = instant
    for a, b in zip(transitions, transitions[1:]):
        assert b - a >= 7 * DAY, 'transitions closer than the sampling step allows'
    return transitions, offsets


def fields(zone, milliseconds):
    local = local_time(zone, milliseconds)
    iso_year, iso_week, _ = local.isocalendar()
    return {
        'year': local.year,
        'month': local.month,
        'dayOfMonth': local.day,
        'hour': local.hour,
        'minute': local.minute,
        'weekday': local.weekday(),
        'dayOfYear': local.timetuple().tm_yday,
        'isoWeek': iso_week,
        'isoWeekYear': iso_year,
        'quarter': (local.month - 1) // 3 + 1,
        'offsetSeconds': int(local.utcoffset().total_seconds()),
    }


def database_version():
    try:
        import tzdata  # type: ignore

        return 'tzdata package ' + tzdata.IANA_VERSION
    except ImportError:
        pass
    for root in zoneinfo.TZPATH:
        path = os.path.join(root, '+VERSION')
        if os.path.exists(path):
            return open(path).read().strip()
        path = os.path.join(root, 'tzdata.zi')
        if os.path.exists(path):
            return open(path).readline().strip()
    return 'system tz database (version file not found)'


def main():
    zones = {}
    for name in ZONES:
        zone = zoneinfo.ZoneInfo(name)
        transitions, offsets = find_transitions(zone)
        instants = {RANGE_START, RANGE_END, 0, -1, 1}
        for transition in transitions:
            instants.update((transition - 1, transition, transition + 1))
        samples = [
            {'time': t, **fields(zone, t)} for t in sorted(instants)
        ]
        zones[name] = {'transitions': transitions, 'offsets': offsets, 'samples': samples}
        print(name, len(transitions), 'transitions', len(samples), 'samples', file=sys.stderr)
    fixture = {
        'generator': 'test/oracles/dataframe-time-zones.py',
        'python': sys.version.split()[0],
        'tzDatabase': database_version(),
        'rangeStart': RANGE_START,
        'rangeEnd': RANGE_END,
        'zones': zones,
    }
    with open(OUTPUT, 'w') as handle:
        json.dump(fixture, handle, separators=(',', ':'))
        handle.write('\n')


main()
