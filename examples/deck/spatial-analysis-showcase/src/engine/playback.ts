// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OptionSpec, SceneContext, SceneFrame} from '../scenes/scene';

/**
 * Playback clock for scenes that animate time.
 *
 * The clock advances a time option while a Play toggle is on and writes the slider back with
 * `ctx.setOptions`, so the slider follows the animation. The option state is the single source of
 * truth for scrubbing: if the value in `ctx.options` differs from the last value the clock wrote,
 * the user (or a story step or the URL) moved it and the clock jumps there. Declare the options
 * with `playbackOptions` and call `clock.advance(frame)` first in `encode`.
 */

/** Option ids the clock reads and writes. All are keys of the scene's option state. */
export type PlaybackOptionIds = {
  /** Slider holding the playhead. */
  time: string;
  /** Toggle: the clock runs while true. */
  play: string;
  /** Slider or select multiplying the base rate (optional). Select values are parsed as numbers. */
  speed?: string;
  /** Toggle: wrap at the end instead of stopping (optional; default is to loop). */
  loop?: string;
};

export type PlaybackClockConfig = {
  /** Playhead range in the units of the time option (seconds, hours, days...). */
  range: readonly [number, number];
  /**
   * Time units advanced per real second at speed 1. Give this or `secondsPerLoop`. For a
   * "playback speed" slider that is already simulated seconds per real second, use `rate: 1`.
   */
  rate?: number;
  /** Real seconds one pass over the whole range takes at speed 1 (alternative to `rate`). */
  secondsPerLoop?: number;
  /** Slider step; written values are rounded to it so the slider stays on its grid. Default 0 (no rounding). */
  step?: number;
  /** When true (default) the clock wraps at the end; otherwise it stops and turns Play off. Overridden by the loop option. */
  loop?: boolean;
  /**
   * What a drag of the slider while playing does: `'jump'` (default) keeps playing from the new
   * time; `'pause'` turns Play off.
   */
  scrub?: 'jump' | 'pause';
  /**
   * Call the scene's `setOption('<time id>')` on every slider write-back (`ctx.setOptions`
   * `notify`). Use it when changing the time must trigger work outside `encode` (default false).
   */
  notify?: boolean;
  /** Minimum real seconds between slider write-backs (default 1/20). The internal time is exact regardless. */
  writeIntervalSeconds?: number;
};

export type PlaybackClock = {
  /**
   * Call once per frame at the start of `encode`. Reads the options, advances the time while
   * playing, writes the slider back (coalesced) and returns the playhead in time-option units.
   */
  advance: (frame: SceneFrame) => number;
  /** Playhead of the last `advance`. */
  readonly time: number;
  /** Position in the range, 0 to 1. */
  readonly progress: number;
  /** True when the last `advance` changed the playhead (by playing or by a scrub). Use it to skip work. */
  readonly moved: boolean;
  /** Current range. */
  readonly range: readonly [number, number];
  /** Changes the range (for example after a dataset or level change). Clamps the time into it. */
  setRange: (t0: number, t1: number) => void;
  /** Sets the playhead from code and writes the slider immediately. */
  seek: (time: number) => void;
  /** Back to the start of the range (writes the slider). Does not change Play. */
  reset: () => void;
  /** Turns Play on or off from code (writes the toggle). */
  setPlaying: (playing: boolean) => void;
  /** Real seconds of one pass over the range at the current speed. */
  getLoopSeconds: () => number;
};

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/**
 * Creates the clock. `ctx.options[ids.time]` is the initial playhead. The `ids` must be keys of
 * the scene's option state; the context type is loose so any scene can pass its `ctx`.
 */
export function createPlaybackClock<O extends object>(
  ctx: Pick<SceneContext<O>, 'options' | 'setOptions'>,
  ids: PlaybackOptionIds,
  config: PlaybackClockConfig
): PlaybackClock {
  const read = (id: string) => (ctx.options as Record<string, unknown>)[id];
  const write = (id: string, value: number | boolean) =>
    ctx.setOptions({[id]: value} as unknown as Partial<O>, {
      notify: Boolean(config.notify) && id === ids.time
    });

  let t0 = config.range[0];
  let t1 = config.range[1];
  const step = config.step ?? 0;
  const writeInterval = config.writeIntervalSeconds ?? 1 / 20;
  const baseRate = () =>
    config.rate ?? (config.secondsPerLoop ? (t1 - t0) / config.secondsPerLoop : 1);

  let time = clamp(Number(read(ids.time) ?? t0), t0, t1);
  let lastWritten = Number(read(ids.time) ?? time);
  let lastWriteSeconds = -Infinity;
  let moved = true;
  let wasPlaying = false;

  const quantize = (value: number) => {
    if (step <= 0) return value;
    const snapped = t0 + Math.round((value - t0) / step) * step;
    return clamp(snapped, t0, t1);
  };
  const writeTime = (now: number, force: boolean) => {
    const value = quantize(time);
    if (value === lastWritten) return;
    if (!force && now - lastWriteSeconds < writeInterval) return;
    lastWritten = value;
    lastWriteSeconds = now;
    write(ids.time, value);
  };
  const speedMultiplier = () => {
    if (!ids.speed) return 1;
    const value = Number(read(ids.speed));
    return Number.isFinite(value) ? value : 1;
  };

  const clock: PlaybackClock = {
    advance(frame) {
      const previous = time;
      const optionTime = Number(read(ids.time));
      // The option differs from what we last wrote: a user drag, a story step, the URL or reset.
      if (Number.isFinite(optionTime) && optionTime !== lastWritten) {
        time = clamp(optionTime, t0, t1);
        lastWritten = optionTime;
        if (config.scrub === 'pause' && read(ids.play)) write(ids.play, false);
      }
      const playing = Boolean(read(ids.play));
      if (playing) {
        // Starting from the end (a step turned Play on after a stop): restart.
        if (!wasPlaying && time >= t1) time = t0;
        const looping = ids.loop ? Boolean(read(ids.loop)) : (config.loop ?? true);
        const span = t1 - t0;
        time += frame.deltaSeconds * baseRate() * speedMultiplier();
        if (time >= t1) {
          if (looping && span > 0) {
            time = t0 + ((time - t0) % span);
          } else {
            time = t1;
            write(ids.play, false);
          }
        }
        writeTime(frame.timeSeconds, false);
      } else if (previous !== time) {
        writeTime(frame.timeSeconds, true);
      }
      if (wasPlaying && !playing) {
        // Paused: settle exactly on the slider's grid, so a paused scene is deterministic.
        time = quantize(time);
        writeTime(frame.timeSeconds, true);
      }
      wasPlaying = playing && Boolean(read(ids.play));
      moved = time !== previous;
      return time;
    },
    get time() {
      return time;
    },
    get progress() {
      return t1 > t0 ? (time - t0) / (t1 - t0) : 0;
    },
    get moved() {
      return moved;
    },
    get range() {
      return [t0, t1] as const;
    },
    setRange(low, high) {
      t0 = low;
      t1 = high;
      time = clamp(time, t0, t1);
      moved = true;
    },
    seek(value) {
      time = clamp(value, t0, t1);
      moved = true;
      lastWritten = Number.NaN;
      writeTime(0, true);
    },
    reset() {
      clock.seek(t0);
    },
    setPlaying(playing) {
      write(ids.play, playing);
    },
    getLoopSeconds() {
      const rate = baseRate() * speedMultiplier();
      return rate > 0 ? (t1 - t0) / rate : Infinity;
    }
  };
  return clock;
}

/** Time label formatters for the slider and readouts. */
export const formatPlaybackTime = {
  /** `HH:MM` from seconds since midnight (wraps at 24 h). */
  clock: (seconds: number) => {
    const total = Math.round(seconds / 60) * 60;
    const hours = Math.floor(total / 3600) % 24;
    const minutes = Math.floor((total % 3600) / 60);
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
  },
  /** `HH:MM UTC`. */
  clockUtc: (seconds: number) => `${formatPlaybackTime.clock(seconds)} UTC`,
  /** `HH:MM` from fractional hours since midnight (for example `13.5` is `13:30`). */
  hour: (hours: number) => formatPlaybackTime.clock(hours * 3600),
  /** `12 Jun 2024 14:30` from seconds since `originSeconds` (a Unix timestamp in seconds). */
  dateTime: (seconds: number, originSeconds: number) => {
    const date = new Date((originSeconds + seconds) * 1000);
    const day = date.getUTCDate();
    const month = date.toLocaleString('en-GB', {month: 'short', timeZone: 'UTC'});
    return `${day} ${month} ${date.getUTCFullYear()} ${String(date.getUTCHours()).padStart(2, '0')}:${String(date.getUTCMinutes()).padStart(2, '0')}`;
  },
  /** `12 Jun 2024` from seconds since `originSeconds`. */
  date: (seconds: number, originSeconds: number) => {
    const date = new Date((originSeconds + seconds) * 1000);
    const month = date.toLocaleString('en-GB', {month: 'short', timeZone: 'UTC'});
    return `${date.getUTCDate()} ${month} ${date.getUTCFullYear()}`;
  },
  /** `Day 12` from days since a start (`dayOffset` shifts the first day number, default 1). */
  day: (days: number, dayOffset = 1) => `Day ${Math.floor(days) + dayOffset}`,
  /** `3 h 20 min`, `45 min`, `2 d 4 h` from a duration in seconds. */
  duration: (seconds: number) => {
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes} min`;
    const hours = Math.floor(minutes / 60);
    if (hours < 48) return minutes % 60 ? `${hours} h ${minutes % 60} min` : `${hours} h`;
    const days = Math.floor(hours / 24);
    return hours % 24 ? `${days} d ${hours % 24} h` : `${days} d`;
  }
};

export type PlaybackOptionsConfig<O> = {
  /** Option ids; defaults `time`, `play`, `speed`, `loop`. */
  ids?: Partial<PlaybackOptionIds>;
  /** Time slider range, step and default. */
  time: {
    min: number;
    max: number;
    step: number;
    default?: number;
    label?: string;
    unit?: string;
    /** Value label, for example `formatPlaybackTime.clockUtc`. */
    format?: (value: number) => string;
    /** Extra sentence for the help text, such as the time zone. */
    help?: string;
  };
  /** Start playing. Default true. */
  playing?: boolean;
  /** Speed control. Omit for none. */
  speed?:
    | {
        kind?: 'slider';
        min: number;
        max: number;
        step: number;
        default: number;
        unit?: string;
        label?: string;
        format?: (value: number) => string;
        help?: string;
      }
    | {
        kind: 'select';
        options: readonly {value: string; label: string}[];
        default: string;
        label?: string;
        help?: string;
      };
  /** Add a Loop toggle (default off: the clock loops unless you pass `false` to `createPlaybackClock`'s `loop`). */
  loop?: boolean | {default: boolean};
  /** Panel group heading. Default `'Playback'`. */
  group?: string;
  /** Disables the Play, time, speed and loop controls when this returns true. */
  disabledWhen?: (state: O) => boolean;
  /** Marks the type for inference only. */
  _state?: O;
};

/**
 * Returns the Play toggle, time slider and optional speed and loop options, all `param`
 * options with help text. Spread the result into the scene's `options`:
 * `options: [...playbackOptions<Options>({time: {...}}), ...]`.
 */
export function playbackOptions<O extends object>(
  config: PlaybackOptionsConfig<O>
): OptionSpec<O>[] {
  const ids = {
    time: 'time',
    play: 'play',
    speed: 'speed',
    loop: 'loop',
    ...config.ids
  };
  const group = config.group ?? 'Playback';
  const disabledWhen = config.disabledWhen;
  const specs: Record<string, unknown>[] = [
    {
      kind: 'toggle',
      id: ids.play,
      label: 'Play',
      group,
      disabledWhen,
      apply: 'param',
      default: config.playing ?? true,
      help: 'Runs the clock. The time slider follows it; drag the slider to jump to a moment and keep playing from there. At the end the clock wraps to the start, or stops when Loop is off.'
    },
    {
      kind: 'slider',
      id: ids.time,
      label: config.time.label ?? 'Time',
      group,
      disabledWhen,
      apply: 'param',
      min: config.time.min,
      max: config.time.max,
      step: config.time.step,
      default: config.time.default ?? config.time.min,
      unit: config.time.unit,
      format: config.time.format,
      help: `The playhead. It moves while Play is on; dragging it jumps the clock to that time.${config.time.help ? ` ${config.time.help}` : ''}`
    }
  ];
  if (config.speed) {
    const speed = config.speed;
    if (speed.kind === 'select') {
      specs.push({
        kind: 'select',
        id: ids.speed,
        label: speed.label ?? 'Playback speed',
        group,
        disabledWhen,
        apply: 'param',
        options: speed.options,
        default: speed.default,
        help: speed.help ?? 'How fast the clock runs relative to real time.'
      });
    } else {
      specs.push({
        kind: 'slider',
        id: ids.speed,
        label: speed.label ?? 'Playback speed',
        group,
        disabledWhen,
        apply: 'param',
        min: speed.min,
        max: speed.max,
        step: speed.step,
        default: speed.default,
        unit: speed.unit,
        format: speed.format,
        help: speed.help ?? 'How fast the clock runs relative to real time.'
      });
    }
  }
  if (config.loop !== undefined && config.loop !== false) {
    specs.push({
      kind: 'toggle',
      id: ids.loop,
      label: 'Loop',
      group,
      disabledWhen,
      apply: 'param',
      default: typeof config.loop === 'object' ? config.loop.default : true,
      help: 'Wrap to the start at the end of the range. Off: the clock stops at the end and Play turns off.'
    });
  }
  return specs as unknown as OptionSpec<O>[];
}
