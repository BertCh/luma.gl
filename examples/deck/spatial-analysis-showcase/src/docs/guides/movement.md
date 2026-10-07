---
title: Movement and trajectories
summary: Playback, stops, encounters, zone dwell, simplification and route similarity for GPS and AIS tracks, and which tool answers which question.
order: 14
---

## What this family does

A track is a list of timed positions: a ferry's AIS fixes, a gull's GPS tag, a taxi trip. Movement analysis asks the questions that only make sense for tracks, and each has a GPU contributor that works on every track at once.

1. **Where is everyone right now?** Interpolate every track at a clock: [`GPUTrajectoryPlayhead`](#/reference/GPUTrajectoryPlayhead). Add the recent past with [`GPUTimeWindowFilter`](#/reference/GPUTimeWindowFilter). Story: [Who is moving in New York Harbor?](#/story/harbor-playback).
2. **How fast, how far, where did it stop?** Per-track lengths, speeds and stop detection: [`GPUTrajectoryMetrics`](#/reference/GPUTrajectoryMetrics). It appears in three stories.
3. **Which tracks meet, and are they alike?** Put every track on one clock and list the pairs within a distance of each other: [`addClockEncounters`](#/reference/addClockEncounters) over [`GPUTrajectoryEncounters`](#/reference/GPUTrajectoryEncounters), then compare whole routes with [`GPUTrackSimilarity`](#/reference/GPUTrackSimilarity). Story: [Which vessels meet, and do they share a route?](#/story/vessel-encounters).
4. **How long does traffic spend in each zone?** [`GPUZoneEvents`](#/reference/GPUZoneEvents) writes the enter and exit events; the recipes [`addFleetDwellZoneEventsRecipe`](#/reference/addFleetDwellZoneEventsRecipe) and [`addFleetDwellRecipe`](#/reference/addFleetDwellRecipe) turn them, or stops, into dwell per zone. Story: [How long do vessels spend in each harbor zone?](#/story/zone-dwell).
5. **How do I lighten, compare and thin out tracks?** [`GPULineSimplification`](#/reference/GPULineSimplification) (Douglas-Peucker or time-aware TD-TR), [`GPUTrajectoryResample`](#/reference/GPUTrajectoryResample) (fixed-size tracks) and [`GPUTemporalReduction`](#/reference/GPUTemporalReduction) (one row per track and time bucket). Story: [Where do the gulls go, and do they all go the same way?](#/story/gull-migration).

Two real datasets run through the four stories: **897 AIS tracks of 471 vessels in New York and New Jersey Harbor on 12 June 2024** (NOAA and the Coast Guard, 63 harbor zones from NOAA charts), and **31 lesser black-backed gulls tracked hourly in autumn 2015** (UvA-BiTS, LifeWatch, INBO).

## Choosing between them

| You want to know | Reach for | It gives you |
| --- | --- | --- |
| Where every track is at time *t* | `GPUTrajectoryPlayhead` | Position, heading, speed and a status (before, active, after, gap) per track, plus a compact list of the active ones |
| The segments live in a moving window | `GPUTimeWindowFilter` | Compact ids, fade weights and clip fractions for trails, with the count written into the draw call |
| Length, speed, stopovers | `GPUTrajectoryMetrics` | Per-track and per-step columns and a bounded stop list with centroids and durations |
| Tracks of equal length | `GPUTrajectoryResample` | A dense tracks x samples table, spaced by time, by distance or on a shared clock |
| Who came close to whom | `addClockEncounters` | Pairs with first meeting, closest approach and time together |
| How alike two whole routes are | `GPUTrackSimilarity` | Discrete Hausdorff and Frechet distances for any list of pairs |
| When a track crosses a boundary | `GPUZoneEvents` | Enter and exit events with interpolated time and position, dwell per (track, zone) |
| Workload per zone | the two fleet dwell recipes | One dense row per zone: count, total, mean and longest dwell |
| A lighter copy of each line | `GPULineSimplification` | Kept vertex ids for any tolerance from one importance column |
| Fewer rows for a scrubbing view | `GPUTemporalReduction` | Count, minimum, maximum, first and last per (cell, time bucket) |

## Key options and how they behave

- **The playhead and every threshold are parameter buffers.** Moving the clock, the maximum gap, a stop threshold, a trail length, an encounter distance or a simplification tolerance writes a few numbers and re-encodes; nothing recompiles. The panel marks the exceptions: the number and spacing of resampled samples, the number of events kept per track and the importance metric are compile-time.
- **Gaps are not guessed.** The playhead's *maximum gap* flags a track as being in a gap instead of drawing it at an interpolated position that may be wrong. AIS moored vessels report only every five minutes, so a short gap limit hides them.
- **Stops are a rule, not a place.** A stop is a run of slow steps lasting at least a minimum duration. The speed and duration you choose define the answer, so look at both sliders before quoting a number.
- **A shared clock trades accuracy for scale.** Encounters are tested only at the clock buckets, so choose a step small compared with the distance a track travels in one step.
- **Hausdorff ignores order; Frechet does not.** Two vessels on the same lane in opposite directions are identical to Hausdorff and far apart to Frechet. Both are computed vertex-to-vertex on resampled routes, matching Shapely `hausdorff_distance` and `frechet_distance` without densification.
- **Importance once, tolerance often.** Simplification computes a per-vertex importance a single time; the tolerance only re-runs a mask and a compaction. With the TD-TR metric the importance also carries time, which keeps the vertices around stops.
- **Two dwell definitions.** *Any time inside* counts a vessel sailing through a channel; *time spent stopped* does not. They answer different questions, so pick the one that matches your question and say which.

## Pitfalls

- **Units follow the data.** Timestamps are float32 seconds from a dataset epoch (exact below about 194 days of seconds), positions are planar meters. The harbor stories project around the harbor; the gull story, spanning 40 degrees of latitude, projects with an azimuthal-equidistant projection so that distances are true, and draws from the degrees.
- **Paths are not vessels.** A vessel that stops reporting for more than 20 minutes starts a new track, so one ship may appear as two tracks and a long stay can look like two visits.
- **Straight-line interpolation.** Between two fixes the playhead draws a straight line. A vessel rounding a corner is drawn on the chord.
- **Approximate zones.** Only the anchorages and channels are official chart polygons; the terminal, gate, ferry-lane and tour areas are hand-drawn and marked approximate.
- **Capacities are bounded.** Pair lists, stop lists and event lists have fixed capacities and an overflow flag. The stories show the flag; if it is set, results are truncated.

Try them in order: start with the harbor playback to learn what a track is, find who meets in the encounters story, then see where the harbor spends its time, and finish with the gulls to see how the same tools scale to a continent.
