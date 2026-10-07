---
title: Space-time analysis
summary: Time as a dimension of the map - cubes, interaction tests, scan statistics, Markov chains and calendar decoding.
order: 40
---

## What the family does

A map of one period tells you where. The tools in this chapter add *when*, and they answer four different kinds of question. Pick the tool by the question, not by the data.

| Question | Tool | Story |
| --- | --- | --- |
| Which places are newly hot, persistently hot or fading? | [`GPUEmergingHotSpots`](#/reference/GPUEmergingHotSpots) over a space-time cube | [Emerging hot spots](#/story/emerging-hot-spots) |
| Do events near each other in space also happen near each other in time? | [`GPUKnoxTest`](#/reference/GPUKnoxTest), [`GPUMantelTest`](#/reference/GPUMantelTest) | [Space-time clusters](#/story/space-time-clusters) |
| Where and when is there a burst of observations the baseline does not explain? | [`GPUSpatialScanStatistic`](#/reference/GPUSpatialScanStatistic) | [Space-time clusters](#/story/space-time-clusters) |
| Do places stay in their class over time, and does the neighborhood matter? | [`GPUTransitionMatrix`](#/reference/GPUTransitionMatrix), [`GPUSpatialMarkov`](#/reference/GPUSpatialMarkov), [`GPULISAMarkov`](#/reference/GPULISAMarkov) | [Election dynamics](#/story/election-dynamics) |
| What does the week look like, and where at which hour? | [`GPUCalendarBuckets`](#/reference/GPUCalendarBuckets), [`GPUTimeWindowFilter`](#/reference/GPUTimeWindowFilter), [`GPUGroupStatistics`](#/reference/GPUGroupStatistics) | [Calendar patterns](#/story/calendar-patterns) |

## Building the cube

Most space-time statistics start from a dense **cube**: one count (or value) per cell and time slice, indexed `cell * sliceCount + slice`. Two contributors build it on the GPU, and they cut time differently.

- [`GPUTemporalReduction`](#/reference/GPUTemporalReduction) buckets rows by *equal-width* time from an origin: the 52 seven-day weeks of the emerging hot spot story. The bucket origin and width are parameters; the number of buckets is compile-time.
- [`addSpaceTimeHotSpotsRecipe`](#/reference/addSpaceTimeHotSpotsRecipe) cuts time by *calendar field* (month, hour, ISO week, weekday) using `GPUCalendarBuckets`, so the slices are months or hours of the day, not fixed-length intervals.

Both produce a `uint32` count cube that `GPUEmergingHotSpots` reads directly. The same layout (`zone * timeBuckets + bucket`) is what `GPUSpatialScanStatistic` expects as its `cases`.

## Key options and what they change

- **Neighborhoods.** Emerging hot spots take either a lattice radius in cells or arbitrary spatial weights (a `GPUNeighborSearch` table). With binary weights the two agree; weights mode is the way in for H3 cells, polygons and points, and the only way to decay neighbors by distance.
- **Temporal window.** A bin also gathers its neighbors in the previous *k* slices. Larger *k* smooths flicker and raises the bar for "new".
- **Confidence and persistence.** The confidence level sets the Gi* critical z; the persistence share (default 90%) decides when a spot is persistent, intensifying, diminishing or historical.
- **Permutations and seeds.** Knox, Mantel and the scan statistic judge themselves against Monte Carlo replicates. The smallest possible p-value is `1 / (replicates + 1)`; a new seed shows how stable a result is.
- **Baselines.** A scan statistic only finds "more than expected", so the expectation is the analysis. Population, history-based independence and a uniform baseline give different clusters from the same cases.
- **Classes and lags.** Markov chains need one set of class breaks for all periods (pooled breaks), a transition step (one or several periods) and, for the spatial Markov, a classification of the neighbors' average.
- **Clock handling.** `GPUCalendarBuckets` takes a fixed UTC offset and the first day of the week as parameters; daylight saving needs a per-row offset column.

## Which options rebuild

Anything that changes a *shape* is compile-time: the number of time slices, the number of periods, the row count of an event table, whether a per-row offset column exists. Everything else is a buffer write. The options panel marks compile-time options with a rebuild badge, and the **Under the hood** drawer counts rebuilds. Each story compiles a variant on first use and keeps it, so switching back is free.

## Pitfalls

- **Observers are not wildlife.** iNaturalist observations measure where and when people look as much as what is there: parks, the lakefront and weekends dominate, and the City Nature Challenge weekend (28 April to 1 May 2023) is a spike made by a campaign. 311 complaints have the same problem. Hot spots of records can be hot spots of effort.
- **Seasons look like trends.** Almost every group has a season (birds in May, insects in July, fungi into October), so a 12-month or 52-week trend test mostly reports the season. Filter to one group, or read the persistent classes first.
- **Clock times are not instants.** The observation timestamps are Chicago local clock time stored as if it were UTC; the story says so and the calendar scene lets you break it on purpose.
- **Cyclic time is not linear.** Slices of the hour of day wrap around. Mean hours of observations that straddle midnight are misleading; use the median or a percentile on a rotated clock.
- **Missing periods change the sample.** The 2024 county returns are missing for five states, so including 2024 drops those counties from every election.
- **Many tests, some false alarms.** Thousands of Gi* tests at 90% will flag cells by chance. Raise the confidence level or read the persistent and intensifying classes first.

## Reading the maps

Hot spots are warm and cold spots are cool; new, consecutive and sporadic classes are paler, entrenched ones are saturated. Clusters from the scan statistic are colored by rank, with the time span in the readouts. In the calendar story the hour by weekday matrix sits in Lake Michigan east of the Loop, rows from the first day of the week at the top, columns hours 0 to 23.
