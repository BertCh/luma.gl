# Movement chapter handoff — 2026-10-07

## Scope and status

All eleven movement stories have a redesigned narrative, live computation, data-derived readouts
and both light/dark-theme rendering paths. This handoff deliberately does not alter Networks or
Earth. Work remains uncommitted; preserve the other chapter WIP in this checkout.

| Story | Direct review URL | Main teaching surface |
| --- | --- | --- |
| Harbor playback | `#/story/harbor-playback/the-question` | GPU playhead, time-window tails, resampling and stopped vessels |
| Vessel encounters | `#/story/vessel-encounters/clock` | shared-clock encounter search, spatial lattice, route similarity |
| US shipping day | `#/story/us-shipping-day/the-lanes` | national density, receiver edge and gate crossings |
| Zone dwell | `#/story/zone-dwell/crossing-the-line` | dwell thresholds and zone-size sensitivity |
| Migration season | `#/story/migration-season/africa` | playhead-linked species season comparison |
| Migration timing | `#/story/migration-timing/the-slope-is-speed` | time × latitude reduction and migration speed |
| Migration flyways | `#/story/migration-flyways/sample` | sampling evidence, line density, bottlenecks and fidelity |
| Migration stopovers | `#/story/migration-stopovers/clustering` | stop detection and spatial clustering |
| Gull migration | `#/story/gull-migration/douglas-peucker` | simplification, time ratio and route families |
| Flight corridors | `#/story/flight-corridors/direction` | national airways, direction balance and altitude extrusion |
| Jet stream | `#/story/flight-corridors-jet-stream/east-west` | step metrics and east/west ground-speed contrast |

Append each fragment to the running showcase URL, for example
`http://localhost:5173/#/story/migration-flyways/sample`.

## Final interrupted pass

The three unfinished designs are now implemented rather than only described in metadata.

- **Vessel encounters** now derives a visible 3 × 3, 400 m search lattice from a returned pair,
  a true-distance rule ring, its live connector, and a representative route-sample leash. The
  annotations update from the pair readback and work with both themes. Review:
  `#/story/vessel-encounters/clock`, `#/story/vessel-encounters/distance`,
  `#/story/vessel-encounters/selected`.
- **Migration season** now has a folded-calendar, five-panel small-multiple comparison linked to
  the map playhead and full-year chart. The 35 N counting threshold is marked on the map. Review:
  `#/story/migration-season/africa`.
- **Migration flyways** now exposes its actual convenience sample: per-species tagged years that
  contribute to the selected season, unique-bird and fix totals, plus map labels at the species'
  mean first recorded locations. The new step explicitly distinguishes track density from a
  population census. Review: `#/story/migration-flyways/sample` and
  `#/story/migration-flyways/line-density`.

New movement-local helpers are `vessel-encounters-overlays.ts`,
`migration-season-comparison.ts`, and `migration-flyways-evidence.ts`.

## Deviations and honest limits

- Every migration year is folded onto calendar 2024. A dot, density segment or seasonal share is
  an animal-year / recorded fix, never a population estimate. The new flyway evidence panel makes
  that sampling basis inspectable.
- AIS close approaches are bucketed, planar and based on recorded positions: sub-bucket passes can
  be missed and tiny separations are within position noise. The vessel lattice illustrates exactly
  what is searched; the route-sample leash is an explanatory witness, not a replacement for the
  GPU's discrete Frechet result.
- National shipping and ADS-B maps are receiver observations, not a census. Receiver-edge and
  coverage language is retained rather than interpreting thin coverage as empty traffic.
- Flight altitude is intentionally visually exaggerated only in the pitched altitude step; this is
  documented and covered by the strict-cartography pitch waiver.

## Manual visual review still needed

1. **Flight corridors low-traffic mask — unresolved.** The direction-balance raster currently
   renders all cells with `(east - west) / (east + west)`, including a cell containing one or a
   few pieces. Its vivid ±1 values can read as strong directional evidence. Inspect
   `#/story/flight-corridors/direction` at 0.1° and 0.5°. If this is misleading, add a
   data-derived minimum-total-length mask / transparent no-data treatment and expose the threshold
   in the legend/readout; do not silently clamp it.
2. **East/west altitude profile — unresolved.** Inspect
   `#/story/flight-corridors-jet-stream/where` with chart mode *Altitude (500 m bins)*. The
   cruise filter uses the lower endpoint of each step while the profile bins by the ending
   altitude; at climb/descent edges this may slightly mis-bin an accepted step. Confirm that it is
   visually negligible at the 9 km default, or bin by the same lower/mean altitude used by the
   filter and state that choice.
3. Inspect the new encounter lattice at narrow and wide radii in both themes. It deliberately
   draws a 3 × 3 neighbourhood around a real pair and may need label collision tuning at mobile
   widths.
4. Inspect seasonal multiple-panel legibility on a phone and the flyway sample labels at zoom 3;
   labels are intentionally secondary to the density and are hidden below their minimum zoom.

## Toolkit / library notes

- `addClockEncounters` requires `distance <= cellSize`; the scene exposes a 400 m maximum because
  the contributor's neighbourhood is the adjacent 3 × 3 lattice.
- `GPUTrajectoryPlayhead` and `GPUTimeWindowFilter` keep the migration season playback on the GPU;
  only compact status/summary buffers return to the CPU.
- `GPULineDensity` computes spherical great-circle length and density; map Web Mercator is display
  only. `GPUTrackSimilarity` works on 96 resampled migration points and reports route shape, not
  every detour.
- The current flight direction-balance capability does not provide a confidence/denominator mask;
  see the first manual review item before presenting it as evidence.

## Gates run after the final pass

- `biome check --write` on the nine touched movement scene/helper files: pass.
- `node scripts/check-story-steps.mjs movement`: pass (all 11 movement scenes).
- `node scripts/check-wgsl-identifiers.mjs src/scenes/movement`: pass (532 templates scanned).
- `node scripts/check-cartography.mjs --strict movement`: pass (the flight altitude pitch waiver
  is declared).
- `node scripts/check-colours.mjs`: pass, with repository-wide palette warnings only.
- `tsc --noEmit -p examples/deck/spatial-analysis-showcase/tsconfig.json`: pass after the final
  recheck; no movement or foreign errors remain in that project typecheck.
