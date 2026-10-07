## What does a Friday rush hour look like, route by route? {#the-question}

Each glowing line is the path of one yellow taxi, drawn for the last few minutes and fading behind its head, on **Friday 2 January 2015 between 08:00 and 08:30**. The head is the taxi's position now. About 1,650 of the 9,000 trips in this window are on the road at the same moment, and together they trace the street grid: Manhattan's avenues, the bridges and the Brooklyn and Queens arterials.

Two compute graphs do the work and nothing is drawn from the CPU: `GPUTrajectoryPlayhead` interpolates every taxi to the clock, and `GPUTimeWindowFilter` selects the route segments whose time falls in the trail window and gives each a fade weight. The renderer reads their output buffers directly.

## Trail length and fade {#trails}

Slide **Trail length** below: it is the width of the time window, in minutes of taxi time, written into a four-number parameter buffer every frame, so changing it never recompiles anything. A short trail shows where each taxi is and which way it is going; a long one shows the corridor it just used and the avenues start to glow.

**Fade** sets how much of the trail's tail dims. At 0 every segment is drawn at full strength until it leaves the window; at 1 the oldest part is already invisible. The window also clips the segment that straddles its edge (the contributor's *clip fractions*), so a trail ends mid-block instead of at the nearest vertex.

## A clock you can scrub {#clock}

Press **Play** and the clock advances at **Playback speed** simulated seconds per real second; the **Clock** slider follows it, and you can drag it to jump anywhere. The line chart shows how many trips are on the road in each half minute; the marker is the clock. The curve is lower at both ends because the window cuts trips that were already under way at 08:00 or have not finished by 08:30, so the first minutes undercount.

At 08:15 about 1,650 of the sampled trips are active. The sample is a seeded half of the 17,797 trips in the window, so the real count is about double that. Try a slow speed to follow a single taxi from kerb to kerb.

## Colour by what the trip earns {#fare}

Colour the trails and heads by **Fare** or **Trip distance** with **Colour by** below, and raise **Hide trips below** to keep only the long and expensive ones. A fare of $20 or more is one trip in six (1,408 of 9,000), and they are the long trips. Hiding the cheap ones leaves a thin network of long corridors.

The ramp is a per-trip buffer read through the segment's track index, and hiding is a threshold in the shader, so neither recompiles nor rewrites geometry.

## Routes, not traces {#routes}

These are not GPS traces. The poopdeck.gl archive routes each trip with OSRM between its recorded pickup and dropoff and spreads the time along the route, so every taxi appears to follow the best road path at a smooth pace. Turn on **Show every route** to see all 402,000 route segments at once as a faint network: every trip that was routed along the same avenue lands on the same line.

That has two consequences. Congestion is absent: a taxi never waits at a light, so speeds describe the road model, not the real morning. And the trips that look identical are not: two taxis on the same avenue were routed separately. Treat the picture as where trips *went*, not as how they drove.

## Limits {#limits}

**Caveats.** Yellow taxis only, in 2015, before ride-hail dominated. Trips are clipped to the 08:00 to 08:30 window, so a trip that began at 07:50 appears from 08:00 at the place it had reached. The count capacity of the compact lists is fixed at the number of trips; the readout **List overflow** would say if it were exceeded. Times are local New York time (the archive stores them as if UTC).
