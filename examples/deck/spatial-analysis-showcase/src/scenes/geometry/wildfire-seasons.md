## Four fire years, ordered by perimeter date {#all-fires}

Every polygon in the archive carries the date of its final NIFC perimeter. This step shows all 90 fires at once, colored by year, with a running total in the readouts: about 2.7 million acres, of which the 2020 season and the 2021 Dixie Fire make up almost all.

Read the dates carefully: they are the date of the **perimeter record**, usually the last mapping near containment, not the ignition. The Dixie Fire started on 13 July 2021 but its perimeter is dated 25 October. So the playback below shows when fires were mapped and closed out, not how they grew. The archive also lacks some famous fires (the August Complex, Caldor), and NIFC's daily progression layer holds multiple perimeters for only a handful of fires of these years, so a real growth animation is not possible from key-free data.

## Press play {#play}

The playhead runs through 1,300 days at 55 days per second. A fire appears when the playhead passes its date; fires from the last 365 days stay on, fading with age (the magma ramp: bright is newly mapped). **Play** starts and pauses the clock, **Date** is the playhead (drag it to jump, and the clock continues from there) and **Playback speed** scales the rate.

The cumulative chart steps up with every perimeter: it climbs in the summer and autumn of 2020, jumps with Dixie in October 2021, and then barely moves in 2022 and 2023. By the end of September 2020, the readout shows about 80% of that year's acres already mapped.

## A time window on the GPU {#window}

`GPUTimeWindowFilter` decides which fires are in view. It is told a window `[date - Trail length, date]` and a fade, and for every fire writes an accepted flag and a fade weight, so no fire list is ever rebuilt on the CPU. Moving the playhead only rewrites an eight-number parameter buffer.

Here the playhead rests in the busy second half of September 2020 with a 60-day trail. **Trail length** is the window; **Fade of the old end** is the share of the window over which fires dim from the old end. The **GPU window count** is what the filter itself reports; **Fires in the window** is the accepted flags read back.

## How many acres, measured on the GPU? {#acres}

`GPUGeometryMeasures` measures the area of every fire and, in the same pass, the area and number of fires per *group*: **Group fires by** year or acreage class (rewriting the group id buffer, no rebuild). The bars show the total area per group. The measured total and the NIFC total agree when the **Area system** is `wgs84`; `planar` Web Mercator overstates by about 1.6 times, which is the same trap as in the shapes scene. Changing the area system is a compile-time option, so it rebuilds the measures graph.

Sorting the fires by class shows the skew: three fires over 300,000 acres, most of the 90 fires under 10,000 acres. By year the bars show the difference between a bad fire season and a bad single fire.

## Follow the fires {#follow}

With **Camera follows the newest fire** on, every new perimeter pulls the camera to it, at most once a second, and the yellow outline marks the fire that just appeared. At twice the speed, September 2020 becomes a tour of California, Oregon and Idaho. Turn the toggle off to look around yourself.

A caveat for all of it: the dates are the dates of the NIFC records, the fires are those the archive carries, and the acres are the polygons' own.
