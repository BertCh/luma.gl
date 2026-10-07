## Where does Manhattan fill up and empty? {#the-tide}

Every taxi trip leaves one hexagon and ends in another. Each **{{cellSize}}** hexagon shows arrivals minus departures in the window **{{window}}**: purple took in riders, orange sent them out, pale is about balanced. The biggest gain is **{{gain}}** and the biggest loss **{{loss}}**. Where does the city fill, and where does it empty? Slide **Window start** or **Window length** below.

## Without a midpoint the map loses direction {#midpoint}

Left of the divider is the size of the net on one sequential ramp, the wrong map: it shows where something happens, never which way riders move. Right is the same numbers with zero as a real class, and **{{balanced}}** hexagons with trips are about balanced. Drag the divider, or move **Window start** below.

*Diverging colour is for data with a meaningful middle.*

## Busy places look balanced, quiet ones do not {#share}

A net count follows volume, so the busiest places always shout. The imbalance share, net over arrivals plus departures, compares places fairly. Switch **Show** to share: **{{hidden}}** hexagons fall under the minimum and are hatched, because a few trips make a share meaningless. The most one-sided busy place is **{{sharpest}}**. Lower **Minimum trips** to see the speckle.

*Counts follow volume; shares compare places.*

## The tide turns through the day {#play-the-day}

Press **Play** and the window sweeps both days. The breaks never change, so one colour always means one net number. The lines chart Midtown and the Upper East Side hour by hour with night shaded, and the small panels are the static proof for Friday morning: the animation shows that it moves, the panels show that it turns. **Playback speed** sets the hours per second.

## Hexagon size changes the answer {#hex-size}

The same hour at several sizes. The dashed ring is one hexagon radius, **{{cellSize}}**. The largest net is **{{maxNet}}** and **{{nonBalanced}}** hexagons are out of balance. The smallest hexagons speckle and the largest merge the core into a few cells. Press each **Hexagon size** chip.

*The size of the zone is part of the answer.*

## Evening flows leave from the airports {#airports}

In the window **{{window}}** LaGuardia nets **{{lgaNet}}** and JFK **{{jfkNet}}**: far more cabs leave the terminals than arrive. The arrows are the heaviest flows of the window. Departures follow the pickup clock and arrivals the drop-off clock, a median trip of **{{medianTrip}}** apart. Now try your own: **Window start**, **Window length**, **Show**.
