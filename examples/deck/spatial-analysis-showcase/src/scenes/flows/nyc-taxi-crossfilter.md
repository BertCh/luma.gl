## Which trips make a taxi city? {#the-question}

These are **440,000 yellow-taxi trips** from the first 38 hours of 2015: New Year's night, New Year's Day and Friday morning. Every dot is one pickup, coloured by the fare. The question is the one a fleet analyst asks first: *which kinds of trips, when and where, carry the money?* With 440,000 rows and five filters the answer has to update as fast as you can drag, so nothing is filtered on the CPU.

`GPUCrossfilter` keeps the rows, the five brushes and every chart's counts on the GPU. The charts below the map are histograms of the same rows; the numbers in **Selected trips** and **Mean fare** are a few hundred values read back after each brush. Pick **Colour points by** and **Show** below to change the encoding, then move on to brush something.

## Brush the clock: the Friday rush {#brush-time}

Drag the **Pickup hours** range below to Friday 07:00 to 10:00 (hours 31 to 34). Only the trips picked up in that window stay on the map; the other charts redraw for exactly those trips. This is the *linked* part of a crossfilter: each chart shows the rows that pass every brush **except its own**, so the hour chart keeps showing the whole day while you brush it, and the highlighted bars are the selection.

The rush hour is 6% of all trips with an ordinary fare ($11.70, $3.80 a mile). Slide the window to the first two hours of the year instead (hours 0 to 2): that is 11% of all trips in two hours, and the fare per mile is higher ($4.21) because slow, crowded streets add waiting time to the meter. **Pickup day** below switches a whole day off.

## Long trips carry a quarter of the money {#long-trips}

Brush **Trip distance** to 8 miles and over. Show **both ends** of each trip: orange is the pickup, cyan the dropoff. Only 8.5% of trips are this long, but at an average fare of $37.61 they bring in about a quarter of all fares. Nearly one in four of them ends at JFK or LaGuardia.

The fare histogram updates as you move the distance brush: the long right tail of expensive fares is made of these trips. Per mile they are cheaper ($2.94, against $3.87 overall) because the flag drop and waiting charges are spread over many more miles.

## Draw a rectangle: JFK {#airports}

The map is a dimension too. This step brushes a rectangle around JFK (**Map area** below): 8,825 pickups, 2% of all trips, at a mean fare of $45 for 16 miles. **Shift-drag** anywhere on the map, or switch on **Drag to brush the map**, to draw your own rectangle; it becomes the area **Drawn on the map**. The rectangle is an inclusive range test on the pickup coordinates, evaluated for every row on the GPU.

Try the other areas: Midtown pickups are 52% of all trips at $10.95, because most rides are short hops inside Manhattan; LaGuardia pickups are $28 for 9 miles. Switch **Show** to dropoffs to see where those same rides end.

## Does party size change the fare? {#party-size}

Brush **Passengers** to 4 to 6. Larger parties are 12.5% of trips and their mean fare ($12.76) is almost the same as the single riders' ($12.12): a taxi is metered by distance and time, not by head count. The bar chart under **Mean fare by party size** is a `GPUCrossfilter` *group* view, a sum of fare per passenger count divided by the count.

The passenger histogram also keeps showing all party sizes while you brush it, which is the self-excluding rule at work. Try combining this brush with the hour range: the late-night groups are larger than the commuters.

## Why the charts ignore their own brush, and what to trust {#limits}

Switch **Histograms ignore their own brush** below off: the hour chart now shows only the selected hours and the other bars vanish. Self-exclusion is a *compile-time* choice (the effective mask of each view changes), so the panel marks it with a rebuild badge and **Under the hood** counts a new graph. **Pickup day** is a live mask: rows of a day that is switched off do not enter any chart, count or list at all.

**Caveats.** These are yellow taxis only, in 2015, before ride-hail dominated; they under-represent the outer boroughs. The trips are the 500,000 in the poopdeck.gl route archive, sampled to 440,000, and the archive stops mid-afternoon on 2 January. Drop-off times are the OSRM route time, not the metered one. Fares exclude tips and tolls, and distances above 15 miles or fares above $60 fall off the histogram axes (the brush still includes them).
