## Which docks drain at 8:30, and where do the bikes pile up? {#the-question}

Each dot is a BIXI station coloured by its **net balance** in the time window: rides arriving minus rides leaving, summed over the 22 weekdays of August 2024. Blue stations lose bikes (more rides start there than end there), red stations gain them. The window is 08:00 to 09:00 on weekdays, the morning commute: over the month, Ville-Marie gains a net 19,000 rides in that hour while the Plateau loses 10,700 and Rosemont 4,500.

The question is the operator's: *where must trucks move bikes, and when?* The answer is not constant. `GPUFlowAggregation` recomputes it for any window from 685,000 records on the GPU: it counts every station pair, ranks them, and writes per-station departure and arrival totals the dots are coloured from. Orange-to-cyan arcs are the largest flows in the same window.

## Slide the window {#window}

Every record carries a start hour and a weekday-or-weekend flag, and the contributor gates records with a **time window**: four numbers in a parameter buffer, so moving it re-runs the same compiled graph without a rebuild. **Window start**, **Window length** and **Day type** below change it; the arcs re-rank as the window moves.

Try 08:00 for one hour: the residential boroughs (Plateau, Rosemont, Villeray, the Sud-Ouest) lose bikes while downtown fills. Slide the window to 17:00 and downtown loses a net 5,900 rides as the Sud-Ouest (+3,600) and Hochelaga-Maisonneuve (+1,200) fill: the commute home ends in the east and south-west, not where it began. The readouts give the busiest pairs in the window and the share of rides the drawn arcs carry. Pairs with fewer than 10 rides in August are folded into one "other stations" zone so every station total is still exact; their arcs are not drawn.

## Play the day {#play}

Press **Play** and the clock sweeps the window across 24 hours: the slider follows it, the shading and the arcs re-rank as it moves, and **Loop** repeats. **Play speed** is hours of the day per real second.

Watch for three things. Before 06:00 almost nothing moves. At 07:30 to 09:00 the commute drains the residential stations toward downtown and the universities, and the rides-per-hour chart climbs. At 17:00 to 18:30 the tide turns, and the stations that filled in the morning empty. On **Weekends** the pulse is smoother and peaks in the afternoon, with no commute.

## Click a station {#station}

Click any station to follow it. The sparkline shows its net balance for each hour of an average day, with a dot at the start of the window, and the text gives the hour it loses the most bikes and the hour it gains the most. Raise **Hide balances below** to leave only the stations with a large imbalance, and compare the two bar groups: the five biggest drains and the five biggest sinks of the window.

In the 08:00 weekday hour the biggest sink is Square-Victoria (+1,766 rides over the month), and the biggest drains are Plateau stations such as Parc Jeanne-Mance and Laval / Duluth. St-Dominique / St-Viateur Nord fills in the morning (+642) and drains at 17:00 (-602). That asymmetry is what an operator rebalances against.

## Limits and things to try {#limits}

**Caveats.** The balance is arrivals minus departures from the trip table only. BIXI rebalances with trucks and with the "bonus" docks, and those moves are not in the data, so the real stock at a dock is not this number: it is the pressure the riders create. Rides under a minute (failed docks) and rides missing a station are dropped, and the full-day totals are averages over 22 weekdays and 9 weekend days of one month. Rare pairs are folded into residual rows (the arcs omit them, the station totals include them).

**Try it.** Set **Day type** to weekends and **Window start** to 14:00: the weekday morning tide shrinks to a few dozen rides per station. Switch **Station shading** to departures to see volume rather than balance. Set **Window length** to 24 for the whole day: only the persistent sinks and sources remain, and the hour at which they fill is lost, which is why the hourly view matters. Switch **Summation order** to atomic and note that the totals can differ in the last digit.
