## A month of rides, drawn straight {#straight}

Each line joins two stations that riders connected during the month. **Pairs drawn** sets how many of the busiest enter the map: {{drawn}}, carrying {{ridesShare}} of all rides between two stations. Width follows rides, but straight lines pile into a hairball.

The cut is a bias: the pairs drawn are shorter than the rest, median length {{lengthBias}} (drawn, left out). Where does the traffic really run?

## Busy pairs gather into trunks {#bundle}

Press Play on **Iterations**. Each round `GPUEdgeBundling` splats every control point onto a density grid and pulls it uphill toward its neighbours, as in the airline story. Pairs that run together merge into trunks.

The dashed ring is the kernel's reach, {{radiusNow}} this round. It shrinks every round, so trunks start coarse. The price is distance: the bundled lines are {{stretch}} times as long as the straight ones.

## The radius sets the scale of a trunk {#radius}

Choose **Radius presets**. The ring is the kernel when it starts, {{radiusStart}} on the ground: lines closer than that attract. Fine radii keep neighbourhoods apart; at the coarsest, neighbouring corridors fall into one blob.

Now set **Colour by** to crossing: {{crossingShare}} of the rides cross a borough line, drawn in orange; the rest stay home, in slate.

*The radius is the level of generalisation.*

## Bundles are not streets {#not-streets}

Press **Play**: {{bikes}} routed rides are on the road at {{clock}}, Montreal time. They follow the street network; the bundles, faint now, only summarise pairs. **Speed** sets minutes per second.

Of the bundles' length, {{offStreetBundled}} lies farther than {{streetDistance}} from any street these rides used, against {{offStreetStraight}} for the straight lines.

*A bundle is a summary of pairs; bikes follow streets.*

## Bundling saves ink and costs truth {#trust}

Drag the divider: straight lines left, bundles right, same widths and classes. Bundling touches {{areaBundled}} of the map against {{areaStraight}} straight, but the lines are {{stretch}} times as long.

Move **Iterations**, **Kernel radius** or **Pairs drawn** to find where the picture stops being honest.

*Bundling saves ink and costs truth.*
