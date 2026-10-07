## Equal size reveals shape, not acreage {#gallery}

The gallery takes the six lowest and six highest live Polsby-Popper rows from the current year and acreage filter. Each loaded outline is north-up and scaled to one extent, so the comparison is about perimeter shape rather than acreage. Hover or select a silhouette for its loaded name, acreage, parts and compactness.

## One fire may contain many polygons {#multipart}

`GPUGeometryMeasures` follows each feature's rings and parts. The selected multipart row is resolved from the loaded sample; its area, ring count and alternate hole-rule error are live readouts. Winding adds exterior rings and subtracts holes, while treating every later ring as a hole is a deliberately unsafe comparison for multipart geometry.

## Compactness compares area with perimeter {#compactness}

Polsby-Popper is `4 pi A / P²`: a circle is 1 and a thin or ragged mapped perimeter approaches zero. The fixed reference classes are shared by map and legend: <.05, .05–.15, .15–.30 and .30–.50, with the strongest colour reserved for the stringiest class. The selected outline and its readout make the area/perimeter relationship inspectable.

## Larger mapped fires look less compact {#trend}

The scatter and rank statistic are rebuilt from the current qualifying rows. They describe this mapped-perimeter sample only: vertex density, multipart treatment and source detail can all move compactness, so the plot is not evidence about fire behaviour.

## More vertices can lengthen perimeter {#detail}

This archive has multipart rings but no prepared topology-safe simplification hierarchy. Instead of dropping every nth vertex, the detail view reports the live vertices-per-kilometre strata and keeps the original ring visible. That fallback is honest about what this data can support: more mapped detail can change measured perimeter and compactness, but this scene does not claim a remeasured simplification result.

## One index cannot describe every form {#other-shapes}

The selected loaded example has a second-moment major axis and a convex-hull comparison. Switch the metric tab between compactness, elongation and convexity; each describes a different geometric property. Hull-minus-fire is an outline comparison, not a process explanation.
