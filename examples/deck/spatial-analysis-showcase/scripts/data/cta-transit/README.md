# cta-transit
Source: CTA GTFS, https://www.transitchicago.com/downloads/sch_data/google_transit.zip (feed 2026-09-11..19).
Licence: CTA Developer License Agreement (bundled in the zip as developers_license_agreement.htm): limited, revocable licence to use, reproduce, distribute and create derivative works "for the sole purpose of assisting mass transportation riders or in furtherance of promoting public transportation"; no stand-alone sale; credit "Data provided by Chicago Transit Authority" optional. We ship only derived aggregates (stops, route counts, frequencies, shapes, hop graph) inside a demo about transit accessibility, not the timetable. Not affiliated with or endorsed by CTA.
Run: `build.py google_transit.zip community-areas.geojson OUT`.
