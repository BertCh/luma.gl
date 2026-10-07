[out:json][timeout:180];
(
  way["natural"="glacier"](45.86,7.58,46.12,7.98);
  relation["natural"="glacier"](45.86,7.58,46.12,7.98);
  node["natural"="peak"](45.86,7.58,46.12,7.98);
  node["natural"="saddle"](45.86,7.58,46.12,7.98);
  node["place"~"^(town|village|hamlet|isolated_dwelling)$"](45.86,7.58,46.12,7.98);
  node["railway"~"^(station|halt)$"](45.86,7.58,46.12,7.98);
  node["aerialway"="station"](45.86,7.58,46.12,7.98);
  node["tourism"="alpine_hut"](45.86,7.58,46.12,7.98);
  way["railway"~"^(station|halt)$"](45.86,7.58,46.12,7.98);
  way["aerialway"="station"]["name"](45.86,7.58,46.12,7.98);
  way["natural"="water"](45.86,7.58,46.12,7.98);
  relation["natural"="water"](45.86,7.58,46.12,7.98);
  way["railway"="rack"](45.86,7.58,46.12,7.98);
  relation["route"="railway"]["name"~"Gornergrat"](45.86,7.58,46.12,7.98);
);
out geom;
