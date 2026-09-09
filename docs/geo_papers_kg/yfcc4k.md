---
title: "YFCC4K Benchmark"
authors: (derived from YFCC100M, Thomee et al.)
venue: (benchmark subset)
year: 2016
arxiv: (YFCC100M)
topics: [benchmark, Flickr photos, worldwide geoloc]
base_model: n/a (dataset)
benchmarks: [YFCC4K]
role: PRIMARY evaluation benchmark for this thesis
---

# YFCC4K

4K-image geolocation benchmark drawn from Flickr (YFCC100M). Photographer-biased toward landmarks and North America/Europe.

## Relation to this thesis
- The PRIMARY benchmark the thesis reports on. Distance thresholds 1/25/200/750/2500 km = Street/City/Region/Country/Continent.
- Reference numbers: [[geobayes]] 4.9/16.1/30.9/55.8/75.4; [[geoclip]] 11.82/29.04/56.13/79.13; thesis same-code baseline 5.0/14.73/24.0/42.07/64.2.
- Its North-America bias is part of why [[pure-mllm-country-bias]] surfaces; contrast with OOD [[gws15k]].
- Sibling benchmark [[im2gps3k]].
