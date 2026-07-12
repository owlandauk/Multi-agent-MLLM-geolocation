# v6b Full Diagnosis Notes

Source result: `geo_pipeline/results/v6b_full.json` on 4536 YFCC4K records.

## Metrics

```text
Street <1km        5.51%
City <25km        16.45%
Region <200km     26.54%
Country <750km    44.14%
Continent <2500km 62.74%
```

## Diagnosis

The main bottleneck is not geocoding. `missing` geocode count was only 4 and
country-child conflict rate was 3.57%.

The dominant error is country-level North America bias, especially United States:

```text
North America false positives: n=825
  by GT continent: Africa:33, Asia:102, Europe:575, Oceania:52, South America:63
  by predicted country: united states:735, canada:59, mexico:23, costa rica:8
```

Predicted Europe is reliable (`pred Europe acc@2500=87.34%`), while predicted
North America is weak (`pred North America acc@2500=49.48%`). Many European
images are being over-routed to United States/Canada.

## v7 Changes To Evaluate

- Require at least one country/city verification step before POMDP threshold stop.
- Add prompt guard against defaulting to United States/Canada from English text,
  generic roads, vegetation, architecture, media, or product branding alone.
- Pass top-3 country candidates into city reasoning and top country/city
  candidates into street reasoning instead of only the MAP parent.
- Parse and merge wrapper responses shaped like `{general, cue_responses}`.
- In evaluation geocoding, if city/street already embeds a country that conflicts
  with the parent country, geocode the embedded-country child first.

## Compare Next Result Against v6b

When `v7_full.json` is available, compare:

```text
Continent <2500km
Country <750km
North America false positives
pred=North America row in predicted_vs_gt_continent
pred Europe / pred North America acc@2500
Country-child conflict rate
geocode_source
```

Primary target: `North America false positives` should drop meaningfully below
825 without hurting Europe accuracy.

## v7 Full Result And Regression

`v7_full.json` reduced North America false positives sharply, but over-corrected
and hurt overall accuracy:

```text
Total records: 4516
Street <1km        3.57%
City <25km        12.98%
Region <200km     21.30%
Country <750km    37.93%
Continent <2500km 56.71%
North America false positives: n=120
Country-child conflict rate: 51.84%
```

The key regression is not North America bias anymore. It is child-level country
override: `child_country_override: 1877`, `street_embedded_country: 1871`, and a
51.84% country-child conflict rate. The v7 prompt encouraged child levels to
introduce alternative countries, and evaluation trusted embedded child countries
over the parent country. This often geocoded an incorrect child country even
when the country-level prediction was usable.

## v8 Fix Direction

- Keep wrapper parsing and the "at least one verification step before stop"
  rule.
- Remove evaluation-time child-country override.
- Skip conflicting child country geocodes and fall back to lower-risk city or
  country-level geocoding instead of querying mixed-country strings.
- Tighten city/street context so child levels refine within top country/city
  candidates and do not introduce countries outside those candidates unless
  directly visible in the image.

For v8, compare against both v6b and v7. Desired pattern: North America false
positives stay far below 825, while country-child conflict returns near the v6b
range instead of 51.84%, and continent/country accuracy recover above v7.
