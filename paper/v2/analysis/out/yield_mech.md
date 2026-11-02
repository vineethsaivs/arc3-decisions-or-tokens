| run | yield | usage | turn-start invocations | resume (structural; wording tools/bare) | nudges | openers incl. nudges | turns | req/turn | req/inv | yield ends | executed ends | other ends |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| franzen-p448s14-v1 | 2048 | True | 1205 | 565 (492/68) | 0 | 1770 | 1205 | 2.13 | 1.45 | 565 | 1188 | 17 |
| franzen-p448s14rep-v1 | 2048 | True | 1131 | 634 (596/31) | 4 | 1769 | 1131 | 2.31 | 1.48 | 634 | 1116 | 15 |
| franzen-l1a1tbfc-v1 | 2048 | True | 1084 | 572 (568/0) | 1 | 1657 | 1084 | 2.19 | 1.43 | 572 | 1069 | 15 |
| franzen-l1a1y8-v1 | 8192 | True | 1154 | 185 (149/35) | 1 | 1340 | 1154 | 2.27 | 1.96 | 185 | 1144 | 10 |
| franzen-l1a1y8r-v1 | 8192 | False | 1145 | 151 (124/25) | 1 | 1297 | 1145 | 2.13 | 1.88 | 151 | 1130 | 15 |
| franzen-l1a1y8sv2-v1 | 8192 | False | 1121 | 196 (149/45) | 0 | 1317 | 1121 | 2.25 | 1.91 | 196 | 1104 | 17 |
| franzen-l1a1y8sv2o16-v1 | 8192 | False | 1216 | 171 (158/12) | 0 | 1387 | 1216 | 2.03 | 1.78 | 171 | 1199 | 17 |
| franzen-l1a1y8sv2r1-v1 | 8192 | False | 1324 | 194 (145/48) | 0 | 1518 | 1324 | 2.17 | 1.89 | 194 | 1309 | 15 |
| franzen-l1a1y8sv2r2-v1 | 8192 | False | 1321 | 200 (166/34) | 0 | 1521 | 1321 | 2.18 | 1.90 | 200 | 1306 | 15 |
| franzen-l1a1y8sv2r3-v1 | 8192 | False | 1235 | 215 (178/36) | 0 | 1450 | 1235 | 2.32 | 1.97 | 215 | 1219 | 16 |
| franzen-l1a1y8sv2r3s4-v1 | 8192 | False | 1257 | 194 (155/39) | 1 | 1452 | 1257 | 2.24 | 1.94 | 194 | 1241 | 16 |
| franzen-l1a1y0sv2-v1 | 0 | False | 1209 | 141 (100/0) | 41 | 1391 | 1209 | 2.16 | 1.93 | 0 | 1191 | 159 |

| run | completion (M) | growth (M) | resume opener tokens (est.) | resume share of growth (est. / upper) | new prefill (M) | of which first-request-after-resume (mean per request) | trims | gen/turn |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| franzen-p448s14-v1 | 5.30 | 8.51 | 901 | 5.6% / 9.1% | 12.54 | 3.48 (6205; fresh 6354, within 1775) | 102 | 4396 |
| franzen-p448s14rep-v1 | 5.29 | 8.49 | 937 | 6.6% / 10.2% | 12.69 | 4.26 (6769; fresh 6097, within 1812) | 104 | 4679 |
| franzen-l1a1tbfc-v1 | 5.62 | 8.58 | 909 | 5.5% / 9.0% | 10.35 | 3.64 (6420; fresh 5354, within 1266) | 108 | 5182 |
| franzen-l1a1y8-v1 | 5.68 | 8.48 | 976 | 1.9% / 2.9% | 8.85 | 1.29 (6998; fresh 4618, within 1748) | 98 | 4924 |

Replay of resume boundaries inside clean turns (observed vs re-segmented at Y):
| run | own Y | clean turns | observed resumes | replay Y=2048 | replay Y=8192 | replay Y=0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| franzen-p448s14-v1 | 2048 | 1188 of 1205 | 546 | 546 | 135 | 0 |
| franzen-p448s14rep-v1 | 2048 | 1116 of 1131 | 587 | 587 | 144 | 0 |
| franzen-l1a1tbfc-v1 | 2048 | 1069 of 1084 | 523 | 523 | 154 | 0 |
| franzen-l1a1y8-v1 | 8192 | 1144 of 1154 | 185 | 625 | 185 | 0 |

Replay-predicted saving in new prefill when a 2,048 run is re-cut at Y=8,192, against the observed gap to the y8 run:
| 2,048 run | avoided resume boundaries | new prefill per resume start minus per in-invocation request | predicted saving (M) | observed: this run minus y8 (M) | share |
| --- | ---: | ---: | ---: | ---: | ---: |
| franzen-p448s14-v1 | 411 | 4431 | 1.82 | 3.69 | 49% |
| franzen-p448s14rep-v1 | 443 | 4958 | 2.20 | 3.85 | 57% |
| franzen-l1a1tbfc-v1 | 369 | 5154 | 1.90 | 1.51 | 126% |
p448s14 and p448s14rep lack the reply-cache fix that the y8 run has, so their observed gap includes that lineage difference; l1a1tbfc is the same lineage.
