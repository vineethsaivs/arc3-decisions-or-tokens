- franzen-p448s14-v1 (Y 2048): 1370 decision points, 560 yields; rows where x>=0 disagrees with the observed yield: 0
- franzen-p448s14rep-v1 (Y 2048): 1490 decision points, 630 yields; rows where x>=0 disagrees with the observed yield: 0
- franzen-l1a1tbfc-v1 (Y 2048): 1292 decision points, 567 yields; rows where x>=0 disagrees with the observed yield: 0
- franzen-l1a1y8-v1 (Y 8192): 1475 decision points, 185 yields; rows where x>=0 disagrees with the observed yield: 0

### Observational: share of requests that execute an action, by depth into the turn and by what preceded them

| depth (generated tokens in the turn before the request) | after tool result, Y=8,192 run | after resume opener, Y=8,192 run | after tool result, Y=2,048 runs | after resume opener, Y=2,048 runs |
| --- | ---: | ---: | ---: | ---: |
| 8K to 16K | 42/138 (30%) | 53/131 (40%) | 70/288 (24%) | 134/399 (34%) |
| 16K to 32K | 20/95 (21%) | 9/38 (24%) | 27/171 (16%) | 46/225 (20%) |
| 32K to inf | 10/50 (20%) | 2/16 (12%) | 11/76 (14%) | 14/124 (11%) |

(Here depth is measured after request j, i.e. the depth at which the decision was taken; the report's table bins the depth before the next request, which is the same quantity.)

### RD at the yield threshold, Y=2,048, three runs pooled

Density check, decision points per 200-token bin of x around the cutoff: [-800,-600) 239, [-600,-400) 211, [-400,-200) 214, [-200,0) 209, [0,200) 185, [200,400) 162, [400,600) 155, [600,800) 128

| bandwidth (tokens) | n (yield / continue) | P(next request acts): yield, continue | diff in means | local-linear tau (95% bootstrap CI) | tokens still needed after j: local-linear tau (95% CI) |
| ---: | --- | --- | ---: | --- | --- |
| 400 | 348 / 423 | 0.385, 0.345 | +0.040 | +0.113 (-0.027 to +0.224) | -2180 (-5252 to +957) |
| 700 | 569 / 745 | 0.376, 0.349 | +0.027 | +0.081 (-0.021 to +0.188) | -418 (-2775 to +1764) |
| 1200 | 814 / 1326 | 0.380, 0.360 | +0.019 | +0.046 (-0.032 to +0.121) | +284 (-1374 to +1978) |

### RD at the yield threshold, Y=8,192, one run

Density check, decision points per 200-token bin of x around the cutoff: [-800,-600) 10, [-600,-400) 18, [-400,-200) 18, [-200,0) 17, [0,200) 20, [200,400) 14, [400,600) 10, [600,800) 9

| bandwidth (tokens) | n (yield / continue) | P(next request acts): yield, continue | diff in means | local-linear tau (95% bootstrap CI) | tokens still needed after j: local-linear tau (95% CI) |
| ---: | --- | --- | ---: | --- | --- |
| 400 | 34 / 35 | 0.412, 0.143 | +0.269 | +0.396 (-0.026 to +0.863) | -1786 (-10221 to +5281) |
| 700 | 48 / 60 | 0.333, 0.183 | +0.150 | +0.392 (+0.140 to +0.668) | -7318 (-14694 to -286) |
| 1200 | 71 / 103 | 0.366, 0.204 | +0.162 | +0.196 (-0.023 to +0.424) | -3060 (-7688 to +1560) |

