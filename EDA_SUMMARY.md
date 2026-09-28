# EDA summary

Real-data run: 951,552 rows, 107 columns, 99 anonymized features.

- **Distribution shape:** 99/99 features pass the bell-shape screen. The screen tests unimodality (Hartigan dip), robust quartile skew, and point-mass/discreteness; kurtosis is intentionally ignored.
- **Same-bar return sanity check:** 45/99 features have |Pearson or Spearman correlation| >= 0.10 with `adjMid.pct_change()`. Strongest are x58 (Pearson -0.439, Spearman -0.351), x61 (+0.434, +0.365), x43 (+0.398, +0.326), x46 (+0.370, +0.287), and x37 (+0.367, +0.305). This establishes market-related structure, not forward predictability.
- **Feature redundancy:** 152 pairs have |Spearman rho| >= 0.80. The strongest are x49/x52 (0.999996), x50/x53 (0.999878), x51/x54 (0.999661), and x13/x16 (0.997614).
- **Naming pattern:** adjacent features are especially related (index-gap 1 median |rho| 0.821; 63% exceed 0.80). Multiples of three are also enriched: same-mod-3 pairs have mean |rho| 0.339 / median 0.354 versus 0.281 / 0.270 for other pairs. Gap-3 pairs have a 90th-percentile |rho| of 0.954. The numbering therefore contains clear grouping structure, but it is not simply an every-third-feature rule.
- **Autocorrelation:** 75/99 features reject no autocorrelation at Ljung-Box(15) after BH-FDR. Largest raw statistic is x42 at 620,899 (ACF1 0.990), followed by x78 524,856 and x6 515,979.
- **Longer-lag ACF:** 75/99 features have sufficient data. The largest post-15 peaks occur at lag 16, so this primarily reflects persistence extending past lag 15 rather than a distinct seasonal period.
- **Structural shifts:** x67, x68, x69 have the largest adjacent-month level shifts at 7.25, 6.93, and 5.96 full-sample standard deviations.
- **Cashflow vs named variables only:** essentially no material Spearman relationship: ret_5m -0.023, wmid +0.013, adjMid +0.013, volume +0.001.
