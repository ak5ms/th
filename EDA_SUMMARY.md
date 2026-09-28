# Training-only EDA

Executed in 266.6 seconds; 13 code cells; 10 figures.

Only the first 80% of the labeled time span is analyzed. Earlier notebook versions used the full data; this is a holdout for subsequent work, not a retrospectively untouched sample.

```text
TIME-SPAN SPLIT (msgStamp)
first_label               2014-01-02 06:05:00-05:00
cutoff                    2022-07-14 00:33:00-04:00
last_label                2024-08-30 16:55:00-04:00
train_rows                                   640734
test_rows                                    160325
train_fraction_of_time                          0.8
Training rows actually analyzed: 640,734; feature columns: 99
Training index: 2014-01-02 06:05:00-05:00 through 2022-07-14 00:30:00-04:00
No row subsampling in plots, shape screen, correlations or ACF.
Pass shape screen: 98/99
Usable Ljung-Box features: 99/99
Largest LB(15): 4578698.6 (x57)
Strong feature pairs (|rho| >= .8): 153
Cashflow/volume absolute value > 1: 551 rows
Zero-volume training rows: 6,471
Non-null flow diagnostic contributions: 584,302
Last cumulative flow diagnostic value: -802.578
Cashflow is likely signed trade flow or quote imbalance (hypothesis, not verified).
The cashflow curve is descriptive; return alignment/executability remain unverified.

Top contemporaneous price-return correlations
feature      n   pearson  spearman
    x61 168937  0.415557  0.350083
    x58 338832 -0.411100 -0.337529
    x43 168935  0.381231  0.313393
    x37 168823  0.351904  0.295867
    x46  99355  0.345192  0.282360
    x34 161112  0.332063  0.264749
    x10  16560  0.286760  0.328594
    x67   9126  0.321950  0.314194
    x52 167411  0.319535  0.285254
    x19 163843  0.315361  0.261706

Top supplied ret_5m correlations
feature      n   pearson  spearman
    x58 338777  0.018288  0.025212
     x7 346205 -0.001722 -0.023301
    x10  16560 -0.007924 -0.021978
    x40 446673 -0.001265 -0.021057
    x34 161114 -0.020514 -0.016435
    x55 480723 -0.000528 -0.020364
    x13 152348 -0.003123 -0.019904
    x16 152348 -0.002247 -0.019792
    x61 168885 -0.018851 -0.018883
    x35 161114 -0.018762 -0.007524

Largest Ljung-Box statistics
feature      n     acf1      lb_stat
    x57 480744 0.989220 4.578699e+06
    x42 446749 0.989065 4.307341e+06
    x66 374281 0.988780 3.650854e+06
    x78 338776 0.990617 3.395004e+06
     x6 329833 0.987226 3.214314e+06
     x3 333875 0.987453 3.203368e+06
    x60 338841 0.986322 3.135615e+06
     x9 346261 0.976988 2.935727e+06
    x56 480744 0.963029 2.612777e+06
    x41 446749 0.963445 2.431948e+06

Strong redundant pairs
feature1 feature2      corr  abs_corr  index_gap
     x49      x52  0.999996  0.999996          3
     x50      x53  0.999857  0.999857          3
     x51      x54  0.999642  0.999642          3
     x13      x16  0.997249  0.997249          3
      x7      x16  0.994874  0.994874          9
      x7      x13  0.994009  0.994009          6
     x49      x55  0.988759  0.988759          6
     x52      x55  0.988740  0.988740          3
     x58      x61 -0.987044  0.987044          3
     x81      x93  0.981349  0.981349         12

Largest monthly shifts (training only)
     max_abs_monthly_mean_shift_z  max_abs_monthly_log_std_ratio
x27                      1.194728                       1.544659
x24                      1.163821                       1.561496
x99                      1.127276                       1.652901
x96                      1.091659                       1.706036
x30                      1.075000                       2.899247
x33                      1.064781                       1.325685
x39                      1.005112                       1.154962
x63                      0.966948                       1.200168
x36                      0.896158                       1.049421
x29                      0.888844                       2.891253

```
