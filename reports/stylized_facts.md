# GAFS stylized-facts report (test split)

Generated windows: 480 of length 30 steps.

| Asset | Metric | Real | Generated |
|---|---|---:|---:|
| EQ_LARGE | Annualised volatility | 0.3363 | 0.2573 |
| EQ_LARGE | Skewness | -1.0027 | 0.3875 |
| EQ_LARGE | Excess kurtosis (fat tails) | 7.0720 | 3.4319 |
| EQ_LARGE | Hill tail exponent | 2.4354 | 3.5903 |
| EQ_LARGE | Leverage corr(r_t, |r_t+l|) | -0.0307 | 0.0360 |
| EQ_LARGE | Mean abs ACF of returns | 0.0297 | 0.0260 |
| EQ_LARGE | ACF of abs returns (lags 1-5) | 0.0234 | -0.0269 |
| EQ_LARGE | Wasserstein distance | | 0.003952 |
| EQ_TECH | Annualised volatility | 0.3660 | 0.3079 |
| EQ_TECH | Skewness | -0.6852 | 0.0377 |
| EQ_TECH | Excess kurtosis (fat tails) | 3.9416 | 2.1997 |
| EQ_TECH | Hill tail exponent | 2.6386 | 3.9624 |
| EQ_TECH | Leverage corr(r_t, |r_t+l|) | -0.0304 | -0.0095 |
| EQ_TECH | Mean abs ACF of returns | 0.0242 | 0.0259 |
| EQ_TECH | ACF of abs returns (lags 1-5) | 0.0168 | -0.0264 |
| EQ_TECH | Wasserstein distance | | 0.002653 |
| CMD_GOLD | Annualised volatility | 0.3576 | 0.3462 |
| CMD_GOLD | Skewness | -0.1320 | -0.3170 |
| CMD_GOLD | Excess kurtosis (fat tails) | 0.8968 | 1.6754 |
| CMD_GOLD | Hill tail exponent | 6.0816 | 4.3040 |
| CMD_GOLD | Leverage corr(r_t, |r_t+l|) | -0.0041 | -0.0096 |
| CMD_GOLD | Mean abs ACF of returns | 0.0332 | 0.0247 |
| CMD_GOLD | ACF of abs returns (lags 1-5) | 0.0243 | -0.0304 |
| CMD_GOLD | Wasserstein distance | | 0.001327 |

Correlation structure: Frobenius distance 0.1726, mean absolute entry difference 0.0374.

Reading guide: generated values should sit near the real column. Excess kurtosis well above 0 and a slowly decaying ACF of absolute returns indicate fat tails and volatility clustering; a negative leverage correlation reproduces the equity leverage effect; the mean absolute ACF of raw returns should stay near zero (no free lunch).