**Table 3.** Extrapolation errors for GCRR-GC European calls ($S_0=100$, $T=0.5$, $r=0.06$, $\sigma=0.20$). Raw errors are $C_n - C_{BS}$ at the even count $n$ and $C_{n+1} - C_{BS}$ at the adjacent odd count. Same-phase Richardson uses $[(n+2)C_{n+2}-nC_n]/2$, while adjacent Richardson uses $(n+1)C_{n+1}-nC_n$. The phase-adjusted column applies adjacent Richardson after $C_n^{PA}=C_n-\eta\Delta A/n$, where $\Delta A=A_1-A_0$ is given by Corollary 3.3. All prices and extrapolations are evaluated using 40-digit arithmetic.

### Panel A: Out-of-the-Money (K=105)

| n | Raw even | Raw odd | Same phase | Adjacent | Adjacent PA |
|---:|---:|---:|---:|---:|---:|
| 50 | -2.196e-02 | +3.373e-02 | -3.652e-05 | +2.818e+00 | +2.845e-03 |
| 100 | -1.099e-02 | +1.701e-02 | -8.862e-06 | +2.817e+00 | +1.490e-03 |
| 500 | -2.199e-03 | +3.425e-03 | -3.452e-07 | +2.816e+00 | +3.090e-04 |
| 1000 | -1.100e-03 | +1.714e-03 | -8.600e-08 | +2.816e+00 | +1.552e-04 |
| 5000 | -2.200e-04 | +3.431e-04 | -3.430e-09 | +2.816e+00 | +3.115e-05 |

### Panel B: In-the-Money (K=95)

| n | Raw even | Raw odd | Same phase | Adjacent | Adjacent PA |
|---:|---:|---:|---:|---:|---:|
| 50 | -2.444e-02 | +2.092e-02 | -4.247e-05 | +2.289e+00 | -1.273e-03 |
| 100 | -1.223e-02 | +1.056e-02 | -1.034e-05 | +2.290e+00 | -5.807e-04 |
| 500 | -2.448e-03 | +2.128e-03 | -4.038e-07 | +2.290e+00 | -1.070e-04 |
| 1000 | -1.224e-03 | +1.065e-03 | -1.006e-07 | +2.290e+00 | -5.294e-05 |
| 5000 | -2.448e-04 | +2.132e-04 | -4.015e-09 | +2.290e+00 | -1.050e-05 |

