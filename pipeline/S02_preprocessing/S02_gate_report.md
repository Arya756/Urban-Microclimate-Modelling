# Stage S02 Gate Report — UrbanTALES Profile Preprocessing

- **Total Catalog Cases:** 538
- **Total Cleaned Cases:** 538 (Idealized: 224, Realistic: 314)
- **Canonical Frame Shape:** (68864, 44) (68,864 rows, 44 columns)
- **Levels Per Case:** Exactly 128 vertical levels (0.25 m to 123.57 m)
- **Duplicate (case, z) Pairs:** 0
- **Momentum Identity Pass Rate (diff <= 1e-5):** 100.0% (68,864 / 68,864 points)
- **Km Diffusivity Pass Rate (diff <= 1e-4):** 100.0% (68,319 / 68,319 valid finite-gradient points)
- **Km Boundary Policy:** Expected `uGrad = NaN` and `Km = NaN` at the lowest level ($z = 0.25\text{ m}$) across all 538 cases due to unmeasured ground row ($z = 0.0\text{ m}$); exactly 7 interior levels with $|\text{uGrad}| \le 10^{-6}\text{ s}^{-1}$
- **Scope Note:** Processes 1D vertical profiles only; no 2D spatial resampling pipeline or 2% tolerance test is implemented in S02
- **Stage Gate Result:** PASSED
