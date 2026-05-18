# Policy Terminal

**Live app:** policy-terminal.streamlit.app

A VAR-based macroeconomic shock simulator examining monetary policy 
transmission across India, Singapore, and the United Kingdom.

## What it does

- FX shock simulation using a VAR(2) model estimated on monthly CPI 
  data (Jan 2012 – Sep 2025)
- Markov-switching regime detection identifying High Growth, 
  Stagflation, and Depression macroeconomic states
- Rolling 12-month out-of-sample forecast with 95% confidence intervals
- Scenario stress-testing across three growth regimes

## Research basis

Built on the findings of:

> Gokhale, A. (2026). "Cross-Country Macroeconomic Dynamics: Inflation, 
> Growth, and Monetary Policy — India, Singapore, and the United Kingdom." 
> SSRN Working Paper. ssrn.com/abstract=6514338

Key finding: India's CPI Granger-causes Singapore's with a two-month 
lag (p = 0.028), and Singapore's leads the UK's (p = 0.039). This 
tool operationalises that transmission structure as a live simulator.

## Methods

- VAR(2) estimation: Statsmodels
- Markov-switching: Statsmodels MarkovRegression
- Rolling forecast with Diebold-Mariano accuracy testing
- Data: MAS Statistics, RBI DBIE, UK ONS

## Stack

Python · Statsmodels · Pandas · Matplotlib · Streamlit

## Author

Anuja A. Gokhale — anujagokhale.github.io
MSc Applied Economics, National University of Singapore (Merit Scholar)
