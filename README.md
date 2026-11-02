# Air-Quality-Forecasting-Model-Robustness-Test

## Project Overview
 
This project evaluates how a Random‑Forest‑based PM10‑forecasting model behaves when real‑world sensor‑data imperfections are introduced. It uses public hourly air‑quality measurements from the São Paulo ‑ São José do Rio Preto monitoring station to test prediction‑reliability under missing‑value and sudden‑spike‑outlier conditions, relevant for digital‑twin environmental monitoring workflows.
 
## Key Files
 
‑  robustness_analysis.py : Main Python script for data preparation, model training, robustness testing and output generation
‑  robustness_rmse_plot.png : Bar‑chart comparing baseline‑clean‑data RMSE against RMSE with introduced data‑quality issues
‑  project_summary.txt : Concise written overview of the project setup, results and practical implications
 
## Core Results
 
‑ Baseline clean‑data RMSE = 3.35 µg/m³
‑ RMSE with 5 % missing lag‑values = 4.02 µg/m³
‑ RMSE with 5 % positive‑spike outliers = 4.65 µg/m³
Outliers produced a larger accuracy drop than missing lag‑values in this test. The finding highlights the need for robustness screening before deploying similar forecasting models for urban‑digital‑twin monitoring use‑cases.
 
## Tools Used
 
Python | pandas | scikit‑learn | matplotlib
