**README**

This code package reproduces the simulation and analysis for the study.

**Files**

1. `01_darca_v24_integrated_agent_core_gravity_fixed.py`
   Core simulation code for DARCA_ONLY, FULL_INTEGRATED, rule baselines, and component controls.

2. `02_run_darca_n10_full_integrated_comparison.sh`
   Runs the fixed N10 main comparison.

3. `03_run_darca_density_sweep_N1_N3_N5_N10.sh`
   Runs the N1, N3, N5, and N10 density-sweep validation.

4. `04_darca_causal_analysis_from_density_sweep.py`
   Computes causal and component contrasts from the density-sweep outputs.

5. `05_darca_robustness_analysis_from_density_outputs.py`
   Computes robustness analyses across seeds, densities, scenarios, outlier-resistant estimators, and negative controls.

**Run order**

Run the scripts in numerical order.

```bash
bash 02_run_darca_n10_full_integrated_comparison.sh
bash 03_run_darca_density_sweep_N1_N3_N5_N10.sh
python3 -u 04_darca_causal_analysis_from_density_sweep.py
python3 -u 05_darca_robustness_analysis_from_density_outputs.py
```

**Environment**

The code was written for Python 3 and standard scientific Python packages. No language-model API or external online service is required for the simulations or analyses.

**Notes**

Figure-generation scripts are not included in this review package. The submitted code is limited to model execution, causal analysis, and robustness analysis.
