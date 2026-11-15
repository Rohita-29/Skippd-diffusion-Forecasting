# Probabilistic short-term solar PV forecasting from sky images

Code for my UROP project at NUS on 15-minute-ahead probabilistic forecasting of PV output from sky images, using the [SKIPP'D dataset](https://huggingface.co/datasets/solarbench/SKIPPD). A conditional diffusion model is compared with smart persistence and with Gaussian and quantile-regression CNN baselines.

The notebooks were run in Google Colab with a GPU runtime. Data is loaded from Hugging Face and results are saved to Google Drive under `MyDrive/UROP/`.

## Notebooks

Development:
- `01_data_audit_and_baselines`: data audit, timestamp check and correction, cleaning, persistence and CNN baselines
- `02_feature_study_and_ensembles`: feature study, deep ensemble, MC dropout
- `03_correction_cnn_diffusion_calibration`: correction CNN, conditional diffusion model, conformal calibration, reserve sizing
- `04_robustness_and_horizon_study`: adaptive calibration, clipping and timestamp ablations, forecast horizons, inference time

Controlled re-evaluation:
- `05_reeval_setup_and_cnn_baselines`: exact-horizon windows, train/select/calib split, evaluation protocol, MSE, Gaussian and quantile CNN baselines
- `05b_reeval_quantile_cnn_fix`: retraining of the quantile CNN after a bug in the first version
- `06_reeval_diffusion_image_ablation`: diffusion model with and without sky images
- `07_reeval_sampling_bounds`: clipping variants and cap sensitivity
- `08_reeval_saved_results_audit`: checks on the saved results
- `09_reeval_evaluation`: scores, calibration, bootstrap tests, reserve sizing and figures

## Running

- Development results: 01 → 02 → 03 → 04.
- Re-evaluation from scratch: 01 → 02 → 05 → 06 → 07 → 09. Notebook 05 first stops after a timing benchmark; set `STOP_AFTER_BENCHMARK = False` and run it again to train the baselines.
- 05b and 08 are only needed for results produced before the quantile fix and the added checks.

Notebook 01 writes `day1b_prep.npz` and notebook 02 writes `clock_fix.json`, which the later notebooks read. Notebooks 05 to 09 write to `MyDrive/UROP/review/`.

## Notes
- Output files keep the prefixes used during the project (`day1b_`, `feat_`, `day3_`, `day4_` for notebooks 01 to 04 and `p1_` to `p5_` for 05 to 09), and a few labels in saved results refer to the development notebooks by those names.
- Notebooks 05 to 09 include checks added after the reported runs (exact-horizon endpoint check, broad test-set mask, experiment ID, required seeds, non-negative interval widths). Notebook 08 confirmed that the saved results pass them.

## Data
Nie, Y., Li, X., Scott, A., Sun, Y., Venugopal, V. and Brandt, A. (2023). SKIPP'D: A SKy Images and Photovoltaic Power Generation Dataset for short-term solar forecasting. *Solar Energy*, 255, 171–179.
