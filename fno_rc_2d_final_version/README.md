# FNO with CFT-based Residual Correction (FNO-RC) - Final 2D Version

This folder contains the final, successful version of the 2D Fourier Neural Operator model enhanced with a CFT-based Residual Correction mechanism (FNO-RC). This model demonstrated a **73.6% relative improvement** over the baseline FNO on the 2D Navier-Stokes benchmark.

## Files Included

-   `fourier_2d_cft_residual.py`: **(Main Model)** Defines the `FNO_RC` architecture.
-   `train_cft_residual_ns_2d.py`: **(Training Script)** Script to train the `FNO_RC` model.
-   `compare_final_models.py`: **(Comparison Script)** Script to compare the trained `FNO_RC` against the baseline FNO and generate error plots.
-   `fourier_2d_baseline.py`: The definition for the baseline `FNO2d` model, required by the comparison script.
-   `utilities3.py`, `chebyshev.py`, `Adam.py`: Core utility and library files required for the code to run.

## How to Use

### 1. Place Model Weights

-   Place your trained FNO-RC model file into the `models/` directory and name it `fno_rc.pt`.
-   For comparison, ensure your pre-trained baseline FNO model is accessible at the path specified in `compare_final_models.py` (default: `/content/drive/MyDrive/my_fno_models/fno_ns_2d_N600.pt`). You may need to adjust this path.

### 2. Train a New Model

To replicate the training process, navigate into this directory and run:

```bash
python train_cft_residual_ns_2d.py --data_path /path/to/your/ns_data.pt
```

The trained model will be saved as `models/fno_rc.pt`.

### 3. Run Comparison and Generate Plots

Once you have both the trained FNO-RC model and the baseline FNO model, run the comparison script:

```bash
python compare_final_models.py --data_path /path/to/your/ns_data.pt --fno_model_path /path/to/baseline/fno.pt
```

This will print the final performance comparison to the console and generate detailed error plots in a new `error_comparison_plots/` directory.

---
*This folder represents a self-contained, validated, and high-performance version of the 2D model.* 