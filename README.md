# Comparative Analysis of Advanced Machine Learning Models for Forecasting CO₂ Across Ecosystems and Land Covers Using ICOS Data Products

> **Preprint** – This repository contains the official code and data processing workflows supporting the scientific article:  
> _Comparative Analysis of Advanced Machine Learning Models for Forecasting CO₂ in Different Ecosystems and Land Covers using ICOS Data and Products_.

👥 Authors and Affiliations
This repository and the underlying research were developed by the authors of the preprint:

Pablo Catret Ruber<sup>a</sup>, David Garcia-Rodriguez<sup>a</sup>, Domingo J. Iglesias Fuente<sup>b</sup>,
Juan José Martínez Durá<sup>a</sup>, J. Javier Samper-Zapater<sup>a</sup>, Ernesto López-Baez<sup>c</sup>

<sup>a</sup>University Research Institute on Robotics and Information and Communication Technologies (IRTIC), Universitat de València, Paterna, 46980, Spain
<sup>b</sup>Valencian Institute of Agricultural Research (IVIA), Moncada, 46113, Spain
<sup>c</sup>Faculty of Physics, Universitat de València, Burjassot, 46100, Spain

---

## 📁 Repository Structure

```
├── data/                       # Raw ICOS data, metadata, and auxiliary station files
├── resultados/                 # CSV summaries: clusters, representatives, metrics, tuning logs
├── modelos_tuning/             # Best tuned models and prediction outputs (validation phase)
├── EntrenamientoModelos/       # Final trained models, test predictions, and evaluation metrics
├── graphs/                     # Final plots and graphs for paper or reports
├── SeleccionHiperparametros/   # Intermediate tuning results and validation forecasts
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
```

---

## 🧾 Key Scripts

| Script                         | Location | Description                                                                                  |
| ------------------------------ | -------- | -------------------------------------------------------------------------------------------- |
| `representatives_selection.py` | Root     | Clusters stations based on metadata and selects a representative per cluster                 |
| `hyperparameter_tuning.py`     | Root     | Performs random hyperparameter search for each forecasting model per representative station  |
| `train.py`                     | Root     | Trains models for all stations using best hyperparameters from their cluster representatives |
| `train_metalearner.py`         | Root     | Builds and evaluates a global meta-learner (stacking) using validation predictions           |
| `hybridization.py`             | Root     | Combines model predictions using hybrid and ensemble strategies; evaluates final performance |
| `statistical_tests.py`         | Root     | Performs statistical tests (e.g., Nemenyi test) and generates critical difference plots      |
| `analysis_LMP.py`              | Root     | Runs exploratory and comparative analyses of model performance across ecosystems             |
| `graphs.py`                    | Root     | Generates visualizations of model metrics and predictions for selected stations              |


---

## 🔍 Overview

This repository provides a comparative analysis of various advanced machine learning techniques for forecasting atmospheric CO₂ concentrations across diverse European ecosystems and land cover types. All data used are official ICOS (Integrated Carbon Observation System) products, processed and harmonized for machine learning pipelines.

---

## 📦 Installation & Dependencies

Before running any scripts or notebooks, ensure you have Python 3.9 or later installed. All required libraries can be installed using the provided `requirements.txt` file.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

---

## 🌍 Data

All the datasets used in this study are included in the `data/` folder. These datasets have been downloaded from the official [ICOS Data Portal](https://data.icos-cp.eu/portal), and each file is accompanied by its metadata file for full transparency and traceability.

> ⚠️ **Note**: Please ensure you respect the data usage policies of ICOS when reusing any part of this dataset.

---

## 🚧 How to Use This Repository

This section provides step-by-step instructions to reproduce the analysis. It will explain how to reproduce the analysis and results step-by-step, including model training, evaluation, and visualization.

---

## 📄 License and Citation

*This preprint is currently under peer review. Citation and licensing details will be updated once the final version is published.*

---

## 🧭 Step 1 – Station Clustering & Representative Selection

The first step in the analysis involves grouping ICOS stations into distinct clusters based on both **categorical** (e.g., land cover, climate zone) and **numerical** features (e.g., sampling height), and selecting **one representative station per cluster**. This ensures model generalizability and data diversity across ecosystems and climate conditions.

### 📂 Script: `representatives_selection.py`

This script performs the following tasks:

1. **Feature construction**: Merges station metadata and classification info.
2. **Mixed clustering**: Applies `KPrototypes` to group stations based on numerical and categorical variables.
3. **Data validity check**: Reconstructs daily CO₂ series for each station, filters out anomalies, and counts valid days.
4. **Representative selection**: Picks one station per cluster with sufficient data and minimal outlier influence.
5. **Result export**: Saves results into CSV files for subsequent modeling stages.

### 📥 Inputs required

Ensure the following files are available under the `data/` folder:

* `ClasificacionesEstacion.csv` – Station metadata and classifications
* `estaciones_alturas.txt` – List of station names and sampling heights
* `paths_datos_antiguos_2024.txt` and `paths_datos_nuevos_2024.txt` – Paths to CO₂ data files

### 📤 Outputs generated

* `resultados/representantes_por_cluster.csv` – Final list of selected representative stations
* `resultados/resumen_estaciones.csv` – Summary of stations with valid days and cluster assignments

### ▶️ How to run

You can execute the script from the project root:

```bash
python representatives_selection.py
```

> ℹ️ The script logs detailed progress into `mi_log.log` and prints the final selection in markdown format.

---

## ⚙️ Step 2 – Hyperparameter Tuning via Random Search

After selecting representative stations, this script performs randomized hyperparameter tuning on various time series models using daily CO₂ data. The goal is to optimize each model's configuration for accurate medium-term prediction (1-year horizon).

### 📂 Script: `hyperparameter_tuning.py`

This script carries out the following tasks:

1. **Loads and preprocesses** daily CO₂ data for each selected station.
2. **Splits the time series** into training, validation, and test segments (365 days each).
3. **Defines model-specific search spaces** for:

   * Classical methods: Exponential Smoothing, Prophet
   * Tree-based models: LightGBM, Random Forest
   * Deep learning models: N-HiTS, TCN, TiDE, GRU, LSTM
4. **Performs N iterations (default: 20)** of random search for each model and station.
5. **Tracks the best model per station/model pair** based on lowest validation MSE.
6. **Saves predictions and model parameters** for reproducibility and analysis.

### 🔧 Models included

| Model                | Type                | Library |
| -------------------- | ------------------- | ------- |
| ExponentialSmoothing | Classical           | `darts` |
| Prophet              | Classical           | `darts` |
| LightGBM             | Tree-based          | `darts` |
| RandomForest         | Tree-based          | `darts` |
| NHiTS                | Deep Learning       | `darts` |
| TCN                  | Deep Learning       | `darts` |
| TiDE                 | Deep Learning       | `darts` |
| GRU                  | RNN (Deep Learning) | `darts` |
| LSTM                 | RNN (Deep Learning) | `darts` |

### 📥 Inputs required

* `data/paths_datos_nuevos_2024.txt` and `data/paths_datos_antiguos_2024.txt` with file paths for the selected stations
* Output of Step 1: selected station names (`GAT 216.0`, `HPB 131.0`, etc.)

### 📤 Outputs Generated

* `SeleccionHiperparametros/Hiperparametros/`: per-station CSV log with best hyperparameters and validation MSE
* `SeleccionHiperparametros/Predicciones/`: CSV files with real vs predicted values on the validation set

### ▶️ How to run

To run the full hyperparameter tuning for the 7 representative stations:

```bash
python hyperparameter_tuning.py
```

You can also tune a specific subset of stations:

```bash
python hyperparameter_tuning.py "JFJ 13.9" "TRN 50.0"
```

> ⚠️ This process may take several hours depending on hardware. Each station × model combination runs 20 random trials.

---
## 🧠 Step 3 – Final Training & Evaluation

After selecting representative stations and tuning models on them, this script applies the **best hyperparameters from each cluster representative** to train forecasting models for all stations in the dataset.

### 📂 Script: `train.py`

This script automates the full training, evaluation, and model export workflow for each station.

### 🔍 What it does

1. **Maps each station to its cluster** using the summary file from Step 1.
2. **Loads the best hyperparameters** tuned for the cluster representative (from Step 2).
3. **Preprocesses hourly CO₂ data**:

   * Merges old/new files
   * Aggregates to daily mean
   * Removes long NaN gaps and outliers
   * Interpolates using PCHIP
4. **Trains all supported models** using Darts and evaluates them on a 1-year test set.
5. **Saves**:

   * Model object (`.pkl`)
   * Forecast vs. real values (`.csv`)
   * Evaluation metrics (MAE, RMSE, MAPE, etc.)

### 📥 Inputs required

* All CO₂ files listed in `data/paths_datos_antiguos_2024.txt` and `data/paths_datos_nuevos_2024.txt`
* Station list in `data/estaciones_alturas.txt`
* Cluster assignments from `resumen_estaciones.csv`
* Best hyperparameters from `SeleccionHiperparametros/Hiperparametros/<STA>/tuning_log.csv` being <STA> the station abbreviation of the asociated representant.

### 📤 Outputs generated

All outputs are stored under the `EntrenamientoModelos/` directory:

* `modelos/`: Trained models (one per station and model)
* `predicciones/`: 1-year forecast vs. real values for test set
* `metricas/`: Evaluation metrics (per model and station), including an aggregated file `ALL_METRICS.csv`

### ▶️ How to run

To train and evaluate all stations in the dataset:

```bash
python train.py
```

To run the pipeline on specific stations only:

```bash
python train.py "TRN_50.0" "OXK 23.0"
```

> Use underscores or spaces interchangeably in station names.

---

## 🧩 Step 4 – Global Meta-Learner via Stacking

To further improve forecast accuracy, this step builds a **global meta-model** that learns to combine the predictions from multiple base models across stations using supervised ensembling (stacking). Three candidate algorithms are compared using cross-validation.

### 📂 Script: `train_metalearner.py`

This script creates a unified tabular dataset from the individual station-level validation predictions (Step 2) and trains a meta-learner.

### 🧠 What it does

1. **Collects all `*_VAL.csv` files** containing real vs. predicted validation values.
2. **Merges predictions by model** for each station (e.g., `pred_LSTM`, `pred_ETS`, ...).
3. **Concatenates all stations** into a global dataframe.
4. **Adds simple temporal features** (`month`, `dayofyear`, etc.).
5. **Trains and compares meta-models** via 5-fold cross-validation:

   * `LinearRegression`
   * `LightGBMRegressor` (with randomized hyperparameter search)
   * `MLPRegressor` (with randomized hyperparameter search)
6. **Selects the best model** based on RMSE and saves it with metadata.

### 🔍 Features used

Two feature configurations are evaluated:

* Only base model predictions (`pred_*`)
* Base model predictions + `month` (basic seasonality proxy)

> 💡 More temporal/contextual features can be added to improve meta-modeling.

### 📥 Inputs required

* Validation predictions: `SeleccionHiperparametros/Predicciones/*_VAL.csv` (from Step 2)

### 📤 Outputs generated

* `meta_global_<Model>.pkl`: best meta-model saved with feature names and config
* `resultados_stacking_global.csv`: full table of RMSE and MAPE per model and feature set

### ▶️ How to run

From the root directory:

```bash
python train_metalearner.py
```

After training, you’ll see output like:

```text
Meta-modelo global guardado en: meta_global_LightGBM.pkl
Métricas detalladas → resultados_stacking_global.csv
```

---

## 🔀 Step 5 – Hybrid Models & Final Evaluation

To wrap up the modeling pipeline, this script generates and evaluates **hybrid models** and compares them against the **global meta-learner** and **simple averaging**. It consolidates final metrics and forecasts for each station.

### 📂 Script: `hybridization.py`

This script loads station-level predictions from the test set and applies several ensemble strategies.

### 🔍 What it does

1. **Loads the trained meta-model** from `meta_global_<model>.pkl`.
2. **Scans the test predictions** from Step 3 for each station and base model.
3. **Creates hybrid models** by averaging predictions:

   * `ProHiTS` = Prophet + N-HiTS
   * `LGBProphet` = LightGBM + Prophet
   * `ProphetTCN` = Prophet + TCN
4. **Adds a simple ensemble baseline**:

   * `MediaSimple` = mean of all available base model predictions
5. **Applies the meta-model** (if required features are present).
6. **Saves predictions and evaluation metrics** to the appropriate folders.

### 📥 Inputs required

* Test set predictions: `EntrenamientoModelos/predicciones/*_TEST.csv` (from `train.py`)
* Trained meta-model: `meta_global_<model>.pkl` (from `train_metalearner.py`)

### 📤 Outputs generated

Stored inside `EntrenamientoModelos/`:

* `predicciones/`: Test forecasts for:

  * Hybrid models: `*_ProHiTS_TEST.csv`, `*_LGBProphet_TEST.csv`, etc.
  * `*_MediaSimple_TEST.csv`
  * `*_Meta_TEST.csv`
* `metricas/`: Performance files with RMSE, MAE, sMAPE, MAPE, RMSLE for all the above

### ▶️ How to run

Once Steps 3 and 4 have been completed, simply execute:

```bash
python hybridization.py
```

Output will confirm which ensemble strategies were saved per station, and highlight any missing components.

---

## 📊 Step 6 – Visual Analysis of Forecast Performance and Seasonality

This script generates a comprehensive set of **visualizations** to assess forecast accuracy, error distributions, seasonal CO₂ cycles, and performance variability across ecosystems, climate zones, and geophysical attributes.

### 📂 Script: `graphs.py`

This script executes a full visualization pipeline, producing both **station-specific** and **aggregated** plots.

### 🧠 What it does

1. **Forecast vs Actual (Test Set Only)**

   * Loads prediction files from `EntrenamientoModelos/predicciones/`
   * Plots actual vs. predicted CO₂ values for all representative stations using the best trained model
   * Output:
     📈 `graphs/forecast_vs_actual_all_stations_test_only.png`

2. **Boxplot of Prediction Errors**

   * Shows the distribution of prediction residuals (error = prediction − actual) per station
   * Output:
     📊 `graphs/boxplot_prediction_errors.png`

3. **Seasonal Cycle Visualization**

   * Reconstructs raw daily CO₂ values from old and new ICOS data
   * Aggregates by day-of-year and smooths using a 7-day rolling average
   * Plots seasonal curves per station, annotated by land cover type
   * Output:
     🌱 `graphs/stations_co2_mean_by_day_ordered.jpg`

4. **Radar Plots of 1−MAPE (%)**

   * Loads all `*_METRICS.csv` files from `EntrenamientoModelos/metricas/`
   * Aggregates median model performance (1−MAPE) across:

     * ESA land cover types
     * Köppen climate zones
     * Altitude ranges
     * Latitude bands
     * Combined ESA + Köppen classification
   * Outputs:

     * `graphs/radar_esa_koppen.png`
     * `graphs/radar_alt_lat.png`
     * `graphs/radar_koppen_esa_combo.png`

5. **Violin Plot of MAPE Distributions**

   * Plots full distribution of MAPE scores per model using violin plots
   * Output:
     🎻 `graphs/violin_mape.png`

### 📥 Inputs required

* `EntrenamientoModelos/predicciones/*_TEST.csv` – Test set predictions
* `EntrenamientoModelos/metricas/*_METRICS.csv` – Evaluation metrics
* `data/ClasificacionesEstacion.csv` – Station metadata
* `data/estaciones_alturas.txt` – Station names
* `data/paths_datos_antiguos_2024.txt` and `paths_datos_nuevos_2024.txt` – Raw ICOS files

### 📤 Outputs generated

All plots are saved to the `graphs/` folder. These plots are directly usable for publication or reporting.

### ▶️ How to run

To generate all figures:

```bash
python graphs.py
```

> ℹ️ If any required input is missing, the script will log warnings and skip the corresponding visualizations.

---

## 📈 Step 7 – Statistical Comparison of Forecasting Models

This script performs a rigorous **statistical evaluation** of the forecasting models using robust summaries, bootstrapped confidence intervals, pairwise significance testing, and **Critical Difference (CD) diagrams** to assess and compare performance across models and ecosystem groups.

### 📂 Script: `statistical_tests.py`

This script produces comprehensive statistical comparisons and interpretable visual diagnostics.

### 🧠 What it does

1. **Robust Summary Statistics**

   * Computes robust indicators (median, IQR, MAD, trimmed mean) for each model and grouping variable:

     * ESA, Köppen, altitude bin, latitude bin, and their combination
   * Output: `AnalisisAgrupaciones/robust_*.csv`

2. **Bootstrap Confidence Intervals**

   * Generates 95% CI for model MAPE percentiles (p25, median, p75)
   * Output: `AnalisisTests/bootstrap_IC_valores.csv` and:

     * `IC_p25_MAPE.png`
     * `IC_median_MAPE.png`
     * `IC_p75_MAPE.png`

3. **Pairwise Superiority Testing**

   * Performs bootstrap tests for all model pairs to determine if model A significantly outperforms B
   * Output: binary heatmaps:

     * `super_MAPE_median.png`, `super_RMSE_p25.png`, etc.

4. **Nemenyi Post-Hoc Tests**

   * Applies **Nemenyi’s test** to identify statistically significant differences after ranking models
   * Output: significance heatmaps (`CD_<metric>_<quantile>_alt.png`)

5. **Critical Difference (CD) Diagrams**

   * Visualizes model average ranks and statistically indistinguishable groups (Demšar-style diagrams)
   * Output: `CD_<metric>_<quantile>.png`, e.g.:

     * `CD_MAPE_median.png`
     * `CD_MSE_p75.png`

6. **Group-Wise Comparison: Hybrid vs Traditional**

   * Evaluates where hybrid models (ensembles/meta-models) outperform traditional ones
   * Groups: ESA, Köppen, latitude, and elevation bins
   * Output:

     * Summary tables: `AnalisisWins/diff_MAPE_<group>.csv`
     * Final visualization: `AnalisisWins/grid_heatmap_delta_MAPE.png`
       showing ∆MAPE (Best Traditional – Best Hybrid)

### 📥 Inputs required

* Model metrics: `EntrenamientoModelos/metricas/*_METRICS.csv`
* Station metadata: `data/ClasificacionesEstacion.csv`

### 📤 Outputs generated

* `AnalisisAgrupaciones/`: Group-wise robust summaries
* `AnalisisTests/`: Bootstrap confidence intervals, pairwise tests, CD diagrams
* `AnalisisWins/`: Hybrid vs. traditional comparison tables and heatmap

### ▶️ How to run

From the root directory:

```bash
python statistical_tests.py
```

> ℹ️ The script will automatically generate all plots and summary files in their corresponding folders. Some analyses require a minimum number of models per station, and warnings will be shown if data is insufficient.

---

## 🧪 Step 8 – Drift Detection for Station LMP 8.0

This final step implements a **drift analysis** on the CO₂ series from the station **LMP 8.0**, investigating whether the data distribution has changed between historical and test periods. It combines visual inspection and statistical testing to assess potential distribution shifts.

### 📂 Script: `analysis_LMP.py`

This script isolates LMP 8.0 from the full dataset and applies a set of **distributional comparisons** and **visualizations**.

### 🧠 What it does

1. **Loads and filters** the hourly CO₂ data for LMP 8.0 from the raw files listed in `data/paths_datos_antiguos_2024.txt` and `paths_datos_nuevos_2024.txt`
2. **Interpolates** daily values using IQR filtering and PCHIP
3. **Splits the data** into consecutive 365-day blocks, with the last one treated as the test set
4. **Plots the CO₂ distributions** (histogram + KDE overlay) for each yearly pack, highlighting the test year in red
5. **Runs statistical tests** to detect significant drift:

   * Welch’s t-test
   * Kolmogorov–Smirnov test
   * Mann–Whitney U test
6. **Overlays time series** of all yearly blocks for direct visual comparison
7. **Exports results** and test outputs to `analysisLMP/`

### 📤 Outputs generated

All outputs are stored under the `analysisLMP/` directory:

* `LMP_hist_kde_overlay.png` – KDE and histogram overlay for all 365-day blocks
* `LMP_train_test_timeseries.png` – Daily CO₂ time series, with training vs. test contrast
* `LMP_drift_tests.csv` – Table with test statistics, p-values, and significance flag

### 📥 Inputs required

* `data/paths_datos_antiguos_2024.txt`
* `data/paths_datos_nuevos_2024.txt`
* `data/estaciones_alturas.txt`

### ▶️ How to run

To perform the drift analysis for station LMP 8.0:

```bash
python analysis_LMP.py
```

> 🔎 This script is tailored for LMP 8.0 but can be adapted to other stations by modifying the `STATION_NAME` variable.

---



