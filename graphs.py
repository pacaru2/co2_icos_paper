# -*- coding: utf-8 -*-
# =============================================================================
# Title       : Visualization of CO₂ Series for ICOS-Atmosphere
# Author      : Pablo Catret
# =============================================================================

# =============================================================================
# Import libraries
# =============================================================================

from pathlib import Path
from collections import defaultdict
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# Paths and directories
# =============================================================================
BASE_DIR   = Path('.')
PRED_DIR   = BASE_DIR / 'EntrenamientoModelos' / 'predicciones'
GRAPH_DIR  = BASE_DIR / 'graphs'
DATA_DIR   = BASE_DIR / 'data'
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = BASE_DIR / 'EntrenamientoModelos' / 'metricas'
CLASSIF_FILE = DATA_DIR / 'ClasificacionesEstacion.csv'

# Paleta amplia (se amplía si hay más modelos)
MODEL_COLORS = [
    "#FF6F00", "#FFD700", "#C6FF00", "#4CAF50", "#1DE9B6", "#00BFFF",
    "#4361EE", "#283593", "#9B59B6", "#E91E63", "#FF4081", "#E63946",
    "#8D6E63", "#26A69A", "#AD1457", "#5E35B1", "#42A5F5", "#7CB342"
]

# =============================================================================
# Station order and plot colors
# =============================================================================
ORDERED_STATIONS = [
    "LMP 8.0", "SAC 60.0", "OPE 50.0", "HPB 50.0", "PAL 12.0",
    "JFJ 13.9", "JUE 50.0", "PUI 47.0", "BIR 50.0",
]
PALETTE = [
    "#FF6F00", "#FFD700", "#C6FF00", "#4CAF50", "#1DE9B6",
    "#00BFFF", "#4361EE", "#283593", "#9B59B6",
]
# --- bins para altitud y latitud --------------------------------------------
ALT_BINS    = [0, 200, 500, 1000, 2000, float('inf')]
ALT_LABELS  = ['0-200', '200-500', '500-1000', '1000-2000', '>2000']
LAT_BINS    = [34, 43, 50, 57, 64, float('inf')]
LAT_LABELS  = ['34-43°N', '43-50°N', '50-57°N', '57-64°N', '>64°N']
# =============================================================================
# Expected trained model per station
# =============================================================================
EXPECTED_MODEL = {
    "LMP 8.0" : ["ProHiTS"],
    "SAC 60.0": ["ProHiTS"],
    "OPE 50.0": ["MediaSimple"],
    "HPB 50.0": ["LSTM"],
    "PAL 12.0": ["ProHiTS"],
    "JFJ 13.9": ["ETS"],
    "JUE 50.0": ["MediaSimple"],
    "PUI 47.0": ["ProHiTS"],
    "BIR 50.0": ["Prophet"],
}

# =============================================================================
# Land cover information (abbreviated codes)
# =============================================================================
abbreviation_to_landcover = {
    'BIR': 'Tree Cover (Evergreen)', 'CBW': 'Grassland',    'CMN': 'Grassland',
    'GAT': 'Tree Cover (Evergreen)', 'HEL': 'Grassland',    'HPB': 'Grassland',
    'HTM': 'Tree Cover (Evergreen)', 'IPR': 'Tree Cover (Mixed)',
    'JFJ': 'Snow',                   'JUE': 'Tree Cover (Deciduous)',
    'KIT': 'Tree Cover (Evergreen)', 'KRE': 'Cropland',     'LIN': 'Grassland',
    'LMP': 'Bare',                   'LUT': 'Cropland',     'NOR': 'Tree Cover (Evergreen)',
    'OPE': 'Cropland',               'OXK': 'Tree Cover (Evergreen)',
    'PAL': 'Moss and Lichen',        'PRS': 'Snow',         'PUI': 'Tree Cover (Mixed)',
    'PUY': 'Grassland',              'RGL': 'Cropland',     'SAC': 'Built-up',
    'SMR': 'Tree Cover (Evergreen)', 'SNO': 'Snow',         'SSL': 'Tree Cover (Evergreen)',
    'STE': 'Tree Cover (Evergreen)', 'SVB': 'Tree Cover (Evergreen)',
    'TOH': 'Tree Cover (Evergreen)', 'TRN': 'Tree Cover (Mixed)',
    'UTO': 'Grassland',              'WAO': 'Grassland',    'WES': 'Tree Cover (Evergreen)',
    'ZEP': 'Snow',                   'ZSF': 'Snow'
}

# =============================================================================
# Utility functions
# =============================================================================

def station_and_model(filename: str):
    """Extrae (estación, modelo) de un nombre de CSV ``<est>_<mod>_TEST.csv``."""
    stem  = Path(filename).stem
    parts = stem.split('_')
    if len(parts) < 3:
        raise ValueError(f"Nombre malformado: {filename}")
    station = ' '.join(parts[:-2])  # re-insertar espacios
    model   = parts[-2]
    return station, model

# =============================================================================
# Load prediction CSVs
# =============================================================================

def load_prediction_data():
    if not PRED_DIR.exists():
        logging.error(f"Carpeta {PRED_DIR} inexistente")
        return {}, {}, {}

    real_dict, pred_dict, err_dict = defaultdict(list), defaultdict(list), defaultdict(list)

    for csv_path in sorted(PRED_DIR.glob('*.csv')):
        station, model = station_and_model(csv_path.name)
        if station not in EXPECTED_MODEL:
            continue  # estación ajena
        if model not in EXPECTED_MODEL[station]:
            continue  # modelo no entrenado originalmente
        if station in real_dict:
            logging.warning(f"Varias predicciones para {station}; se ignora {csv_path.name}")
            continue

        df = pd.read_csv(csv_path, parse_dates=['time'])
        if not {'real', 'pred'}.issubset(df.columns):
            logging.error(f"Columnas faltantes en {csv_path.name}")
            continue

        r, p = df['real'].to_numpy(), df['pred'].to_numpy()
        real_dict[station] = r
        pred_dict[station] = p
        err_dict[station]  = p - r
        logging.info(f"Predicción cargada: {station} · {model} · N={len(r)}")

    for st in ORDERED_STATIONS:
        if st not in real_dict:
            logging.warning(f"Sin CSV de predicción para {st} → no aparecerá en el gráfico")
    return real_dict, pred_dict, err_dict

# =============================================================================
# Plot 1 & 2: Forecast vs Actual + Boxplot of prediction errors
# =============================================================================

def plot_predictions(real_dict, pred_dict, err_dict):
    plt.figure(figsize=(20, 15))
    letters = [chr(ord('a') + i) for i in range(len(ORDERED_STATIONS))]
    sub = 1
    for idx, station in enumerate(ORDERED_STATIONS):
        if station not in real_dict:
            continue
        plt.subplot(3, 3, sub); sub += 1
        x = np.arange(1, len(real_dict[station]) + 1)
        plt.plot(x, real_dict[station], label='Actual', color='k', lw=1.5, alpha=0.8)
        plt.plot(x, pred_dict[station], label='Forecast', color=PALETTE[idx], lw=2.5)
        plt.title(f"{letters[idx]}) {station}", fontsize=20)
        plt.xticks(np.arange(1, len(x)+1, 73).tolist() + [len(x)])
        plt.xlim((1, len(x)))
        plt.ylim((400, 460))
        plt.grid(True, ls='--', lw=0.7)
        plt.tick_params(axis='both', labelsize=16)
        if idx >= 6:
            plt.xlabel('Time Step (days)', fontsize=20)
        if idx % 3 == 0:
            plt.ylabel('CO₂ Concentration (ppm)', fontsize=20)
        plt.legend(fontsize=18)
        plt.gca().set_facecolor('white')
        sns.despine()
    plt.tight_layout()
    out1 = GRAPH_DIR / 'forecast_vs_actual_all_stations_test_only.png'
    plt.savefig(out1, dpi=300, bbox_inches='tight'); plt.close()
    logging.info(f"Gráfico guardado → {out1.relative_to(BASE_DIR)}")

    err_data  = [err_dict[s] for s in ORDERED_STATIONS if s in err_dict]
    err_labels = [s for s in ORDERED_STATIONS if s in err_dict]
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=err_data, palette=PALETTE[:len(err_data)])
    plt.axhline(0, color='red', ls='--', lw=1)
    plt.xticks(range(len(err_labels)), err_labels, rotation=45, fontsize=20)
    plt.xlabel('Station', fontsize=24)
    plt.ylabel('Error (Pred – Actual) (ppm)', fontsize=24)
    plt.ylim((-30, 30)); plt.yticks(fontsize=20)
    plt.grid(True, ls='--', lw=0.7, which='both'); plt.tight_layout()
    out2 = GRAPH_DIR / 'boxplot_prediction_errors.png'
    plt.savefig(out2); plt.close()
    logging.info(f"Gráfico guardado → {out2.relative_to(BASE_DIR)}")

# =============================================================================
# Load and process raw data for seasonal cycle
# =============================================================================

def process_station_raw(nom_est, path_old, path_new):
    """Devuelve DataFrame con media de CO₂ por Day-of-Year."""
    cols = ['Site', 'SamplingHeight', 'Year', 'Month', 'Day', 'Hour', 'Minute',
            'DecimalDate', 'co2', 'Stdev', 'NbPoints', 'Flag', 'InstrumentId',
            'QualityId', 'LTR', 'CMR', 'STTB', 'QcBias', 'QcBiasUncertainty',
            'co2-WithoutSpikes', 'Stdev-WithoutSpikes', 'NbPoints-WithoutSpikes']
    try:
        df_old = pd.read_csv(DATA_DIR / path_old, sep=';', comment='#'); df_old.columns = cols[:len(df_old.columns)]
        df_new = pd.read_csv(DATA_DIR / path_new, sep=';', comment='#'); df_new.columns = cols[:len(df_new.columns)]
        df = pd.concat([df_old, df_new], ignore_index=True)
    except FileNotFoundError:
        logging.error(f"Fichero no encontrado para {nom_est}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error leyendo {nom_est}: {e}")
        return pd.DataFrame()

    df.replace([-999.99, -9.99], np.nan, inplace=True)
    df['TIMESTAMP'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')
    df.set_index('TIMESTAMP', inplace=True)
    df_daily = df[['co2']].resample('D').mean()
    df_daily['DayOfYear'] = df_daily.index.dayofyear
    df_mean_doy = df_daily.groupby('DayOfYear')['co2'].mean().reset_index()
    df_mean_doy['Station'] = nom_est[:3]
    df_mean_doy['Height']  = nom_est
    return df_mean_doy

# =============================================================================
# Plot seasonal cycle by station
# =============================================================================

def plot_seasonal_cycle():
    paths_file = DATA_DIR / 'paths_datos_nuevos_2024.txt'
    if not paths_file.exists():
        logging.warning("No se encuentran los ficheros de rutas de datos crudos; se omite ciclo estacional")
        return

    # Leer listas de nombres y rutas
    with open(DATA_DIR / 'estaciones_alturas.txt') as f1, \
         open(DATA_DIR / 'paths_datos_nuevos_2024.txt') as f2, \
         open(DATA_DIR / 'paths_datos_antiguos_2024.txt') as f3:
        nombres   = [l.strip() for l in f1]
        nuevos    = [l.strip() for l in f2]
        antiguos  = [l.strip() for l in f3]

    tasks = [(n, a, b) for n, a, b in zip(nombres, antiguos, nuevos) if n in ORDERED_STATIONS]
    if not tasks:
        logging.warning("Ninguna estación válida para el ciclo estacional")
        return

    all_df = pd.concat([process_station_raw(n, a, b) for n, a, b in tasks], ignore_index=True)
    if all_df.empty:
        logging.warning("Sin datos para graficar ciclo estacional")
        return

    plt.style.use('default')
    plt.figure(figsize=(20, 15))
    letters = [chr(ord('a') + i) for i in range(len(ORDERED_STATIONS))]
    sub = 1
    for idx, station in enumerate(ORDERED_STATIONS):
        df_st = all_df[all_df['Height'] == station]
        if df_st.empty:
            continue
        plt.subplot(3, 3, sub); sub += 1
        df_st = df_st.sort_values('DayOfYear')
        smooth = df_st['co2'].rolling(window=7, min_periods=1).mean()
        landcover = abbreviation_to_landcover.get(station[:3], 'Unknown')
        plt.plot(df_st['DayOfYear'], smooth, color=PALETTE[idx], lw=3.5, label='CO₂')
        plt.plot(df_st['DayOfYear'], smooth, color='k', lw=3.5, alpha=0.1)
        plt.title(f"{letters[idx]}) {station} ({landcover})", fontsize=20)
        plt.xticks(np.arange(1, 366, 73).tolist() + [365])
        plt.xlim((1, 365)); plt.ylim((405, 445))
        plt.grid(True, ls='--', lw=0.7, which='both')
        plt.tick_params(axis='both', labelsize=16)
        if idx >= 6:
            plt.xlabel('Day of Year', fontsize=20)
        if idx % 3 == 0:
            plt.ylabel('CO₂ Concentration (ppm)', fontsize=20)
        plt.gca().set_facecolor('white')
        sns.despine()
    plt.tight_layout()
    out3 = GRAPH_DIR / 'stations_co2_mean_by_day_ordered.jpg'
    plt.savefig(out3, dpi=300, bbox_inches='tight'); plt.close()
    logging.info(f"Gráfico guardado → {out3.relative_to(BASE_DIR)}")




# -----------------------------------------------------------------------------
def _load_all_metrics() -> pd.DataFrame:
    """Devuelve DF con todas las métricas + clasificación de la estación."""
    if not METRICS_DIR.exists():
        logging.error(f'Carpeta métricas inexistente: {METRICS_DIR}')
        return pd.DataFrame()

    # --- leer métricas -------------------------------------------------------
    frames = []
    for csv in METRICS_DIR.glob('*_METRICS.csv'):
        try:
            frames.append(pd.read_csv(csv, dtype=str))
        except Exception as e:
            logging.warning(f'No se pudo leer {csv.name}: {e}')
    if not frames:
        logging.error('No se cargó ningún CSV de métricas')
        return pd.DataFrame()

    dfm = pd.concat(frames, ignore_index=True)
    # convertir numéricos
    num_cols = ['MAE', 'MSE', 'RMSE', 'MAPE', 'sMAPE', 'RMSLE']
    dfm[num_cols] = dfm[num_cols].apply(pd.to_numeric, errors='coerce')

    # renombrar modelo MediaSimple → Consensus
    dfm['Model'].replace({'MediaSimple': 'Consensus'}, inplace=True)

    # --- leer clasificaciones -----------------------------------------------
    if not CLASSIF_FILE.exists():
        logging.error(f'Falta fichero de clasificaciones: {CLASSIF_FILE}')
        return pd.DataFrame()

    # --- leer clasificaciones -----------------------------------------------
    dfc = pd.read_csv(CLASSIF_FILE, sep=';', decimal=',')
    dfc.rename(columns=str.strip, inplace=True)

    # eliminar posibles filas repetidas por abreviatura
    dfc = dfc.drop_duplicates(subset='Abbreviation', keep='first')

    # asegurar numéricos
    dfc['Level']   = pd.to_numeric(dfc['Level'], errors='coerce')
    dfc['Latitud'] = pd.to_numeric(dfc['Latitud'],
                                    errors='coerce')

    # clave de unión (primeras 3 letras)
    dfm['Abbreviation'] = dfm['Station'].str[:3]

    # fusión muchos-a-uno (cada Abbreviation única en dfc)
    df = dfm.merge(dfc, on='Abbreviation', how='left', validate='m:1')


    # --- grupos de altitud / latitud ----------------------------------------
    df['level_group']    = pd.cut(df['Level'], ALT_BINS, labels=ALT_LABELS, right=False)
    df['latitude_group'] = pd.cut(df['Latitud'], LAT_BINS, labels=LAT_LABELS, right=False)

    # combinación Köppen + ESA (con salto de línea)
    df['Koppen_ESA'] = df['Koppen'].str.strip() + '\n' + df['ESA'].str.strip()

    return df


# -----------------------------------------------------------------------------
def _make_pivots(df: pd.DataFrame):
    """Calcula medianas MAPE y genera pivots 1-MAPE."""
    pivots = {}
    # lista de (nombre_col, nombre_fig)
    cols = ['ESA', 'Koppen', 'latitude_group', 'level_group', 'Koppen_ESA']
    for col in cols:
        med = (df.groupby(['Model', col])['MAPE'].median().reset_index())
        med['1-MAPE'] = 100 - med['MAPE']
        pivots[col] = med.pivot(index='Model', columns=col, values='1-MAPE').reset_index()
    return pivots


# -----------------------------------------------------------------------------
def _radar_from_pivot(ax, df_pivot, colors):
    """Dibuja un radar plot en ax a partir de df_pivot (Model + categorías)."""
    import numpy as np

    cats = list(df_pivot.columns[1:])
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    for ang in angles[:-1]:
        ax.plot([ang, ang], [97, 100], color='black', lw=3, alpha=0.5)

    handles, labels = [], []
    for i, row in df_pivot.iterrows():
        vals = row.drop('Model').values.astype(float).tolist()
        vals += vals[:1]
        color = colors[i % len(colors)]
        line, = ax.plot(angles, vals, lw=5, label=row['Model'], color=color)
        ax.fill(angles, vals, color=color, alpha=0.1)
        handles.append(line); labels.append(row['Model'])

    ax.set_ylim(97, 100)
    ax.spines['polar'].set_visible(False)
    radial_ticks = [97.5, 98, 98.5, 99, 99.5, 100]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f'{t:.1f}%' for t in radial_ticks], size=28, fontweight='bold')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, size=36, fontweight='bold')
    for tick in ax.get_xticklabels():
        tick.set_y(tick.get_position()[1] - 0.14)

    return handles, labels


def _plot_all_radars(pivots):
    """Genera y guarda las 3 figuras de radar."""
    colors = MODEL_COLORS

    # --- (ESA + Köppen) ------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(32, 24), subplot_kw=dict(polar=True))
    handles, labels = _radar_from_pivot(axs[0], pivots['ESA'], colors)
    _radar_from_pivot(axs[1], pivots['Koppen'], colors)
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.125), ncol=len(labels)/2, prop={'size': 40, 'weight': 'bold'})
    plt.tight_layout(rect=[0, 0.00, 1, 1]); plt.subplots_adjust(wspace=0.5)
    plt.savefig(GRAPH_DIR / 'radar_esa_koppen.png', dpi=500, bbox_inches='tight')
    plt.close()

    # --- (Altitud + Latitud) -------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(32, 24), subplot_kw=dict(polar=True))
    handles, labels = _radar_from_pivot(axs[0], pivots['level_group'], colors)
    _radar_from_pivot(axs[1], pivots['latitude_group'], colors)
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.125), ncol=len(labels)/2, prop={'size': 40, 'weight': 'bold'})
    plt.tight_layout(rect=[0, 0.00, 1, 1]); plt.subplots_adjust(wspace=0.5)
    plt.savefig(GRAPH_DIR / 'radar_alt_lat.png', dpi=500, bbox_inches='tight')
    plt.close()

    # --- combinación Köppen × ESA -------------------------------------------
    piv = pivots['Koppen_ESA']
    cats = list(piv.columns[1:])
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(32, 32), subplot_kw=dict(polar=True))
    # líneas radiales
    for ang in angles[:-1]:
        ax.plot([ang, ang], [97, 100], color='black', lw=3, alpha=0.5)

    handles, labels = [], []
    for i, row in piv.iterrows():
        vals = row.drop('Model').values.astype(float).tolist() + \
               [row.drop('Model').values[0]]
        color = colors[i % len(colors)]
        line, = ax.plot(angles, vals, lw=5, color=color, label=row['Model'])
        ax.fill(angles, vals, color=color, alpha=0.1)
        handles.append(line); labels.append(row['Model'])

    ax.set_ylim(97, 100); ax.spines['polar'].set_visible(False)
    ticks = [97.5, 98, 98.5, 99, 99.5, 100]
    ax.set_yticks(ticks); ax.set_yticklabels([])
    # poner etiquetas de ticks radiales manualmente
    theta = np.deg2rad(46)
    for t in ticks:
        ax.text(theta, t, f'{t:.1f}%', ha='left', va='center', size=44, fontweight='bold')

    ax.grid(ls='--', lw=1.5, alpha=0.5)
    ax.set_xticks(angles[:-1]);ax.set_xticklabels(cats, size=45, fontweight='bold')
    for tick in ax.get_xticklabels():
        tick.set_y(tick.get_position()[1] - 0.14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(GRAPH_DIR / 'radar_koppen_esa_combo.png', dpi=500, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
def _plot_violin(df):
    """Violín de los valores MAPE (no medianas) por modelo."""
    modelos = sorted(df['Model'].unique())
    palette = (MODEL_COLORS * (len(modelos)//len(MODEL_COLORS)+1))[:len(modelos)]
    plt.figure(figsize=(16, 7))
    sns.violinplot(x='Model', y='MAPE', data=df, palette=palette,
                   cut=0, inner='box')
    plt.xlabel('Modelos', fontsize=16)
    plt.ylabel('MAPE', fontsize=16)
    plt.ylim(0, df['MAPE'].max()*1.1)
    plt.grid(axis='y', ls='--', alpha=0.6); sns.despine()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'violin_mape.png', dpi=400, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
def make_radar_and_violin():
    """Pipeline completo."""
    df_all = _load_all_metrics()
    if df_all.empty:
        return
    pivots = _make_pivots(df_all)
    _plot_all_radars(pivots)
    _plot_violin(df_all)
# =============================================================================
# Main
# =============================================================================
def main():
    real_dict, pred_dict, err_dict = load_prediction_data()
    if real_dict:
        plot_predictions(real_dict, pred_dict, err_dict)
    plot_seasonal_cycle()
    try:
        make_radar_and_violin()
    except Exception as e:
        logging.error(f'Error generando radar/violin: {e}')

if __name__ == '__main__':
    main()
