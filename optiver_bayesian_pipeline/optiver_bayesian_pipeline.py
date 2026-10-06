import os
import logging
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import mlflow
import optuna
from typing import Tuple, List, Dict
from joblib import Parallel, delayed

# ---------------------------------------------------------
# 1. Configuration & Setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOOK_DATA_PATH = r"C:\Users\msk80743\Use Case Design\book_train_mini.csv"
TRADE_DATA_PATH = r"C:\Users\msk80743\Use Case Design\trade_train_mini.csv"
TRAIN_LABELS_PATH = r"C:\Users\msk80743\Use Case Design\train_mini.csv"

# ---------------------------------------------------------
# 2. Parallel Feature Engineering
# ---------------------------------------------------------
def calc_wap1(df: pd.DataFrame) -> pd.Series:
    return (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])

def calc_wap2(df: pd.DataFrame) -> pd.Series:
    return (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])

def calc_realized_volatility(series: pd.Series) -> float:
    return np.sqrt(np.sum(series**2))

def create_book_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df_s = df[df['seconds_in_bucket'] >= window].copy()
    if df_s.empty: return pd.DataFrame()
        
    df_s['log_return1'] = df_s.groupby('time_id')['wap1'].apply(lambda x: np.log(x).diff()).reset_index(level=0, drop=True)
    df_s['spread1'] = (df_s['ask_price1'] - df_s['bid_price1']) / df_s['wap1']
    df_s['vol_imbalance'] = np.abs(df_s['bid_size1'] - df_s['ask_size1']) / (df_s['bid_size1'] + df_s['ask_size1'])
    df_s['total_depth'] = df_s['bid_size1'] + df_s['ask_size1'] + df_s['bid_size2'] + df_s['ask_size2']
    
    prefix = f"{window}_" if window > 0 else ""
    feat = df_s.groupby(['stock_id', 'time_id']).agg({
        'log_return1': [calc_realized_volatility],
        'spread1': ['mean'],
        'vol_imbalance': ['mean'],
        'total_depth': ['sum']
    })
    feat.columns = [f"{prefix}book_{c[0]}_{c[1]}" for c in feat.columns]
    return feat.reset_index()

def create_trade_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df_s = df[df['seconds_in_bucket'] >= window].copy()
    if df_s.empty: return pd.DataFrame()
        
    df_s['log_return'] = df_s.groupby('time_id')['price'].apply(lambda x: np.log(x).diff()).reset_index(level=0, drop=True)
    prefix = f"{window}_" if window > 0 else ""
    feat = df_s.groupby(['stock_id', 'time_id']).agg(
        trade_realized_vol=('log_return', calc_realized_volatility),
        trade_size_sum=('size', 'sum')
    )
    feat.columns = [f"{prefix}{c}" for c in feat.columns]
    return feat.reset_index()

def process_single_stock(stock_id: int, df_book: pd.DataFrame, df_trade: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    if not df_book.empty:
        df_book['wap1'], df_book['wap2'] = calc_wap1(df_book), calc_wap2(df_book)
    time_ids = pd.Series(df_book['time_id'].unique(), name='time_id').to_frame()
    time_ids['stock_id'] = stock_id
    df_merged = time_ids
    for w in windows:
        b_feat = create_book_features(df_book, w)
        t_feat = create_trade_features(df_trade, w)
        if not b_feat.empty: df_merged = df_merged.merge(b_feat, on=['stock_id', 'time_id'], how='left')
        if not t_feat.empty: df_merged = df_merged.merge(t_feat, on=['stock_id', 'time_id'], how='left')
    return df_merged

def process_data_parallel() -> Tuple[pd.DataFrame, int, int]:
    logger.info("Loading Raw Data...")
    df_book, df_trade, labels = pd.read_csv(BOOK_DATA_PATH), pd.read_csv(TRADE_DATA_PATH), pd.read_csv(TRAIN_LABELS_PATH)
    
    b_grp, t_grp = dict(tuple(df_book.groupby('stock_id'))), dict(tuple(df_trade.groupby('stock_id')))
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_single_stock)(sid, b_grp.get(sid, pd.DataFrame()), t_grp.get(sid, pd.DataFrame()), [0, 300]) 
        for sid in b_grp.keys()
    )
    
    final_df = labels.merge(pd.concat(results, ignore_index=True), on=['stock_id', 'time_id'], how='left').fillna(0)
    
    # Sort strictly chronologically by time_id to ensure proper temporal ordering
    final_df = final_df.sort_values('time_id').reset_index(drop=True)
    
    # Globally categorize IDs to prevent index-out-of-bounds in PyMC
    final_df['stock_idx'] = pd.factorize(final_df['stock_id'])[0]
    final_df['time_idx'] = pd.factorize(final_df['time_id'])[0]
    
    num_stocks = final_df['stock_idx'].nunique()
    num_times = final_df['time_idx'].nunique()
    
    return final_df, num_stocks, num_times

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

# ---------------------------------------------------------
# 3. Strict Temporal Splitting Logic
# ---------------------------------------------------------
def get_temporal_split(df: pd.DataFrame, test_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data strictly chronologically based on unique time_ids."""
    unique_times = df['time_id'].unique()
    split_idx = int(len(unique_times) * (1 - test_size))
    
    # The threshold time_id separating past from future
    time_threshold = unique_times[split_idx]
    
    past_df = df[df['time_id'] < time_threshold].copy()
    future_df = df[df['time_id'] >= time_threshold].copy()
    
    return past_df, future_df

# ---------------------------------------------------------
# 4. Dynamic Optuna Bayesian Model
# ---------------------------------------------------------
def build_and_train_model(X_train: pd.DataFrame, y_train: pd.Series, params: Dict, num_stocks: int, num_times: int, n_iterations=20000):
    feature_cols = [c for c in X_train.columns if c not in ['stock_id', 'time_id', 'stock_idx', 'time_idx']]
    
    stock_idx = X_train['stock_idx'].values
    time_idx = X_train['time_idx'].values
    
    X_matrix = X_train[feature_cols].values
    X_scaled = (X_matrix - np.mean(X_matrix, axis=0)) / (np.std(X_matrix, axis=0) + 1e-8)
    
    with pm.Model() as model:
        sigma_w = pm.InverseGamma('sigma_w', alpha=2, beta=params['beta_w'])
        rw_sigma = (1 - params['delta']) / params['delta']
        time_trend = pm.GaussianRandomWalk('time_trend', sigma=rw_sigma, shape=num_times, init_dist=pm.Normal.dist(mu=0, sigma=0.1))
        
        tau = pm.HalfCauchy('tau', beta=params['tau_scale'])
        lam = pm.HalfCauchy('lam', beta=1, shape=len(feature_cols))
        betas = pm.Normal('betas', mu=0, sigma=tau * lam, shape=len(feature_cols))
        
        k = params['k']
        time_factors = pm.Normal('time_factors', mu=0, sigma=1, shape=(num_times, k))
        stock_loadings = pm.Normal('stock_loadings', mu=0, sigma=1, shape=(num_stocks, k))
        latent_effect = pm.math.sum(time_factors[time_idx] * stock_loadings[stock_idx], axis=1)
        
        baseline = pm.Normal('baseline', mu=0, sigma=sigma_w)
        mu = baseline + time_trend[time_idx] + latent_effect + pm.math.dot(X_scaled, betas)
        
        y_obs = pm.StudentT('y_obs', nu=params['nu'], mu=mu, sigma=0.1, observed=y_train.values)
        
        mean_field = pm.fit(n=n_iterations, method='advi', obj_optimizer=pm.adam(learning_rate=0.01), progressbar=False)
        trace = mean_field.sample(500)
        
    return trace, feature_cols, np.mean(X_matrix, axis=0), np.std(X_matrix, axis=0)

def predict_model(trace, X_test, feat_cols, mean_train, std_train, k):
    post = az.summary(trace)['mean']
    
    beta_weights = np.array([post[f'betas[{i}]'] for i in range(len(feat_cols))])
    X_scaled = (X_test[feat_cols].values - mean_train) / (std_train + 1e-8)
    base_pred = np.dot(X_scaled, beta_weights)
    
    preds = []
    for row_num, (_, row) in enumerate(X_test.iterrows()):
        s_idx, t_idx = int(row['stock_idx']), int(row['time_idx'])
        
        baseline = post.get('baseline', 0)
        # Using .get with default 0 handles unseen times in the far future nicely
        trend = post.get(f'time_trend[{t_idx}]', 0)
        
        latent = 0
        for fac in range(k):
            t_fac = post.get(f'time_factors[{t_idx}, {fac}]', 0)
            s_load = post.get(f'stock_loadings[{s_idx}, {fac}]', 0)
            latent += t_fac * s_load
                
        pred = baseline + trend + latent + base_pred[row_num]
        preds.append(max(pred, 0.00001))
        
    return np.array(preds)

# ---------------------------------------------------------
# 5. Optuna & Splitting Pipeline
# ---------------------------------------------------------
global_train_val_df = None
global_num_stocks = 0
global_num_times = 0

def objective(trial):
    global global_train_val_df, global_num_stocks, global_num_times
    
    params = {
        'nu': trial.suggest_int('nu', 3, 8),
        'beta_w': trial.suggest_float('beta_w', 1e-5, 1e-2, log=True),
        'tau_scale': trial.suggest_float('tau_scale', 1e-4, 1e-1, log=True),
        'delta': trial.suggest_float('delta', 0.90, 0.995),
        'k': trial.suggest_categorical('k', [2, 3, 5]) 
    }
    
    # Strictly temporal split for Validation (using last 20% of the training block)
    train_df, val_df = get_temporal_split(global_train_val_df, test_size=0.20)
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        try:
            logger.info(f"Trial {trial.number}: Training with {params}")
            trace, feat_cols, m_tr, s_tr = build_and_train_model(
                train_df.drop(columns=['target']), train_df['target'], params, global_num_stocks, global_num_times, n_iterations=15000
            )
            preds_val = predict_model(trace, val_df.drop(columns=['target']), feat_cols, m_tr, s_tr, params['k'])
            score = rmspe(val_df['target'].values, preds_val)
            mlflow.log_metric("val_rmspe", score)
            logger.info(f"Trial {trial.number} Validation Score: {score:.4f}")
            return score
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 999.0 

if __name__ == "__main__":
    mlflow.set_experiment("Optiver_Bayesian_Optuna")
    
    # 1. Process Data
    df, global_num_stocks, global_num_times = process_data_parallel()
    
    # 2. Extract strict chronological Hold-out Test Set
    logger.info("Creating Strict Temporal Hold-out Test split...")
    global_train_val_df, test_df = get_temporal_split(df, test_size=0.15)
    
    logger.info(f"Train/Val shape: {global_train_val_df.shape} | Test shape: {test_df.shape}")
    
    # 3. Run Optuna Hyperparameter Tuning
    study = optuna.create_study(direction="minimize", study_name="Bayesian_Vol_Tuning")
    logger.info("Starting Optuna Hyperparameter Optimization...")
    study.optimize(objective, n_trials=5)
    
    logger.info(f"Best Parameters Found: {study.best_params}")
    
    # 4. Final Evaluation on Hold-Out Test Set
    logger.info("--- Training Final Model on combined Train+Val Set with Best Params ---")
    with mlflow.start_run(run_name="Final_Test_Evaluation"):
        mlflow.log_params(study.best_params)
        
        final_trace, feat_cols, m_tr, s_tr = build_and_train_model(
            global_train_val_df.drop(columns=['target']), global_train_val_df['target'], study.best_params, global_num_stocks, global_num_times, n_iterations=30000
        )
        
        logger.info("Evaluating on pure Chronological Hold-Out Test Set...")
        test_preds = predict_model(final_trace, test_df.drop(columns=['target']), feat_cols, m_tr, s_tr, study.best_params['k'])
        
        final_test_rmspe = rmspe(test_df['target'].values, test_preds)
        mlflow.log_metric("final_test_rmspe", final_test_rmspe)
        
        logger.info(f"==================================================")
        logger.info(f"FINAL TEMPORAL HOLD-OUT TEST RMSPE: {final_test_rmspe:.4f}")
        logger.info(f"==================================================")