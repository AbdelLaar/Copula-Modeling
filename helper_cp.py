import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


def t_winsorize(series, alpha = 0.005):
    x = series.dropna()
    df_t, loc_t, scale_t = t.fit(x)

    lower_bound = t.ppf(alpha, df_t, loc_t, scale_t)
    upper_bound = t.ppf(1 - alpha, df_t, loc_t, scale_t)

    # Do not modify assets with no extremes
    if x.min() >= lower_bound and x.max() <= upper_bound:
        return series.copy()
    
    series_new = series.copy()
    series_new = series_new.clip(lower_bound, upper_bound)

    return series_new

def to_log_returns(df):
    return np.log(df).diff().dropna()  

def recalibrate_returns(dicito, mu, sigma, print_names: bool = False):
    dic = dicito.copy()
    for key in dic.keys():
        dic[key] = dic[key] * sigma[key] + mu[key]
        if print_names:
            print(key)
            
    return dic


def prepare_dictionary_of_assets(simulation_dictionary, return_type: str = "log", start_value = 1.0, include_step0: bool = True):
    asset_index_dict = {}
    for asset, df in simulation_dictionary.items():
        if df is None or df.empty:
            asset_index_dict[asset] = df.copy
            continue
        data = df.copy().astype(float)
        if return_type == "arithmetic":
            growht = 1.0 + data
            growth = groth.mask(growth <= 0.0, np.nan)
            index_df = start_value * growth.cumprod()
        elif return_type == "log":
            index_df = start_value * np.exp(data.cumsum())
        else:
            raise ValueError(" return type given is invalid")
        
        if include_step0:
            if 0 not in index_df.index:
                baseline = pd.DataFrame([start_value] * index_df.shape[1], index = index_df.columns).T
                baseline.index = [0]
                index_df = pd.concat([baseline, index_df], axis = 0).sort_index()
        asset_index_dict[asset] = index_df
    return asset_index_dict

def calculate_mdd_for_sim(dictionary):
    mdd_by_asset = {}
    mdd_step_by_asset = {}
    for asset, df_levels in dictionary.items():
        mdd, mdd_step, peak_step = compute_mdd_single_asset(df_levels)
        mdd_by_asset[asset] = mdd
        mdd_step_by_asset[asset] = mdd_step
    return mdd_by_asset, mdd_step_by_asset

def compute_mdd_single_asset(df: pd.DataFrame):
    running_peak = df.cummax()
    drawdowns = (df / running_peak) - 1.0
    mdd_neg = drawdowns.min(axis = 0)
    mdd = -mdd_neg
    mdd_step = drawdowns.idxmin(axis = 0)
    peak_step = pd.Series(index = df.columns, dtype = df.index.dtype)
    for col in df.columns:
        t_star = mdd_step[col]
        sub = df.loc[:t_star, col]
        peak_val = running_peak.loc[t_star, col]
        peak_step[col] = sub.idxmax() if np.isfinite(peak_val) else t_star
    
    return mdd, mdd_step, peak_step


# -----------------------------
# 1) VaR per simulation (per column) for one asset
# -----------------------------
def var_per_simulation(asset_returns_df: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """
    Computes VaR for each simulation column from an asset returns panel.
    Convention: returns are arithmetic returns. VaR is a positive loss number.
    VaR_alpha = - quantile_alpha(returns)  (e.g. alpha=0.05 -> 5% left tail)
    """
    q = asset_returns_df.quantile(alpha, axis=0)
    return (-q).rename("VaR")


# -----------------------------
# 2) VaR per simulation for all assets in a dict
# -----------------------------
def var_dict(returns_dict: dict[str, pd.DataFrame], alpha: float = 0.05) -> dict[str, pd.Series]:
    """
    returns_dict: {asset_name: DataFrame(T x N_sims)} of returns
    output: {asset_name: Series(N_sims)} VaR per simulation
    """
    return {k: var_per_simulation(df, alpha=alpha) for k, df in returns_dict.items()}


# -----------------------------
# 3) 95th percentile (or any percentile) across assets of the VaR distribution
#     (done per simulation index; then also returns an overall scalar if you want)
# -----------------------------
def var_percentile_across_assets(var_by_asset: dict[str, pd.Series], pct: float = 0.95):
    """
    Takes {asset: Series(VaR per sim)} and returns:
      - a Series with the pct-quantile across assets for each simulation (row-wise percentile)
      - an overall scalar percentile across *all* VaR values pooled (optional handy summary)
    """
    var_matrix = pd.concat(var_by_asset, axis=1)  # index: sim, columns: asset
    across_assets = var_matrix.quantile(pct, axis=1).rename(f"VaR_p{int(pct*100)}_across_assets")
    pooled = var_matrix.stack().quantile(pct)
    return across_assets, pooled


# -----------------------------
# 4) Plot VaR distributions for each asset
# -----------------------------
def plot_var_distributions(var_by_asset: dict[str, pd.Series], bins: int = 40):
    """
    Plots one histogram per asset (overlayed) of VaR across simulations.
    """
    plt.figure(figsize=(11, 6))
    for asset, s in var_by_asset.items():
        plt.hist(s.to_numpy(), bins=bins, alpha=0.35, density=True, label=asset)
    plt.xlabel("VaR")
    plt.ylabel("Density")
    plt.title("VaR distribution across simulations (per asset)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- 1) Horizon return per simulation (column) ---
def horizon_return_df(df: pd.DataFrame, kind: str = "log") -> pd.Series:
    """
    df: rows=time, cols=scenarios (simulations)
    kind: "simple" (arithmetic returns) or "log" (log returns)

    returns: Series indexed by scenario, one horizon return per scenario
    """
    if kind == "simple":
        return (1.0 + df).prod(axis=0) - 1.0
    elif kind == "log":
        return df.sum(axis=0).apply(np.exp) - 1.0
    else:
        raise ValueError("kind must be 'simple' or 'log'")

# --- 2) Portfolio horizon returns from asset simulations + weights ---
def portfolio_horizon_returns(
    returns_dict: dict[str, pd.DataFrame],
    weights: dict[str, float],
    kind: str = "simple",
) -> pd.Series:
    """
    Builds portfolio horizon returns per scenario by taking the SAME scenario index across assets.
    (Assumes scenario alignment across assets.)

    returns: Series indexed by scenario, one portfolio horizon return per scenario
    """
    # per-asset horizon returns (Series per asset, indexed by scenario)
    hr = {a: horizon_return_df(df, kind=kind) for a, df in returns_dict.items()}

    # stack into matrix: rows=scenario, cols=asset
    hr_df = pd.DataFrame(hr)

    # align weights to columns
    w = pd.Series(weights).reindex(hr_df.columns)

    # weighted sum of horizon returns (linear portfolio)
    port_hr = hr_df.mul(w, axis=1).sum(axis=1)
    port_hr.name = "portfolio_horizon_return"
    return port_hr

# --- 3) VaR from a horizon-return Series (positive loss number) ---
def var_from_horizon_returns(horizon_returns: pd.Series, alpha: float = 0.05) -> float:
    """
    VaR_alpha = -quantile_alpha(returns)  (positive loss convention)
    """
    return float(-horizon_returns.quantile(alpha))

# --- 4) Plot distribution of portfolio VaR (really: distribution of horizon losses/returns) ---
def plot_portfolio_return_distribution(port_hr: pd.Series, bins: int = 50):
    plt.figure(figsize=(10, 5))
    plt.hist(port_hr.to_numpy(), bins=bins, alpha=0.7, density=True)
    plt.axvline(port_hr.quantile(0.05), linewidth=1.5, linestyle="--", label="5% quantile")
    plt.xlabel("Portfolio horizon return")
    plt.ylabel("Density")
    plt.title("Portfolio horizon return distribution (across simulations)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()