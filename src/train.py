from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

# Paths are built from this file's location (src/train.py), so they work
# no matter which folder you run the script from.
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"


# ==========================================
# PHASES 4, 5 & 6: POISSON & DIXON-COLES (STAGE 1)
# ==========================================
def tau_function(x, y, lam, mu, rho):
    """Dixon-Coles adjustment factor for low-scoring games (0-0, 1-0, 0-1, 1-1)."""
    if x == 0 and y == 0:
        return 1.0 - (lam * mu * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lam * rho)
    elif x == 1 and y == 0:
        return 1.0 + (mu * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel:
    """
    Fits either Simple Independent Poisson (use_dc=False) or full Dixon-Coles
    (use_dc=True) with exponential time decay xi (in days).
    """

    def __init__(self, xi=0.0018, use_dc=True, max_goals=10):
        self.xi = xi
        self.use_dc = use_dc
        self.max_goals = max_goals
        self.teams = []
        self.params_ = {}

    def fit(self, df):
        # Train on the most recent 750 matches (about 2 seasons)
        train_df = df.tail(750).copy()
        self.teams = sorted(set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"])))
        n_teams = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        h_idx = train_df["HomeTeam"].map(team_idx).values
        a_idx = train_df["AwayTeam"].map(team_idx).values
        h_goals = train_df["FTHG"].values
        a_goals = train_df["FTAG"].values

        # Exponential time decay: recent games weigh about 1.0, older games fade
        max_date = train_df["Date"].max()
        days_ago = (max_date - train_df["Date"]).dt.days.values
        weights = np.exp(-self.xi * days_ago)

        # Initial guess: [attack_0..N, defense_0..N, home_adv, rho]
        init_params = np.concatenate([np.zeros(n_teams), np.zeros(n_teams), [0.25, -0.05]])

        def neg_log_likelihood(params):
            att = params[:n_teams]
            dfn = params[n_teams : 2 * n_teams]
            hfa = params[2 * n_teams]
            rho = params[2 * n_teams + 1] if self.use_dc else 0.0

            # Keeps the average attack rating centered at 0.0
            constraint_penalty = 100.0 * (np.mean(att) ** 2)

            lam = np.exp(att[h_idx] + dfn[a_idx] + hfa)
            mu = np.exp(att[a_idx] + dfn[h_idx])

            log_p_home = poisson.logpmf(h_goals, lam)
            log_p_away = poisson.logpmf(a_goals, mu)

            if self.use_dc:
                tau = np.ones_like(lam)
                m00 = (h_goals == 0) & (a_goals == 0)
                m01 = (h_goals == 0) & (a_goals == 1)
                m10 = (h_goals == 1) & (a_goals == 0)
                m11 = (h_goals == 1) & (a_goals == 1)
                tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
                tau[m01] = 1.0 + lam[m01] * rho
                tau[m10] = 1.0 + mu[m10] * rho
                tau[m11] = 1.0 - rho
                tau = np.clip(tau, 1e-6, None)
                log_tau = np.log(tau)
            else:
                log_tau = 0.0

            ll = np.sum(weights * (log_p_home + log_p_away + log_tau))
            # Mild L2 regularization stops extreme values for newly promoted clubs
            l2_reg = 0.01 * np.sum(att**2 + dfn**2)
            return -ll + constraint_penalty + l2_reg

        bounds = [(-2.5, 2.5)] * (2 * n_teams) + [(0.0, 1.0), (-0.4, 0.4)]
        res = minimize(neg_log_likelihood, init_params, method="L-BFGS-B", bounds=bounds)

        att = res.x[:n_teams]
        dfn = res.x[n_teams : 2 * n_teams]
        self.params_ = {
            "attack": {t: att[i] for i, t in enumerate(self.teams)},
            "defense": {t: dfn[i] for i, t in enumerate(self.teams)},
            "hfa": res.x[2 * n_teams],
            "rho": res.x[2 * n_teams + 1] if self.use_dc else 0.0,
        }
        return self

    def predict_match(self, home_team, away_team):
        """Outputs expected goals (lam, mu), the scoreline matrix, and P(H), P(D), P(A)."""
        # Newly promoted teams not in params_ get slightly below-average ratings
        att_h = self.params_["attack"].get(home_team, -0.15)
        dfn_h = self.params_["defense"].get(home_team, 0.15)
        att_a = self.params_["attack"].get(away_team, -0.15)
        dfn_a = self.params_["defense"].get(away_team, 0.15)
        hfa = self.params_["hfa"]
        rho = self.params_["rho"]

        lam = float(np.exp(att_h + dfn_a + hfa))
        mu = float(np.exp(att_a + dfn_h))

        goals = np.arange(self.max_goals)
        p_home = poisson.pmf(goals, lam)
        p_away = poisson.pmf(goals, mu)
        matrix = np.outer(p_home, p_away)

        if self.use_dc:
            for x in [0, 1]:
                for y in [0, 1]:
                    matrix[x, y] = max(0.0, matrix[x, y] * tau_function(x, y, lam, mu, rho))

        matrix /= matrix.sum()  # All cells sum to 1.0

        p_h = float(np.sum(np.tril(matrix, -1)))  # Home goals > Away goals
        p_d = float(np.sum(np.diag(matrix)))      # Home goals == Away goals
        p_a = float(np.sum(np.triu(matrix, 1)))   # Home goals < Away goals

        over_25 = float(
            sum(matrix[i, j] for i in range(self.max_goals) for j in range(self.max_goals) if i + j > 2)
        )

        return {
            "lam": lam,
            "mu": mu,
            "p_H": p_h,
            "p_D": p_d,
            "p_A": p_a,
            "over_25": over_25,
            "matrix": matrix,
        }


# ==========================================
# PHASE 10 METRICS: LOG LOSS, BRIER & RPS
# ==========================================
def compute_metrics(y_true_int, probs):
    """
    Multi-class Log Loss, Brier Score, and Ranked Probability Score (RPS).
    y_true_int: 0 = Home Win, 1 = Draw, 2 = Away Win (ordered outcomes).
    """
    n = len(y_true_int)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), y_true_int] = 1.0

    ll = log_loss(y_true_int, probs, labels=[0, 1, 2])
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    cum_probs = np.cumsum(probs, axis=1)
    cum_true = np.cumsum(onehot, axis=1)
    rps = float(np.mean(0.5 * np.sum((cum_probs[:, :2] - cum_true[:, :2]) ** 2, axis=1)))
    return {"LogLoss": round(ll, 4), "Brier": round(brier, 4), "RPS": round(rps, 4)}


# ==========================================
# STAGE 1 WALK-FORWARD (separate function so tests can call it)
# ==========================================
def generate_stage1_predictions(df, warmup=380, step=190):
    """
    Walks forward through time. Each chunk of matches is predicted by models
    fitted only on the matches BEFORE that chunk.
    """
    dc_records, pois_records = [], []

    for start in range(warmup, len(df), step):
        train_slice = df.iloc[:start]
        test_slice = df.iloc[start : start + step]

        dc_model = DixonColesModel(use_dc=True).fit(train_slice)
        pois_model = DixonColesModel(use_dc=False).fit(train_slice)

        for _, row in test_slice.iterrows():
            dc_out = dc_model.predict_match(row["HomeTeam"], row["AwayTeam"])
            pois_out = pois_model.predict_match(row["HomeTeam"], row["AwayTeam"])

            dc_records.append({
                "Match_ID": row["Match_ID"],
                "dc_lam": dc_out["lam"],
                "dc_mu": dc_out["mu"],
                "dc_p_H": dc_out["p_H"],
                "dc_p_D": dc_out["p_D"],
                "dc_p_A": dc_out["p_A"],
                "dc_over_25": dc_out["over_25"],
            })
            pois_records.append({
                "Match_ID": row["Match_ID"],
                "pois_p_H": pois_out["p_H"],
                "pois_p_D": pois_out["p_D"],
                "pois_p_A": pois_out["p_A"],
            })

    return pd.DataFrame(dc_records), pd.DataFrame(pois_records)


# ==========================================
# PHASES 7, 8, 9 & 10: STAGE 2 LIGHTGBM & WALK-FORWARD BACKTEST
# ==========================================
STAGE2_FEATURES = [
    "dc_lam", "dc_mu", "dc_p_H", "dc_p_D", "dc_p_A", "dc_over_25",
    "HomeElo", "AwayElo", "EloDiff",
    "HomeRestDays", "AwayRestDays", "RestDiff",
    "Home_Roll_Points", "Away_Roll_Points", "FormPointsDiff",
    "Home_Roll_xGDiff", "Away_Roll_xGDiff", "xGDiff_Diff",
    "Home_Roll_GoalDiff", "Away_Roll_GoalDiff",
]


def make_lgbm():
    """Shallow trees prevent overfitting on noisy football scores."""
    return LGBMClassifier(
        n_estimators=80,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,  # needed, otherwise subsample is ignored
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def make_calibrated_lgbm():
    """Platt (sigmoid) calibration with time-ordered folds instead of shuffled ones."""
    return CalibratedClassifierCV(
        estimator=make_lgbm(),
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
    )


def run_walk_forward_pipeline():
    in_path = PROCESSED_DIR / "features_matches.csv"
    if not in_path.is_file():
        raise FileNotFoundError(f"{in_path} not found. Run `python -m src.features` first.")

    df = pd.read_csv(in_path, parse_dates=["Date"])
    df = df.dropna(subset=["Home_Roll_Points", "Away_Roll_Points"]).reset_index(drop=True)

    # Ordered outcomes: 0 = Home, 1 = Draw, 2 = Away
    target_map = {"H": 0, "D": 1, "A": 2}
    df["Target"] = df["FTR"].map(target_map)

    # Step 1: out-of-sample Stage 1 predictions for historical matches
    warmup, step = 380, 190
    print(f"[INFO] Generating walk-forward Stage 1 predictions across {len(df)} matches...")
    dc_df, pois_df = generate_stage1_predictions(df, warmup, step)
    eval_df = df.iloc[warmup:].merge(dc_df, on="Match_ID").merge(pois_df, on="Match_ID")

    # Step 2: walk-forward backtest for Stage 2 (LightGBM + calibration)
    s2_warmup = 1140
    s2_step = 190

    lgbm_uncal_probs, lgbm_cal_probs = [], []
    test_indices = []

    print("[INFO] Running walk-forward Stage 2 LightGBM & calibration backtest...")
    for start in range(s2_warmup, len(eval_df), s2_step):
        s2_train = eval_df.iloc[:start]
        s2_test = eval_df.iloc[start : start + s2_step]

        X_train, y_train = s2_train[STAGE2_FEATURES], s2_train["Target"].values
        X_test = s2_test[STAGE2_FEATURES]

        base_lgbm = make_lgbm()
        base_lgbm.fit(X_train, y_train)
        lgbm_uncal_probs.append(base_lgbm.predict_proba(X_test))

        cal_lgbm = make_calibrated_lgbm()
        cal_lgbm.fit(X_train, y_train)
        lgbm_cal_probs.append(cal_lgbm.predict_proba(X_test))

        test_indices.extend(s2_test.index.tolist())

    backtest_df = eval_df.loc[test_indices].copy()
    y_true = backtest_df["Target"].values

    uncal_probs = np.vstack(lgbm_uncal_probs)
    cal_probs = np.vstack(lgbm_cal_probs)
    pois_probs = backtest_df[["pois_p_H", "pois_p_D", "pois_p_A"]].values
    dc_probs = backtest_df[["dc_p_H", "dc_p_D", "dc_p_A"]].values

    # Baseline: always predict the historical H/D/A rates (from rows before every test row)
    hist_rates = eval_df.iloc[:s2_warmup]["Target"].value_counts(normalize=True).reindex([0, 1, 2]).values
    naive_probs = np.tile(hist_rates, (len(y_true), 1))

    comparison = {
        "1. Naive Base Rate": compute_metrics(y_true, naive_probs),
        "2. Simple Poisson (Stage 1 Baseline)": compute_metrics(y_true, pois_probs),
        "3. Dixon-Coles (Stage 1 Scout)": compute_metrics(y_true, dc_probs),
        "4. LightGBM Uncalibrated (Stage 2)": compute_metrics(y_true, uncal_probs),
        "5. LightGBM Calibrated (Stage 2 Manager)": compute_metrics(y_true, cal_probs),
    }

    # Bookmaker benchmark on rows with valid Bet365 odds
    if {"B365H", "B365D", "B365A"}.issubset(backtest_df.columns):
        odds_cols = ["B365H", "B365D", "B365A"]
        valid_odds_mask = (
            backtest_df[odds_cols].notna().all(axis=1)
            & (backtest_df[odds_cols] > 1.0).all(axis=1)
        ).values
        if valid_odds_mask.sum() > 0:
            odds_vals = backtest_df.loc[valid_odds_mask, odds_cols].values
            raw_inv = 1.0 / odds_vals
            bookie_probs = raw_inv / raw_inv.sum(axis=1, keepdims=True)
            comparison["6. Bookmaker Implied (Benchmark)"] = compute_metrics(
                y_true[valid_odds_mask], bookie_probs
            )

    results_table = pd.DataFrame(comparison).T
    print("\n=== PHASE 10: HONEST WALK-FORWARD MODEL COMPARISON ===")
    print(results_table.to_string())
    results_table.to_csv(PROCESSED_DIR / "model_comparison.csv")
    stage2_out = backtest_df[["Match_ID", "Date", "Season", "HomeTeam", "AwayTeam", "Target"]].copy()
    for k, name in enumerate(["H", "D", "A"]):
        stage2_out[f"lgbm_p_{name}"] = uncal_probs[:, k]
        stage2_out[f"cal_p_{name}"] = cal_probs[:, k]
    stage2_out.to_csv(PROCESSED_DIR / "backtest_stage2.csv", index=False)

    # Fit final production models on all available data and save them
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_dc = DixonColesModel(use_dc=True).fit(df)
    final_cal_lgbm = make_calibrated_lgbm()
    final_cal_lgbm.fit(eval_df[STAGE2_FEATURES], eval_df["Target"].values)

    joblib.dump(final_dc, MODELS_DIR / "dixon_coles_stage1.pkl")
    joblib.dump(final_cal_lgbm, MODELS_DIR / "lightgbm_stage2.pkl")
    eval_df.to_csv(PROCESSED_DIR / "backtest_predictions.csv", index=False)
    print(f"\n[SUCCESS] Saved trained Stage 1 & Stage 2 artifacts to '{MODELS_DIR}'")

if __name__ == "__main__":
    from src.train import run_walk_forward_pipeline as _run

    _run()