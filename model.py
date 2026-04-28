import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap

warnings.filterwarnings("ignore", message="ntree_limit is deprecated.*", category=UserWarning)

RANDOM_STATE = 42
MODEL_PATH = "model.pkl"
DATA_PATH = "leasing_default.csv"

STATS = {
    "age":                    {"mean": 41.86, "min": 21,   "max": 70,    "sigma": 10.03, "p5": 25,   "p95": 58},
    "years_cdl":              {"mean": 18.03, "min": 0,    "max": 45,    "sigma": 10.33, "p5": 0,    "p95": 34},
    "years_owner_op":         {"mean": 13.69, "min": 0,    "max": 40,    "sigma": 9.42,  "p5": 0,    "p95": 31},
    "fico_score":             {"mean": 639.47,"min": 368,  "max": 850,   "sigma": 71.14, "p5": 525,  "p95": 759},
    "violations_3y":          {"mean": 0.82,  "min": 0,    "max": 6,     "sigma": 0.91,  "p5": 0,    "p95": 3},
    "avg_weekly_miles":       {"mean": 2597,  "min": 981,  "max": 4200,  "sigma": 456,   "p5": 1844, "p95": 3344},
    "loads_per_month":        {"mean": 11.9,  "min": 2,    "max": 28,    "sigma": 3.65,  "p5": 6,    "p95": 18},
    "unique_brokers_6m":      {"mean": 5.04,  "min": 1,    "max": 14,    "sigma": 2.13,  "p5": 2,    "p95": 9},
    "on_time_delivery_pct":   {"mean": 80.05, "min": 28.5, "max": 99.9,  "sigma": 11.76, "p5": 57.2, "p95": 95.9},
    "avg_days_between_loads": {"mean": 3.01,  "min": 0.5,  "max": 15,    "sigma": 2.03,  "p5": 0.53, "p95": 7.2},
    "gross_weekly_revenue":   {"mean": 5947,  "min": 1916, "max": 11544, "sigma": 1248,  "p5": 4002, "p95": 8108},
    "net_weekly_income":      {"mean": 1419,  "min": 12,   "max": 3581,  "sigma": 540,   "p5": 607,  "p95": 2384},
    "payment_to_income_ratio":{"mean": 0.589, "min": 0.19, "max": 5,     "sigma": 0.258, "p5": 0.29, "p95": 1.14},
    "avg_account_balance":    {"mean": 3269,  "min": 100,  "max": 80000, "sigma": 3350,  "p5": 286,  "p95": 11308},
    "prior_late_payments":    {"mean": 1.19,  "min": 0,    "max": 7,     "sigma": 0.91,  "p5": 0,    "p95": 3},
    "contract_start_month":   {"mean": 6.54,  "min": 1,    "max": 12,    "sigma": 3.34,  "p5": 1,    "p95": 12},
}

FREIGHT_TYPES  = ["dry_van", "reefer", "flatbed"]
FREIGHT_LABELS = {"dry_van": "Dry Van", "reefer": "Reefer", "flatbed": "Flatbed"}
REGIONS        = ["midwest", "south", "southeast", "west", "northeast"]
REGION_LABELS  = {"midwest": "Midwest", "south": "South", "southeast": "Southeast",
                  "west": "West", "northeast": "Northeast"}

FEATURE_LABELS = {
    "age":                    "Age",
    "years_cdl":              "Years with CDL",
    "years_owner_op":         "Years as owner-operator",
    "fico_score":             "FICO Score",
    "is_foreign_born":        "Foreign-born",
    "violations_3y":          "Traffic violations (3y)",
    "avg_weekly_miles":       "Avg weekly miles",
    "loads_per_month":        "Loads per month",
    "freight_type":           "Freight type",
    "unique_brokers_6m":      "Unique brokers (6m)",
    "on_time_delivery_pct":   "On-time delivery (%)",
    "avg_days_between_loads": "Avg days between loads",
    "gross_weekly_revenue":   "Gross weekly revenue",
    "net_weekly_income":      "Net weekly income",
    "payment_to_income_ratio":"Payment-to-income ratio",
    "avg_account_balance":    "Avg account balance",
    "prior_late_payments":    "Prior late payments",
    "region":                 "Region",
    "contract_start_month":   "Contract start month",
}

CAT_COLS = ["freight_type", "region"]

ACTIONABLE_FEATURES = {
    "payment_to_income_ratio": {"target": 0.35, "direction": "reduce",   "label": "payment-to-income ratio"},
    "on_time_delivery_pct":    {"target": 92,   "direction": "increase", "label": "on-time delivery"},
    "unique_brokers_6m":       {"target": 8,    "direction": "increase", "label": "broker diversification"},
    "avg_days_between_loads":  {"target": 2,    "direction": "reduce",   "label": "days between loads"},
}


def generate_random_driver():
    rng = np.random.default_rng()

    def randn():
        return rng.standard_normal()

    def sample(key, is_int=False, mean_shift=0.0, scale=1.0):
        s = STATS[key]
        v = float(np.clip(s["mean"] + mean_shift + s["sigma"] * scale * randn(), s["min"], s["max"]))
        return int(round(v)) if is_int else v

    def sample_cat(arr, weights):
        return rng.choice(arr, p=weights)

    # Risk-tier sampling: shifts key features toward riskier values for
    # medium/high tiers so the generator produces a more balanced mix of
    # APPROVE / REVIEW / DECLINE outcomes across random draws.
    # Weights: 30% standard · 50% medium-risk · 20% high-risk
    # "medium" targets the REVIEW band (42–64%) with moderate shifts
    # "high" targets DECLINE (≥ 64%) with stronger shifts
    tier = rng.choice(["standard", "medium", "high"], p=[0.30, 0.50, 0.20])
    # non-actionable: moderate shifts so FICO doesn't monopolise the waterfall
    fico_shift    = {"standard":   0, "medium":  -25, "high":  -50}[tier]
    late_shift    = {"standard": 0.0, "medium":  0.30, "high":  0.65}[tier]
    balance_mult  = {"standard": 1.0, "medium":  0.82, "high":  0.55}[tier]
    # actionable (block 05): stronger shifts so they appear prominently in waterfall
    opex_add      = {"standard": 0.0, "medium":  0.09, "high":  0.17}[tier]
    otd_shift     = {"standard": 0.0, "medium": -11.0, "high": -20.0}[tier]
    brokers_shift = {"standard": 0.0, "medium":  -2.0, "high":  -3.0}[tier]
    days_shift    = {"standard": 0.0, "medium":  +1.5, "high":  +3.5}[tier]

    age            = sample("age", True)
    years_cdl      = max(0, min(age - 21, sample("years_cdl", True)))
    years_owner_op = max(0, min(years_cdl, sample("years_owner_op", True)))
    fico_score     = sample("fico_score", True, mean_shift=fico_shift)
    avg_weekly_miles = sample("avg_weekly_miles", True)
    freight_type   = sample_cat(FREIGHT_TYPES, [0.55, 0.29, 0.16])

    rate = (2.4 if freight_type == "reefer" else 2.55 if freight_type == "flatbed" else 2.15) + randn() * 0.15
    gross_weekly_revenue = int(np.clip(avg_weekly_miles * rate, 1916, 11544))
    opex_ratio           = 0.55 + opex_add + rng.random() * 0.17
    net_weekly_income    = int(max(12, gross_weekly_revenue * (1 - opex_ratio) - 750))
    payment_to_income_ratio = round(min(5.0, 692 / max(50, net_weekly_income)), 3)

    s_bal = STATS["avg_account_balance"]
    avg_account_balance = int(np.clip(
        s_bal["mean"] * balance_mult + s_bal["sigma"] * 0.4 * randn(),
        s_bal["min"], s_bal["max"]
    ))

    return {
        "age":                    age,
        "years_cdl":              years_cdl,
        "years_owner_op":         years_owner_op,
        "fico_score":             fico_score,
        "is_foreign_born":        int(rng.random() < 0.55),
        "violations_3y":          sample("violations_3y", True),
        "avg_weekly_miles":       avg_weekly_miles,
        "loads_per_month":        sample("loads_per_month", True),
        "freight_type":           freight_type,
        "unique_brokers_6m":      sample("unique_brokers_6m", True, mean_shift=brokers_shift),
        "on_time_delivery_pct":   round(float(np.clip(sample("on_time_delivery_pct", mean_shift=otd_shift), 28.5, 99.9)), 1),
        "avg_days_between_loads": round(float(np.clip(sample("avg_days_between_loads", mean_shift=days_shift), 0.5, 15)), 2),
        "gross_weekly_revenue":   gross_weekly_revenue,
        "net_weekly_income":      net_weekly_income,
        "payment_to_income_ratio":payment_to_income_ratio,
        "avg_account_balance":    avg_account_balance,
        "prior_late_payments":    sample("prior_late_payments", True, mean_shift=late_shift),
        "region":                 sample_cat(REGIONS, [0.31, 0.24, 0.15, 0.15, 0.15]),
        "contract_start_month":   sample("contract_start_month", True),
    }


def train_and_save():
    df = pd.read_csv(DATA_PATH)
    df_model = df.copy()

    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders[col] = le

    X = df_model.drop(columns=["default_12m"])
    y = df_model["default_12m"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)

    scale_pos_weight = float((y_tr == 0).sum()) / float((y_tr == 1).sum())

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    explainer = shap.TreeExplainer(model)

    bundle = {
        "model":         model,
        "encoders":      encoders,
        "explainer":     explainer,
        "feature_names": list(X.columns),
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_and_save()
    return joblib.load(MODEL_PATH)


def _driver_to_df(driver, encoders, feature_names):
    row = {}
    for feat in feature_names:
        val = driver[feat]
        if feat in encoders:
            val = int(encoders[feat].transform([val])[0])
        row[feat] = val
    return pd.DataFrame([row])


def predict(driver, bundle):
    X_row  = _driver_to_df(driver, bundle["encoders"], bundle["feature_names"])
    prob   = float(bundle["model"].predict_proba(X_row)[0, 1])
    sv     = bundle["explainer"].shap_values(X_row)
    contribs = dict(zip(bundle["feature_names"], sv[0].tolist()))
    ev     = bundle["explainer"].expected_value
    base   = float(ev.item()) if hasattr(ev, "item") else float(ev)
    return {
        "probability":  prob,
        "base_value":   base,
        "contributions": contribs,
        "total_shap":   sum(contribs.values()),
    }


def format_value(key, val):
    if key == "is_foreign_born":
        return "Yes" if val == 1 else "No"
    if key == "freight_type":
        return FREIGHT_LABELS.get(val, str(val))
    if key == "region":
        return REGION_LABELS.get(val, str(val))
    if key in ("gross_weekly_revenue", "net_weekly_income", "avg_account_balance"):
        return f"US$ {int(val):,}"
    if key == "payment_to_income_ratio":
        return f"{float(val):.3f}"
    if key == "on_time_delivery_pct":
        return f"{float(val):.1f}%"
    if key == "avg_days_between_loads":
        return f"{float(val):.2f}"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _qualifier(key, val):
    s = STATS.get(key)
    if not s or not isinstance(val, (int, float)):
        return ""
    if val < s["p5"]:            return "very low"
    if val < s["mean"] - s["sigma"] * 0.5: return "low"
    if val > s["p95"]:           return "very high"
    if val > s["mean"] + s["sigma"] * 0.5: return "high"
    return "near average"


def get_insights(driver, contributions):
    top3 = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    out  = []
    for key, sv in top3:
        val       = driver[key]
        label     = FEATURE_LABELS[key]
        direction = "increased" if sv > 0 else "decreased"
        qualifier = _qualifier(key, val if isinstance(val, (int, float)) else 0)
        val_str   = format_value(key, val)
        intensity = "strongly" if abs(sv) > 0.5 else "significantly" if abs(sv) > 0.2 else "mildly"
        sign      = "+" if sv > 0 else ""
        out.append({
            "key":          key,
            "label":        label,
            "shap":         sv,
            "shap_display": f"{sign}{sv:.3f}",
            "driver_value": val_str,
            "sentence":     f"{label} {qualifier} ({val_str}) {direction} default risk {intensity}.",
        })
    return out


def get_actionables(driver, contributions, bundle):
    """
    Ceteris-paribus simulation: for each operationally controllable feature,
    check whether the driver's current value is already at (or better than) the
    recommended target. If not, simulate moving to the target and report the
    change in predicted probability in percentage points.
    Only features with at least 0.5 pp improvement are shown.
    """
    suggestions = []
    base_prob = predict(driver, bundle)["probability"]
    for key, cfg in ACTIONABLE_FEATURES.items():
        val = driver[key]
        # skip if driver is already at or beyond the recommended target
        if cfg["direction"] == "reduce"   and float(val) <= float(cfg["target"]):
            continue
        if cfg["direction"] == "increase" and float(val) >= float(cfg["target"]):
            continue
        modified = {**driver, key: cfg["target"]}
        new_prob = predict(modified, bundle)["probability"]
        delta    = (new_prob - base_prob) * 100
        if delta < -0.5:
            suggestions.append({
                "key":         key,
                "label":       cfg["label"],
                "current":     val,
                "current_fmt": f"{val:.2f}" if isinstance(val, float) else str(val),
                "target":      cfg["target"],
                "delta":       f"{delta:.1f}",
                "direction":   cfg["direction"],
                "shap":        contributions.get(key, 0),
            })
    # sort by biggest absolute improvement first
    return sorted(suggestions, key=lambda x: float(x["delta"]))[:3]
