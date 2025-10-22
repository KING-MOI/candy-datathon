"""Candy analysis: which candy should we make?

Reads data/candy-data.csv (Kaggle: fivethirtyeight/the-ultimate-halloween-candy-power-ranking).
Writes build/data.json and site/index.html (one file, works offline).

Run: ../.venv/bin/python -I analysis.py
"""

import base64
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(538)

FEATURES = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
            "crispedricewafer", "hard", "bar", "pluribus"]
LABEL = {
    "chocolate": "Chocolate", "fruity": "Fruit flavor", "caramel": "Caramel",
    "peanutyalmondy": "Peanut or almond", "nougat": "Nougat",
    "crispedricewafer": "Crisp rice or wafer", "hard": "Hard candy",
    "bar": "Bar shape", "pluribus": "Many per pack",
    "sugarpercent": "Sugar (bottom to top)", "pricepercent": "Price (bottom to top)",
}
NUMERIC = ["sugarpercent", "pricepercent"]
TERMS = FEATURES + NUMERIC
NOT_CANDY = {"One dime", "One quarter"}


def load():
    df = pd.read_csv(HERE / "data" / "candy-data.csv", encoding="utf-8")
    df["competitorname"] = df["competitorname"].str.replace("Õ", "'", regex=False)
    df = df.rename(columns={"competitorname": "name"})
    df["is_candy"] = ~df["name"].isin(NOT_CANDY)
    return df


def perm_p(a, b, n=20000):
    """Two-sided permutation p-value for a difference in means."""
    pooled = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    k = len(a)
    hits = 0
    for _ in range(n):
        RNG.shuffle(pooled)
        if abs(pooled[:k].mean() - pooled[k:].mean()) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


def boot_diff(a, b, n=5000):
    d = [RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5]).tolist()


def lifts(c):
    out = []
    for f in FEATURES:
        a = c.loc[c[f] == 1, "winpercent"].to_numpy()
        b = c.loc[c[f] == 0, "winpercent"].to_numpy()
        out.append({
            "key": f, "label": LABEL[f], "n_with": int(len(a)), "n_without": int(len(b)),
            "mean_with": a.mean(), "mean_without": b.mean(), "diff": a.mean() - b.mean(),
            "ci": boot_diff(a, b), "p": perm_p(a, b),
        })
    return out


def design(df, terms=TERMS):
    return sm.add_constant(df[terms].astype(float), has_constant="add")


def fit_ols(c, terms=TERMS):
    X, y = design(c, terms), c["winpercent"].astype(float)
    m = sm.OLS(y, X).fit()
    rob = m.get_robustcov_results(cov_type="HC3")
    return m, rob


def loo(c, terms=TERMS):
    X, y = design(c, terms).to_numpy(), c["winpercent"].to_numpy(float)
    preds = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        preds[i] = X[i] @ beta
    return preds


def forest_check(c):
    """Does a nonlinear model find structure the additive model misses?"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    X, y = c[TERMS].astype(float).to_numpy(), c["winpercent"].to_numpy(float)
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=538)
    pred = cross_val_predict(rf, X, y, cv=LeaveOneOut())
    return pred


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def pair_map(c):
    rows = []
    for i, a in enumerate(FEATURES):
        for j, b in enumerate(FEATURES):
            sub = c[(c[a] == 1) & (c[b] == 1)]
            rows.append({"a": a, "b": b, "n": int(len(sub)),
                         "mean": float(sub["winpercent"].mean()) if len(sub) else None,
                         "names": sub.sort_values("winpercent", ascending=False)["name"].tolist()})
    return rows


def recipe_space(c, beta, sugar, price):
    """Every combination of the nine yes/no traits, scored by the model,
    with how many real candies sit within one trait of it."""
    real = c[FEATURES].to_numpy(int)
    out = []
    for bits in itertools.product([0, 1], repeat=len(FEATURES)):
        b = np.array(bits)
        dist = np.abs(real - b).sum(axis=1)
        x = np.concatenate([[1.0], b, [sugar, price]])
        out.append({"bits": bits, "pred": float(x @ beta), "exact": int((dist == 0).sum()),
                    "near": int((dist <= 1).sum())})
    return out


def bootstrap_recipes(c, sugar, price, n=2000, min_near=3):
    """Refit on resampled candies and record which supported recipe ranks first."""
    X, y = design(c).to_numpy(), c["winpercent"].to_numpy(float)
    real = c[FEATURES].to_numpy(int)
    grid = np.array(list(itertools.product([0, 1], repeat=len(FEATURES))))
    dist = np.abs(real[None, :, :] - grid[:, None, :]).sum(axis=2)
    supported = (dist <= 1).sum(axis=1) >= min_near
    G = np.column_stack([np.ones(len(grid)), grid, np.full(len(grid), sugar), np.full(len(grid), price)])
    top = {}
    coefs = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        beta, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        coefs.append(beta)
        score = np.where(supported, G @ beta, -np.inf)
        k = tuple(grid[int(np.argmax(score))])
        top[k] = top.get(k, 0) + 1
    return top, np.array(coefs), supported, grid


def bits_to_names(bits):
    return [FEATURES[i] for i, v in enumerate(bits) if v]


LINEUP = [
    {"id": "crispy", "name": "The Crispy Cup", "role": "Our pick",
     "pitch": "A milk chocolate peanut butter cup with crisped rice folded into the filling. One cup per wrapper, sweet, not hard.",
     "on": ["chocolate", "peanutyalmondy", "crispedricewafer"], "sugar": 0.75, "price": 0.5},
    {"id": "cup", "name": "Plain peanut butter cup", "role": "Safe fallback",
     "pitch": "The same cup without the crisp. This recipe already exists and owns the top two spots.",
     "on": ["chocolate", "peanutyalmondy"], "sugar": 0.75, "price": 0.5},
    {"id": "hybrid", "name": "Chocolate and fruit", "role": "The host's example",
     "pitch": "Chocolate-covered chewy fruit. The simple model likes it, but only one real candy has tried the pair.",
     "on": ["chocolate", "fruity"], "sugar": 0.75, "price": 0.5},
    {"id": "sink", "name": "Everything bar", "role": "Stack every winner",
     "pitch": "Chocolate, caramel, peanut, nougat and crisp in one bar. Every add-in that scored well, all at once.",
     "on": ["chocolate", "caramel", "peanutyalmondy", "nougat", "crispedricewafer", "bar"], "sugar": 0.75, "price": 0.5},
    {"id": "fruit", "name": "Chewy fruit bites", "role": "Best without chocolate",
     "pitch": "Soft fruit chews, many per pack, never hard. The best fruit-only recipe in the data.",
     "on": ["fruity", "pluribus"], "sugar": 0.75, "price": 0.5},
]


def vec(on, sugar, price):
    return np.array([1.0] + [1.0 if f in on else 0.0 for f in FEATURES] + [sugar, price])


def nearest(c, on, sugar, k=4):
    b = np.array([1 if f in on else 0 for f in FEATURES])
    d = np.abs(c[FEATURES].to_numpy(int) - b).sum(axis=1)
    t = c.assign(dist=d, sd=(c.sugarpercent - sugar).abs()).sort_values(["dist", "sd", "winpercent"], ascending=[True, True, False])
    return [{"name": r.name_, "win": float(r.winpercent), "dist": int(r.dist)}
            for r in t.rename(columns={"name": "name_"}).head(k).itertuples()]


def weakest_pair(c, on):
    best = None
    for a, b in itertools.combinations(on, 2):
        n = int(((c[a] == 1) & (c[b] == 1)).sum())
        if best is None or n < best["n"]:
            best = {"a": a, "b": b, "n": n}
    return best


def main():
    df = load()
    c = df[df.is_candy].reset_index(drop=True)
    y = c["winpercent"].to_numpy(float)
    quarter = float(df.loc[df.name == "One quarter", "winpercent"].iloc[0])
    dime = float(df.loc[df.name == "One dime", "winpercent"].iloc[0])

    L = lifts(c)
    m, rob = fit_ols(c)
    beta = np.asarray(rob.params)
    cov = np.asarray(rob.cov_params())
    ci = np.asarray(rob.conf_int())
    s = float(np.sqrt(m.scale))

    p_loo = loo(c)
    base_pred = np.array([np.delete(y, i).mean() for i in range(len(y))])
    p_rf = forest_check(c)

    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=538)
    rf.fit(c[TERMS].astype(float).to_numpy(), y)

    # bootstrap the coefficients
    X = design(c).to_numpy()
    B = []
    for _ in range(4000):
        idx = RNG.integers(0, len(y), len(y))
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        B.append(b)
    B = np.array(B)
    term_idx = {t: i + 1 for i, t in enumerate(TERMS)}

    lineup = []
    for item in LINEUP:
        x = vec(item["on"], item["sugar"], item["price"])
        pred = float(x @ beta)
        se = float(np.sqrt(x @ cov @ x))
        boots = B @ x
        rfp = float(rf.predict([x[1:]])[0])
        lineup.append({**item,
                       "pred": pred, "lo": pred - 1.96 * se, "hi": pred + 1.96 * se,
                       "pi_lo": pred - 1.96 * np.sqrt(se ** 2 + s ** 2), "pi_hi": pred + 1.96 * np.sqrt(se ** 2 + s ** 2),
                       "boot_lo": float(np.percentile(boots, 2.5)), "boot_hi": float(np.percentile(boots, 97.5)),
                       "p_beat_quarter": float((boots > quarter).mean()),
                       "rf": rfp, "rank": int((y > pred).sum() + 1),
                       "weakest": weakest_pair(c, item["on"]), "nearest": nearest(c, item["on"], item["sugar"])})
    by = {r["id"]: r for r in lineup}
    crisp_b = B[:, term_idx["crispedricewafer"]]
    boot = {
        "n": len(B),
        "p_pos": {t: float((B[:, term_idx[t]] > 0).mean()) for t in TERMS},
        "crisp_gain_lo": float(np.percentile(crisp_b, 2.5)), "crisp_gain_hi": float(np.percentile(crisp_b, 97.5)),
        "p_crispy_beats_sink": float(((B @ vec(by["crispy"]["on"], .75, .5)) > (B @ vec(by["sink"]["on"], .75, .5))).mean()),
    }

    # hybrid: how the additive model scores the one real chocolate-and-fruit candy
    tp = c[(c.chocolate == 1) & (c.fruity == 1)].iloc[0]
    tp_idx = int(c.index[c.name == tp["name"]][0])
    hybrid = {"name": tp["name"], "win": float(tp.winpercent), "pred_loo": float(p_loo[tp_idx]),
              "rank": int((y > tp.winpercent).sum() + 1), "hard": int(tp.hard),
              "choc_mean": float(c.loc[(c.chocolate == 1) & (c.fruity == 0), "winpercent"].mean()),
              "fruit_mean": float(c.loc[(c.fruity == 1) & (c.chocolate == 0), "winpercent"].mean()),
              "cramers_v": float(np.sqrt(stats.chi2_contingency(pd.crosstab(c.chocolate, c.fruity), correction=False)[0] / len(c)))}

    ch = c[c.chocolate == 1].copy()
    ch["addins"] = ch[["caramel", "peanutyalmondy", "nougat", "crispedricewafer"]].sum(axis=1)
    addins = [{"count": int(k), "n": int(len(g)), "mean": float(g.winpercent.mean()),
               "candies": [{"name": r.name, "win": float(r.winpercent)} for r in g.sort_values("winpercent", ascending=False).itertuples()]}
              for k, g in ch.groupby("addins")]
    crisper = c[c.name == "Snickers Crisper"].index[0]

    price = {"r_all": float(stats.pearsonr(c.pricepercent, c.winpercent)[0]),
             "p_all": float(stats.pearsonr(c.pricepercent, c.winpercent)[1]),
             "r_choc": float(stats.pearsonr(ch.pricepercent, ch.winpercent)[0]),
             "p_choc": float(stats.pearsonr(ch.pricepercent, ch.winpercent)[1]),
             "r_sugar": float(stats.pearsonr(c.sugarpercent, c.winpercent)[0]),
             "choc_price": float(c.loc[c.chocolate == 1, "pricepercent"].mean()),
             "other_price": float(c.loc[c.chocolate == 0, "pricepercent"].mean())}

    candies = []
    for i, r in df.iterrows():
        row = {"name": r["name"], "t": [int(r[f]) for f in FEATURES], "sugar": float(r.sugarpercent),
               "price": float(r.pricepercent), "win": float(r.winpercent), "candy": bool(r.is_candy)}
        if r.is_candy:
            j = int(c.index[c.name == r["name"]][0])
            row["pred"] = float(p_loo[j])
            row["resid"] = float(y[j] - p_loo[j])
        candies.append(row)

    data = {
        "source": {"kaggle": "https://www.kaggle.com/datasets/fivethirtyeight/the-ultimate-halloween-candy-power-ranking",
                   "story": "https://fivethirtyeight.com/videos/the-ultimate-halloween-candy-power-ranking/",
                   "matchups": 269000, "voters": 8371, "year": 2017},
        "features": [{"key": f, "label": LABEL[f]} for f in FEATURES],
        "terms": [{"key": t, "label": LABEL[t]} for t in TERMS],
        "candies": candies,
        "quarter": quarter, "dime": dime,
        "lifts": L,
        "model": {"beta": beta.tolist(), "cov": cov.tolist(), "s": s, "n": int(m.nobs),
                  "r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj), "f_p": float(m.f_pvalue),
                  "loo_r2": float(r2(y, p_loo)), "loo_mae": float(np.abs(y - p_loo).mean()),
                  "base_mae": float(np.abs(y - base_pred).mean()),
                  "rf_loo_r2": float(r2(y, p_rf)), "rf_loo_mae": float(np.abs(y - p_rf).mean()),
                  "coefs": [{"key": t, "label": LABEL[t], "b": float(beta[i + 1]), "lo": float(ci[i + 1][0]),
                             "hi": float(ci[i + 1][1]), "p": float(rob.pvalues[i + 1])} for i, t in enumerate(TERMS)],
                  "const": float(beta[0])},
        "boot": boot,
        "pairs": pair_map(c),
        "addins": addins,
        "crisper": {"win": float(y[crisper]), "pred": float(p_loo[crisper])},
        "hybrid": hybrid,
        "price": price,
        "lineup": lineup,
        "summary": {"n": len(c), "below_quarter": int((y < quarter).sum()), "mean": float(y.mean()),
                    "n_choc": int(c.chocolate.sum()), "n_fruit": int(((c.fruity == 1) & (c.chocolate == 0)).sum()),
                    "n_neither": int(((c.fruity == 0) & (c.chocolate == 0)).sum()),
                    "top": c.sort_values("winpercent", ascending=False).head(3)["name"].tolist()},
    }

    (HERE / "build").mkdir(exist_ok=True)
    (HERE / "build" / "data.json").write_text(json.dumps(data, indent=1))
    print_summary(data)
    if (HERE / "template.html").exists():
        build_html(data)


def print_summary(d):
    m = d["model"]
    print(f"OLS R2 {m['r2']:.3f}  LOO R2 {m['loo_r2']:.3f}  LOO MAE {m['loo_mae']:.2f} (mean-only {m['base_mae']:.2f})  forest LOO R2 {m['rf_loo_r2']:.3f}")
    for r in d["lineup"]:
        w = r["weakest"]
        print(f"{r['name']:24s} pred {r['pred']:5.1f} [{r['lo']:.1f}, {r['hi']:.1f}] boot [{r['boot_lo']:.1f}, {r['boot_hi']:.1f}] forest {r['rf']:5.1f} rank {r['rank']:2d}  weakest {w['a'] if w else '-'}x{w['b'] if w else '-'} n={w['n'] if w else '-'}  near {[(n['name'], round(n['win'])) for n in r['nearest']]}")
    print("bootstrap P(effect>0):", {k: round(v, 3) for k, v in d["boot"]["p_pos"].items()})
    print("crisp gain 95%:", round(d["boot"]["crisp_gain_lo"], 1), round(d["boot"]["crisp_gain_hi"], 1), " P(crispy > sink)", round(d["boot"]["p_crispy_beats_sink"], 3))
    print("hybrid:", d["hybrid"])
    print("price:", {k: round(v, 3) for k, v in d["price"].items()})


def build_html(data):
    template = (HERE / "template.html").read_text()
    d3 = (HERE / "vendor" / "d3.min.js").read_text()
    font = base64.b64encode((HERE / "vendor" / "Geist-Variable.woff2").read_bytes()).decode()
    payload = json.dumps(data, separators=(",", ":"))
    html = (template.replace("/*__D3__*/", d3)
            .replace("/*__DATA__*/", f"const DATA = {payload};")
            .replace("__FONT_B64__", font))
    out = HERE / "site" / "index.html"
    out.write_text(html)
    print(f"wrote {out.relative_to(HERE)} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
