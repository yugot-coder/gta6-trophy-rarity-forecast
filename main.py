"""GTA VI trophy-rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on Exophase, the percent
of tracked players who will hold it at the case's target, as five quantiles,
scored by mean pinball loss on the logit scale.

Method (all per trophy, in logit space, rates clamped to 0.01..99.99):

1.  Compress the history to its distinct points (a read is kept when any
    rate or the tracked count moved since the last kept read).  Exophase
    refreshes only every few hours, so long flat stretches carry no trend
    information and would swamp a fit.
2.  Fit trends against LOG TIME, x = ln(1 + hours since the release
    began): launch growth decelerates roughly like 1/t, so a line in
    log time extrapolates with built-in saturation, where a line in
    linear time would run away over a 4.5-day horizon.  Fits use the
    last 24 hours of distinct points (recent growth, not the explosive
    launch-night average), least-squares slopes (the house curve's
    estimator), shrunk toward the tier median slope.
    Two views of the future, blended in logit space:
    (a) earned/tracked decomposition -- forecast log(earned) and
        log(players_tracked) separately in log time and take the ratio.
        Earned counts only grow, tracked counts only grow, and their
        ratio is the rate; this keeps the forecast sane when a refresh
        floods the tracked pool with new players (dilution) while
        earned counts keep climbing.
    (b) direct logit-rate trend in log time.
3.  A mild pull of each median toward its tier's cross-sectional median.
4.  Structural caps, per quantile level (if X <= Y almost surely then
    quantile_tau(X) <= quantile_tau(Y)): the platinum is capped by every
    other trophy, and the story chain "Complete Chapter 1..4", "final
    mission", "100% completion" is capped monotonically along the chain.
    The chain is parsed from the trophies' own descriptions at runtime.
5.  Spread: a random-walk term (0.5*sqrt(H) logit, as the house
    baseline), plus trend-extrapolation uncertainty proportional to
    |slope| * (x_target - x_last), plus the trophy's residual noise,
    plus a refresh-shock term that is large on short horizons (a refresh
    can move every rate at once) and fades on long ones (refreshes
    average into the trend).

Deterministic: fixed iteration order, no randomness, no wall-clock use.
Standard library only.  Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuned constants (see scratch/NOTES.md for the simulation evidence) ---
WINDOW_HOURS = 24.0   # recent window for trend fits
RW_SIGMA = 0.5        # logit per sqrt-day, random-walk spread (house baseline)
TREND_UNC = 0.40      # spread per unit |log-time slope| * log-time gap
SHOCK_SIGMA = 0.55    # refresh-shock spread at horizon 0
TAU_SHOCK = 1.0       # days; decay scale of the refresh-shock spread term
SIGMA_FLOOR = 0.45    # minimum logit spread
SIGMA_CAP = 2.2       # maximum logit spread
BLEND_EN = 0.5        # weight of the earned/tracked view in the logit blend
TIER_PULL = 0.0       # pull of each median toward its tier median
SHRINK_K = 2.0        # shrinkage strength: w = n_obs / (n_obs + SHRINK_K)
SLOPE_CLIP = 4.0      # logit per log-hour, sanity clip on rate slopes
GROWTH_CLIP = 5.0     # log per log-hour, sanity clip on earned growth
TRACKED_CLIP = 2.0    # log per log-hour, sanity clip on tracked growth
MAX_FIT_POINTS = 48   # subsample cap for slope-fit pairwise / LS points
FIT_METHOD = "ls"      # "ls" (least squares, house-curve style) or "theil"
# Plausible week-one ceiling (as a fraction 0..1) for a trophy's rate,
# by tier: rates saturate toward these, so the trend is fitted in
# u = logit(rate / ceiling) space and can never extrapolate past them.
CEILING = {"platinum": 0.05, "gold": 0.30, "silver": 0.40, "bronze": 0.85}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def logit_frac(rate_pct, ceiling):
    """logit of the rate as a fraction of its ceiling."""
    f = min(1.0 - 1e-6, max(1e-6, (rate_pct / 100.0) / ceiling))
    return math.log(f / (1.0 - f))


def inv_logit_frac(u, ceiling):
    """Ceiling * sigmoid(u), in percent."""
    if u >= 0:
        f = 1.0 / (1.0 + math.exp(-u))
    else:
        e = math.exp(u)
        f = e / (1.0 + e)
    return clamp_pct(100.0 * ceiling * f)


def round4(x):
    """Round to 4 decimals, half up (as the house baseline rounds)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds (float)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def fit_slope(xs, ys):
    """Slope + residual scale of y against x.

    Least squares by default (the house curve's estimator, and the one
    that wins in simulation); Theil-Sen when FIT_METHOD says so.
    Returns (slope, resid_sd).
    """
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    if FIT_METHOD == "theil":
        return theil_sen(xs, ys)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-12:
        return 0.0, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    inter = my - slope * mx
    if n > 2:
        resid = math.sqrt(sum((y - (inter + slope * x)) ** 2
                              for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return slope, resid


def theil_sen(xs, ys):
    """Robust slope: median of pairwise slopes. Returns (slope, resid_sd).

    resid_sd is a robust residual scale (1.4826 * MAD) around the line
    through the median point, floored later by the caller.
    """
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    slopes = []
    for a in range(n):
        xa, ya = xs[a], ys[a]
        for b in range(a + 1, n):
            dx = xs[b] - xa
            if dx > 1e-9:
                slopes.append((ys[b] - ya) / dx)
    if not slopes:
        return 0.0, 0.0
    slope = median(slopes)
    mx, my = median(xs), median(ys)
    intercept = my - slope * mx
    resid = [abs(y - (intercept + slope * x)) for x, y in zip(xs, ys)]
    return slope, 1.4826 * median(resid)


def subsample(xs, ys, cap):
    """Evenly subsample (keeping first/last) so len <= cap, deterministically."""
    n = len(xs)
    if n <= cap:
        return xs, ys
    step = (n - 1) / (cap - 1)
    idx = sorted(set(int(round(k * step)) for k in range(cap)))
    return [xs[i] for i in idx], [ys[i] for i in idx]


def distinct_points(times, rates_by_trophy, tracked):
    """Indices of reads where something moved (first read always kept)."""
    keep = []
    prev_rates = None
    prev_n = None
    for j, t in enumerate(times):
        rates = rates_by_trophy[j]
        n = tracked[j]
        moved = prev_rates is None
        if not moved:
            if prev_n is not None and n is not None and prev_n > 0:
                if abs(n - prev_n) / prev_n >= 0.001:
                    moved = True
            if not moved:
                for a, b in zip(rates, prev_rates):
                    if a is None or b is None:
                        if a is not b:
                            moved = True
                            break
                    elif abs(a - b) >= 0.005:
                        moved = True
                        break
        if moved:
            keep.append(j)
            prev_rates = rates
            prev_n = n
    return keep


def chapter_rank(trophy):
    """Story-chain rank parsed from the description, or None.

    Chapter trophies rank by chapter number; the final mission ranks
    after every numbered chapter; 100% completion ranks after that.
    """
    desc = (trophy.get("description") or "").lower()
    m = re.search(r"complete chapter (\d+)", desc)
    if m:
        return int(m.group(1))
    if "final mission" in desc:
        return 99
    if "100% completion" in desc or "100% complete" in desc:
        return 100
    return None


def window_points(xs, ys, times, window_hours):
    """Restrict (xs, ys) to the recent window; fall back to the last 8
    points, then to all points, when the window holds too few."""
    if not xs:
        return xs, ys
    t_last = times[-1]
    cut = t_last - window_hours / 24.0
    wx = [x for x, t in zip(xs, times) if t >= cut]
    wy = [y for y, t in zip(ys, times) if t >= cut]
    if len(wx) >= 3:
        return wx, wy
    if len(xs) >= 3:
        return xs[-8:], ys[-8:]
    return xs, ys


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    n_tr = len(trophies)

    at_epoch = parse_time(case["at"]) if case.get("at") else 0.0
    target_epoch = parse_time(case["target"]) if case.get("target") else at_epoch

    # release start: game.release_date at 00:00Z, else first history read
    rel_epoch = None
    game = case.get("game") or {}
    if game.get("release_date"):
        try:
            rel_epoch = datetime.fromisoformat(
                game["release_date"] + "T00:00:00+00:00").timestamp()
        except Exception:
            rel_epoch = None

    # --- history series -------------------------------------------------
    times = []          # days relative to the checkpoint (<= 0)
    rates_rows = []     # per read, list of rates in trophy order
    tracked = []        # per read, players_tracked or None
    for read in history:
        try:
            t = (parse_time(read["at"]) - at_epoch) / 86400.0
        except Exception:
            continue
        rates = read.get("rates") or []
        if len(rates) < n_tr:
            rates = list(rates) + [None] * (n_tr - len(rates))
        times.append(t)
        rates_rows.append(list(rates[:n_tr]))
        tracked.append(read.get("players_tracked"))

    if rel_epoch is None:
        rel_epoch = at_epoch + (times[0] * 86400.0 if times else 0.0)

    def logtime(epoch_seconds):
        hours = max(0.0, (epoch_seconds - rel_epoch) / 3600.0)
        return math.log1p(hours)

    x_target = logtime(target_epoch)

    # last known rate per trophy (searching back over nulls)
    last_rate = []
    for i in range(n_tr):
        r = None
        for row in reversed(rates_rows):
            if row[i] is not None:
                r = row[i]
                break
        last_rate.append(r)

    # tier fallback for trophies never observed: tier median of last rates
    tier_last = {}
    for i, tr in enumerate(trophies):
        if last_rate[i] is not None:
            tier_last.setdefault(tr.get("tier", "bronze"), []).append(last_rate[i])
    tier_fallback = {tier: median(vals) for tier, vals in tier_last.items()}
    global_fallback = median([r for r in last_rate if r is not None]) if any(
        r is not None for r in last_rate
    ) else 1.0
    for i, tr in enumerate(trophies):
        if last_rate[i] is None:
            last_rate[i] = tier_fallback.get(
                tr.get("tier", "bronze"), global_fallback
            )

    # --- distinct (changed) points --------------------------------------
    keep_idx = distinct_points(times, rates_rows, tracked)
    if not keep_idx and times:
        keep_idx = [len(times) - 1]
    d_times = [times[j] for j in keep_idx]
    d_x = [logtime(at_epoch + t * 86400.0) for t in d_times]
    x_last = d_x[-1] if d_x else logtime(at_epoch)
    dx_target = max(0.0, x_target - x_last)

    # tracked-count trend in log time (recent window)
    n_xs, n_ys, n_ts = [], [], []
    for k, j in enumerate(keep_idx):
        n = tracked[j]
        if n is not None and n > 0:
            n_xs.append(d_x[k])
            n_ys.append(math.log(n))
            n_ts.append(d_times[k])
    n_xs, n_ys = window_points(n_xs, n_ys, n_ts, WINDOW_HOURS)
    n_xs, n_ys = subsample(n_xs, n_ys, MAX_FIT_POINTS)
    if len(n_xs) >= 3:
        g_n, _ = fit_slope(n_xs, n_ys)
    elif len(n_xs) == 2 and n_xs[1] > n_xs[0]:
        g_n = (n_ys[1] - n_ys[0]) / (n_xs[1] - n_xs[0])
    else:
        g_n = 0.0
    g_n = min(TRACKED_CLIP, max(0.0, g_n))
    n_last = None
    for v in reversed(tracked):
        if v is not None and v > 0:
            n_last = float(v)
            break

    # --- per-trophy fits ------------------------------------------------
    med_logit = [0.0] * n_tr
    sigma = [0.0] * n_tr
    slopes = []
    g_earned_list = []
    n_obs_list = []
    ceilings = []
    for i, tr in enumerate(trophies):
        tier = tr.get("tier", "bronze")
        ceil = CEILING.get(tier, 0.85)
        # the ceiling must sit above the last observed rate (a trophy
        # already at 100% keeps a ceiling of ~1)
        ceil = max(ceil, min(0.999, last_rate[i] / 100.0 * 1.5 + 0.01))
        if last_rate[i] >= 99.99:
            ceil = 0.9999
        ceilings.append(ceil)
        # rate-trend series in ceiling space: u = logit(rate / ceiling)
        xs, ys, ts = [], [], []
        for k, j in enumerate(keep_idx):
            r = rates_rows[j][i]
            if r is not None:
                xs.append(d_x[k])
                ys.append(logit_frac(r, ceil))
                ts.append(d_times[k])
        n_obs_list.append(len(xs))
        xs, ys = window_points(xs, ys, ts, WINDOW_HOURS)
        xs, ys = subsample(xs, ys, MAX_FIT_POINTS)
        if len(xs) >= 3:
            slope, resid = fit_slope(xs, ys)
        elif len(xs) == 2 and xs[1] > xs[0]:
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            resid = 0.0
        else:
            slope, resid = 0.0, 0.0
        slope = min(SLOPE_CLIP, max(-SLOPE_CLIP, slope))
        slopes.append(slope)

        # earned-count series: log(rate * N) vs log time
        exs, eys, ets = [], [], []
        for k, j in enumerate(keep_idx):
            r = rates_rows[j][i]
            n = tracked[j]
            if r is not None and n is not None and n > 0 and r > 0:
                exs.append(d_x[k])
                eys.append(math.log(r / 100.0 * n))
                ets.append(d_times[k])
        exs, eys = window_points(exs, eys, ets, WINDOW_HOURS)
        exs, eys = subsample(exs, eys, MAX_FIT_POINTS)
        if len(exs) >= 3:
            g_e, _ = fit_slope(exs, eys)
        elif len(exs) == 2 and exs[1] > exs[0]:
            g_e = (eys[1] - eys[0]) / (exs[1] - exs[0])
        else:
            g_e = 0.0
        g_e = min(GROWTH_CLIP, max(-0.2, g_e))
        g_earned_list.append(g_e)

        # spread (uses the raw per-trophy slope; shrinkage below only
        # moves the median)
        rw = RW_SIGMA * math.sqrt(horizon_days)
        trend_unc = TREND_UNC * abs(slope) * dx_target
        shock = SHOCK_SIGMA * math.exp(-horizon_days / TAU_SHOCK)
        resid_term = min(1.0, max(0.10, resid))
        extra = 0.25 if len(xs) < 3 else 0.0
        s = math.sqrt(rw * rw + trend_unc * trend_unc
                      + resid_term * resid_term + shock * shock
                      + extra * extra)
        sigma[i] = min(SIGMA_CAP, max(SIGMA_FLOOR, s))

    # --- shrinkage toward tier medians, then medians --------------------
    tier_slopes = {}
    for i, tr in enumerate(trophies):
        tier_slopes.setdefault(tr.get("tier", "bronze"), []).append(slopes[i])
    global_slope = median(slopes) if slopes else 0.0
    tier_med = {t: (median(v) if len(v) >= 3 else global_slope)
                for t, v in tier_slopes.items()}
    tier_ge = {}
    for i, tr in enumerate(trophies):
        tier_ge.setdefault(tr.get("tier", "bronze"), []).append(g_earned_list[i])
    global_ge = median(g_earned_list) if g_earned_list else 0.0
    tier_ge_med = {t: (median(v) if len(v) >= 3 else global_ge)
                   for t, v in tier_ge.items()}

    for i, tr in enumerate(trophies):
        tier = tr.get("tier", "bronze")
        n_obs = n_obs_list[i]
        w = n_obs / (n_obs + SHRINK_K) if n_obs else 0.0
        slope_s = w * slopes[i] + (1.0 - w) * tier_med.get(tier, global_slope)
        ge_s = w * g_earned_list[i] + (1.0 - w) * tier_ge_med.get(tier, global_ge)
        # direct rate-trend view in ceiling space (saturating)
        u_last = logit_frac(last_rate[i], ceilings[i])
        rate_rt = inv_logit_frac(u_last + slope_s * dx_target, ceilings[i])
        # earned/tracked view in log time, capped at the ceiling
        rate_en = None
        if n_last is not None and last_rate[i] > 0:
            e_last = last_rate[i] / 100.0 * n_last
            log_e_f = math.log(e_last) + ge_s * dx_target
            log_n_f = math.log(n_last) + g_n * dx_target
            e_f = math.exp(log_e_f)
            if e_f < e_last:
                e_f = e_last
            n_f = math.exp(log_n_f)
            if n_f < n_last:
                n_f = n_last
            rate_en = clamp_pct(100.0 * e_f / n_f)
            cap_pct = 100.0 * ceilings[i]
            if rate_en > cap_pct:
                rate_en = cap_pct
        if rate_en is not None:
            med_logit[i] = (BLEND_EN * logit(rate_en)
                            + (1.0 - BLEND_EN) * logit(rate_rt))
        else:
            med_logit[i] = logit(rate_rt)

    # --- tier cross-sectional pull --------------------------------------
    tier_meds = {}
    for i, tr in enumerate(trophies):
        tier_meds.setdefault(tr.get("tier", "bronze"), []).append(med_logit[i])
    tier_med_logit = {t: median(v) for t, v in tier_meds.items()}
    for i, tr in enumerate(trophies):
        tm = tier_med_logit.get(tr.get("tier", "bronze"))
        if tm is not None:
            med_logit[i] = ((1.0 - TIER_PULL) * med_logit[i]
                            + TIER_PULL * tm)

    # --- quantiles -------------------------------------------------------
    rows = []
    for i, tr in enumerate(trophies):
        qs = []
        for z in Z:
            q = inv_logit(med_logit[i] + z * sigma[i])
            qs.append(round4(clamp_pct(q)))
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})

    # --- structural caps, per quantile level -----------------------------
    by_id = {row["id"]: row for row in rows}
    plat = [tr for tr in trophies if tr.get("tier") == "platinum"]
    if plat:
        others = [r for r in rows if r["id"] != plat[0]["id"]]
        if others:
            prow = by_id[plat[0]["id"]]
            for k in range(5):
                cap = min(r["q"][k] for r in others)
                if prow["q"][k] > cap:
                    prow["q"][k] = cap
            for k in range(1, 5):
                if prow["q"][k] < prow["q"][k - 1]:
                    prow["q"][k] = prow["q"][k - 1]
    chain = []
    for tr in trophies:
        rank = chapter_rank(tr)
        if rank is not None:
            chain.append((rank, tr["id"]))
    chain.sort()
    for pos in range(1, len(chain)):
        prev_row = by_id[chain[pos - 1][1]]
        cur_row = by_id[chain[pos][1]]
        for k in range(5):
            if cur_row["q"][k] > prev_row["q"][k]:
                cur_row["q"][k] = prev_row["q"][k]
        for k in range(1, 5):
            if cur_row["q"][k] < cur_row["q"][k - 1]:
                cur_row["q"][k] = cur_row["q"][k - 1]

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
