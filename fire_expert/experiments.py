"""Чисельні експерименти та побудова графіків для розрахунково-графічної роботи.

Запуск: python -m fire_expert.experiments
Результати зберігаються у каталозі figures/ (графіки) та results.json (числа).
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from .expert_system import ExpertSystem, classify
from .fuzzy import DEFUZZ_METHODS, defuzzify
from .knowledge_base import FIRE_INDEX, HUMIDITY, TEMPERATURE, WIND, DRY_DAYS
from .nesterov import nesterov_class, nesterov_index

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 100})
COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
ES = ExpertSystem()
M = ES.model
BASE = dict(temperature=30.0, humidity=30.0, wind=5.0, dry_days=10.0)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def idx(values, alg="mamdani", tnorm="min", method="centroid"):
    return M.infer(values, alg, tnorm, method).value


# ---------------------------------------------------------------- рисунки
def fig_membership():
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6))
    for ax, var in zip(axes.ravel(), (TEMPERATURE, HUMIDITY, WIND, DRY_DAYS)):
        x = var.universe()
        for k, (_, mf) in enumerate(var.terms.items()):
            ax.plot(x, mf(x), color=COLORS[k], lw=2, label=mf.label)
        ax.set_title(f"{var.label}, {var.unit}", fontsize=9)
        ax.set_ylim(-0.02, 1.08); ax.legend(fontsize=7, loc="center right")
    fig.tight_layout(); save(fig, "fig_mf_inputs.png")

    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    x = FIRE_INDEX.universe()
    for k, (_, mf) in enumerate(FIRE_INDEX.terms.items()):
        ax.plot(x, mf(x), color=COLORS[k], lw=2, label=mf.label)
    ax.set_xlabel("Індекс пожежної небезпеки, бал"); ax.set_ylabel("μ")
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(); save(fig, "fig_mf_output.png")


def _box(ax, x, y, w, h, text, fc="#e8f0fe", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#34495e", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, p, q):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", color="#34495e", lw=1.3))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.8, 4.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2)
    _box(ax, 0.2, 4.9, 2.4, 0.9, "Користувач\n(консольний інтерфейс)", "#fdebd0")
    _box(ax, 3.5, 4.9, 3.0, 0.9, "Введення та валідація\nвхідних даних")
    _box(ax, 7.2, 4.9, 2.6, 0.9, "Фазифікація")
    _box(ax, 0.2, 2.9, 3.4, 1.3, f"База знань\n(лінгвістичні змінні,\n{len(M.rules)} нечітких правил)", "#d5f5e3")
    _box(ax, 4.3, 2.9, 3.0, 1.3, "Механізм нечіткого виводу\n(активація правил,\nМамдані / Сугено)", "#d5f5e3")
    _box(ax, 7.9, 2.9, 1.9, 1.3, "Дефазифікація\nта класифікація")
    _box(ax, 0.2, 0.5, 3.4, 1.2, "Підсистема пояснення\n(правила, чутливість)", "#fcf3cf")
    _box(ax, 4.3, 0.5, 3.0, 1.2, "Висновок і\nрекомендації", "#fcf3cf")
    _box(ax, 7.9, 0.5, 1.9, 1.2, "Модуль порівняння\n(еталон Нестерова)", "#f5eef8", 7)
    _arrow(ax, (2.6, 5.35), (3.5, 5.35)); _arrow(ax, (6.5, 5.35), (7.2, 5.35))
    _arrow(ax, (8.5, 4.9), (6.4, 4.2)); _arrow(ax, (3.6, 3.55), (4.3, 3.55))
    _arrow(ax, (7.3, 3.55), (7.9, 3.55)); _arrow(ax, (8.8, 2.9), (6.4, 1.7))
    _arrow(ax, (5.8, 2.9), (5.8, 1.7)); _arrow(ax, (4.3, 1.1), (3.6, 1.1)); _arrow(ax, (1.9, 2.9), (1.9, 1.7))
    fig.tight_layout(); save(fig, "fig_architecture.png")


def fig_classes():
    fig, ax = plt.subplots(figsize=(6.8, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    items = [
        (0.1, 5.2, 3.0, 1.7, "MembershipFunction\n──────────────\nlabel, kind, params\n__call__(x)"),
        (3.5, 5.2, 3.0, 1.7, "LinguisticVariable\n──────────────\nkey, label, unit, lo, hi,\nterms; degree(term, x)"),
        (6.9, 5.2, 3.0, 1.7, "FuzzyRule\n──────────────\ncode, antecedents,\nconsequent, weight"),
        (3.5, 2.9, 3.0, 1.9, "FuzzyModel\n──────────────\ninputs, output, rules\nfuzzify(), activate(),\naggregate(), sugeno(), infer()"),
        (6.9, 2.9, 3.0, 1.9, "InferenceResult\n──────────────\nvalue, degrees,\nstrengths, aggregated"),
        (0.1, 0.2, 3.0, 2.1, "ExpertSystem\n──────────────\nmodel\nassess(), sensitivity(),\nexplain()"),
        (3.5, 0.2, 3.0, 2.1, "Assessment\n──────────────\nindex, level, level_name,\nrecommendation, fired,\ncomparison"),
        (6.9, 0.2, 3.0, 2.1, "FiredRule\n──────────────\ncode, text, alpha,\nexplanation"),
    ]
    for x, y, w, h, t in items:
        _box(ax, x, y, w, h, t, "#eaf2f8", 7.5)
    for p, q in [((3.1, 6.0), (3.5, 6.0)), ((5.0, 5.2), (5.0, 4.8)), ((6.5, 6.0), (6.9, 6.0)),
                 ((6.5, 3.85), (6.9, 3.85)), ((3.5, 1.3), (3.1, 1.3)), ((3.5, 3.4), (2.0, 2.3)),
                 ((6.5, 1.3), (6.9, 1.3))]:
        _arrow(ax, p, q)
    fig.tight_layout(); save(fig, "fig_classes.png")


def fig_mamdani_steps(values):
    res = M.infer(values, "mamdani", "min", "centroid")
    agg, parts = M.aggregate(res.strengths, "min")
    x = M.universe
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    ax = axes[0]
    for rule, shaped in parts:
        if shaped.max() > 0.01:
            ax.fill_between(x, shaped, alpha=0.35, label=rule.code)
    ax.set_title("Обрізані висновки правил", fontsize=9); ax.set_xlabel("Індекс небезпеки")
    ax.legend(fontsize=6, ncol=2); ax.set_ylabel("μ")
    ax = axes[1]
    ax.fill_between(x, agg, color="#aec7e8", label="Агрегована множина")
    marks = {"centroid": "Центр ваги", "bisector": "Бісектриса", "mom": "Сер. максимумів"}
    for k, (m, lab) in enumerate(marks.items()):
        v = defuzzify(x, agg, m)
        ax.axvline(v, color=COLORS[k + 1], lw=1.6, ls="--", label=f"{lab}: {v:.1f}")
    ax.set_title("Агрегація та дефазифікація", fontsize=9); ax.set_xlabel("Індекс небезпеки")
    ax.legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "fig_mamdani_steps.png")


def grid_surface(n=41):
    T = np.linspace(-10, 45, n); H = np.linspace(0, 100, n)
    Zm = np.zeros((n, n)); Zs = np.zeros((n, n))
    for i, h in enumerate(H):
        for j, t in enumerate(T):
            v = dict(BASE, temperature=t, humidity=h, dry_days=12.0)
            Zm[i, j] = idx(v, "mamdani"); Zs[i, j] = idx(v, "sugeno")
    return T, H, Zm, Zs


def fig_surfaces(T, H, Zm, Zs):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.9), layout="constrained")
    for ax, Z, title in ((axes[0], Zm, "Мамдані"), (axes[1], Zs, "Сугено")):
        cs = ax.contourf(T, H, Z, levels=np.linspace(0, 100, 11), cmap="YlOrRd")
        ax.set_title(title, fontsize=9); ax.set_xlabel("Температура, °C"); ax.grid(False)
    axes[0].set_ylabel("Вологість, %")
    fig.colorbar(cs, ax=axes[:2], shrink=0.9, pad=0.02, label="Індекс")
    c2 = axes[2].contourf(T, H, Zs - Zm, levels=np.linspace(-20, 20, 9), cmap="coolwarm")
    axes[2].set_title("Сугено − Мамдані", fontsize=9); axes[2].set_xlabel("Температура, °C"); axes[2].grid(False)
    fig.colorbar(c2, ax=axes[2], shrink=0.9, pad=0.02, label="Різниця")
    save(fig, "fig_surfaces.png")


def fig_dry_days():
    d = np.linspace(0, 30, 61)
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for k, h in enumerate((20, 40, 60)):
        for alg, ls in (("mamdani", "-"), ("sugeno", "--")):
            y = [idx(dict(BASE, humidity=h, dry_days=v), alg) for v in d]
            ax.plot(d, y, ls, color=COLORS[k], lw=1.8,
                    label=f"{'Мамдані' if alg == 'mamdani' else 'Сугено'}, φ = {h} %")
    ax.set_xlabel("Тривалість періоду без опадів, діб"); ax.set_ylabel("Індекс небезпеки")
    ax.legend(fontsize=7, ncol=3); fig.tight_layout(); save(fig, "fig_dry_days.png")


def fig_sensitivity():
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4)); out = {}
    for ax, var in zip(axes.ravel(), (TEMPERATURE, HUMIDITY, WIND, DRY_DAYS)):
        xs = np.linspace(var.lo, var.hi, 61)
        ym = [idx(dict(BASE, **{var.key: v}), "mamdani") for v in xs]
        ys = [idx(dict(BASE, **{var.key: v}), "sugeno") for v in xs]
        ax.plot(xs, ym, color=COLORS[0], lw=2, label="Мамдані"); ax.plot(xs, ys, "--", color=COLORS[2], lw=2, label="Сугено")
        ax.set_xlabel(f"{var.label}, {var.unit}", fontsize=8); ax.set_ylabel("Індекс", fontsize=8)
        ax.legend(fontsize=7); out[var.key] = round(float(max(ym) - min(ym)), 1)
    fig.tight_layout(); save(fig, "fig_sensitivity.png")
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def dataset(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    return [dict(temperature=float(rng.uniform(5, 40)), humidity=float(rng.uniform(10, 95)),
                 wind=float(rng.uniform(0, 15)), dry_days=float(rng.integers(0, 31))) for _ in range(n)]


def calibrate(index, ref, start=(20, 40, 60, 80)):
    """Покоординатний підбір порогів класів, що максимізує збіг із еталонними класами."""
    th = list(start)

    def acc(t):
        return float(((np.digitize(index, t) + 1) == ref).mean())
    best = acc(th)
    for _ in range(6):
        for i in range(4):
            lo = th[i - 1] + 1 if i > 0 else 3
            hi = th[i + 1] - 1 if i < 3 else 97
            for v in range(int(lo), int(hi) + 1):
                cand = th[:i] + [v] + th[i + 1:]
                a = acc(cand)
                if a > best + 1e-9:
                    best, th = a, cand
    return th, best


def fig_confusion(conf_m, conf_s):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    for ax, C, title in ((axes[0], conf_m, "Мамдані (центр ваги)"), (axes[1], conf_s, "Сугено")):
        ax.imshow(C, cmap="Blues"); ax.grid(False); ax.set_title(title, fontsize=9)
        ax.set_xticks(range(5)); ax.set_xticklabels(["I", "II", "III", "IV", "V"])
        ax.set_yticks(range(5)); ax.set_yticklabels(["I", "II", "III", "IV", "V"])
        ax.set_xlabel("Клас за нечіткою моделлю"); ax.set_ylabel("Клас за Нестеровим")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, int(C[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if C[i, j] > C.max() / 2 else "black")
    fig.tight_layout(); save(fig, "fig_confusion.png")


def fig_scatter(g, im, is_):
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    lg = np.log10(np.array(g) + 1)
    ax.scatter(lg, im, s=6, alpha=0.5, label="Мамдані", color=COLORS[0])
    ax.scatter(lg, is_, s=6, alpha=0.5, label="Сугено", color=COLORS[2])
    for lim in (300, 1000, 3000, 5000):
        ax.axvline(np.log10(lim + 1), color="gray", ls=":", lw=1)
    ax.set_xlabel("lg(G + 1), G - показник Нестерова"); ax.set_ylabel("Індекс небезпеки")
    ax.legend(fontsize=8); fig.tight_layout(); save(fig, "fig_scatter.png")


# ------------------------------------------------------------- розрахунки
SCENARIOS = [
    ("Після дощу", 18, 85, 2, 1), ("Тепла волога погода", 24, 70, 3, 4),
    ("Типове літо", 27, 50, 4, 8), ("Суха тепла погода", 29, 30, 5, 12),
    ("Спека з посухою", 34, 20, 4, 20), ("Спека, посуха і вітер", 36, 15, 14, 25),
    ("Холодна суха осінь", 8, 30, 6, 22), ("Сильний вітер, помірні умови", 25, 45, 15, 9),
]
VARIANTS = (("mamdani", "min", "centroid"), ("mamdani", "min", "bisector"), ("mamdani", "min", "mom"),
            ("mamdani", "prod", "centroid"), ("sugeno", "min", None), ("sugeno", "prod", None))


def main():
    res = {}
    fig_membership(); fig_architecture(); fig_classes()
    fig_mamdani_steps(dict(temperature=30, humidity=30, wind=4, dry_days=10))

    # E1. Типові сценарії.
    rows = []
    for name, t, h, w, d in SCENARIOS:
        v = dict(temperature=t, humidity=h, wind=w, dry_days=d)
        vals = [idx(v, a, tn, m or "centroid") for a, tn, m in VARIANTS]
        g = nesterov_index(t, h, d)
        rows.append(dict(name=name, inputs=[t, h, w, d], values=[round(x, 1) for x in vals],
                         level=classify(vals[0])[1], nesterov_g=round(g), nesterov_class=nesterov_class(g)))
    res["scenarios"] = rows

    # E2. Поверхні.
    T, H, Zm, Zs = grid_surface(); fig_surfaces(T, H, Zm, Zs)
    res["surface"] = dict(mean_abs_diff=round(float(np.abs(Zs - Zm).mean()), 2),
                          max_abs_diff=round(float(np.abs(Zs - Zm).max()), 2),
                          mamdani_min=round(float(Zm.min()), 1), mamdani_max=round(float(Zm.max()), 1),
                          sugeno_min=round(float(Zs.min()), 1), sugeno_max=round(float(Zs.max()), 1))
    fig_dry_days(); res["sensitivity_range"] = fig_sensitivity()

    # E5-E8. Порівняння з показником Нестерова на 1000 випадкових ситуаціях.
    data = dataset()
    g = [nesterov_index(v["temperature"], v["humidity"], v["dry_days"]) for v in data]
    cls_n = np.array([nesterov_class(x) for x in g])
    im = np.array([idx(v, "mamdani") for v in data]); is_ = np.array([idx(v, "sugeno") for v in data])
    cm = np.array([classify(x)[0] for x in im]); cs = np.array([classify(x)[0] for x in is_])
    conf = lambda c: np.array([[int(((cls_n == i) & (c == j)).sum()) for j in range(1, 6)] for i in range(1, 6)])
    cmat_m, cmat_s = conf(cm), conf(cs)
    fig_confusion(cmat_m, cmat_s); fig_scatter(g, im, is_)

    def stats(c, x):
        return dict(exact=round(float((c == cls_n).mean() * 100), 1),
                    within1=round(float((np.abs(c - cls_n) <= 1).mean() * 100), 1),
                    mae=round(float(np.abs(c - cls_n).mean()), 3),
                    spearman=round(spearman(x, g), 3))
    res["accuracy"] = dict(n=len(data), mamdani=stats(cm, im), sugeno=stats(cs, is_),
                           class_dist_nesterov=[int((cls_n == k).sum()) for k in range(1, 6)],
                           conf_mamdani=cmat_m.tolist(), conf_sugeno=cmat_s.tolist())

    # Калібрування порогів на окремій вибірці (seed=7) і перевірка на тестовій (seed=42).
    cal = dataset(1000, seed=7)
    g_cal = [nesterov_index(v["temperature"], v["humidity"], v["dry_days"]) for v in cal]
    ref_cal = np.array([nesterov_class(x) for x in g_cal])
    calib = {}
    for alg, arr in (("mamdani", im), ("sugeno", is_)):
        idx_cal = np.array([idx(v, alg) for v in cal])
        th, a_cal = calibrate(idx_cal, ref_cal)
        c_test = np.digitize(arr, th) + 1
        calib[alg] = dict(thresholds=th, train_exact=round(a_cal * 100, 1),
                          test_exact=round(float((c_test == cls_n).mean() * 100), 1),
                          test_within1=round(float((np.abs(c_test - cls_n) <= 1).mean() * 100), 1),
                          test_mae=round(float(np.abs(c_test - cls_n).mean()), 3))
    res["calibration"] = calib

    # Порівняння варіантів виводу відносно «Мамдані, min, центр ваги».
    variants = []
    for a, tn, m in VARIANTS:
        arr = np.array([idx(v, a, tn, m or "centroid") for v in data])
        variants.append(dict(alg=a, tnorm=tn, method=m, mean=round(float(arr.mean()), 2),
                             mad=round(float(np.abs(arr - im).mean()), 2),
                             agree=round(float((np.array([classify(x)[0] for x in arr]) == cm).mean() * 100), 1)))
    res["variants"] = variants
    for m in ("bisector", "som", "lom"):
        arr = np.array([idx(v, "mamdani", "min", m) for v in data[:200]])
        res.setdefault("defuzz_extra", {})[m] = round(float(np.abs(arr - im[:200]).mean()), 2)

    # Швидкодія.
    sample = data[:300]; timing = {}
    for alg in ("mamdani", "sugeno"):
        t0 = time.perf_counter()
        for v in sample: idx(v, alg)
        timing[alg] = round((time.perf_counter() - t0) / len(sample) * 1000, 3)
    res["timing_ms"] = timing

    # Частота спрацювання правил та повнота бази знань.
    fire = {r.code: 0 for r in M.rules}; mx = {r.code: 0.0 for r in M.rules}; min_total = 1e9
    for v in data:
        st = M.activate(M.fuzzify(v))
        min_total = min(min_total, sum(a for _, a in st))
        for r, a in st:
            if a > 0.01: fire[r.code] += 1
            mx[r.code] = max(mx[r.code], a)
    res["rules"] = dict(fire_pct={k: round(c / len(data) * 100, 1) for k, c in fire.items()},
                        min_total_strength=round(min_total, 3))
    # Додаткові дані для тексту записки.
    demo = dict(temperature=30, humidity=30, wind=4, dry_days=10)
    st = M.activate(M.fuzzify(demo))
    res["demo"] = dict(inputs=[30, 30, 4, 10],
                       values=[round(idx(demo, a, tn, m or "centroid"), 1) for a, tn, m in VARIANTS],
                       top=[[r.code, round(a, 2)] for r, a in sorted(st, key=lambda p: -p[1])[:5]])
    hot = (T[None, :] >= 28) & (H[:, None] <= 60)
    res["surface"]["hot_dry_mean_diff"] = round(float((Zs - Zm)[hot].mean()), 2)
    res["rule_table"] = [dict(code=r.code, text=ES.rule_text(r).replace("ЯКЩО ", "").replace(", ТО індекс небезпеки", " →"),
                              weight=r.weight) for r in M.rules]
    here = os.path.dirname(OUT)
    res["loc"] = {f: sum(1 for _ in open(os.path.join(here, "fire_expert", f), encoding="utf-8"))
                  for f in ("fuzzy.py", "knowledge_base.py", "expert_system.py", "nesterov.py", "experiments.py")}
    res["loc"]["main.py"] = sum(1 for _ in open(os.path.join(here, "main.py"), encoding="utf-8"))
    res["loc"]["tests"] = sum(1 for _ in open(os.path.join(here, "tests", "test_fire_expert.py"), encoding="utf-8"))
    res["drydays"] = {str(h): {str(dd): [round(idx(dict(BASE, humidity=h, dry_days=dd), a), 1) for a in ("mamdani", "sugeno")]
                               for dd in (0, 3, 6, 10, 15, 20, 30)} for h in (20, 40, 60)}
    res["wind_off"] = round(idx(dict(temperature=25, humidity=45, wind=4, dry_days=9)), 1)
    json.dump(res, open(os.path.join(os.path.dirname(OUT), "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
