"""Власна реалізація нечіткого логічного виводу (алгоритми Мамдані та Сугено).

Модуль залежить лише від NumPy й не використовує готових бібліотек нечіткої логіки.
Складається з функцій належності, лінгвістичних змінних, правил та механізму виводу.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 1. Функції належності
# ---------------------------------------------------------------------------


def _ramp_up(x, a, b):
    """Висхідна гілка: 0 до a, 1 від b (при a == b - «сходинка»)."""
    if b <= a:
        return (x >= b).astype(float)
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def _ramp_down(x, c, d):
    """Низхідна гілка: 1 до c, 0 від d (при c == d - «сходинка»)."""
    if d <= c:
        return (x <= c).astype(float)
    return np.clip((d - x) / (d - c), 0.0, 1.0)


def trapmf(x, a, b, c, d):
    """Трапецієвидна функція належності з параметрами a <= b <= c <= d."""
    x = np.asarray(x, dtype=float)
    return np.minimum(_ramp_up(x, a, b), _ramp_down(x, c, d))


def trimf(x, a, b, c):
    """Трикутна функція належності - окремий випадок трапеції (b == c)."""
    return trapmf(x, a, b, b, c)


@dataclass(frozen=True)
class MembershipFunction:
    """Терм лінгвістичної змінної: підпис, тип ('tri' або 'trap') і параметри."""
    label: str
    kind: str
    params: Tuple[float, ...]

    def __call__(self, x):
        if self.kind == "tri":
            return trimf(x, *self.params)
        if self.kind == "trap":
            return trapmf(x, *self.params)
        raise ValueError(f"Невідомий тип функції належності: {self.kind}")


@dataclass
class LinguisticVariable:
    """Лінгвістична змінна: назва, одиниця виміру, діапазон і набір термів."""
    key: str
    label: str
    unit: str
    lo: float
    hi: float
    terms: Dict[str, MembershipFunction]

    def universe(self, n: int = 1001) -> np.ndarray:
        """Дискретний універсум (рівномірна сітка з n вузлів)."""
        return np.linspace(self.lo, self.hi, n)

    def degree(self, term: str, x: float) -> float:
        """Ступінь належності чіткого значення x терму term."""
        return float(self.terms[term](np.array([x]))[0])


# ---------------------------------------------------------------------------
# 2. Правила та дефазифікація
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuzzyRule:
    """Продукційне правило «ЯКЩО умови ТО висновок» з вагою та поясненням."""
    code: str
    antecedents: Tuple[Tuple[str, str], ...]   # пари (змінна, терм)
    consequent: str                            # терм вихідної змінної
    weight: float = 1.0                        # ступінь довіри експерта
    operator: str = "and"                      # 'and' або 'or'
    explanation: str = ""                      # обґрунтування правила


DEFUZZ_METHODS = ("centroid", "bisector", "mom", "som", "lom")


def defuzzify(x: np.ndarray, mu: np.ndarray, method: str = "centroid") -> float:
    """Перетворює агреговану нечітку множину на чітке число."""
    total = mu.sum()
    if total <= 0:
        raise ValueError("Агрегована множина порожня: жодне правило не спрацювало")
    if method == "centroid":                  # центр ваги
        return float((x * mu).sum() / total)
    if method == "bisector":                  # бісектриса: ділить площу навпіл
        cum = np.cumsum(mu)
        return float(x[np.argmax(cum >= total / 2.0)])
    top = x[np.isclose(mu, mu.max())]         # точки, де досягається максимум
    if method == "mom":                       # середнє з максимумів
        return float(top.mean())
    if method == "som":                       # найменший з максимумів
        return float(top.min())
    if method == "lom":                       # найбільший з максимумів
        return float(top.max())
    raise ValueError(f"Невідомий метод дефазифікації: {method}")


# ---------------------------------------------------------------------------
# 3. Механізм виводу
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Результат одного виводу з усіма проміжними даними (для пояснення)."""
    value: float
    algorithm: str
    degrees: Dict[str, Dict[str, float]]
    strengths: List[Tuple[FuzzyRule, float]]
    aggregated: Optional[np.ndarray] = None    # лише для алгоритму Мамдані


class FuzzyModel:
    """Нечітка модель з кількома входами та одним виходом."""

    def __init__(self, inputs: Dict[str, LinguisticVariable], output: LinguisticVariable,
                 rules: List[FuzzyRule], sugeno_constants: Dict[str, float],
                 n_points: int = 1001):
        self.inputs, self.output, self.rules = inputs, output, rules
        self.sugeno_constants = sugeno_constants
        self.universe = output.universe(n_points)
        # Функції належності вихідних термів обчислюються один раз.
        self.output_mfs = {t: mf(self.universe) for t, mf in output.terms.items()}

    # --- перевірка та фазифікація ------------------------------------------
    def validate(self, values: Dict[str, float]) -> None:
        for key, var in self.inputs.items():
            if key not in values:
                raise ValueError(f"Не задано значення змінної «{var.label}»")
            if not var.lo <= float(values[key]) <= var.hi:
                raise ValueError(
                    f"Значення «{var.label}» = {values[key]} поза діапазоном "
                    f"[{var.lo}; {var.hi}] {var.unit}")

    def fuzzify(self, values: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Фазифікація: ступені належності до всіх термів усіх змінних."""
        return {k: {t: var.degree(t, float(values[k])) for t in var.terms}
                for k, var in self.inputs.items()}

    # --- активація правил --------------------------------------------------
    @staticmethod
    def _and(vals, tnorm):
        return min(vals) if tnorm == "min" else float(np.prod(vals))

    @staticmethod
    def _or(vals, tnorm):
        if tnorm == "min":
            return max(vals)
        res = 0.0
        for v in vals:                          # ймовірнісна сума
            res = res + v - res * v
        return res

    def activate(self, degrees, tnorm: str = "min"):
        """Сила спрацювання кожного правила з урахуванням його ваги."""
        out = []
        for rule in self.rules:
            mus = [degrees[v][t] for v, t in rule.antecedents]
            raw = self._and(mus, tnorm) if rule.operator == "and" else self._or(mus, tnorm)
            out.append((rule, raw * rule.weight))
        return out

    # --- алгоритм Мамдані ----------------------------------------------------
    def aggregate(self, strengths, tnorm: str = "min"):
        """Імплікація (обрізання або масштабування) та агрегація максимумом."""
        agg = np.zeros_like(self.universe)
        parts = []
        for rule, alpha in strengths:
            mf = self.output_mfs[rule.consequent]
            shaped = np.fmin(alpha, mf) if tnorm == "min" else alpha * mf
            parts.append((rule, shaped))
            agg = np.fmax(agg, shaped)
        return agg, parts

    # --- алгоритм Сугено нульового порядку -----------------------------------
    def sugeno(self, strengths) -> float:
        total = sum(a for _, a in strengths)
        if total <= 0:
            raise ValueError("Жодне правило не спрацювало")
        return float(sum(a * self.sugeno_constants[r.consequent] for r, a in strengths) / total)

    # --- загальна точка входу ------------------------------------------------
    def infer(self, values, algorithm="mamdani", tnorm="min", method="centroid") -> InferenceResult:
        self.validate(values)
        degrees = self.fuzzify(values)
        strengths = self.activate(degrees, tnorm)
        if algorithm == "mamdani":
            agg, _ = self.aggregate(strengths, tnorm)
            return InferenceResult(defuzzify(self.universe, agg, method), algorithm,
                                   degrees, strengths, agg)
        if algorithm == "sugeno":
            return InferenceResult(self.sugeno(strengths), algorithm, degrees, strengths)
        raise ValueError(f"Невідомий алгоритм: {algorithm}")
