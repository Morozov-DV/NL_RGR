"""Експертна система: оцінювання, пояснення результату та рекомендації."""
from dataclasses import dataclass, field
from typing import Dict, List

from .fuzzy import FuzzyModel
from .knowledge_base import build_model

# Пороги класифікації індексу: (верхня межа, номер класу, назва).
LEVELS = ((20, 1, "Мінімальна"), (40, 2, "Низька"), (60, 3, "Помірна"),
          (80, 4, "Висока"), (101, 5, "Надзвичайна"))

RECOMMENDATIONS = {
    1: "Особливих заходів не потрібно; здійснюється звичайний моніторинг.",
    2: "Посилити інформування відвідувачів лісу про правила пожежної безпеки.",
    3: "Організувати періодичне патрулювання, перевірити готовність протипожежних засобів.",
    4: "Обмежити відвідування лісу, посилити патрулювання, підготувати до виїзду пожежну техніку.",
    5: "Заборонити відвідування лісу та розведення вогню, перевести служби в режим підвищеної готовності.",
}


def classify(index: float) -> tuple:
    """Повертає (номер класу, назва) за значенням індексу."""
    for upper, number, name in LEVELS:
        if index < upper:
            return number, name
    return 5, "Надзвичайна"


@dataclass
class FiredRule:
    code: str
    text: str
    alpha: float
    explanation: str


@dataclass
class Assessment:
    """Повний результат оцінювання разом із поясненням."""
    inputs: Dict[str, float]
    algorithm: str
    index: float
    level: int
    level_name: str
    recommendation: str
    degrees: Dict[str, Dict[str, float]]
    fired: List[FiredRule]
    comparison: Dict[str, float] = field(default_factory=dict)


class ExpertSystem:
    """Експертна система з базою знань, механізмом виводу та підсистемою пояснення."""

    def __init__(self, model: FuzzyModel = None):
        self.model = model or build_model()

    def rule_text(self, rule) -> str:
        """Текстове подання правила у вигляді «ЯКЩО ... ТО ...»."""
        parts = []
        for var, term in rule.antecedents:
            v = self.model.inputs[var]
            parts.append(f"{v.label.lower()} - {v.terms[term].label.lower()}")
        glue = " І " if rule.operator == "and" else " АБО "
        cons = self.model.output.terms[rule.consequent].label.lower()
        return f"ЯКЩО {glue.join(parts)}, ТО індекс небезпеки {cons}"

    def assess(self, temperature, humidity, wind, dry_days, algorithm="mamdani",
               tnorm="min", method="centroid") -> Assessment:
        values = {"temperature": temperature, "humidity": humidity,
                  "wind": wind, "dry_days": dry_days}
        res = self.model.infer(values, algorithm, tnorm, method)
        level, name = classify(res.value)
        fired = [FiredRule(r.code, self.rule_text(r), round(a, 3), r.explanation)
                 for r, a in sorted(res.strengths, key=lambda p: -p[1]) if a > 0.01]
        comparison = {}
        for alg in ("mamdani", "sugeno"):
            comparison[alg] = self.model.infer(values, alg, tnorm, method).value
        return Assessment(values, algorithm, res.value, level, name,
                          RECOMMENDATIONS[level], res.degrees, fired, comparison)

    def sensitivity(self, a: Assessment, fraction: float = 0.1) -> Dict[str, float]:
        """Зміна індексу при збільшенні кожного входу на fraction діапазону
        (підсистема пояснення: який чинник найсильніше впливає на результат)."""
        out = {}
        for key, var in self.model.inputs.items():
            shifted = dict(a.inputs)
            shifted[key] = min(var.hi, shifted[key] + fraction * (var.hi - var.lo))
            new = self.model.infer(shifted, a.algorithm).value
            out[key] = new - a.index
        return out

    def explain(self, a: Assessment, top: int = 5) -> str:
        """Текстовий звіт: вхідні дані, спрацювання правил, висновок, рекомендація."""
        lines = ["ВХІДНІ ДАНІ:"]
        for key, var in self.model.inputs.items():
            lines.append(f"  {var.label}: {a.inputs[key]} {var.unit}")
        lines.append("ФАЗИФІКАЦІЯ (ненульові ступені належності):")
        for key, terms in a.degrees.items():
            var = self.model.inputs[key]
            shown = ", ".join(f"{var.terms[t].label} {d:.2f}" for t, d in terms.items() if d > 0)
            lines.append(f"  {var.label}: {shown}")
        lines.append(f"ПРАВИЛА, ЩО СПРАЦЮВАЛИ (перші {top}):")
        for fr in a.fired[:top]:
            lines.append(f"  {fr.code} (сила {fr.alpha:.2f}): {fr.text}. {fr.explanation}")
        lines.append(f"РЕЗУЛЬТАТ ({a.algorithm}): індекс {a.index:.1f}; "
                     f"клас небезпеки {a.level} - {a.level_name}")
        sens = self.sensitivity(a)
        key = max(sens, key=lambda k: abs(sens[k]))
        lines.append(f"ВИЗНАЧАЛЬНИЙ ЧИННИК: {self.model.inputs[key].label.lower()} "
                     f"(+10% діапазону змінює індекс на {sens[key]:+.1f})")
        lines.append(f"РЕКОМЕНДАЦІЯ: {a.recommendation}")
        return "\n".join(lines)
