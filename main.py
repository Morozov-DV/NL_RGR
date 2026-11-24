"""Консольний інтерфейс експертної системи оцінки ризику лісових пожеж.

Приклади:
    python main.py --temperature 34 --humidity 18 --wind 10 --dry-days 20
    python main.py --temperature 30 --humidity 30 --wind 4 --dry-days 10 --algorithm sugeno
    python main.py --interactive
"""
import argparse
import sys

from fire_expert.expert_system import ExpertSystem
from fire_expert.fuzzy import DEFUZZ_METHODS
from fire_expert.knowledge_base import build_model

PROMPTS = (("temperature", "Температура повітря, °C", 25.0),
           ("humidity", "Відносна вологість, %", 50.0),
           ("wind", "Швидкість вітру, м/с", 4.0),
           ("dry_days", "Діб без значних опадів", 5.0))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Експертна система оцінки ризику лісових пожеж (нечітка логіка)")
    p.add_argument("--temperature", type=float, help="температура повітря, °C")
    p.add_argument("--humidity", type=float, help="відносна вологість, %%")
    p.add_argument("--wind", type=float, help="швидкість вітру, м/с")
    p.add_argument("--dry-days", type=float, help="кількість діб без значних опадів")
    p.add_argument("--algorithm", choices=("mamdani", "sugeno"), default="mamdani")
    p.add_argument("--tnorm", choices=("min", "prod"), default="min", help="операція «І»")
    p.add_argument("--method", choices=DEFUZZ_METHODS, default="centroid", help="метод дефазифікації")
    p.add_argument("--interactive", action="store_true", help="ввести значення у діалозі")
    return p.parse_args(argv)


def ask(prompt, default):
    """Запитує число; порожній ввід означає значення за замовчуванням."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip().replace(",", ".")
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Введіть число.")


def main(argv=None):
    args = parse_args(argv)
    es = ExpertSystem(build_model())
    given = [args.temperature, args.humidity, args.wind, args.dry_days]
    if args.interactive or any(v is None for v in given):
        values = [ask(text, default) if cur is None else cur
                  for (key, text, default), cur in zip(PROMPTS, given)]
    else:
        values = given
    try:
        result = es.assess(*values, algorithm=args.algorithm, tnorm=args.tnorm, method=args.method)
    except ValueError as err:                       # значення поза діапазоном
        print(f"Помилка: {err}")
        return 1
    print(es.explain(result))
    print(f"ПОРІВНЯННЯ: Мамдані {result.comparison['mamdani']:.1f}; Сугено {result.comparison['sugeno']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
