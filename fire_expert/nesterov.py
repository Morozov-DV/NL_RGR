"""Показник пожежної небезпеки за В. Г. Нестеровим (еталон для порівняння)."""
import math

# Верхні межі показника для класів I-IV; понад 5000 - клас V.
NESTEROV_LIMITS = (300.0, 1000.0, 3000.0, 5000.0)


def dew_point(t: float, rh: float) -> float:
    """Точка роси за наближенням Магнуса з коефіцієнтами Алдучова-Ескріджа (t у °C, rh у %)."""
    a, b = 17.625, 243.04
    gamma = a * t / (b + t) + math.log(max(rh, 1.0) / 100.0)
    return b * gamma / (a - gamma)


def nesterov_index(t: float, rh: float, dry_days: float) -> float:
    """Спрощений показник G = N * t * (t - d) для сталих умов протягом N діб.
    Для t <= 0 °C показник дорівнює нулю."""
    if t <= 0:
        return 0.0
    return dry_days * t * max(t - dew_point(t, rh), 0.0)


def nesterov_class(g: float) -> int:
    """Клас пожежної небезпеки I-V (повертається число 1-5)."""
    for i, limit in enumerate(NESTEROV_LIMITS, start=1):
        if g <= limit:
            return i
    return 5
