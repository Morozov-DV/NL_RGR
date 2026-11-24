"""База знань: лінгвістичні змінні та правила оцінки пожежної небезпеки лісу."""
from .fuzzy import FuzzyModel, FuzzyRule, LinguisticVariable, MembershipFunction as MF

# Вхідні змінні (діапазони відповідають кліматичним умовам помірного поясу).
TEMPERATURE = LinguisticVariable("temperature", "Температура повітря", "°C", -10.0, 45.0, {
    "cool": MF("Прохолодна", "trap", (-10, -10, 10, 18)),
    "warm": MF("Тепла", "tri", (14, 22, 30)),
    "hot": MF("Спекотна", "trap", (26, 33, 45, 45)),
})
HUMIDITY = LinguisticVariable("humidity", "Відносна вологість повітря", "%", 0.0, 100.0, {
    "dry": MF("Низька", "trap", (0, 0, 20, 35)),
    "moderate": MF("Помірна", "tri", (25, 45, 65)),
    "humid": MF("Висока", "trap", (55, 75, 100, 100)),
})
WIND = LinguisticVariable("wind", "Швидкість вітру", "м/с", 0.0, 20.0, {
    "calm": MF("Слабкий", "trap", (0, 0, 2, 5)),
    "moderate": MF("Помірний", "tri", (3, 7, 11)),
    "strong": MF("Сильний", "trap", (9, 13, 20, 20)),
})
DRY_DAYS = LinguisticVariable("dry_days", "Тривалість періоду без значних опадів", "діб", 0.0, 30.0, {
    "recent": MF("Короткий", "trap", (0, 0, 2, 6)),
    "medium": MF("Середній", "tri", (3, 8, 15)),
    "long": MF("Тривалий", "trap", (10, 18, 30, 30)),
})

# Вихідна змінна - індекс пожежної небезпеки (0-100).
FIRE_INDEX = LinguisticVariable("fire_index", "Індекс пожежної небезпеки", "бал", 0.0, 100.0, {
    "minimal": MF("Мінімальний", "trap", (0, 0, 10, 22)),
    "low": MF("Низький", "tri", (15, 30, 45)),
    "moderate": MF("Помірний", "tri", (38, 52, 66)),
    "high": MF("Високий", "tri", (56, 70, 84)),
    "extreme": MF("Екстремальний", "trap", (72, 88, 100, 100)),
})

# Константи висновків для алгоритму Сугено (приблизно центри вихідних термів).
SUGENO_CONSTANTS = {"minimal": 5.0, "low": 30.0, "moderate": 52.0, "high": 70.0, "extreme": 94.0}


def _r(code, ante, cons, weight=1.0, why=""):
    return FuzzyRule(code, tuple(ante), cons, weight, "and", why)


RULES = [
    _r("R1", [("temperature", "hot"), ("humidity", "dry"), ("dry_days", "long")], "extreme", 1.0,
       "Спека, сухе повітря та тривала посуха максимально підсушують лісову підстилку."),
    _r("R2", [("temperature", "hot"), ("humidity", "dry"), ("dry_days", "medium")], "high", 1.0,
       "Спека та сухе повітря за середньої тривалості посушливого періоду."),
    _r("R3", [("temperature", "hot"), ("humidity", "dry"), ("dry_days", "recent")], "moderate", 0.9,
       "Нещодавні опади частково компенсують вплив спеки й сухого повітря."),
    _r("R4", [("temperature", "hot"), ("humidity", "moderate"), ("dry_days", "long")], "high", 0.9,
       "Тривала посуха за спеки підсушує горючі матеріали навіть за помірної вологості."),
    _r("R5", [("temperature", "hot"), ("humidity", "moderate"), ("dry_days", "medium")], "moderate", 0.8,
       "Спека за помірної вологості та середнього посушливого періоду."),
    _r("R6", [("temperature", "hot"), ("humidity", "humid")], "low", 0.9,
       "Висока вологість повітря стримує висихання підстилки навіть у спеку."),
    _r("R7", [("temperature", "warm"), ("humidity", "dry"), ("dry_days", "long")], "high", 0.9,
       "Тривала посуха та сухе повітря за теплої погоди."),
    _r("R8", [("temperature", "warm"), ("humidity", "dry"), ("dry_days", "medium")], "moderate", 0.9,
       "Сухе повітря та середній посушливий період за теплої погоди."),
    _r("R9", [("temperature", "warm"), ("humidity", "moderate"), ("dry_days", "long")], "moderate", 0.8,
       "Тривала відсутність опадів підвищує небезпеку за теплої погоди."),
    _r("R10", [("temperature", "warm"), ("humidity", "moderate"), ("dry_days", "medium")], "low", 0.9,
        "Типові умови теплої погоди без ознак посухи."),
    _r("R11", [("temperature", "warm"), ("humidity", "humid")], "minimal", 0.9,
       "Тепла та волога погода: займання малоймовірне."),
    _r("R12", [("temperature", "cool"), ("humidity", "dry"), ("dry_days", "long")], "moderate", 0.7,
       "Тривала посуха за низької температури: небезпека обмежена."),
    _r("R13", [("temperature", "cool"), ("humidity", "humid")], "minimal", 1.0,
       "Прохолодна й волога погода майже виключає займання."),
    _r("R14", [("temperature", "cool"), ("humidity", "moderate")], "minimal", 0.8,
       "Прохолодна погода помірної вологості."),
    _r("R15", [("dry_days", "recent")], "minimal", 0.8,
       "Нещодавні значні опади зволожили лісову підстилку."),
    _r("R16", [("humidity", "humid"), ("dry_days", "recent")], "minimal", 1.0,
       "Висока вологість після опадів: небезпеки практично немає."),
    _r("R17", [("wind", "strong"), ("humidity", "dry"), ("temperature", "hot")], "extreme", 1.0,
       "Сильний вітер за спеки й сухого повітря швидко поширює вогонь."),
    _r("R18", [("wind", "strong"), ("humidity", "dry"), ("dry_days", "long")], "extreme", 0.9,
       "Сильний вітер за посухи та сухого повітря."),
    _r("R19", [("wind", "strong"), ("humidity", "dry"), ("dry_days", "medium")], "high", 0.9,
       "Сильний вітер посилює небезпеку за сухого повітря."),
    _r("R20", [("wind", "strong"), ("humidity", "moderate"), ("dry_days", "long")], "high", 0.8,
       "Вітер підсилює небезпеку на тлі тривалої посухи."),
    _r("R21", [("wind", "calm"), ("humidity", "humid")], "minimal", 0.8,
       "Штиль і висока вологість обмежують поширення вогню."),
    _r("R22", [("humidity", "dry")], "moderate", 0.25,
       "Низька вологість повітря сама по собі підвищує небезпеку."),
    _r("R23", [("dry_days", "long")], "moderate", 0.25,
       "Тривалий посушливий період сам по собі підвищує небезпеку."),
    _r("R24", [("humidity", "humid")], "low", 0.3,
       "Висока вологість повітря знижує небезпеку."),
    _r("R25", [("humidity", "moderate")], "low", 0.2,
       "Помірна вологість повітря: небезпека невисока."),
    _r("R26", [("wind", "strong"), ("humidity", "moderate"), ("dry_days", "medium")], "moderate", 0.7,
       "Сильний вітер за помірної вологості та середнього посушливого періоду."),
    _r("R27", [("wind", "strong"), ("temperature", "warm"), ("humidity", "moderate")], "moderate", 0.6,
       "Сильний вітер у теплу погоду підсушує підстилку та сприяє поширенню вогню."),
    _r("R28", [("wind", "moderate"), ("humidity", "dry"), ("dry_days", "medium")], "high", 0.6,
       "Помірний вітер за сухого повітря та середнього посушливого періоду."),
    _r("R29", [("wind", "strong")], "moderate", 0.2,
       "Сильний вітер сам по собі ускладнює гасіння та підсилює поширення вогню."),
]


def build_model() -> FuzzyModel:
    """Створює нечітку модель із базою знань за замовчуванням."""
    inputs = {v.key: v for v in (TEMPERATURE, HUMIDITY, WIND, DRY_DAYS)}
    return FuzzyModel(inputs, FIRE_INDEX, RULES, SUGENO_CONSTANTS)
