# Експертна система оцінки ризику лісових пожеж (нечітка логіка)
РГР, варіант 4. Власна реалізація алгоритмів Мамдані та Сугено на NumPy.

## Запуск
```
pip install -r requirements.txt
python main.py --temperature 34 --humidity 18 --wind 10 --dry-days 20
python main.py --temperature 30 --humidity 30 --wind 4 --dry-days 10 --algorithm sugeno
python main.py --interactive
python -m unittest                  # 16 автоматичних тестів
python -m fire_expert.experiments   # експерименти, графіки у figures/, числа у results.json
```

## Структура
- `fire_expert/fuzzy.py` - ядро нечіткого виводу (функції належності, Мамдані, Сугено, дефазифікація)
- `fire_expert/knowledge_base.py` - лінгвістичні змінні та 29 правил
- `fire_expert/expert_system.py` - оцінювання, пояснення, рекомендації
- `fire_expert/nesterov.py` - еталонний показник Нестерова
- `fire_expert/experiments.py` - чисельні експерименти та графіки
- `main.py` - консольний інтерфейс; `tests/` - тести
