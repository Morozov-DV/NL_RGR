"""Автоматичні тести ядра нечіткого виводу та експертної системи."""
import itertools
import unittest

import numpy as np

from fire_expert.expert_system import ExpertSystem, classify
from fire_expert.fuzzy import defuzzify, trapmf, trimf
from fire_expert.knowledge_base import build_model
from fire_expert.nesterov import nesterov_class, nesterov_index

SAFE = dict(temperature=15, humidity=85, wind=2, dry_days=1)      # безпечні умови
DANGER = dict(temperature=36, humidity=15, wind=14, dry_days=25)  # небезпечні умови


class MembershipTests(unittest.TestCase):
    def test_triangle_vertex_and_edges(self):
        self.assertAlmostEqual(float(trimf(5, 0, 5, 10)), 1.0)
        self.assertAlmostEqual(float(trimf(0, 0, 5, 10)), 0.0)
        self.assertAlmostEqual(float(trimf(2.5, 0, 5, 10)), 0.5)

    def test_trapezoid_shoulders(self):
        self.assertAlmostEqual(float(trapmf(0, 0, 0, 5, 10)), 1.0)     # ліве «плече»
        self.assertAlmostEqual(float(trapmf(10, 0, 5, 10, 10)), 1.0)   # праве «плече»
        self.assertAlmostEqual(float(trapmf(7.5, 0, 0, 5, 10)), 0.5)


class DefuzzTests(unittest.TestCase):
    def test_symmetric_set_gives_center(self):
        x = np.linspace(0, 10, 1001)
        mu = trimf(x, 2, 5, 8)
        for method in ("centroid", "bisector", "mom"):
            self.assertAlmostEqual(defuzzify(x, mu, method), 5.0, delta=0.05)

    def test_som_not_greater_than_lom(self):
        x = np.linspace(0, 10, 1001)
        mu = trapmf(x, 1, 3, 6, 9)
        self.assertLessEqual(defuzzify(x, mu, "som"), defuzzify(x, mu, "lom"))

    def test_empty_set_raises(self):
        with self.assertRaises(ValueError):
            defuzzify(np.linspace(0, 1, 11), np.zeros(11))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.es = ExpertSystem(build_model())

    def test_safe_conditions_give_low_index(self):
        for alg in ("mamdani", "sugeno"):
            r = self.es.assess(*SAFE.values(), algorithm=alg)
            self.assertLess(r.index, 20)
            self.assertEqual(r.level, 1)

    def test_dangerous_conditions_give_extreme_class(self):
        for alg in ("mamdani", "sugeno"):
            r = self.es.assess(*DANGER.values(), algorithm=alg)
            self.assertGreaterEqual(r.index, 80)
            self.assertEqual(r.level, 5)

    def test_algorithms_are_consistent(self):
        # Результати Мамдані та Сугено не розходяться більш ніж на 15 пунктів.
        grids = [np.linspace(v.lo, v.hi, 5) for v in self.es.model.inputs.values()]
        for point in itertools.product(*grids):
            r = self.es.assess(*point)
            self.assertLess(abs(r.comparison["mamdani"] - r.comparison["sugeno"]), 15, msg=str(point))

    def test_rule_base_covers_input_space(self):
        grids = [np.linspace(v.lo, v.hi, 9) for v in self.es.model.inputs.values()]
        for point in itertools.product(*grids):
            values = dict(zip(self.es.model.inputs, point))
            strengths = self.es.model.activate(self.es.model.fuzzify(values))
            self.assertGreater(sum(a for _, a in strengths), 0, msg=str(values))

    def test_index_grows_with_dry_days(self):
        base = dict(temperature=30, humidity=30, wind=5)
        vals = [self.es.assess(dry_days=d, **base, algorithm="sugeno").index for d in (0, 10, 20, 30)]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])))

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            self.es.assess(60, 50, 5, 5)

    def test_explanation_lists_rules(self):
        r = self.es.assess(*DANGER.values())
        self.assertEqual(r.fired[0].code, "R1")
        self.assertIn("РЕКОМЕНДАЦІЯ", self.es.explain(r))

    def test_levels(self):
        self.assertEqual(classify(5)[0], 1)
        self.assertEqual(classify(55)[0], 3)
        self.assertEqual(classify(100)[0], 5)


class NesterovTests(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(nesterov_class(100), 1)
        self.assertEqual(nesterov_class(2000), 3)
        self.assertEqual(nesterov_class(9000), 5)

    def test_zero_for_freezing(self):
        self.assertEqual(nesterov_index(-3, 50, 10), 0.0)

    def test_index_grows_with_dryness(self):
        self.assertGreater(nesterov_index(30, 20, 10), nesterov_index(30, 80, 10))


if __name__ == "__main__":
    unittest.main()
