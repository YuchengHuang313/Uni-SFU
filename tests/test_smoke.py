import math
import unittest

import optimize_piecewise_activations as pipeline


class UniSFUSmokeTest(unittest.TestCase):
    def test_paper_activation_set(self):
        self.assertEqual(
            list(pipeline.ACTIVATION_FUNCTIONS),
            ["GELU", "SiLU", "Sigmoid", "Tanh", "Softplus", "ELU"],
        )
        for name in pipeline.ACTIVATION_FUNCTIONS:
            self.assertEqual(pipeline.helper_activation_domain(name), (-8.0, 8.0))

    def test_degree_count_conversion(self):
        degrees = [0, 1, 2, 3]
        self.assertEqual(pipeline.helper_degrees_to_counts(degrees), (4, 3, 2, 1))
        degree_counts = pipeline.helper_degree_type_counts(degrees, degree_max=3)
        self.assertEqual(degree_counts, [1, 1, 1, 1])
        self.assertEqual(
            pipeline.helper_degree_counts_to_c_counts(degree_counts),
            (4, 3, 2, 1),
        )

    def test_area_model_reference_value(self):
        area = pipeline.helper_hardware_area(4, 3, 2, 1)
        self.assertTrue(math.isclose(area, 3627.08, rel_tol=0.0, abs_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
