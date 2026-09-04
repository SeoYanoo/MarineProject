import unittest

from classification import _find_classification_result


class ClassificationResponseTests(unittest.TestCase):
    def test_selects_highest_confidence_classification_result(self):
        payload = {
            "outputs": [
                {"top": "50톤급", "confidence": 0.71},
                {"predictions": [{"class": "100톤급", "confidence": 0.88}]},
            ]
        }

        self.assertEqual(
            _find_classification_result(payload),
            ("100톤급", 0.88),
        )

    def test_does_not_treat_detection_box_as_classification(self):
        payload = {
            "predictions": [
                {
                    "class": "ship",
                    "confidence": 0.99,
                    "x": 10,
                    "y": 20,
                    "width": 30,
                    "height": 40,
                }
            ]
        }

        self.assertIsNone(_find_classification_result(payload))


if __name__ == "__main__":
    unittest.main()
