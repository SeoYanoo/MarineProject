import unittest
from unittest.mock import patch

from PIL import Image

import detection


class DetectionPipelineTests(unittest.TestCase):
    def test_unconfigured_pipeline_does_not_silently_create_demo_results(self):
        image = Image.new("RGB", (640, 360), "black")
        with (
            patch.object(detection, "_roboflow_detect", return_value=None),
            patch.object(detection, "_load_yolo", return_value=None),
            patch.object(detection, "ALLOW_DEMO_MODE", False),
        ):
            with self.assertRaisesRegex(RuntimeError, "실제 객체 탐지 모델"):
                detection.run_pipeline(image)

    def test_explicit_demo_mode_still_available_for_development(self):
        image = Image.new("RGB", (640, 360), "black")
        with (
            patch.object(detection, "_roboflow_detect", return_value=None),
            patch.object(detection, "_load_yolo", return_value=None),
            patch.object(detection, "ALLOW_DEMO_MODE", True),
        ):
            result = detection.run_pipeline(image)

        self.assertEqual(result.mode, "demo")
        self.assertEqual(len(result.predictions), 3)

    def test_nms_removes_overlapping_same_class_box(self):
        boxes = [
            (10, 10, 100, 100, 0.9, "drone"),
            (15, 15, 100, 100, 0.8, "drone"),
            (15, 15, 100, 100, 0.7, "ship"),
        ]

        result = detection._filter_detection_boxes(boxes, 640, 360)

        self.assertEqual(len(result), 2)
        self.assertEqual({item[5] for item in result}, {"drone", "ship"})


if __name__ == "__main__":
    unittest.main()
