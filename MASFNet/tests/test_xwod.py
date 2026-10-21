import tempfile
import unittest
from pathlib import Path

from PIL import Image

from utils.coco_eval import (build_coco_ground_truth, condition_from_path,
                             evaluate_coco_predictions)


class CocoEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temporary_dir.name) / 'heavy_rain_test_00001.jpg'
        Image.new('RGB', (100, 80)).save(str(self.image_path))
        self.lines = ['{} 10,20,50,60,0\n'.format(self.image_path)]
        self.classes = ['car', 'bus']

    def tearDown(self):
        self.temporary_dir.cleanup()

    def test_weather_prefix_keeps_underscores(self):
        self.assertEqual(condition_from_path(str(self.image_path)), 'heavy_rain')

    def test_coco_metrics_include_map_precision_recall_and_class_ap(self):
        coco_gt, records = build_coco_ground_truth(self.lines, self.classes)
        predictions = [{
            'image_id': 1,
            'category_id': 1,
            'bbox': [10, 20, 40, 40],
            'score': 0.9,
        }]
        metrics = evaluate_coco_predictions(coco_gt, predictions, records, self.classes)

        self.assertAlmostEqual(metrics['map_50'], 1.0)
        self.assertAlmostEqual(metrics['map_50_95'], 1.0)
        self.assertAlmostEqual(metrics['precision'], 1.0)
        self.assertAlmostEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['per_class']['car']['instances'], 1)
        self.assertEqual(metrics['per_class']['bus']['ap_50'], None)
        self.assertAlmostEqual(metrics['per_condition']['heavy_rain']['map_50'], 1.0)

    def test_empty_predictions_return_zero_precision_and_recall(self):
        coco_gt, records = build_coco_ground_truth(self.lines, self.classes)
        metrics = evaluate_coco_predictions(coco_gt, [], records, self.classes)

        self.assertEqual(metrics['map_50'], 0.0)
        self.assertEqual(metrics['precision'], 0.0)
        self.assertEqual(metrics['recall'], 0.0)
        self.assertEqual(metrics['false_negatives'], 1)


if __name__ == '__main__':
    unittest.main()
