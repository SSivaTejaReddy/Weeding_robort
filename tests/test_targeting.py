import unittest

from ai_validation.targeting import (
    Detection,
    TargetingConfig,
    assign_crop_complement_targets,
    assign_extraction_targets,
)


class AssignExtractionTargetsTests(unittest.TestCase):
    def test_selects_highest_confidence_target_per_zone(self):
        detections = [
            Detection("not_crop", 0.82, 35, 70, 65, 110),
            Detection("not_crop", 0.94, 45, 75, 75, 115),
            Detection("not_crop", 0.89, 275, 80, 315, 120),
        ]

        targets = assign_extraction_targets(detections, 400, 200)

        self.assertEqual([target.zone_index for target in targets], [0, 2])
        self.assertEqual(targets[0].detection.confidence, 0.94)

    def test_rejects_upper_frame_and_low_confidence_candidates(self):
        detections = [
            Detection("not_crop", 0.91, 120, 10, 160, 40),
            Detection("not_crop", 0.40, 120, 140, 160, 170),
        ]

        targets = assign_extraction_targets(detections, 400, 200)

        self.assertEqual(targets, [])

    def test_crop_guard_rejects_nearby_not_crop_center(self):
        detections = [
            Detection("crop", 0.90, 180, 90, 240, 155),
            Detection("not_crop", 0.93, 202, 110, 222, 145),
            Detection("not_crop", 0.88, 300, 110, 330, 145),
        ]
        config = TargetingConfig(crop_guard_px=12)

        targets = assign_extraction_targets(detections, 400, 200, config)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].zone_index, 3)

    def test_accepts_not_crop_candidate(self):
        targets = assign_extraction_targets(
            [Detection("not_crop", 0.80, 96, 130, 120, 160)],
            400,
            200,
        )

        self.assertEqual(len(targets), 1)

    def test_crop_complement_marks_open_zone_points(self):
        targets = assign_crop_complement_targets([], 400, 200)

        self.assertEqual([target.zone_index for target in targets], [0, 1, 2, 3])
        self.assertTrue(all(target.detection.label == "not_crop" for target in targets))

    def test_crop_complement_blocks_protected_working_point(self):
        crop_at_zone_two_point = Detection("crop", 0.90, 140, 130, 160, 180)

        targets = assign_crop_complement_targets([crop_at_zone_two_point], 400, 200)

        self.assertEqual([target.zone_index for target in targets], [0, 2, 3])

    def test_low_confidence_crop_still_protects_working_point(self):
        crop_at_zone_two_point = Detection("crop", 0.30, 140, 130, 160, 180)

        targets = assign_crop_complement_targets([crop_at_zone_two_point], 400, 200)

        self.assertEqual([target.zone_index for target in targets], [0, 2, 3])

    def test_wider_default_guard_protects_nearby_working_point(self):
        crop_near_zone_two_point = Detection("crop", 0.80, 105, 130, 125, 180)

        targets = assign_crop_complement_targets([crop_near_zone_two_point], 400, 200)

        self.assertEqual([target.zone_index for target in targets], [0, 2, 3])


if __name__ == "__main__":
    unittest.main()
