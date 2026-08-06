import unittest

from src.action_adapter import resolve_object_ids


class ResolveObjectIdsTest(unittest.TestCase):
    def test_explicit_object_id_is_not_replaced_by_object_type(self):
        objects = [
            {"objectId": "Drawer|near", "objectType": "Drawer", "visibleBounds2D": [0, 0, 1, 1]},
            {"objectId": "Drawer|target", "objectType": "Drawer", "visibleBounds2D": [1, 0, 2, 1]},
        ]

        resolved, warning = resolve_object_ids(
            "OpenObject",
            {"objectId": "Drawer|target", "objectType": "Drawer"},
            objects,
        )

        self.assertIsNone(warning)
        self.assertEqual({"objectId": "Drawer|target"}, resolved)
