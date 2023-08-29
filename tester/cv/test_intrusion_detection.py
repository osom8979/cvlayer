# -*- coding: utf-8 -*-

from unittest import TestCase, main

from cvlayer.cv.intrusion_detection import CounterConfig, HierarchicalIntrusionDetection


class HierarchicalIntrusionDetectionTestCase(TestCase):
    def setUp(self):
        # =============================
        #              +y             |
        #               |             |
        #               |             |
        #        (-1,1) |             |
        #           *-------*         |
        #           |   |   |         |
        # -x -------|---*---|---*--- +x
        #           |[0]|   |   |     | <- Level 0
        #           *-------*   |     |
        #               |[1]    |     | <- Level 1
        #               *-------*     |
        #               |     (2,-2)  |
        #              -y             |
        # =============================
        roi1 = -1, 1, 1, -1
        roi2 = 0, 0, 2, -2

        self.threshold = 4
        self.maxvalue = 5
        self.minvalue = 1
        self.upgrade_period = 1
        self.downgrade_period = 1
        self.upgrade_init = 2
        self.downgrade_init = 3

        self.config = CounterConfig(
            self.threshold,
            self.maxvalue,
            self.minvalue,
            self.upgrade_period,
            self.downgrade_period,
            self.upgrade_init,
            self.downgrade_init,
        )

        self.hid = HierarchicalIntrusionDetection(roi1, roi2, config=self.config)

    def test_hierarchical_intrusion_detection_default(self):
        outside_first = (-2, 2, -1, 1)
        inside_first = (-1, 1, 0, 0)
        inside_union = (0, 0, 1, -1)
        inside_second = (1, -1, 2, -2)
        outside_second = (2, -2, 3, -3)

        # OUTSIDE #1
        res1 = self.hid.run(outside_first)
        self.assertEqual(0, res1.level)
        self.assertEqual(1, res1.counter)  # minvalue

        # OUTSIDE #1 -> INSIDE #1
        res2 = self.hid.run(inside_first)
        self.assertEqual(0, res2.level)
        self.assertEqual(2, res2.counter)

        res3 = self.hid.run(inside_first)
        self.assertEqual(0, res3.level)
        self.assertEqual(3, res3.counter)

        res4 = self.hid.run(inside_first)
        self.assertEqual(0, res4.level)
        self.assertEqual(4, res4.counter)

        res5 = self.hid.run(inside_first)
        self.assertEqual(0, res5.level)
        self.assertEqual(5, res5.counter)

        res6 = self.hid.run(inside_first)
        self.assertEqual(0, res6.level)
        self.assertEqual(5, res6.counter)

        # INSIDE #1 -> UNION
        res7 = self.hid.run(inside_union)
        self.assertEqual(1, res7.level)
        self.assertEqual(2, res7.counter)

        # UNION -> INSIDE #2
        res8 = self.hid.run(inside_second)
        self.assertEqual(1, res8.level)
        self.assertEqual(3, res8.counter)

        res9 = self.hid.run(inside_second)
        self.assertEqual(1, res9.level)
        self.assertEqual(4, res9.counter)

        res10 = self.hid.run(inside_second)
        self.assertEqual(1, res10.level)
        self.assertEqual(5, res10.counter)

        res11 = self.hid.run(inside_second)
        self.assertEqual(1, res11.level)
        self.assertEqual(5, res11.counter)

        # INSIDE #2 -> OUTSIDE #2
        res12 = self.hid.run(outside_second)
        self.assertEqual(1, res12.level)
        self.assertEqual(4, res12.counter)

        res13 = self.hid.run(outside_second)
        self.assertEqual(1, res13.level)
        self.assertEqual(3, res13.counter)

        res14 = self.hid.run(outside_second)
        self.assertEqual(1, res14.level)
        self.assertEqual(2, res14.counter)

        res15 = self.hid.run(outside_second)
        self.assertEqual(1, res15.level)
        self.assertEqual(1, res15.counter)

        res16 = self.hid.run(outside_second)
        self.assertEqual(0, res16.level)
        self.assertEqual(3, res16.counter)

        res17 = self.hid.run(outside_second)
        self.assertEqual(0, res17.level)
        self.assertEqual(2, res17.counter)

        res18 = self.hid.run(outside_second)
        self.assertEqual(0, res18.level)
        self.assertEqual(1, res18.counter)

        res19 = self.hid.run(outside_second)
        self.assertEqual(0, res19.level)
        self.assertEqual(1, res19.counter)


if __name__ == "__main__":
    main()
