# -*- coding: utf-8 -*-

from cvlayer.cv.tracker import Tracker
from cvlayer.cv.tracking import TrackingMethod


class CvlTracker:
    @staticmethod
    def create_tracker_dasiamrpn(*args, **kwargs):
        return Tracker(TrackingMethod.DaSiamRPN, *args, **kwargs)

    @staticmethod
    def create_tracker_boosting(*args, **kwargs):
        return Tracker(TrackingMethod.Boosting, *args, **kwargs)

    @staticmethod
    def create_tracker_mil(*args, **kwargs):
        return Tracker(TrackingMethod.MIL, *args, **kwargs)

    @staticmethod
    def create_tracker_kcf(*args, **kwargs):
        return Tracker(TrackingMethod.KCF, *args, **kwargs)

    @staticmethod
    def create_tracker_tld(*args, **kwargs):
        return Tracker(TrackingMethod.TLD, *args, **kwargs)

    @staticmethod
    def create_tracker_medianflow(*args, **kwargs):
        return Tracker(TrackingMethod.MedianFlow, *args, **kwargs)

    @staticmethod
    def create_tracker_goturn(*args, **kwargs):
        return Tracker(TrackingMethod.GOTURN, *args, **kwargs)

    @staticmethod
    def create_tracker_csrt(*args, **kwargs):
        return Tracker(TrackingMethod.CSRT, *args, **kwargs)

    @staticmethod
    def create_tracker_mosse(*args, **kwargs):
        return Tracker(TrackingMethod.MOSSE, *args, **kwargs)
