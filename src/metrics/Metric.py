#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Abstract class for metrics.
"""

from interface.Interface import InterfaceException as IException

from abc import ABCMeta, abstractmethod


class Metric(object):

    __metaclass__ = ABCMeta

    def obtain(self):
        """Obtain metric 
        """
        raise IException("Metric is not available!")

    