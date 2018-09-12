#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Abstract class for interface.
    
    Name: Interface.py
"""

import traceback

class InterfaceException(Exception):
    """Customized class for handle exceptions."""
    
    DEBUG = True
    
    @staticmethod
    def format_exception(message = None):
        """Format a exception message.

        Returns
        ----------
        fmt_message : string
            A formatted exception message.
        """
        if message is not None:
            return "Unexpected error:\n%s" % message.replace('%', '%%')
        elif InterfaceException.DEBUG == True:
            return "Unexpected error:\n%s" % traceback.format_exc().replace('%', '%%')
        else:
            return "Unexpected error\n"