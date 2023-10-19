# Reference: https://github.com/tensorflow/tensorflow/blob/6935c8f706dde1906e388b3142906c92cdcc36db/tensorflow/python/platform/tf_logging.py

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# @Time    : 2023/5/12 23:43 CST
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/logger/_logging.py
# @Software: PyCharm

from typing import Optional

import inspect

import logging as _logging
import threading as _threading
import sys as _sys
import traceback as _traceback

# Don't use this directly. Use get_logger() instead.
_logger = None
_logger_lock = _threading.Lock()

_caller_cache = _threading.local()


def _get_caller(offset=3):
    # Get the name of the calling function
    caller_func_name = inspect.stack()[1].function

    # Try to find the calling function in the stack trace
    for i, frame in enumerate(inspect.stack()):
        if frame.function == caller_func_name:
            offset = i + 2
            break
    # Get the calling frame
    f = _sys._getframe(offset)

    # Iterate over the stack to find the lowest non-logging frame
    our_file = f.f_code.co_filename
    f = f.f_back
    while f:
        code = f.f_code
        if code.co_filename != our_file:
            return code, f
        f = f.f_back
    return None, None


# The definition of `findCaller` changed in Python 3.2,
# and further changed in Python 3.8
if _sys.version_info.major >= 3 and _sys.version_info.minor >= 8:
    def _logger_find_caller(stack_info=False,
                            stacklevel=1):  # pylint: disable=g-wrong-blank-lines
        code, frame = _get_caller()
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return '(unknown file)', 0, '(unknown function)', sinfo
elif _sys.version_info.major >= 3 and _sys.version_info.minor >= 2:
    def _logger_find_caller(
            stack_info=False):  # pylint: disable=g-wrong-blank-lines
        code, frame = _get_caller()
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return '(unknown file)', 0, '(unknown function)', sinfo
else:
    def _logger_find_caller():  # pylint: disable=g-wrong-blank-lines
        code, frame = _get_caller()
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name)
        else:
            return '(unknown file)', 0, '(unknown function)'


def _get_logger(_name: Optional[str] = None, stacklevel: Optional[int] = 3):
    global _logger

    # Use double-checked locking to avoid taking lock unnecessarily.
    if _logger:
        return _logger

    _logger_lock.acquire()

    try:
        if _logger:
            return _logger

        # Scope the TensorFlow logger to not conflict with users' loggers.
        logger = _logging.getLogger(_name)

        # Override findCaller on the logger to skip internal helper functions
        logger.findCaller = _logger_find_caller

        # Don't further configure the TensorFlow logger if the root logger is
        # already configured. This prevents double logging in those cases.
        if not logger.hasHandlers() and not _logging.getLogger().handlers:
            # Determine whether we are in an interactive environment
            _interactive = False
            try:
                # This is only defined in interactive shells.
                if _sys.ps1:
                    _interactive = True
            except AttributeError:
                # Even now, we may be in an interactive shell with `python -i`.
                _interactive = _sys.flags.interactive

            # If we are in an interactive environment (like Jupyter), set loglevel
            # to INFO and pipe the output to stdout.
            if _interactive:
                logger.setLevel(_logger.INFO)
                _logging_target = _sys.stdout
            else:
                _logging_target = _sys.stderr

            # Add the output handler.
            # if not logger.hasHandlers():
            _handler = _logging.StreamHandler(_logging_target)
            _handler.setFormatter(
                _logging.Formatter(_logging.BASIC_FORMAT, None))
            logger.addHandler(_handler)

        _logger = logger
        return _logger

    finally:
        _logger_lock.release()


if __name__ == '__main__':
    print(_sys.version_info)
