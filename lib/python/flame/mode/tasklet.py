# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""flame tasklet."""

from __future__ import annotations

import logging
from queue import Queue
from typing import Callable

from flame.mode.composer import ComposerContext
from flame.mode.enums import LoopIndicator
from flame.monitor.runtime import time_tasklet

logger = logging.getLogger(__name__)


class Tasklet(object):
    """Tasklet is a class for defining a unit of work."""

    def __init__(self, alias: str, func: Callable, *args, **kwargs) -> None:
        """Initialize the class.

        Parameters
        ----------
        alias: an alias; should be unique within child class of Role class
        func: a method that will be executed as a tasklet
        *args: positional arguments for method func
        **kwargs: keyword arguments for method func
        """
        if not callable(func):
            raise TypeError(f"{func} is not callable")

        self.alias = alias
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.siblings = []

        self.composer = ComposerContext.get_composer()

        self.reset()

    def __str__(self):
        """Return tasklet details."""
        starter = self.loop_starter.func.__name__ if self.loop_starter else ""
        ender = self.loop_ender.func.__name__ if self.loop_ender else ""

        return (
            f"func: {self.func.__name__}"
            + f"\nloop_state: {self.loop_state}"
            + f"\nloop_starter: {starter}"
            + f"\nloop_ender: {ender}"
        )

    def __rshift__(self, other: Tasklet) -> Tasklet:

        # case: t1 >> [t2, t3]
        # make list[0] (e.g. t2) the next tasklet and asign the rest of the list as siblings (e.g. [t3])
        if isinstance(other, list):
            if len(other) == 0:
                raise ValueError(f"empry list is not permitted")
            elif len(other) == 1:
                other = other[0]
            else:
                firstTasklet = other[0]
                firstTasklet.set_siblings(other[1:])
                other = firstTasklet

        """Set up connection."""
        if self not in self.composer.chain:
            self.composer.chain[self] = set()

        if other not in self.composer.chain:
            self.composer.chain[other] = set()

        if self not in self.composer.reverse_chain:
            self.composer.reverse_chain[self] = set()

        if other not in self.composer.reverse_chain:
            self.composer.reverse_chain[other] = set()

        # case 1: t1 >> loop(t2 >> t3)
        # if t1 is self, t3 is other; t3.loop_starter is t2
        if other.loop_starter and other.loop_starter not in self.composer.chain:
            self.composer.chain[other.loop_starter] = set()

        # same as case 1
        if other.loop_starter and other.loop_starter not in self.composer.reverse_chain:
            self.composer.reverse_chain[other.loop_starter] = set()

        if other.loop_state & LoopIndicator.END:
            # same as case 1
            self.composer.chain[self].add(other.loop_starter)
            self.composer.reverse_chain[other.loop_starter].add(self)
        else:
            self.composer.chain[self].add(other)
            self.composer.reverse_chain[other].add(self)

        # clear the unlinked state of this tasklet
        self.composer.clear_unlinked_tasklet_state(self.alias)

        return other

    def reset(self):
        """Reset the tasklet's state."""
        self.cont_fn = None
        self.loop_check_fn = None
        self.loop_starter = None
        self.loop_ender = None
        self.loop_state = LoopIndicator.NONE

    def set_continue_fn(self, cont_fn: Callable) -> None:
        """Set continue function."""
        self.cont_fn = cont_fn

    def get_composer(self):
        """Return composer object."""
        return self.composer

    def set_siblings(self, siblings):
        self.siblings = siblings

    def get_root(self) -> Tasklet:
        """get_root returns a root tasklet."""
        q = Queue()
        q.put(self)
        visited = set()
        visited.add(self)
        while not q.empty():
            root = q.get()

            for parent in self.composer.reverse_chain[root]:
                if parent not in visited:
                    visited.add(parent)
                    q.put(parent)

        return root

    def get_ender(self) -> Tasklet:
        """Get last tasklet in a loop.

        If a tasklet is not in a loop, then loop_ender is None.
        In such a case, the current tasklet object is returned.
        If the current tasklet is indeed in a loop and is the last tasklet,
        it returns itself.
        Otherwise, it returns a its member variable, loop_ender.

        Returns
        -------
        tasklet: a last tasklet in a loop or an entire chain
        """
        if self.is_last_in_loop() or self.loop_ender is None:
            return self

        return self.loop_ender

    @time_tasklet
    def do(self) -> None:
        """Execute tasklet."""
        self.func(*self.args, **self.kwargs)

    def is_loop_done(self) -> bool:
        """Return if loop is done."""
        if not self.loop_check_fn:
            return True

        return self.loop_check_fn()

    def is_last_in_loop(self) -> bool:
        """Return if the tasklet is the last one in a loop."""
        return self.loop_state & LoopIndicator.END

    def is_continue(self) -> bool:
        """Return True if continue condition is met and otherwise False."""
        if not callable(self.cont_fn):
            return False

        return self.cont_fn()

    def update_loop_attrs(self, check_fn=None, state=None, starter=None, ender=None):
        if check_fn:
            self.loop_check_fn = check_fn
        if state is not None:
            self.loop_state = state
        if starter:
            self.loop_starter = starter
        if ender:
            self.loop_ender = ender

    def insert_before(self, tasklet: Tasklet) -> None:
        """Insert a tasklet before another tasklet in the composer."""
        self.composer.insert(self.alias, tasklet, after=False)

    def insert_after(self, tasklet: Tasklet) -> None:
        """Insert a tasklet before another tasklet in the composer."""
        self.composer.insert(self.alias, tasklet, after=True)

    def replace_with(self, tasklet: Tasklet) -> None:
        """Replace a tasklet with another tasklet in the composer."""
        self.alias = tasklet.alias
        self.func = tasklet.func
        self.args = tasklet.args
        self.kwargs = tasklet.kwargs

    def remove(self) -> None:
        """Remove a tasklet from the composer."""
        self.composer.remove_tasklet(self.alias)


class Loop(object):
    """Loop class."""

    def __init__(self, loop_check_fn=None) -> None:
        """Initialize loop object.

        Parameters
        ----------
        loop_check_fn: a function object to check loop exit conditions
        """
        if not callable(loop_check_fn):
            raise TypeError(f"{loop_check_fn} is not callable")

        self.loop_check_fn = loop_check_fn

    def __call__(self, ender: Tasklet) -> Tasklet:
        """Configure boundaries of loop and its exit condition.

        Given an ender of a loop, the starter (i.e., the first tasklet of a loop)
        is obtained. Then, all the tasklets in the loop are obtained by using
        the two tasklets. The ender is specified for every tasklets in the loop.
        The purpose of these updates are to facilitate the traverse of tasklets.

        Parameters
        ----------
        ender: last tasklet in a loop

        Returns
        -------
        ender: last tasklet in a loop
        """
        # composer is universally shared across tasklets
        # let's get it from ender
        composer = ender.get_composer()

        # the tasklet is the sole tasklet in a loop
        # if there are more than one tasklet in the loop,
        # ender should have been added in the chain
        # when rshift (i.e, >>) was handled
        if ender not in composer.chain:
            ender.loop_starter = ender
            ender.loop_ender = ender
            ender.loop_state = LoopIndicator.BEGIN | LoopIndicator.END
            ender.loop_check_fn = self.loop_check_fn

            composer.chain[ender] = set()
            composer.reverse_chain[ender] = set()

            return ender

        # since tasklets in a loop are not yet chained with tasklets outside
        # the loop, calling get_root() returns the first tasklet of the loop.
        starter = ender.get_root()

        starter.loop_starter = starter
        starter.loop_state |= LoopIndicator.BEGIN

        ender.loop_starter = starter
        ender.loop_state |= LoopIndicator.END

        tasklets_in_loop = composer.get_tasklets_in_loop(starter, ender)
        # for each tasklet in loop, loop_check_fn and loop_ender are updated
        for tasklet in tasklets_in_loop:
            if tasklet.loop_starter and tasklet.loop_ender:
                # if both loop_starter and loop_ender are already set,
                # they are set for an inner loop
                # so, don't update loop_starter and loop_ender in that case
                continue

            tasklet.loop_starter = starter
            tasklet.loop_check_fn = self.loop_check_fn
            tasklet.loop_ender = ender

        return ender
