#!/usr/bin/env python3

from transmorph.engine.profiler import Profiler


def test_profiler():

    p = Profiler()
    p.task_start(task_label="Task 1")
    i2 = p.task_start(task_label="Task 2")
    p.task_start(task_label="Task 3")
    i4 = p.task_start(task_label="Task 4")
    p.task_end(i2)
    p.task_end(i4)
