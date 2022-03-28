#!/usr/bin/env python3

from transmorph.layers.profiler import Profiler

p = Profiler()
i1 = p.task_start(task_label="Task 1")
i2 = p.task_start(task_label="Task 2")
i3 = p.task_start(task_label="Task 3")
i4 = p.task_start(task_label="Task 4")
p.task_end(i2)
p.task_end(i4)
print(p.log_tasks())
