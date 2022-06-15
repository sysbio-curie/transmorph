#!/usr/bin/env python3

import time


class Task:
    """
    Defines a timed task that can be started and stopped. Useful to
    profile a layer part.
    """

    def __init__(self, task_id: int, task_start_time: float, task_label: str = ""):
        self.id = task_id
        self.tstart = task_start_time
        self.tend = -1
        self.label = task_label
        self.state = "ongoing"

    def __str__(self) -> str:
        return (
            f"{self.id}\t{self.label}\t{'{:.2e}'.format(self.elapsed())}\t{self.state}"
        )

    def elapsed(self) -> float:
        if not self.is_ended():
            return time.time() - self.tstart
        return self.tend - self.tstart

    def is_ended(self) -> bool:
        return self.tend != -1

    def end(self) -> None:
        self.tend = time.time()
        self.state = "ended"


class Profiler:
    """
    Manages tasks to profile.
    """

    def __init__(self):
        self.tasks = []
        self.n_tasks_ongoing = 0
        self.tstart = -1
        self.tend = -1

    def task_start(self, task_label: str = "") -> int:
        task_id = len(self.tasks)
        if task_id == 0:
            self.tstart = time.time()
        self.tasks.append(Task(task_id, time.time(), task_label))
        self.n_tasks_ongoing += 1
        return task_id

    def task_end(self, task_id: int) -> float:
        assert 0 <= task_id < len(self.tasks), "Task id out of range."
        self.n_tasks_ongoing -= 1
        if self.n_tasks_ongoing == 0:
            self.tend = time.time()
        self.tasks[task_id].end()
        elapsed = self.tasks[task_id].elapsed()
        return elapsed

    def elapsed(self) -> float:
        if self.tstart == -1:
            return 0.0
        if self.tend == -1:
            return time.time() - self.tstart
        return self.tend - self.tstart

    def log_tasks(self) -> str:
        return "TID\tNAME\tELAPSED(s)\tSTATE\n" + "\n".join(
            [str(t) for t in self.tasks]
        )

    def log_stats(self) -> str:
        if len(self.tasks) == 0:
            return ""
        ntask = len(self.tasks)
        ntask_end = sum(t.is_ended() for t in self.tasks)
        ntask_ongoing = ntask - ntask_end
        ltask_id = max(
            range(len(self.tasks)), key=lambda tid: self.tasks[tid].elapsed()
        )
        ltask = self.tasks[ltask_id].label
        ltask_time = self.tasks[ltask_id].elapsed()
        ttime = self.elapsed()
        return (
            f"Total time: {ttime}s\n"
            + f"Longest task: {ltask} [{ltask_id}] ({ltask_time}s)\n"
            + f"Ended tasks: {ntask_end}\n"
            + f"Ongoing tasks: {ntask_ongoing}\n"
            + f"Total tasks: {len(self.tasks)}"
        )


profiler = Profiler()
