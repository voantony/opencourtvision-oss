import time


def hms(s):
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"


class StageTimer:
    """`with T('3 hits'): ...` records wall time per stage in T.times and prints `[name] secs`."""

    def __init__(self):
        self.times = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.time()

            def __exit__(self, *exc):
                timer.times[name] = round(time.time() - self.t0, 2)
                if exc[0] is None:
                    print(f"[{name}] {timer.times[name]} s", flush=True)
                return False

        return _Ctx()
