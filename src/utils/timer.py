import time

def time_step(label, timers=None, verbose=True):
    """
    Decorator to time a function and optionally store timing info.

    Args:
        label (str): Name of the timed step.
        timers (dict, optional): If provided, will store duration as timers[label].
        verbose (bool): Whether to print timing info.

    Returns:
        function: The wrapped function.
    """
    def decorator(fn):
        def wrapped(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            duration = time.time() - start
            if verbose:
                print(f"[TIMER] {label} took {duration:.2f} seconds")
            if timers is not None:
                timers[label] = duration
            return result
        return wrapped
    return decorator
