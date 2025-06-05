import time
from typing import Callable, Any

def timeFtn(ftn: Callable[..., Any], *args, **kwargs) -> None:
    start = time.time()
    ftn(*args, **kwargs)
    end = time.time()
    dur = end - start
    print(f"{ftn.__name__}() ran for {dur:.2f} seconds.")
