import time


def printTime(func):
  def wrapper(*args, **kw):
    start = time.time()
    result = func(*args, **kw)
    end = time.time()
    print(f"Function: {func.__name__} Total Time Used: {end - start}")
    return result
  return wrapper