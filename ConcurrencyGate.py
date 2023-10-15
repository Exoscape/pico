import threading

# Slightly modified version of a semaphore
class ConcurrencyGate:
  def __init__(self, max_concurrency = 1):
    self.MaxConcurrency = max_concurrency
    self.Counter = 0
    self.Lock = threading.Lock()

  def Acquire(self):
    with self.Lock:
      if self.Counter >= self.MaxConcurrency:
        return False
      
      self.Counter += 1
      return True

  def Release(self):
    with self.Lock:
      if self.Counter > 0:
        self.Counter -= 1

  def IsLocked(self):
    return self.Counter >= self.MaxConcurrency