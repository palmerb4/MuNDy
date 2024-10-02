import time
import numpy as np

if __name__ == '__main__':
  field_size = 100
  num_iterations = 1000000
  field1 = np.zeros(field_size)
  field2 = np.zeros(field_size)
  result = np.zeros(field_size)

  start_time = time.time()
  field1 = np.random.rand(field_size)
  field2 = np.random.rand(field_size)
  for i in range(num_iterations):
    result = field1 + field2
  end_time = time.time()
  print("Python direct run time: ", end_time - start_time)
