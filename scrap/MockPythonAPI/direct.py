import time
import mock

field_size = 100
num_iterations = 1000000
state = mock.GlobalState()
field1 = state.declare_field("field1", field_size)
field2 = state.declare_field("field2", field_size)
result = state.declare_field("result", field_size)

start_time = time.time()
mock.direct_randomize_field(field1, 0.0, 1.0)
mock.direct_randomize_field(field2, 0.0, 1.0)
for i in range(num_iterations):
  mock.direct_add_fields(field1, field2, result)
end_time = time.time()
print("Python run time: ", end_time - start_time)

