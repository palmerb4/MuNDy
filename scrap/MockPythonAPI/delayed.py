import time
import mock

field_size = 100
num_iterations = 1000000
state = mock.GlobalState()
field1 = state.declare_field("field1", field_size)
field2 = state.declare_field("field2", field_size)
result = state.declare_field("result", field_size)

# Start a trace to detect and collect all calls to the mock for delayed execution
trace = mock.create_trace()
trace.start()
setup_start_time = time.time()

mock.randomize_field(field1, 0.0, 1.0)
mock.randomize_field(field2, 0.0, 1.0)

def block():
  mock.add_fields(field1, field2, result)

mock.for_loop(0, num_iterations, block)

setup_end_time = time.time()
print("Python setup time: ", setup_end_time - setup_start_time)
trace.stop()

# Run the trace
trace_start_time = time.time()
trace.run()
trace_end_time = time.time()
print("Python trace run time: ", trace_end_time - trace_start_time)
