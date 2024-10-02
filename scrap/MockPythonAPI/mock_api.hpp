#pragma once

#include "mock.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stack>
#include <vector>

namespace mock_api {
// Notice that this is in a different namespace so that we can have the same functions perform delayed execution
class Method {
 public:
  virtual ~Method() = default;
  virtual void run() = 0;
};

class Routine : public Method {
 public:
  void append(std::shared_ptr<Method> action) {
    actions.push_back(action);
  }

  void run() override {
    for (const auto &action : actions) {
      action->run();
    }
  }

 private:
  std::vector<std::shared_ptr<Method>> actions;
};

// Global trace pointer (used to hide the trace from the python API)
class Trace;
std::stack<std::shared_ptr<Trace>> trace_stack;

class Trace : public Method, public std::enable_shared_from_this<Trace> {
 public:
  void start() {
    // Push this trace onto the stack, making it the current active trace
    trace_stack.push(shared_from_this());
  }

  void stop() {
    // Ensure the current trace is the one stopping
    if (trace_stack.empty() || trace_stack.top().get() != this) {
      throw std::runtime_error("Mismatched trace start/stop.");
    }
    trace_stack.pop();  // Remove this trace from the stack
  }

  void run() override {
    routine_.run();
  }

  void append(std::shared_ptr<Method> method) {
    routine_.append(method);
  }

  static std::shared_ptr<Trace> get_active_trace() {
    return trace_stack.empty() ? nullptr : trace_stack.top();
  }

 private:
  // The only way to create a trace is through create_trace()
  Trace() = default;
  Routine routine_;  // Holds the actions for this trace
  friend std::shared_ptr<Trace> create_trace();
};

std::shared_ptr<Trace> create_trace() {
  // Cant use make_shared due to private constructor
  return std::shared_ptr<Trace>(new Trace());
}

std::shared_ptr<Trace> trace_function(const std::function<void()> &func) {
  auto func_trace = create_trace();
  func_trace->start();
  // Run the function. It should automatically append actions to the trace
  func();
  func_trace->stop();
  return func_trace;
}

class ForLoop : public Method {
 public:
  ForLoop(size_t start, size_t end, std::shared_ptr<Method> action) : start_(start), end_(end), action_(action) {
  }

  void run() override {
    for (size_t i = start_; i < end_; ++i) {
      action_->run();
    }
  }

 private:
  size_t start_;
  size_t end_;
  std::shared_ptr<Method> action_;
};

class RandomizeField : public Method {
 public:
  RandomizeField(mock::Field &field, double min, double max) : field_(field), min_(min), max_(max) {
  }

  void run() override {
    mock::randomize_field(field_, min_, max_);
  }

 private:
  mock::Field &field_;
  double min_;
  double max_;
};

class AddFields : public Method {
 public:
  AddFields(mock::Field &field1, mock::Field &field2, mock::Field &result)
      : field1_(field1), field2_(field2), result_(result) {
  }

  void run() override {
    mock::add_fields(field1_, field2_, result_);
  }

 private:
  mock::Field &field1_;
  mock::Field &field2_;
  mock::Field &result_;
};

class PrintField : public Method {
 public:
  PrintField(const mock::Field &field) : field_(field) {
  }

  void run() override {
    mock::print_field(field_);
  }

 private:
  const mock::Field &field_;
};

void randomize_field(mock::Field &field, double min, double max) {
  Trace::get_active_trace()->append(std::make_shared<RandomizeField>(field, min, max));
}

void add_fields(mock::Field &field1, mock::Field &field2, mock::Field &result) {
  Trace::get_active_trace()->append(std::make_shared<AddFields>(field1, field2, result));
}

void print_field(mock::Field &field) {
  Trace::get_active_trace()->append(std::make_shared<PrintField>(field));
}

void for_loop(int start, int stop, const std::shared_ptr<Method> &action) {
  Trace::get_active_trace()->append(std::make_shared<ForLoop>(start, stop, action));
}

void for_loop(int start, int stop, const std::function<void()> &function_block) {
  auto func_trace = trace_function(function_block);
  for_loop(start, stop, func_trace);
}

}  // namespace mock_api
