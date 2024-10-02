#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stack>
#include <vector>

#include "mock.hpp"
#include "mock_api.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mock, m) {
  py::class_<mock::Field, std::shared_ptr<mock::Field>>(m, "Field")
      .def("__getitem__", [](const mock::Field &f, size_t i) { return f[i]; })
      .def("__setitem__", [](mock::Field &f, size_t i, double value) { f[i] = value; })
      .def("size", &mock::Field::size);

  py::class_<mock::GlobalState>(m, "GlobalState")
      .def(py::init<>())
      .def("declare_field", &mock::GlobalState::declare_field, py::return_value_policy::reference)
      .def("get_field", &mock::GlobalState::get_field, py::return_value_policy::reference);

  py::class_<mock_api::Trace, std::shared_ptr<mock_api::Trace>>(m, "Trace")
      .def("start", &mock_api::Trace::start)
      .def("stop", &mock_api::Trace::stop)
      .def("run", &mock_api::Trace::run);

  // Direct
  m.def("direct_randomize_field", &mock::randomize_field);
  m.def("direct_add_fields", &mock::add_fields);
  m.def("direct_print_field", &mock::print_field);

  // Delayed
  m.def("create_trace", &mock_api::create_trace);
  m.def("randomize_field", &mock_api::randomize_field);
  m.def("randomize_field2", &mock_api::randomize_field);
  m.def("add_fields", &mock_api::add_fields);
  m.def("print_field", &mock_api::print_field);
  m.def("for_loop", py::overload_cast<int, int, const std::shared_ptr<mock_api::Method> &>(&mock_api::for_loop));
  m.def("for_loop", py::overload_cast<int, int, const std::function<void()> &>(&mock_api::for_loop));
}
