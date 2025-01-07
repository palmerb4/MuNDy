#pragma once

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stack>
#include <vector>

namespace mock {

class Field {
 public:
  const double &operator[](size_t i) const {
    return data_[i];
  }

  double &operator[](size_t i) {
    return data_[i];
  }

  size_t size() const {
    return data_.size();
  }

 private:
  Field(const std::string &name, const size_t size) : name_(name), data_(size) {
  }
  static std::unique_ptr<Field> create(const std::string &name, const size_t size) {
    // Can't use make_unique because the constructor is private.
    return std::unique_ptr<Field>(new Field(name, size));
  }
  std::string name_;
  std::vector<double> data_;
  friend class GlobalState;
};

class GlobalState {
 public:
  Field &declare_field(const std::string &name, size_t size) {
    fields_[name] = Field::create(name, size);
    return *fields_[name];
  }

  Field &get_field(const std::string &name) {
    return *fields_[name];
  }

 private:
  std::map<std::string, std::unique_ptr<Field>> fields_;
};

void randomize_field(Field &field, double min, double max) {
  for (size_t i = 0; i < field.size(); ++i) {
    field[i] = min + (max - min) * rand() / RAND_MAX;
  }
}

void add_fields(Field &field1, Field &field2, Field &result) {
  for (size_t i = 0; i < field1.size(); ++i) {
    result[i] = field1[i] + field2[i];
  }
}

void print_field(const Field &field) {
  for (size_t i = 0; i < field.size(); ++i) {
    std::cout << field[i] << " ";
  }
  std::cout << std::endl;
}

}  // namespace mock
