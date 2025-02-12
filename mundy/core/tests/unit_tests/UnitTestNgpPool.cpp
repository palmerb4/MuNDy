// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
//                                                 Author: Bryce Palmer
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

// External libs
#include <gmock/gmock.h>  // for EXPECT_THAT, HasSubstr, etc
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <iostream>
#include <stdexcept>  // for logic_error, invalid_argument, etc
#include <vector> 

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize

// Mundy libs
#include <mundy_core/NgpView.hpp>  // for NgpView
#include <mundy_core/NgpPool.hpp>  // for NgpPool

namespace mundy {

namespace core {

namespace {

TEST(NgpPoolTest, DefaultConstruction) {
  // A default constructed pool is empty
  NgpPool<int> pool;
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.capacity(), 0);
}

TEST(NgpPoolTest, Simple) {
  size_t capacity = 10;
  NgpPool<int> pool(capacity);
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.capacity(), capacity);

  // Insert some values into the pool
  int value = 42;
  for (size_t i = 0; i < capacity; ++i) {
    pool.add(value);
    EXPECT_EQ(pool.size(), i + 1);
  }

  // Acquire all the values from the pool
  // In the end, the pool should be empty, but it's capacity should be unchanged.
  for (size_t i = 0; i < capacity; ++i) {
    EXPECT_EQ(pool.acquire(), value) << "Failed on iteration " << i;
  }
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.capacity(), capacity);

  // If we push back a value, the size should increase.
  pool.add(value);
  EXPECT_EQ(pool.size(), 1);
  EXPECT_EQ(pool.capacity(), capacity);
  EXPECT_EQ(pool.acquire(), value);
}

TEST(NgpPoolTest, SimpleHostDevice) {
#ifdef STK_ENABLE_GPU
  // Host and device will differ
  size_t capacity = 10;
  NgpPool<int> pool(capacity);
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.capacity(), capacity);

  // Insert some values into the pool
  int value = 42;
  for (size_t i = 0; i < capacity; ++i) {
    pool.add(value);
    EXPECT_EQ(pool.size(), i + 1);
  }
  // Because we haven't synced the device to the host, the host data should be untouched.
  EXPECT_EQ(pool.size_host(), 0);

  // Acquire all the values from the pool
  // In the end, the pool should be empty, but it's capacity should be unchanged.
  for (size_t i = 0; i < capacity; ++i) {
    EXPECT_EQ(pool.acquire(), value);
  }
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.capacity(), capacity);
  
  // Perform the mod on the host
  for (size_t i = 0; i < capacity; ++i) {
    pool.add_host(value);
    EXPECT_EQ(pool.size_host(), i + 1);
  }

  // Acquire all the values from the pool
  for (size_t i = 0; i < capacity; ++i) {
    EXPECT_EQ(pool.acquire_host(), value);
  }
  EXPECT_EQ(pool.size_host(), 0);
  EXPECT_EQ(pool.capacity(), capacity);

  // If we push back a value, the size should increase.
  int value_to_push_back = 42;
  pool.add(value_to_push_back);
  EXPECT_EQ(pool.size(), 1);
  EXPECT_EQ(pool.size_host(), 0) << "Host should be untouched";
  EXPECT_EQ(pool.capacity(), capacity);
  EXPECT_EQ(pool.acquire(), 42);
  pool.add_host(value_to_push_back);
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.size_host(), 1);
  EXPECT_EQ(pool.capacity(), capacity);
  EXPECT_EQ(pool.acquire_host(), 42);

  // Perform a mod on host and mark/sync to device
  pool.add_host(33);
  pool.add_host(44);
  pool.modify_on_host();
  pool.sync_to_device();
  EXPECT_EQ(pool.size(), 2);
  EXPECT_EQ(pool.size_host(), 2);
  EXPECT_EQ(pool.acquire(), 33);
  EXPECT_EQ(pool.acquire(), 44);
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.size_host(), 2);
  pool.modify_on_device();
  pool.sync_to_host();
  EXPECT_EQ(pool.size_host(), 0);
#else
  // Because we are not using GPU the host and device views are the same
  // Typically, we'll modify on host or device and then perform syncs. The following just affirms that the two are the same.
  size_t capacity = 10;
  NgpPool<int> pool(capacity);
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.size_host(), 0);
  EXPECT_EQ(pool.capacity(), capacity);

  // Insert some values into the pool
  int value = 42;
  for (size_t i = 0; i < capacity; ++i) {
    pool.add(value);
    EXPECT_EQ(pool.size(), i + 1);
    EXPECT_EQ(pool.size_host(), i + 1);
  }

  // Acquire all the values from the pool
  // In the end, the pool should be empty, but it's capacity should be unchanged.
  for (size_t i = 0; i < capacity; ++i) {
    EXPECT_EQ(pool.acquire_host(), value) << "Failed on iteration " << i;
  }
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.size_host(), 0);
  EXPECT_EQ(pool.capacity(), capacity);

  // If we push back a value, the size should increase.
  pool.add_host(value);
  EXPECT_EQ(pool.size(), 1);
  EXPECT_EQ(pool.size_host(), 1);
  EXPECT_EQ(pool.capacity(), capacity);
  EXPECT_EQ(pool.acquire(), value);
#endif
}

TEST(NgpPoolTest, Reserve) {
  size_t capacity = 10;
  NgpPool<int> pool(capacity);
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.capacity(), capacity);

  // Insert some values into the pool
  int value = 42;
  for (size_t i = 0; i < capacity; ++i) {
    pool.add(value);
    EXPECT_EQ(pool.size(), i + 1);
  }

  // Reserve a smaller size
  pool.reserve(capacity - 2);
  EXPECT_EQ(pool.size(), capacity);
  EXPECT_EQ(pool.capacity(), capacity);

  // Reserve a larger size
  size_t new_capacity = 20;
  pool.reserve(new_capacity);
  EXPECT_EQ(pool.size(), capacity);
  EXPECT_EQ(pool.capacity(), new_capacity);

  // Acquire all the values from the pool
  for (size_t i = 0; i < capacity; ++i) {
    EXPECT_EQ(pool.acquire(), value) << "Failed on iteration " << i;
  }
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.capacity(), new_capacity);
}

TEST(NgpPoolTest, OverUnderflow) {
  NgpPool<int> pool;
  EXPECT_EQ(pool.size(), 0);
  EXPECT_EQ(pool.capacity(), 0);
  
#ifndef NDEBUG
  // Only throws in debug
  EXPECT_THROW(pool.acquire(), std::runtime_error) << "Overfetching from the pool should throw in debug";
  EXPECT_THROW(pool.add(42), std::runtime_error) << "Overfilling the pool should throw in debug";
#else
  // Should not throw in release allowing us to enter undefined behavior
  EXPECT_NO_THROW(pool.acquire()) << "Overfetching from the pool should not throw in release";
  EXPECT_NO_THROW(pool.add(42)) << "Overfilling the pool should not throw in release";
#endif
}

TEST(NgpPoolTest, BatchedAdditionsFetches) {
  int value = 42;
  size_t capacity = 10;
  NgpPool<int> pool(capacity);
  for (size_t i = 0; i < capacity; ++i) {
    pool.add(value);
  }

  // Acquire all values
  auto values = pool.acquire(capacity);
  EXPECT_EQ(values.extent(0), capacity);
  EXPECT_EQ(pool.size(), 0);
  for (size_t i = 0; i < values.extent(0); ++i) {
    EXPECT_EQ(values.view_device()(i), value);
    values.view_device()(i) = 33;
  }

  // Add the values back
  pool.add(values);
  EXPECT_EQ(pool.size(), 10);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(pool.acquire(), 33);
  }
  EXPECT_EQ(pool.size(), 0);

  // Acquire a subset of the values
  pool.add(values);
  auto values2 = pool.acquire(capacity - 4);
  EXPECT_EQ(values2.extent(0), capacity - 4);
  for (size_t i = 0; i < values2.extent(0); ++i) {
    EXPECT_EQ(values2.view_device()(i), 33);
  }
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(pool.acquire(), 33);
  }
}


}  // namespace

}  // namespace core

}  // namespace mundy
