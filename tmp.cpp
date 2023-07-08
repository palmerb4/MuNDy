#include <iostream>
#include <gtest/gtest.h>

int main() {
    std::cout << "Google Test Version: " << ::testing::internal::GetGoogleTestVersion() << std::endl;
    return 0;
}

