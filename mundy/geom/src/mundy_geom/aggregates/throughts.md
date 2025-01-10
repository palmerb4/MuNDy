












# Aggregates should be augmentable! 

The add_augment function should check if the specified augment is compatible with the current class.
Compatibility likely comes down to checking topology/rank at compile time





#include <iostream>
#include <vector>
#include <array>

// Base Sphere class
class Sphere {
private:
    std::vector<double> center_data_; // Center coordinates
    double radius_;

public:
    Sphere(const std::vector<double>& center_data, double radius)
        : center_data_(center_data), radius_(radius) {}

    const std::vector<double>& center_data() const { return center_data_; }
    double radius() const { return radius_; }

    void print() const {
        std::cout << "Sphere: Center = (";
        for (size_t i = 0; i < center_data_.size(); ++i) {
            std::cout << center_data_[i];
            if (i + 1 < center_data_.size()) std::cout << ", ";
        }
        std::cout << "), Radius = " << radius_ << '\n';
    }

    // Chaining to add the next augment (supports additional template parameters)
    template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
    auto add_augment(Args&&... args) const {
        return NextAugment<Sphere, AugmentTemplates...>(*this, std::forward<Args>(args)...);
    }
};

// Mobility augment
template <typename Base>
class MobilityAugment : public Base {
private:
    std::vector<double> velocity_data_; // Velocity for each dimension

public:
    MobilityAugment(const Base& base, const std::vector<double>& velocity_data)
        : Base(base), velocity_data_(velocity_data) {}

    const std::vector<double>& velocity_data() const { return velocity_data_; }

    void update_position() {
        auto& center = this->center_data();
        std::cout << "Updating position with velocity: ";
        for (size_t i = 0; i < velocity_data_.size(); ++i) {
            std::cout << velocity_data_[i] << " ";
        }
        std::cout << '\n';
    }

    void print() const {
        Base::print();
        std::cout << "Velocity = (";
        for (size_t i = 0; i < velocity_data_.size(); ++i) {
            std::cout << velocity_data_[i];
            if (i + 1 < velocity_data_.size()) std::cout << ", ";
        }
        std::cout << ")\n";
    }

    // Chaining to add the next augment
    template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
    auto add_augment(Args&&... args) const {
        return NextAugment<MobilityAugment, AugmentTemplates...>(*this, std::forward<Args>(args)...);
    }
};

// Color augment
template <typename Base, typename ColorType>
class ColorAugment : public Base {
private:
    ColorType color_data_; // RGB color

public:
    ColorAugment(const Base& base, const ColorType& color_data)
        : Base(base), color_data_(color_data) {}

    const ColorType& color_data() const { return color_data_; }

    void paint() {
        std::cout << "Painting object with color: (";
        for (size_t i = 0; i < 3; ++i) {
            std::cout << color_data_[i];
            if (i + 1 < 3) std::cout << ", ";
        }
        std::cout << ")\n";
    }

    void print() const {
        Base::print();
        std::cout << "Color = (" << color_data_[0] << ", " << color_data_[1]
                  << ", " << color_data_[2] << ")\n";
    }

    // Chaining to add the next augment
    template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
    auto add_augment(Args&&... args) const {
        return NextAugment<ColorAugment, AugmentTemplates...>(*this, std::forward<Args>(args)...);
    }
};

// Example usage
int main() {
    std::vector<double> center = {1.0, 2.0, 3.0};
    double radius = 5.0;
    std::vector<double> velocity = {0.1, 0.2, 0.3};
    std::array<double, 3> color = {0.5, 0.7, 0.9};

    // Build the object with chaining
    auto augmented_sphere = Sphere(center, radius)
        .add_augment<MobilityAugment>(velocity)
        .add_augment<ColorAugment, std::array<double, 3>>(color);

    // Use the augmented object
    augmented_sphere.print();
    augmented_sphere.update_position();
    augmented_sphere.paint();

    return 0;
}
