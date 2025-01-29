#include <array>
#include <iostream>
#include <vector>

// Aggregates
struct SPHERE {};

// Qualifiers
struct SHARED_RADIUS {};
struct SHARED_LENGTH {};
struct SHARED_MASS {};

// Augments
struct COLORED {};
struct TEXTURED {};
struct MOTILE {};

template <typename... Tags>
struct aggregate_type;

template <typename Base, typename... Tags>
struct augment_type;

template <typename... Tags>
struct entity_view_type;

// Helper aliases
template <typename... Tags>
using aggregate_type_t = typename aggregate_type<Tags...>::type;

template <typename... Tags>
using entity_view_type_t = typename entity_view_type<Tags...>::type;

// Base Sphere class
class SphereData {
   private:
    std::vector<double> center_data_;  // Center coordinates
    double radius_;

   public:
    SphereData(const std::vector<double>& center_data, double radius)
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

    // Chaining to add the next augment (supports additional template
    // parameters)
    template <template <typename, typename...> class NextAugment,
              typename... AugmentTemplates, typename... Args>
    auto add_augment(Args&&... args) const {
        return NextAugment<SphereData, AugmentTemplates...>(
            *this, std::forward<Args>(args)...);
    }
};

enum class TextureType { Matte, Glossy, Metallic };

template <TextureType Texture>
struct TextureWrapper {
    static constexpr TextureType value = Texture;
};

template <typename Base, typename Texture>
class TextureDataAugment : public Base {
   public:
    TextureDataAugment(const Base& base) : Base(base) {}

    void print() const {
        Base::print();
        std::cout << "Texture: ";
        switch (Texture::value) {
            case TextureType::Matte:
                std::cout << "Matte\n";
                break;
            case TextureType::Glossy:
                std::cout << "Glossy\n";
                break;
            case TextureType::Metallic:
                std::cout << "Metallic\n";
                break;
        }
    }

    // Chaining to add the next augment (supports additional template
    // parameters)
    template <template <typename, typename...> class NextAugment,
              typename... AugmentTemplates, typename... Args>
    auto add_augment(Args&&... args) const {
        return NextAugment<TextureDataAugment, AugmentTemplates...>(
            *this, std::forward<Args>(args)...);
    }
};

// Mobility augment
template <typename Base>
class MobilityDataAugment : public Base {
   private:
    std::vector<double> velocity_data_;  // Velocity for each dimension

   public:
    MobilityDataAugment(const Base& base, const std::vector<double>& velocity_data)
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
    template <template <typename, typename...> class NextAugment,
              typename... AugmentTemplates, typename... Args>
    auto add_augment(Args&&... args) const {
        return NextAugment<MobilityDataAugment, AugmentTemplates...>(
            *this, std::forward<Args>(args)...);
    }
};

// Color augment
template <typename Base, typename ColorType>
class ColorDataAugment : public Base {
   private:
    ColorType color_data_;  // RGB color

   public:
    ColorDataAugment(const Base& base, const ColorType& color_data)
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
    template <template <typename, typename...> class NextAugment,
              typename... AugmentTemplates, typename... Args>
    auto add_augment(Args&&... args) const {
        return NextAugment<ColorDataAugment, AugmentTemplates...>(
            *this, std::forward<Args>(args)...);
    }
};

// Specializations for specific tag combinations
template <>
struct aggregate_type<SPHERE> {
    using type = SphereData;
};



// Example usage
int main() {
    std::vector<double> center = {1.0, 2.0, 3.0};
    double radius = 5.0;
    std::vector<double> velocity = {0.1, 0.2, 0.3};
    std::array<double, 3> color = {0.5, 0.7, 0.9};

    // Build the object with chaining
    auto augmented_sphere_data =
        SphereData(center, radius)
            .add_augment<TextureDataAugment, TextureWrapper<TextureType::Matte>>()
            .add_augment<MobilityDataAugment>(velocity)
            .add_augment<ColorDataAugment, std::array<double, 3>>(color);

    // Use the augmented object
    augmented_sphere_data.print();
    augmented_sphere_data.update_position();
    augmented_sphere_data.paint();

    return 0;
}
