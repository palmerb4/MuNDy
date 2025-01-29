template <typename... Tags>
struct aggregate_t;

template <typename... Tags>
struct entity_view_t;

// Specializations for specific tag combinations
template <>
struct aggregate_t<SPHERE, COLORED> {
    using type = ColoredSphereData; // Example aggregate type
};

template <>
struct entity_view_t<SPHERE, COLORED> {
    using type = ColoredSphereView; // Example entity view type
};

// Helper aliases
template <typename... Tags>
using aggregate_t_t = typename aggregate_t<Tags...>::type;

template <typename... Tags>
using entity_view_t_t = typename entity_view_t<Tags...>::type;

template <typename... Tags>
class AggregateRepository {
private:
    using AggregateType = aggregate_t_t<Tags...>;
    using InstanceGenerator = std::function<AggregateType()>;

    static std::vector<InstanceGenerator>& get_instance_generators() {
        static std::vector<InstanceGenerator> generators;
        return generators;
    }

public:
    // Register an instance generator for aggregates with these tags
    static void register_instance_generator(const InstanceGenerator& generator) {
        get_instance_generators().emplace_back(generator);
    }

    // Run a functor on each aggregate and entity
    template <typename AlgorithmToRunPerObject>
    static void for_each_object_run(const AlgorithmToRunPerObject& functor) {
        using EntityViewType = entity_view_t_t<Tags...>;

        for (const auto& instance_generator : get_instance_generators()) {
            // Generate the aggregate
            AggregateType aggregate = instance_generator();

            // Iterate over entities in the aggregate
            aggregate.for_each_entity([&](stk::mesh::Entity entity) {
                EntityViewType view = aggregate.get_entity_view(entity);
                functor(view);
            });
        }
    }
};

auto declare_sphere_data(double position, double radius) {
    using AggregateType = aggregate_t_t<SPHERE>;

    AggregateType aggregate{position, radius};

    // Register the instance generator
    AggregateRepository<SPHERE>::register_instance_generator(
        [aggregate]() -> AggregateType { return aggregate; });

    return aggregate;
}

template <typename AugmentTag, typename AggregateType, typename... Args>
auto add_augment(AggregateType& base, AugmentTag tag, Args&&... args) {
    using AugmentedType = aggregate_t_t<SPHERE, AugmentTag>;

    AugmentedType augmented{base, std::forward<Args>(args)...};

    // Register the augmented instance generator
    AggregateRepository<SPHERE, AugmentTag>::register_instance_generator(
        [augmented]() -> AugmentedType { return augmented; });

    return augmented;
}

struct AABBComputer {
    template <typename EntityView>
    void operator()(EntityView& view) const {
        std::cout << "Computing AABB for entity at position: " << view.get_position() << "\n";
    }
};

int main() {
    using namespace sam::geom;

    // Declare sphere data and augment with color
    auto colored_sphere = declare_sphere_data(0.0, 5.0)
                              .add_augment(COLORED{}, "red");

    // Dispatch the operation
    for_each<SPHERE, COLORED>(AABBComputer{});

    return 0;
}
