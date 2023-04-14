Current, simplest possible approach (simplest to us, not necessarily the users)

1. Each `Method` and `Kernel`will be host-only. 
2. Move the GPU Kernel launch operation inside the `Kernel.execute()`. Not all parallel ops (like parallel_reduce) can be wrapped in a single `for_each_entity_run`. Let each `Kernel` decide if they are for/reduce or scan op.
3. We can't completely abstract away Kokkos from users who'll be writing custom kernels. But I guess that's not too much to ask, all we're asking for is to put a KOKKOS_LAMBDA or KOKKOS_FUNCTION macros before the functions they define.