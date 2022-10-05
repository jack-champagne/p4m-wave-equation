# wave-equation superscaler

This project is part of ongoing research between Professor Wei Zhu and myself exploring super-scaling techniques for computational difficult physics problems.
The goal is to improve current discretization methods of approximating the wave equation over a plane. When the wave speed through the plane varies over space,
the plane is referred to as non-homogenous media. Current techniques require breaking the plane into a fine mesh grid to approximate the non-linear components
of the physics to get an accurate solution. The goal is that using the fine solver, a coarse solver can be used, and a machine learning technique (hypernets,
group-equivariant neural nets, and a new technique (which is still being flushed out), to upscale to a much better solution that scales well in computational
cost relative to accuracy.
