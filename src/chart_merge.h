#pragma once

namespace cumesh {

// Shared by the GPU cost kernel and the standalone host regression test.
// The public entry points validate target >= 2 and strength in [0, 1].
#if defined(__CUDACC__)
__host__ __device__
#endif
inline float chart_merge_cost_multiplier(int faces_a, int faces_b, int target, float strength) {
    int count = faces_a < faces_b ? faces_a : faces_b;
    if (strength == 0.0f || count >= target) return 1.0f;
    if (count < 1) count = 1;
    const float fade = float(target - count) / float(target - 1);
    return 1.0f - strength * fade;
}

} // namespace cumesh
