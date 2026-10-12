#pragma once

// Deterministic random streams for Monte Carlo.
//
// Determinism contract: paths are grouped into fixed-size blocks and block b
// always draws from stream(seed, b). Results are reduced in block order, so a
// run is bit-for-bit identical on 1 thread or 16 -- you can debug a 16-thread
// production run on your laptop.

#include <cmath>
#include <cstdint>

namespace greekstream {

__extension__ using uint128 = unsigned __int128;

class Rng {
public:
    explicit constexpr Rng(std::uint64_t seed) noexcept {
        for (auto& w : s_) {
            w = splitmix64(seed);
        }
    }

    // Independent stream for block `index` of a run seeded with `seed`.
    [[nodiscard]] static constexpr Rng stream(std::uint64_t seed, std::uint64_t index) noexcept {
        std::uint64_t x = seed ^ (index * 0xD1B54A32D192ED03ULL);
        return Rng{splitmix64(x)};
    }

    constexpr std::uint64_t next() noexcept {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform in the open interval (0, 1): never 0, never 1.
    constexpr double uniform_open() noexcept { return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53; }

    double normal() noexcept;

private:
    static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }
    static constexpr std::uint64_t splitmix64(std::uint64_t& x) noexcept {
        std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    std::uint64_t s_[4]{};
};

// Inverse standard normal CDF (Acklam's rational approximation, relative
// error < 1.15e-9). Inversion (rather than Box-Muller) keeps a one-to-one
// map from uniforms to normals, which quasi-random sequences require later.
[[nodiscard]] inline double inverse_normal(double p) noexcept {
    constexpr double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                            1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
    constexpr double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                            6.680131188771972e+01,  -1.328068155288572e+01};
    constexpr double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                            -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
    constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                            3.754408661907416e+00};
    constexpr double p_low = 0.02425;
    constexpr double p_high = 1.0 - p_low;

    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > p_high) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double q = p - 0.5;
    const double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

inline double Rng::normal() noexcept { return inverse_normal(uniform_open()); }

}  // namespace greekstream
