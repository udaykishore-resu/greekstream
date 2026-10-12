#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace greekstream {

enum class OptionType : std::uint8_t { Call, Put };
enum class Exercise : std::uint8_t { European, American };

// Everything the book is sensitive to. One entry per underlying.
struct MarketData {
    std::vector<double> spots;
    std::vector<double> vols;
    double rate{0.0};

    [[nodiscard]] std::size_t underlyings() const noexcept { return spots.size(); }
};

// First-order sensitivities of a scalar value to every market input.
struct Greeks {
    double value{0.0};
    std::vector<double> delta;  // d value / d spot_i
    std::vector<double> vega;   // d value / d vol_i   (per 1.00 of vol, not per 1%)
    double rho{0.0};            // d value / d rate

    [[nodiscard]] std::size_t sensitivities() const noexcept { return delta.size() + vega.size() + 1; }
};

}  // namespace greekstream
