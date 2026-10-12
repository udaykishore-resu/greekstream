#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "greekstream/risk/risk_engine.hpp"

using namespace greekstream;

namespace {

std::vector<Position> make_book(std::size_t underlyings) {
    std::vector<Position> book;
    for (std::size_t u = 0; u < underlyings; ++u) {
        const double k = 90.0 + 5.0 * static_cast<double>(u % 5);
        book.push_back({u, OptionType::Call, Exercise::European, k, 0.5, 10.0});
        book.push_back({u, OptionType::Put, Exercise::European, k + 10, 1.0, -5.0});
        book.push_back({u, OptionType::Put, Exercise::American, k, 0.75, 3.0});
    }
    return book;
}

MarketData make_market(std::size_t underlyings) {
    MarketData m;
    for (std::size_t u = 0; u < underlyings; ++u) {
        m.spots.push_back(95.0 + static_cast<double>(u));
        m.vols.push_back(0.15 + 0.01 * static_cast<double>(u % 10));
    }
    m.rate = 0.03;
    return m;
}

}  // namespace

TEST(RiskEngine, AadMatchesBumpForEverySensitivity) {
    constexpr std::size_t kN = 8;
    RiskEngine engine{make_book(kN), 100};
    const auto mkt = make_market(kN);
    const auto aad = engine.aad(mkt);
    const auto bump = engine.bump(mkt);
    EXPECT_NEAR(aad.value, bump.value, 1e-9);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_NEAR(aad.delta[i], bump.delta[i], 1e-3 * std::max(1.0, std::abs(aad.delta[i]))) << i;
        EXPECT_NEAR(aad.vega[i], bump.vega[i], 1e-2 * std::max(1.0, std::abs(aad.vega[i]))) << i;
    }
    EXPECT_NEAR(aad.rho, bump.rho, 1e-2 * std::max(1.0, std::abs(aad.rho)));
}

TEST(RiskEngine, TapeIsReusedAcrossTicks) {
    constexpr std::size_t kN = 4;
    RiskEngine engine{make_book(kN), 50};
    auto mkt = make_market(kN);
    (void)engine.aad(mkt);
    const auto bytes = engine.tape_bytes();
    const auto nodes = engine.last_tape_nodes();
    for (int tick = 0; tick < 20; ++tick) {
        for (auto& s : mkt.spots) {
            s *= 1.0 + 0.001 * (tick % 3 - 1);
        }
        (void)engine.aad(mkt);
        EXPECT_EQ(engine.tape_bytes(), bytes);
        EXPECT_EQ(engine.last_tape_nodes(), nodes);
    }
}
