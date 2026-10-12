#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include "greekstream/models/basket_mc.hpp"
#include "greekstream/models/black_scholes.hpp"
#include "greekstream/random.hpp"

using namespace greekstream;

TEST(Random, InverseNormalIsAccurate) {
    for (double x : {-5.0, -2.5, -1.0, -0.1, 0.0, 0.3, 1.7, 3.2, 6.0}) {
        EXPECT_NEAR(inverse_normal(ncdf(x)), x, 2e-8 * std::max(1.0, std::abs(x))) << x;
    }
}

TEST(Random, StreamsAreReproducibleAndDistinct) {
    Rng a = Rng::stream(7, 3);
    Rng b = Rng::stream(7, 3);
    Rng c = Rng::stream(7, 4);
    for (int i = 0; i < 100; ++i) {
        const auto va = a.next();
        EXPECT_EQ(va, b.next());
        EXPECT_NE(va, c.next());
    }
}

namespace {
MarketData single(double s, double v, double r) { return MarketData{{s}, {v}, r}; }
}  // namespace

// A one-asset "basket" is a vanilla: MC must agree with Black-Scholes.
TEST(BasketMc, SingleAssetMatchesBlackScholes) {
    const BasketOption opt{{1.0}, 100.0, 1.0, OptionType::Call};
    McConfig cfg;
    cfg.paths = 400'000;
    cfg.correlation = 0.0;
    const double mc = basket_price(opt, single(100, 0.2, 0.03), cfg);
    const double bs = black_scholes<double>(OptionType::Call, 100.0, 100.0, 0.03, 0.2, 1.0);
    EXPECT_NEAR(mc, bs, 0.05);  // ~3 standard errors at 400k antithetic paths
}

TEST(BasketMc, AadGreeksMatchClosedFormForSingleAsset) {
    const BasketOption opt{{1.0}, 105.0, 0.75, OptionType::Put};
    McConfig cfg;
    cfg.paths = 400'000;
    cfg.correlation = 0.0;
    const auto g = basket_greeks_aad(opt, single(100, 0.25, 0.02), cfg);
    const auto bs = black_scholes_greeks(OptionType::Put, 100, 105, 0.02, 0.25, 0.75);
    EXPECT_NEAR(g.value, bs.price, 0.05);
    EXPECT_NEAR(g.delta[0], bs.delta, 0.01);
    EXPECT_NEAR(g.vega[0], bs.vega, 0.5);
    EXPECT_NEAR(g.rho, bs.rho, 0.5);
}

// Pathwise AAD and bump-and-revalue with common random numbers estimate the
// same derivative of the same discrete estimator: they must agree closely.
TEST(BasketMc, AadMatchesBumpOnFiveAssetBasket) {
    const BasketOption opt{{0.3, 0.2, 0.2, 0.15, 0.15}, 100.0, 1.0, OptionType::Call};
    const MarketData mkt{{100, 95, 110, 90, 105}, {0.2, 0.25, 0.3, 0.18, 0.22}, 0.02};
    McConfig cfg;
    cfg.paths = 50'000;
    const auto aad = basket_greeks_aad(opt, mkt, cfg);
    const auto bump = basket_greeks_bump(opt, mkt, cfg, 1e-6);
    EXPECT_NEAR(aad.value, bump.value, 1e-12);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(aad.delta[i], bump.delta[i], 2e-3) << i;
        EXPECT_NEAR(aad.vega[i], bump.vega[i], 2e-2) << i;
    }
    EXPECT_NEAR(aad.rho, bump.rho, 2e-2);
}

TEST(BasketMc, BitIdenticalAcrossThreadCounts) {
    const BasketOption opt{{0.5, 0.5}, 100.0, 1.0, OptionType::Call};
    const MarketData mkt{{100, 100}, {0.2, 0.3}, 0.01};
    McConfig one;
    one.paths = 100'000;
    McConfig four = one;
    four.threads = 4;
    const auto g1 = basket_greeks_aad(opt, mkt, one);
    const auto g4 = basket_greeks_aad(opt, mkt, four);
    // Exact equality on purpose: determinism is a contract, not a tolerance.
    EXPECT_EQ(std::memcmp(&g1.value, &g4.value, sizeof(double)), 0);
    EXPECT_EQ(g1.delta, g4.delta);
    EXPECT_EQ(g1.vega, g4.vega);
    EXPECT_EQ(std::memcmp(&g1.rho, &g4.rho, sizeof(double)), 0);
    EXPECT_EQ(basket_price(opt, mkt, one), basket_price(opt, mkt, four));
}
