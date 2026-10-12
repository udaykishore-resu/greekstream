#include <gtest/gtest.h>

#include <cmath>

#include "greekstream/aad/number.hpp"
#include "greekstream/models/binomial.hpp"
#include "greekstream/models/black_scholes.hpp"

using namespace greekstream;
using aad::ActiveTape;
using aad::Number;
using aad::Tape;

namespace {
struct BsCase {
    OptionType type;
    double s, k, r, v, t;
};
constexpr BsCase kCases[] = {
    {OptionType::Call, 100, 100, 0.05, 0.20, 1.0},  {OptionType::Put, 100, 100, 0.05, 0.20, 1.0},
    {OptionType::Call, 120, 100, 0.01, 0.35, 0.25}, {OptionType::Put, 80, 100, 0.03, 0.15, 2.0},
    {OptionType::Call, 50, 75, 0.00, 0.60, 0.5},
};
}  // namespace

TEST(BlackScholes, KnownValue) {
    // Hull: S=42, K=40, r=10%, vol=20%, T=0.5 -> call 4.76, put 0.81
    EXPECT_NEAR(black_scholes<double>(OptionType::Call, 42.0, 40.0, 0.10, 0.20, 0.5), 4.7594, 1e-4);
    EXPECT_NEAR(black_scholes<double>(OptionType::Put, 42.0, 40.0, 0.10, 0.20, 0.5), 0.8086, 1e-4);
}

TEST(BlackScholes, PutCallParity) {
    for (const auto& c : kCases) {
        const double call = black_scholes<double>(OptionType::Call, c.s, c.k, c.r, c.v, c.t);
        const double put = black_scholes<double>(OptionType::Put, c.s, c.k, c.r, c.v, c.t);
        EXPECT_NEAR(call - put, c.s - c.k * std::exp(-c.r * c.t), 1e-12);
    }
}

// The core claim: the generic template, run on the tape, reproduces the
// closed-form Greeks to machine precision.
TEST(BlackScholes, AadGreeksEqualClosedForm) {
    for (const auto& c : kCases) {
        Tape tape;
        ActiveTape active{tape};
        Number s{c.s}, r{c.r}, v{c.v};
        Number price = black_scholes<Number>(c.type, s, c.k, r, v, c.t);
        price.propagate_to_start();
        const auto g = black_scholes_greeks(c.type, c.s, c.k, c.r, c.v, c.t);
        EXPECT_NEAR(price.value(), g.price, 1e-12);
        EXPECT_NEAR(s.adjoint(), g.delta, 1e-12);
        EXPECT_NEAR(v.adjoint(), g.vega, 1e-10);
        EXPECT_NEAR(r.adjoint(), g.rho, 1e-10);
    }
}

TEST(Binomial, EuropeanConvergesToBlackScholes) {
    for (const auto& c : kCases) {
        const double bs = black_scholes<double>(c.type, c.s, c.k, c.r, c.v, c.t);
        const double tree = binomial_crr<double>(c.type, Exercise::European, c.s, c.k, c.r, c.v, c.t, 2000);
        EXPECT_NEAR(tree, bs, 2e-3 * std::max(1.0, bs)) << static_cast<int>(c.type);
    }
}

TEST(Binomial, AmericanPutCarriesEarlyExercisePremium) {
    const double eu = binomial_crr<double>(OptionType::Put, Exercise::European, 90, 100, 0.08, 0.25, 1.0, 500);
    const double am = binomial_crr<double>(OptionType::Put, Exercise::American, 90, 100, 0.08, 0.25, 1.0, 500);
    EXPECT_GT(am, eu + 0.1);
    EXPECT_GE(am, 10.0);  // never below intrinsic
    // Without dividends an American call is never exercised early.
    const double eu_c = binomial_crr<double>(OptionType::Call, Exercise::European, 90, 100, 0.08, 0.25, 1.0, 500);
    const double am_c = binomial_crr<double>(OptionType::Call, Exercise::American, 90, 100, 0.08, 0.25, 1.0, 500);
    EXPECT_NEAR(am_c, eu_c, 1e-10);
}

TEST(Binomial, AmericanAadDeltaMatchesCentralDifference) {
    constexpr std::size_t kSteps = 300;
    const double s0 = 95.0, k = 100.0, r = 0.05, v = 0.3, t = 1.0;
    Tape tape;
    ActiveTape active{tape};
    Number s{s0}, rr{r}, vv{v};
    Number price = binomial_crr<Number>(OptionType::Put, Exercise::American, s, k, rr, vv, t, kSteps);
    price.propagate_to_start();

    auto px = [&](double spot, double rate, double vol) {
        return binomial_crr<double>(OptionType::Put, Exercise::American, spot, k, rate, vol, t, kSteps);
    };
    EXPECT_NEAR(price.value(), px(s0, r, v), 1e-12);
    const double h = 1e-4;
    EXPECT_NEAR(s.adjoint(), (px(s0 + h, r, v) - px(s0 - h, r, v)) / (2 * h), 1e-5);
    EXPECT_NEAR(vv.adjoint(), (px(s0, r, v + h) - px(s0, r, v - h)) / (2 * h), 1e-4);
    EXPECT_NEAR(rr.adjoint(), (px(s0, r + h, v) - px(s0, r - h, v)) / (2 * h), 1e-4);
}
