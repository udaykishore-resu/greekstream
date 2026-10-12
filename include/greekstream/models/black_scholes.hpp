#pragma once

// Black-Scholes(-Merton) European option, written ONCE as a template on the
// number type. Instantiated with double it prices; instantiated with
// aad::Number it records itself on the tape and yields every Greek in one
// backward sweep. The closed-form Greeks below exist only to test that.

#include <cmath>

#include "greekstream/aad/number.hpp"
#include "greekstream/market.hpp"

namespace greekstream {

template <typename T>
[[nodiscard]] T black_scholes(OptionType type, const T& spot, double strike, const T& rate, const T& vol,
                              double expiry, double dividend = 0.0) {
    using std::exp;
    using std::log;
    using std::sqrt;
    const T std_dev = vol * std::sqrt(expiry);
    const T d1 = (log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * expiry) / std_dev;
    const T d2 = d1 - std_dev;
    const T df = exp(-rate * expiry);
    const double carry = std::exp(-dividend * expiry);
    if (type == OptionType::Call) {
        return spot * carry * ncdf(d1) - strike * df * ncdf(d2);
    }
    return strike * df * ncdf(-d2) - spot * carry * ncdf(-d1);
}

struct AnalyticGreeks {
    double price;
    double delta;
    double gamma;
    double vega;
    double rho;
    double theta;
};

[[nodiscard]] inline AnalyticGreeks black_scholes_greeks(OptionType type, double spot, double strike, double rate,
                                                         double vol, double expiry, double dividend = 0.0) {
    const double sd = vol * std::sqrt(expiry);
    const double d1 = (std::log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * expiry) / sd;
    const double d2 = d1 - sd;
    const double df = std::exp(-rate * expiry);
    const double carry = std::exp(-dividend * expiry);
    const double pdf = npdf(d1);
    AnalyticGreeks g{};
    g.gamma = carry * pdf / (spot * sd);
    g.vega = spot * carry * pdf * std::sqrt(expiry);
    if (type == OptionType::Call) {
        g.price = spot * carry * ncdf(d1) - strike * df * ncdf(d2);
        g.delta = carry * ncdf(d1);
        g.rho = strike * expiry * df * ncdf(d2);
        g.theta = -spot * carry * pdf * vol / (2.0 * std::sqrt(expiry)) - rate * strike * df * ncdf(d2) +
                  dividend * spot * carry * ncdf(d1);
    } else {
        g.price = strike * df * ncdf(-d2) - spot * carry * ncdf(-d1);
        g.delta = carry * (ncdf(d1) - 1.0);
        g.rho = -strike * expiry * df * ncdf(-d2);
        g.theta = -spot * carry * pdf * vol / (2.0 * std::sqrt(expiry)) + rate * strike * df * ncdf(-d2) -
                  dividend * spot * carry * ncdf(-d1);
    }
    return g;
}

}  // namespace greekstream
