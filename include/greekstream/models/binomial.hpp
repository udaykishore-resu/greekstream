#pragma once

// Cox-Ross-Rubinstein binomial tree for European and American options,
// templated on the number type like every model here. With aad::Number the
// backward induction (including the early-exercise max) is recorded on the
// tape, so American Greeks come out of the same single sweep.

#include <cmath>
#include <cstddef>
#include <vector>

#include "greekstream/aad/number.hpp"
#include "greekstream/market.hpp"

namespace greekstream {

template <typename T>
[[nodiscard]] T binomial_crr(OptionType type, Exercise exercise, const T& spot, double strike, const T& rate,
                             const T& vol, double expiry, std::size_t steps) {
    using std::exp;
    using std::sqrt;
    const double dt = expiry / static_cast<double>(steps);
    const double phi = type == OptionType::Call ? 1.0 : -1.0;  // payoff = max(phi (S - K), 0)
    const T up = exp(vol * std::sqrt(dt));
    const T down = 1.0 / up;
    const T ratio = up * up;
    const T growth = exp(rate * dt);
    const T p_up = (growth - down) / (up - down);
    const T disc_up = p_up / growth;
    const T disc_down = (1.0 - p_up) / growth;

    // Terminal layer: S d^n, S d^(n-1) u, ... built multiplicatively.
    std::vector<T> values;
    values.reserve(steps + 1);
    T lowest = spot;
    for (std::size_t i = 0; i < steps; ++i) {
        lowest = lowest * down;
    }
    T node_spot = lowest;
    for (std::size_t j = 0; j <= steps; ++j) {
        values.push_back(fmax(phi * (node_spot - strike), 0.0));
        node_spot = node_spot * ratio;
    }

    // Backward induction. Tape cost matters here: each tree node is written
    // as ONE expression (continuation and exercise value fused), so with
    // aad::Number a node costs one tape record (plus one for its spot on
    // American trees) instead of four or five.
    for (std::size_t step = steps; step-- > 0;) {
        if (exercise == Exercise::American) {
            lowest = lowest * up;  // lowest node of this slice: S d^step
            node_spot = lowest;
            for (std::size_t j = 0; j <= step; ++j) {
                values[j] = fmax(disc_down * values[j] + disc_up * values[j + 1],
                                 fmax(phi * (node_spot - strike), 0.0));
                node_spot = node_spot * ratio;
            }
        } else {
            for (std::size_t j = 0; j <= step; ++j) {
                values[j] = disc_down * values[j] + disc_up * values[j + 1];
            }
        }
    }
    return values[0];
}

}  // namespace greekstream
