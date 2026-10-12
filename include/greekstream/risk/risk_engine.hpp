#pragma once

// Real-time risk for a book of vanilla options on many underlyings.
//
// On every tick the engine revalues the whole book and produces
// d(PV)/d(spot_i), d(PV)/d(vol_i) and d(PV)/d(rate):
//
//   aad()   records each position on a reusable, checkpointed tape and
//           sweeps it once: cost ~ k x (one valuation), independent of the
//           number of underlyings.
//   bump()  revalues the book 2N + 2 times (N spots, N vols, rate, base).
//
// European options use Black-Scholes, American options a CRR tree; both are
// the same templates instantiated with double or aad::Number.

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "greekstream/aad/number.hpp"
#include "greekstream/market.hpp"
#include "greekstream/models/binomial.hpp"
#include "greekstream/models/black_scholes.hpp"

namespace greekstream {

struct Position {
    std::size_t underlying;
    OptionType type;
    Exercise exercise;
    double strike;
    double expiry;
    double quantity;
};

class RiskEngine {
public:
    explicit RiskEngine(std::vector<Position> book, std::size_t tree_steps = 100)
        : book_(std::move(book)), tree_steps_(tree_steps) {}

    template <typename T>
    [[nodiscard]] T price(const Position& p, const T& spot, const T& vol, const T& rate) const {
        if (p.exercise == Exercise::European) {
            return black_scholes<T>(p.type, spot, p.strike, rate, vol, p.expiry);
        }
        return binomial_crr<T>(p.type, p.exercise, spot, p.strike, rate, vol, p.expiry, tree_steps_);
    }

    [[nodiscard]] double pv(const MarketData& m) const {
        double total = 0.0;
        for (const Position& p : book_) {
            total += p.quantity * price<double>(p, m.spots[p.underlying], m.vols[p.underlying], m.rate);
        }
        return total;
    }

    // Every sensitivity from one reverse sweep per position.
    //
    // PV is a sum over positions, so instead of recording the whole book on
    // one tape (hundreds of MB for tree-heavy books) we checkpoint: market
    // inputs are recorded once below the mark; each position is recorded,
    // swept back to the mark with its quantity as the seed adjoint, and
    // rewound. Input adjoints accumulate across positions. The live tape never
    // exceeds one position, so it stays cache-resident, and because the tape
    // is a member that is cleared (not freed), steady-state ticks perform no
    // heap allocation inside it.
    [[nodiscard]] Greeks aad(const MarketData& m) {
        using aad::Number;
        tape_.clear();
        aad::ActiveTape active{tape_};
        std::vector<Number> spots(m.spots.begin(), m.spots.end());
        std::vector<Number> vols(m.vols.begin(), m.vols.end());
        Number rate{m.rate};
        tape_.set_mark();

        double pv = 0.0;
        std::size_t peak_nodes = 0;
        for (const Position& p : book_) {
            Number value = price<Number>(p, spots[p.underlying], vols[p.underlying], rate);
            pv += p.quantity * value.value();
            value.adjoint() = p.quantity;  // seed: d(PV)/d(position value)
            tape_.propagate_to_mark();
            peak_nodes = std::max(peak_nodes, tape_.node_count());
            tape_.rewind_to_mark();
        }

        Greeks g;
        g.value = pv;
        g.delta.resize(spots.size());
        g.vega.resize(vols.size());
        for (std::size_t i = 0; i < spots.size(); ++i) {
            g.delta[i] = spots[i].adjoint();
            g.vega[i] = vols[i].adjoint();
        }
        g.rho = rate.adjoint();
        last_tape_nodes_ = peak_nodes;
        return g;
    }

    // Forward differences: 2N + 2 revaluations. Central differences would be
    // more accurate but cost 4N + 2, so forward keeps the comparison fair to bumping.
    [[nodiscard]] Greeks bump(const MarketData& m, double rel_bump = 1e-5) const {
        Greeks g;
        g.value = pv(m);
        const std::size_t n = m.underlyings();
        g.delta.resize(n);
        g.vega.resize(n);
        MarketData x = m;
        for (std::size_t i = 0; i < n; ++i) {
            const double hs = rel_bump * m.spots[i];
            x.spots[i] += hs;
            g.delta[i] = (pv(x) - g.value) / hs;
            x.spots[i] = m.spots[i];

            const double hv = rel_bump;
            x.vols[i] += hv;
            g.vega[i] = (pv(x) - g.value) / hv;
            x.vols[i] = m.vols[i];
        }
        x.rate += rel_bump;
        g.rho = (pv(x) - g.value) / rel_bump;
        return g;
    }

    [[nodiscard]] std::size_t positions() const noexcept { return book_.size(); }
    // Peak live tape size during the last aad() call (one position + inputs).
    [[nodiscard]] std::size_t last_tape_nodes() const noexcept { return last_tape_nodes_; }
    [[nodiscard]] std::size_t tape_bytes() const noexcept { return tape_.reserved_bytes(); }

private:
    std::vector<Position> book_;
    std::size_t tree_steps_;
    aad::Tape tape_;
    std::size_t last_tape_nodes_{0};
};

}  // namespace greekstream
