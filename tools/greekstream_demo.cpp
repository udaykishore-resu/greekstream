// greekstream-demo: a live risk loop. A book of vanilla options on N
// underlyings is revalued on every simulated tick; all 2N + 1 Greeks come
// from one AAD sweep. Every few ticks the same Greeks are recomputed by
// bump-and-revalue to show agreement and the cost gap.
//
//   greekstream-demo --underlyings 50 --ticks 20 --check-every 5

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "greekstream/random.hpp"
#include "greekstream/risk/risk_engine.hpp"

using namespace greekstream;
using Clock = std::chrono::steady_clock;

namespace {

double micros(Clock::duration d) { return std::chrono::duration<double, std::micro>(d).count(); }

}  // namespace

int main(int argc, char** argv) {
    std::size_t n = 50;
    int ticks = 20;
    int check_every = 5;
    std::size_t tree_steps = 100;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string_view a{argv[i]};
        const char* v = argv[i + 1];
        if (a == "--underlyings") n = std::strtoull(v, nullptr, 10);
        else if (a == "--ticks") ticks = std::atoi(v);
        else if (a == "--check-every") check_every = std::atoi(v);
        else if (a == "--tree-steps") tree_steps = std::strtoull(v, nullptr, 10);
        else {
            std::fprintf(stderr, "usage: greekstream-demo [--underlyings N] [--ticks T] [--check-every K] [--tree-steps S]\n");
            return 2;
        }
    }

    // Book: per underlying, a European call, a European put and an American put.
    Rng rng{2026};
    std::vector<Position> book;
    MarketData mkt;
    mkt.rate = 0.035;
    for (std::size_t u = 0; u < n; ++u) {
        const double spot = 50.0 + 150.0 * rng.uniform_open();
        mkt.spots.push_back(spot);
        mkt.vols.push_back(0.15 + 0.35 * rng.uniform_open());
        book.push_back({u, OptionType::Call, Exercise::European, std::round(spot * 1.05), 0.5, 100.0});
        book.push_back({u, OptionType::Put, Exercise::European, std::round(spot * 0.95), 1.0, -50.0});
        book.push_back({u, OptionType::Put, Exercise::American, std::round(spot), 0.75, 25.0});
    }
    RiskEngine engine{book, tree_steps};
    std::printf("book: %zu positions on %zu underlyings (%zu sensitivities per tick, CRR %zu steps)\n\n",
                engine.positions(), n, 2 * n + 1, tree_steps);
    std::printf("%5s %16s %12s %12s %12s %12s\n", "tick", "PV", "net delta$", "net vega", "rho", "AAD us");

    double aad_total = 0.0;
    double bump_total = 0.0;
    int bump_runs = 0;
    double worst_rel = 0.0;
    for (int tick = 1; tick <= ticks; ++tick) {
        for (std::size_t u = 0; u < n; ++u) {  // GBM-ish tick: ~1bp moves
            mkt.spots[u] *= std::exp(0.001 * rng.normal());
        }
        const auto t0 = Clock::now();
        const Greeks g = engine.aad(mkt);
        const auto t1 = Clock::now();
        aad_total += micros(t1 - t0);

        double delta_dollars = 0.0;
        double vega = 0.0;
        for (std::size_t u = 0; u < n; ++u) {
            delta_dollars += g.delta[u] * mkt.spots[u];
            vega += g.vega[u];
        }
        std::printf("%5d %16.2f %12.0f %12.1f %12.1f %12.0f\n", tick, g.value, delta_dollars, vega, g.rho,
                    micros(t1 - t0));

        if (check_every > 0 && tick % check_every == 0) {
            const auto b0 = Clock::now();
            const Greeks b = engine.bump(mkt);
            const auto b1 = Clock::now();
            bump_total += micros(b1 - b0);
            ++bump_runs;
            for (std::size_t u = 0; u < n; ++u) {
                worst_rel = std::max(worst_rel, std::abs(g.delta[u] - b.delta[u]) / std::max(1.0, std::abs(b.delta[u])));
            }
            std::printf("      bump-and-revalue check: %.0f us, max delta diff %.2e\n", micros(b1 - b0), worst_rel);
        }
    }

    const double aad_avg = aad_total / ticks;
    std::printf("\nAAD   : %.0f us per tick for all %zu Greeks (peak tape %zu nodes, %.1f MB reserved)\n", aad_avg, 2 * n + 1,
                engine.last_tape_nodes(), static_cast<double>(engine.tape_bytes()) / 1e6);
    if (bump_runs > 0) {
        const double bump_avg = bump_total / bump_runs;
        std::printf("bump  : %.0f us per tick (%zu revaluations)\n", bump_avg, 2 * n + 2);
        std::printf("speedup %.1fx; AAD vs bump max relative delta difference %.2e\n", bump_avg / aad_avg, worst_rel);
    }
    return 0;
}
