// The headline chart: cost of ALL first-order Greeks vs number of
// underlyings, AAD vs bump-and-revalue.
//
//   ./build/release/bench/greekstream_bench --benchmark_filter=Basket
//
// Expect bump time to grow ~linearly with N (2N + 2 pricings) and AAD time
// to grow only with the cost of one pricing (which is itself O(N) per path),
// so the ratio bump/AAD grows ~linearly in N.

#include <benchmark/benchmark.h>

#include <vector>

#include "greekstream/models/basket_mc.hpp"
#include "greekstream/models/black_scholes.hpp"
#include "greekstream/risk/risk_engine.hpp"

using namespace greekstream;

namespace {

BasketOption basket(std::size_t n) {
    return BasketOption{std::vector<double>(n, 1.0 / static_cast<double>(n)), 100.0, 1.0, OptionType::Call};
}

MarketData market(std::size_t n) {
    MarketData m;
    for (std::size_t i = 0; i < n; ++i) {
        m.spots.push_back(90.0 + static_cast<double>(i % 20));
        m.vols.push_back(0.15 + 0.01 * static_cast<double>(i % 15));
    }
    m.rate = 0.02;
    return m;
}

McConfig mc_config() {
    McConfig cfg;
    cfg.paths = 8'192;
    return cfg;
}

void BM_BasketPriceOnly(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto opt = basket(n);
    const auto mkt = market(n);
    for (auto _ : state) {
        double p = basket_price(opt, mkt, mc_config());
        benchmark::DoNotOptimize(p);
    }
    state.counters["greeks"] = static_cast<double>(2 * n + 1);
}

void BM_BasketGreeksBump(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto opt = basket(n);
    const auto mkt = market(n);
    for (auto _ : state) {
        auto g = basket_greeks_bump(opt, mkt, mc_config());
        benchmark::DoNotOptimize(g.value);
    }
    state.counters["greeks"] = static_cast<double>(2 * n + 1);
}

void BM_BasketGreeksAAD(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto opt = basket(n);
    const auto mkt = market(n);
    for (auto _ : state) {
        auto g = basket_greeks_aad(opt, mkt, mc_config());
        benchmark::DoNotOptimize(g.value);
    }
    state.counters["greeks"] = static_cast<double>(2 * n + 1);
}

BENCHMARK(BM_BasketPriceOnly)->RangeMultiplier(4)->Range(1, 64)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BasketGreeksBump)->RangeMultiplier(4)->Range(1, 64)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BasketGreeksAAD)->RangeMultiplier(4)->Range(1, 64)->Unit(benchmark::kMillisecond);

// Analytic Black-Scholes book: each pricing is so cheap that AAD's recording
// overhead is visible. For small N, bumping can WIN -- know where the
// crossover is before you claim AAD is always faster.
std::vector<Position> vanilla_book(std::size_t n) {
    std::vector<Position> book;
    for (std::size_t u = 0; u < n; ++u) {
        for (int k = 0; k < 10; ++k) {
            book.push_back({u, k % 2 ? OptionType::Call : OptionType::Put, Exercise::European, 80.0 + 4.0 * k,
                            0.25 + 0.1 * k, 1.0});
        }
    }
    return book;
}

void BM_VanillaBookBump(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    RiskEngine engine{vanilla_book(n)};
    const auto mkt = market(n);
    for (auto _ : state) {
        auto g = engine.bump(mkt);
        benchmark::DoNotOptimize(g.value);
    }
}

void BM_VanillaBookAAD(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    RiskEngine engine{vanilla_book(n)};
    const auto mkt = market(n);
    for (auto _ : state) {
        auto g = engine.aad(mkt);
        benchmark::DoNotOptimize(g.value);
    }
}

BENCHMARK(BM_VanillaBookBump)->RangeMultiplier(4)->Range(1, 256)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_VanillaBookAAD)->RangeMultiplier(4)->Range(1, 256)->Unit(benchmark::kMicrosecond);

}  // namespace
