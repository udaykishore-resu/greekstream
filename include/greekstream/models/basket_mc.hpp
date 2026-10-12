#pragma once

// Monte Carlo basket option under correlated GBM (one-factor correlation),
// with three ways to get Greeks:
//
//   basket_price        double, price only                    cost: 1 pricing
//   basket_greeks_bump  bump-and-revalue, common random nums  cost: 2N + 2 pricings
//   basket_greeks_aad   pathwise AAD                          cost: ~constant x 1 pricing
//
// The AAD version uses the tape checkpoint pattern: per-pricing quantities
// (drifts, vol*sqrt(T), discount) are recorded once below the mark; each path
// is recorded above it, swept back to the mark, and discarded. Only after all
// paths are summed is the mark-to-start region swept once.
//
// Parallelism and determinism: paths are split into fixed blocks; block b
// always uses Rng::stream(seed, b); each block's result (price sum and the
// adjoints of the below-mark quantities) is stored by block index and reduced
// in block order. Same seed => bit-identical output for any thread count.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "greekstream/aad/number.hpp"
#include "greekstream/market.hpp"
#include "greekstream/random.hpp"

namespace greekstream {

struct BasketOption {
    std::vector<double> weights;  // one per underlying
    double strike{100.0};
    double expiry{1.0};
    OptionType type{OptionType::Call};
};

struct McConfig {
    std::size_t paths{65'536};      // rounded up to whole antithetic pairs
    std::size_t block_size{2'048};  // paths per deterministic block
    std::uint64_t seed{7};
    std::size_t threads{1};
    double correlation{0.3};        // one-factor: Z_i = sqrt(rho) M + sqrt(1 - rho) e_i
};

namespace detail {

// Quantities that depend on market inputs only, computed once per pricing.
template <typename T>
struct BasketSetup {
    std::vector<T> log_drift;  // (r - vol^2/2) T
    std::vector<T> vol_sqrt_t; // vol sqrt(T)
    T discount;                // exp(-r T)
};

template <typename T>
BasketSetup<T> make_setup(const std::vector<T>& vols, const T& rate, double expiry) {
    using std::exp;
    BasketSetup<T> s{{}, {}, T(1.0)};
    s.log_drift.reserve(vols.size());
    s.vol_sqrt_t.reserve(vols.size());
    for (const T& v : vols) {
        s.log_drift.push_back(T((rate - 0.5 * v * v) * expiry));
        s.vol_sqrt_t.push_back(T(v * std::sqrt(expiry)));
    }
    s.discount = exp(-rate * expiry);
    return s;
}

// Correlated normals for one path into `z` (size n). The caller prices the
// path twice, with +z and -z (antithetic variates).
inline void draw(Rng& rng, double rho, std::vector<double>& z) {
    const double common = rng.normal();
    const double a = std::sqrt(rho);
    const double b = std::sqrt(1.0 - rho);
    for (double& zi : z) {
        zi = a * common + b * rng.normal();
    }
}

template <typename T>
T path_payoff(const BasketOption& opt, const std::vector<T>& spots, const BasketSetup<T>& s,
              const std::vector<double>& z, double sign) {
    using std::exp;
    T basket = 0.0;
    for (std::size_t i = 0; i < spots.size(); ++i) {
        basket += opt.weights[i] * spots[i] * exp(s.log_drift[i] + s.vol_sqrt_t[i] * (sign * z[i]));
    }
    const T intrinsic = opt.type == OptionType::Call ? T(basket - opt.strike) : T(opt.strike - basket);
    return T(s.discount * fmax(intrinsic, 0.0));
}

struct BlockResult {
    double price_sum{0.0};
    std::vector<double> adj;  // adjoints of the below-mark (per-pricing) nodes
};

inline std::size_t block_count(const McConfig& cfg) {
    const std::size_t pairs = (cfg.paths + 1) / 2;
    const std::size_t pairs_per_block = std::max<std::size_t>(1, cfg.block_size / 2);
    return (pairs + pairs_per_block - 1) / pairs_per_block;
}

inline std::size_t pairs_in_block(const McConfig& cfg, std::size_t b) {
    const std::size_t pairs = (cfg.paths + 1) / 2;
    const std::size_t per = std::max<std::size_t>(1, cfg.block_size / 2);
    const std::size_t begin = b * per;
    return std::min(per, pairs - begin);
}

template <typename Work>
void run_blocks(const McConfig& cfg, std::size_t blocks, Work&& work) {
    const std::size_t threads = std::max<std::size_t>(1, std::min(cfg.threads, blocks));
    if (threads == 1) {
        work(0, 1);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&work, t, threads] { work(t, threads); });
    }
    for (auto& th : pool) {
        th.join();
    }
}

}  // namespace detail

[[nodiscard]] inline double basket_price(const BasketOption& opt, const MarketData& mkt, const McConfig& cfg) {
    const std::size_t n = mkt.underlyings();
    const auto setup = detail::make_setup<double>(mkt.vols, mkt.rate, opt.expiry);
    const std::size_t blocks = detail::block_count(cfg);
    std::vector<double> sums(blocks, 0.0);

    detail::run_blocks(cfg, blocks, [&](std::size_t tid, std::size_t stride) {
        std::vector<double> z(n);
        for (std::size_t b = tid; b < blocks; b += stride) {
            Rng rng = Rng::stream(cfg.seed, b);
            double sum = 0.0;
            for (std::size_t p = 0; p < detail::pairs_in_block(cfg, b); ++p) {
                detail::draw(rng, cfg.correlation, z);
                sum += detail::path_payoff<double>(opt, mkt.spots, setup, z, +1.0);
                sum += detail::path_payoff<double>(opt, mkt.spots, setup, z, -1.0);
            }
            sums[b] = sum;
        }
    });

    double total = 0.0;
    std::size_t paths = 0;
    for (std::size_t b = 0; b < blocks; ++b) {
        total += sums[b];
        paths += 2 * detail::pairs_in_block(cfg, b);
    }
    return total / static_cast<double>(paths);
}

// Bump-and-revalue with common random numbers (same seed for every bump):
// forward differences, 2N + 2 full pricings.
[[nodiscard]] inline Greeks basket_greeks_bump(const BasketOption& opt, const MarketData& mkt, const McConfig& cfg,
                                               double rel_bump = 1e-4) {
    const std::size_t n = mkt.underlyings();
    Greeks g;
    g.value = basket_price(opt, mkt, cfg);
    g.delta.resize(n);
    g.vega.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        MarketData up = mkt;
        const double hs = rel_bump * mkt.spots[i];
        up.spots[i] += hs;
        g.delta[i] = (basket_price(opt, up, cfg) - g.value) / hs;

        MarketData vup = mkt;
        const double hv = rel_bump;
        vup.vols[i] += hv;
        g.vega[i] = (basket_price(opt, vup, cfg) - g.value) / hv;
    }
    MarketData rup = mkt;
    rup.rate += rel_bump;
    g.rho = (basket_price(opt, rup, cfg) - g.value) / rel_bump;
    return g;
}

[[nodiscard]] inline Greeks basket_greeks_aad(const BasketOption& opt, const MarketData& mkt, const McConfig& cfg) {
    using aad::Number;
    const std::size_t n = mkt.underlyings();
    const std::size_t blocks = detail::block_count(cfg);
    std::vector<detail::BlockResult> results(blocks);

    // Below-mark adjoint layout per block: [spot_i..., log_drift_i..., vol_sqrt_t_i..., discount]
    detail::run_blocks(cfg, blocks, [&](std::size_t tid, std::size_t stride) {
        aad::Tape tape;  // one tape per thread: recording needs no locks
        aad::ActiveTape active{tape};
        std::vector<Number> spots(mkt.spots.begin(), mkt.spots.end());
        std::vector<Number> vols(mkt.vols.begin(), mkt.vols.end());
        Number rate{mkt.rate};
        const auto setup = detail::make_setup<Number>(vols, rate, opt.expiry);
        tape.set_mark();

        std::vector<double> z(n);
        for (std::size_t b = tid; b < blocks; b += stride) {
            // Isolate this block's contribution; the reduction across blocks
            // happens later, in block order, so threads cannot change it.
            tape.reset_adjoints();
            Rng rng = Rng::stream(cfg.seed, b);
            double sum = 0.0;
            for (std::size_t p = 0; p < detail::pairs_in_block(cfg, b); ++p) {
                detail::draw(rng, cfg.correlation, z);
                for (const double sign : {+1.0, -1.0}) {
                    Number payoff = detail::path_payoff<Number>(opt, spots, setup, z, sign);
                    sum += payoff.value();
                    payoff.propagate_to_mark();  // sweep this path only
                    tape.rewind_to_mark();       // and forget it: memory stays flat
                }
            }
            auto& r = results[b];
            r.price_sum = sum;
            r.adj.resize(3 * n + 1);
            for (std::size_t i = 0; i < n; ++i) {
                r.adj[i] = spots[i].adjoint();
                r.adj[n + i] = setup.log_drift[i].adjoint();
                r.adj[2 * n + i] = setup.vol_sqrt_t[i].adjoint();
            }
            r.adj[3 * n] = setup.discount.adjoint();
        }
    });

    double total = 0.0;
    std::size_t paths = 0;
    std::vector<double> adj(3 * n + 1, 0.0);
    for (std::size_t b = 0; b < blocks; ++b) {
        total += results[b].price_sum;
        paths += 2 * detail::pairs_in_block(cfg, b);
        for (std::size_t k = 0; k < adj.size(); ++k) {
            adj[k] += results[b].adj[k];
        }
    }
    const double inv_paths = 1.0 / static_cast<double>(paths);

    // One sweep over the mark-to-start region carries the averaged adjoints
    // of the setup quantities back to vols and rate.
    aad::Tape tape;
    aad::ActiveTape active{tape};
    std::vector<Number> vols(mkt.vols.begin(), mkt.vols.end());
    Number rate{mkt.rate};
    auto setup = detail::make_setup<Number>(vols, rate, opt.expiry);
    for (std::size_t i = 0; i < n; ++i) {
        setup.log_drift[i].adjoint() = adj[n + i] * inv_paths;
        setup.vol_sqrt_t[i].adjoint() = adj[2 * n + i] * inv_paths;
    }
    setup.discount.adjoint() = adj[3 * n] * inv_paths;
    tape.propagate_to_start();

    Greeks g;
    g.value = total * inv_paths;
    g.delta.resize(n);
    g.vega.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        g.delta[i] = adj[i] * inv_paths;  // spots are leaves: no further chain rule
        g.vega[i] = vols[i].adjoint();
    }
    g.rho = rate.adjoint();
    return g;
}

}  // namespace greekstream
