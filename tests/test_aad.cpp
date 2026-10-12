#include <gtest/gtest.h>

#include <cmath>
#include <functional>

#include "greekstream/aad/number.hpp"

using greekstream::aad::ActiveTape;
using greekstream::aad::Number;
using greekstream::aad::Tape;

namespace {

// Central finite difference of a double function.
double fd(const std::function<double(double)>& f, double x, double h = 1e-6) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

}  // namespace

TEST(Aad, ElementaryDerivativesMatchFiniteDifferences) {
    struct Case {
        const char* name;
        std::function<Number(const Number&)> active;
        std::function<double(double)> passive;
        double x;
    };
    using greekstream::ncdf;
    const Case cases[] = {
        {"x*x", [](const Number& x) -> Number { return x * x; }, [](double x) { return x * x; }, 1.7},
        {"1/x", [](const Number& x) -> Number { return 1.0 / x; }, [](double x) { return 1.0 / x; }, 0.8},
        {"exp", [](const Number& x) -> Number { return exp(x); }, [](double x) { return std::exp(x); }, 0.3},
        {"log", [](const Number& x) -> Number { return log(x); }, [](double x) { return std::log(x); }, 2.5},
        {"sqrt", [](const Number& x) -> Number { return sqrt(x); }, [](double x) { return std::sqrt(x); }, 3.0},
        {"pow", [](const Number& x) -> Number { return pow(x, 2.5); }, [](double x) { return std::pow(x, 2.5); }, 1.3},
        {"ncdf", [](const Number& x) -> Number { return ncdf(x); }, [](double x) { return ncdf(x); }, -0.4},
        {"neg-sub", [](const Number& x) -> Number { return -(x - 3.0) * x; }, [](double x) { return -(x - 3.0) * x; }, 1.1},
        {"nested",
         [](const Number& x) -> Number { return exp(-x * x / 2.0) * log(1.0 + x) / sqrt(x); },
         [](double x) { return std::exp(-x * x / 2.0) * std::log(1.0 + x) / std::sqrt(x); }, 0.9},
    };
    for (const auto& c : cases) {
        Tape tape;
        ActiveTape active{tape};
        Number x{c.x};
        Number y = c.active(x);
        EXPECT_NEAR(y.value(), c.passive(c.x), 1e-14) << c.name;
        y.propagate_to_start();
        EXPECT_NEAR(x.adjoint(), fd(c.passive, c.x), 1e-7) << c.name;
    }
}

TEST(Aad, ExpressionTemplatesRecordOneNodePerAssignment) {
    Tape tape;
    ActiveTape active{tape};
    Number s{100.0}, k{95.0}, v{0.2}, t{1.0};
    const std::size_t before = tape.node_count();
    // ~12 operations, 4 distinct leaves (v used 3 times, t twice)...
    Number d1 = (log(s / k) + 0.5 * v * v * t) / (v * sqrt(t));
    // ...but exactly one new node.
    EXPECT_EQ(tape.node_count() - before, 1U);
    d1.propagate_to_start();
    EXPECT_NEAR(s.adjoint(), 1.0 / (100.0 * 0.2), 1e-14);  // d d1 / d s = 1/(s v sqrt t)
}

TEST(Aad, SelfAssignmentReadsOldValue) {
    Tape tape;
    ActiveTape active{tape};
    Number x{3.0};
    Number y = x;
    y = y * y + y;  // y = 9 + 3 = 12, dy/dx = 2x + 1 = 7
    y += 1.0;
    EXPECT_DOUBLE_EQ(y.value(), 13.0);
    y.propagate_to_start();
    EXPECT_DOUBLE_EQ(x.adjoint(), 7.0);
}

TEST(Aad, GradientOfManyInputsInOneSweep) {
    Tape tape;
    ActiveTape active{tape};
    constexpr int kN = 100;
    std::vector<Number> xs;
    for (int i = 0; i < kN; ++i) {
        xs.emplace_back(1.0 + 0.01 * i);
    }
    Number sum = 0.0;
    for (int i = 0; i < kN; ++i) {
        sum += static_cast<double>(i) * xs[static_cast<std::size_t>(i)] * xs[static_cast<std::size_t>(i)];
    }
    sum.propagate_to_start();
    for (int i = 0; i < kN; ++i) {
        // d/dx_i of i * x_i^2 = 2 i x_i
        EXPECT_NEAR(xs[static_cast<std::size_t>(i)].adjoint(), 2.0 * i * (1.0 + 0.01 * i), 1e-12);
    }
}

TEST(Aad, MarkAndRewindReuseMemoryAndAccumulate) {
    Tape tape;
    ActiveTape active{tape};
    Number a{2.0};
    Number b = a * 3.0;  // below the mark: d b / d a = 3
    tape.set_mark();
    const std::size_t at_mark = tape.node_count();
    for (int path = 0; path < 1000; ++path) {
        Number y = b * b;  // dy/db = 2b = 12
        y.propagate_to_mark();
        tape.rewind_to_mark();
    }
    EXPECT_EQ(tape.node_count(), at_mark);  // nothing above the mark survives
    EXPECT_DOUBLE_EQ(b.adjoint(), 12'000.0);
    tape.propagate_mark_to_start();
    EXPECT_DOUBLE_EQ(a.adjoint(), 36'000.0);
}

TEST(Aad, MaxRoutesAdjointThroughWinningBranch) {
    Tape tape;
    ActiveTape active{tape};
    Number x{5.0}, y{3.0};
    Number m = fmax(x * 2.0, y * 4.0);  // 10 vs 12 -> second branch
    m.propagate_to_start();
    EXPECT_DOUBLE_EQ(m.value(), 12.0);
    EXPECT_DOUBLE_EQ(x.adjoint(), 0.0);
    EXPECT_DOUBLE_EQ(y.adjoint(), 4.0);
}

TEST(Aad, ClearKeepsReservedMemory) {
    Tape tape;
    ActiveTape active{tape};
    for (int i = 0; i < 100'000; ++i) {
        Number x{static_cast<double>(i)};
    }
    const auto reserved = tape.reserved_bytes();
    tape.clear();
    EXPECT_EQ(tape.node_count(), 0U);
    for (int i = 0; i < 100'000; ++i) {
        Number x{static_cast<double>(i)};
    }
    EXPECT_EQ(tape.reserved_bytes(), reserved);  // no growth on reuse
}
