#pragma once

// Active number + expression templates for reverse-mode AD.
//
// Naive operator-overloading AAD records one tape node per operation:
// `d1 = (log(S/K) + (r + 0.5*v*v)*t) / (v*sqrt(t))` would cost ~10 nodes.
// Here every operator returns a lightweight *expression object* instead of a
// Number. Nothing touches the tape until the expression is assigned to a
// Number; at that point the whole right-hand side becomes ONE node whose
// arguments are the leaf Numbers, with partials computed by the chain rule at
// compile-time-known offsets. Fewer, fatter nodes => less memory traffic in
// the backward sweep.
//
// Rules of use:
//  * Model code is a template on the number type T and works for double and
//    for aad::Number alike (unqualified calls to log/exp/sqrt/ncdf resolve
//    by ADL for Number and via `using std::log` for double).
//  * Never capture an expression with `auto`: it holds references to its
//    operands. Always assign to a Number (or T).

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <type_traits>

#include "greekstream/aad/tape.hpp"

namespace greekstream::aad {

class Number;

template <typename E>
concept Expression = requires(const E& e) {
    { E::num_numbers } -> std::convertible_to<std::size_t>;
    { e.value() } -> std::same_as<double>;
};

namespace detail {
// Leaves are held by reference (they outlive the full-expression); interior
// expression nodes are small value types and are held by value.
template <typename E>
using Held = std::conditional_t<std::is_same_v<E, Number>, const Number&, const E>;
}  // namespace detail

// A passive constant inside an expression: contributes no tape argument.
struct Const {
    static constexpr std::size_t num_numbers = 0;
    double v;
    [[nodiscard]] double value() const noexcept { return v; }
    template <std::size_t Offset>
    void push(Node&, double) const noexcept {}
};

template <typename Op, Expression L, Expression R>
class Binary {
public:
    static constexpr std::size_t num_numbers = L::num_numbers + R::num_numbers;

    Binary(const L& l, const R& r) noexcept : l_(l), r_(r), value_(Op::eval(l.value(), r.value())) {}

    [[nodiscard]] double value() const noexcept { return value_; }

    template <std::size_t Offset>
    void push(Node& node, double weight) const noexcept {
        if constexpr (L::num_numbers > 0) {
            l_.template push<Offset>(node, weight * Op::d_left(l_.value(), r_.value(), value_));
        }
        if constexpr (R::num_numbers > 0) {
            r_.template push<Offset + L::num_numbers>(node, weight * Op::d_right(l_.value(), r_.value(), value_));
        }
    }

private:
    detail::Held<L> l_;
    detail::Held<R> r_;
    double value_;
};

template <typename Op, Expression A>
class Unary {
public:
    static constexpr std::size_t num_numbers = A::num_numbers;

    explicit Unary(const A& a, double param = 0.0) noexcept
        : a_(a), param_(param), value_(Op::eval(a.value(), param)) {}

    [[nodiscard]] double value() const noexcept { return value_; }

    template <std::size_t Offset>
    void push(Node& node, double weight) const noexcept {
        a_.template push<Offset>(node, weight * Op::derivative(a_.value(), value_, param_));
    }

private:
    detail::Held<A> a_;
    double param_;
    double value_;
};

// ---- operation table: value and local partials ----
namespace op {
struct Add {
    static double eval(double l, double r) noexcept { return l + r; }
    static double d_left(double, double, double) noexcept { return 1.0; }
    static double d_right(double, double, double) noexcept { return 1.0; }
};
struct Sub {
    static double eval(double l, double r) noexcept { return l - r; }
    static double d_left(double, double, double) noexcept { return 1.0; }
    static double d_right(double, double, double) noexcept { return -1.0; }
};
struct Mul {
    static double eval(double l, double r) noexcept { return l * r; }
    static double d_left(double, double r, double) noexcept { return r; }
    static double d_right(double l, double, double) noexcept { return l; }
};
struct Div {
    static double eval(double l, double r) noexcept { return l / r; }
    static double d_left(double, double r, double) noexcept { return 1.0 / r; }
    static double d_right(double, double r, double v) noexcept { return -v / r; }
};
struct Pow {
    static double eval(double l, double r) noexcept { return std::pow(l, r); }
    static double d_left(double l, double r, double v) noexcept { return r * v / l; }
    static double d_right(double l, double, double v) noexcept { return std::log(l) * v; }
};
struct Max {
    static double eval(double l, double r) noexcept { return l > r ? l : r; }
    static double d_left(double l, double r, double) noexcept { return l > r ? 1.0 : 0.0; }
    static double d_right(double l, double r, double) noexcept { return l > r ? 0.0 : 1.0; }
};
struct Neg {
    static double eval(double a, double) noexcept { return -a; }
    static double derivative(double, double, double) noexcept { return -1.0; }
};
struct Exp {
    static double eval(double a, double) noexcept { return std::exp(a); }
    static double derivative(double, double v, double) noexcept { return v; }
};
struct Log {
    static double eval(double a, double) noexcept { return std::log(a); }
    static double derivative(double a, double, double) noexcept { return 1.0 / a; }
};
struct Sqrt {
    static double eval(double a, double) noexcept { return std::sqrt(a); }
    static double derivative(double, double v, double) noexcept { return 0.5 / v; }
};
struct PowScalar {  // a^p with passive p
    static double eval(double a, double p) noexcept { return std::pow(a, p); }
    static double derivative(double a, double v, double p) noexcept { return p * v / a; }
};
struct NormalCdf {
    static double eval(double a, double) noexcept { return 0.5 * std::erfc(-a * std::numbers::sqrt2 / 2.0); }
    static double derivative(double a, double, double) noexcept {
        return std::exp(-0.5 * a * a) * (std::numbers::inv_sqrtpi / std::numbers::sqrt2);
    }
};
struct NormalPdf {
    static double eval(double a, double) noexcept {
        return std::exp(-0.5 * a * a) * (std::numbers::inv_sqrtpi / std::numbers::sqrt2);
    }
    static double derivative(double a, double v, double) noexcept { return -a * v; }
};
}  // namespace op

// ---- the active number ----
class Number {
public:
    static constexpr std::size_t num_numbers = 1;

    Number() : Number(0.0) {}
    // Implicit on purpose: lets templated model code write `T x = 0.0;`.
    Number(double v) : value_(v), node_(Tape::active()->record(0)) {}  // NOLINT(google-explicit-constructor)

    template <Expression E>
        requires(!std::is_same_v<E, Number>)
    Number(const E& e) : value_(e.value()), node_(record(e)) {}  // NOLINT(google-explicit-constructor)

    Number(const Number&) = default;  // shares the node: same variable
    Number& operator=(const Number&) = default;
    Number(Number&&) = default;
    Number& operator=(Number&&) = default;
    ~Number() = default;

    template <Expression E>
        requires(!std::is_same_v<E, Number>)
    Number& operator=(const E& e) {
        // Record before overwriting: `x = x * y` reads x's current node/value.
        const double v = e.value();
        Node* n = record(e);
        value_ = v;
        node_ = n;
        return *this;
    }
    Number& operator=(double v) {
        value_ = v;
        node_ = Tape::active()->record(0);
        return *this;
    }

    template <Expression E>
    Number& operator+=(const E& e) { return *this = *this + e; }
    template <Expression E>
    Number& operator-=(const E& e) { return *this = *this - e; }
    template <Expression E>
    Number& operator*=(const E& e) { return *this = *this * e; }
    template <Expression E>
    Number& operator/=(const E& e) { return *this = *this / e; }
    // Defined after the free operators below (a non-template member body
    // only sees names declared before it).
    Number& operator+=(double d);
    Number& operator-=(double d);
    Number& operator*=(double d);
    Number& operator/=(double d);

    [[nodiscard]] double value() const noexcept { return value_; }
    [[nodiscard]] double& adjoint() noexcept { return node_->adjoint; }
    [[nodiscard]] double adjoint() const noexcept { return node_->adjoint; }

    // Re-record as a fresh input leaf on the active tape (e.g. per thread).
    void put_on_tape() { node_ = Tape::active()->record(0); }

    // Seed d(this)/d(this) = 1 and sweep the whole tape backwards.
    void propagate_to_start() {
        node_->adjoint = 1.0;
        Tape::active()->propagate_to_start();
    }
    void propagate_to_mark() {
        node_->adjoint = 1.0;
        Tape::active()->propagate_to_mark();
    }

    template <std::size_t Offset>
    void push(Node& node, double weight) const noexcept {
        node.partials[Offset] = weight;
        node.args[Offset] = &node_->adjoint;
    }

private:
    template <Expression E>
    static Node* record(const E& e) {
        static_assert(E::num_numbers <= Tape::kArgBlock, "expression too large for one tape node");
        Node* node = Tape::active()->record(static_cast<std::uint32_t>(E::num_numbers));
        e.template push<0>(*node, 1.0);
        return node;
    }

    double value_;
    Node* node_;
};

// ---- operators: at least one side must be an Expression ----
#define GREEKSTREAM_AAD_BINARY(SYM, OP)                                                    \
    template <Expression L, Expression R>                                                  \
    [[nodiscard]] inline Binary<op::OP, L, R> operator SYM(const L& l, const R& r) {       \
        return {l, r};                                                                     \
    }                                                                                      \
    template <Expression L>                                                                \
    [[nodiscard]] inline Binary<op::OP, L, Const> operator SYM(const L& l, double r) {     \
        return {l, Const{r}};                                                              \
    }                                                                                      \
    template <Expression R>                                                                \
    [[nodiscard]] inline Binary<op::OP, Const, R> operator SYM(double l, const R& r) {     \
        return {Const{l}, r};                                                              \
    }
GREEKSTREAM_AAD_BINARY(+, Add)
GREEKSTREAM_AAD_BINARY(-, Sub)
GREEKSTREAM_AAD_BINARY(*, Mul)
GREEKSTREAM_AAD_BINARY(/, Div)
#undef GREEKSTREAM_AAD_BINARY

template <Expression A>
[[nodiscard]] inline Unary<op::Neg, A> operator-(const A& a) { return Unary<op::Neg, A>{a}; }

template <Expression A>
[[nodiscard]] inline Unary<op::Exp, A> exp(const A& a) { return Unary<op::Exp, A>{a}; }
template <Expression A>
[[nodiscard]] inline Unary<op::Log, A> log(const A& a) { return Unary<op::Log, A>{a}; }
template <Expression A>
[[nodiscard]] inline Unary<op::Sqrt, A> sqrt(const A& a) { return Unary<op::Sqrt, A>{a}; }
template <Expression A>
[[nodiscard]] inline Unary<op::PowScalar, A> pow(const A& a, double p) { return Unary<op::PowScalar, A>{a, p}; }
template <Expression L, Expression R>
[[nodiscard]] inline Binary<op::Pow, L, R> pow(const L& l, const R& r) { return {l, r}; }
template <Expression A>
[[nodiscard]] inline Unary<op::NormalCdf, A> ncdf(const A& a) { return Unary<op::NormalCdf, A>{a}; }
template <Expression A>
[[nodiscard]] inline Unary<op::NormalPdf, A> npdf(const A& a) { return Unary<op::NormalPdf, A>{a}; }

template <Expression L, Expression R>
[[nodiscard]] inline Binary<op::Max, L, R> fmax(const L& l, const R& r) { return {l, r}; }
template <Expression L>
[[nodiscard]] inline Binary<op::Max, L, Const> fmax(const L& l, double r) { return {l, Const{r}}; }
template <Expression R>
[[nodiscard]] inline Binary<op::Max, Const, R> fmax(double l, const R& r) { return {Const{l}, r}; }

inline Number& Number::operator+=(double d) { return *this = *this + d; }
inline Number& Number::operator-=(double d) { return *this = *this - d; }
inline Number& Number::operator*=(double d) { return *this = *this * d; }
inline Number& Number::operator/=(double d) { return *this = *this / d; }

// Comparisons act on values (branching is not differentiated).
#define GREEKSTREAM_AAD_COMPARE(SYM)                                                                         \
    template <Expression L, Expression R>                                                                    \
    [[nodiscard]] inline bool operator SYM(const L& l, const R& r) { return l.value() SYM r.value(); }       \
    template <Expression L>                                                                                  \
    [[nodiscard]] inline bool operator SYM(const L& l, double r) { return l.value() SYM r; }                 \
    template <Expression R>                                                                                  \
    [[nodiscard]] inline bool operator SYM(double l, const R& r) { return l SYM r.value(); }
GREEKSTREAM_AAD_COMPARE(<)
GREEKSTREAM_AAD_COMPARE(>)
GREEKSTREAM_AAD_COMPARE(<=)
GREEKSTREAM_AAD_COMPARE(>=)
#undef GREEKSTREAM_AAD_COMPARE

// Uniform value access for templated code: value(double) is the identity.
[[nodiscard]] inline double value(double x) noexcept { return x; }
template <Expression E>
[[nodiscard]] inline double value(const E& e) noexcept { return e.value(); }

}  // namespace greekstream::aad

namespace greekstream {

// Passive (double) versions so templated model code reads the same for both.
[[nodiscard]] inline double ncdf(double x) noexcept { return 0.5 * std::erfc(-x * std::numbers::sqrt2 / 2.0); }
[[nodiscard]] inline double npdf(double x) noexcept {
    return std::exp(-0.5 * x * x) * (std::numbers::inv_sqrtpi / std::numbers::sqrt2);
}
[[nodiscard]] inline double fmax(double a, double b) noexcept { return a > b ? a : b; }

}  // namespace greekstream
