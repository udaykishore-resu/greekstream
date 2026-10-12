#pragma once

// The tape: a linear record of every assignment made with an active Number.
//
// Reverse-mode AD (adjoint algorithmic differentiation) evaluates a function
// forward while recording, for each result, the local partial derivatives
// with respect to its arguments. One backward sweep over the tape then
// accumulates d(output)/d(every input). The cost of that sweep is a small
// constant multiple of the forward evaluation, INDEPENDENT of the number of
// inputs -- that is why AAD wins over bump-and-revalue as Greeks multiply.
//
// Memory: nodes and their argument arrays live in fixed-size blocks that are
// kept across rewind()/clear(), so after the first evaluation the tape
// performs no heap allocation at all.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace greekstream::aad {

struct Node {
    double adjoint{0.0};
    std::uint32_t n{0};          // number of arguments (0 = leaf / input)
    double* partials{nullptr};   // d(this)/d(arg_i)
    double** args{nullptr};      // &arg_i.adjoint

    void propagate() const noexcept {
        if (n == 0 || adjoint == 0.0) {
            return;
        }
        for (std::uint32_t i = 0; i < n; ++i) {
            *args[i] += partials[i] * adjoint;
        }
    }
};

template <typename T, std::size_t BlockSize>
class BlockArena {
public:
    struct Position {
        std::size_t block{0};
        std::size_t offset{0};
    };

    T* allocate(std::size_t count) {
        assert(count <= BlockSize);
        if (offset_ + count > BlockSize) {
            ++block_;
            offset_ = 0;
        }
        if (block_ == blocks_.size()) {
            blocks_.push_back(std::make_unique<T[]>(BlockSize));
        }
        T* p = blocks_[block_].get() + offset_;
        offset_ += count;
        return p;
    }

    [[nodiscard]] Position position() const noexcept { return {block_, offset_}; }
    void rewind(Position p) noexcept {
        block_ = p.block;
        offset_ = p.offset;
    }

    // Visit elements in [from, to) from last to first. Only meaningful for
    // single-element allocations (nodes), which never leave block-tail gaps.
    template <typename F>
    void reverse_each(Position from, Position to, F&& f) {
        if (blocks_.empty()) {
            return;
        }
        for (std::size_t b = to.block + 1; b-- > from.block;) {
            if (b >= blocks_.size()) {
                continue;
            }
            const std::size_t begin = (b == from.block) ? from.offset : 0;
            const std::size_t end = (b == to.block) ? to.offset : BlockSize;
            T* base = blocks_[b].get();
            for (std::size_t i = end; i-- > begin;) {
                f(base[i]);
            }
        }
    }

    [[nodiscard]] std::size_t count_between(Position from, Position to) const noexcept {
        return (to.block * BlockSize + to.offset) - (from.block * BlockSize + from.offset);
    }
    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return blocks_.size() * BlockSize * sizeof(T); }

private:
    std::vector<std::unique_ptr<T[]>> blocks_;
    std::size_t block_{0};
    std::size_t offset_{0};
};

class Tape {
public:
    static constexpr std::size_t kNodeBlock = 16'384;
    static constexpr std::size_t kArgBlock = 65'536;  // max arguments in one expression

    struct Mark {
        BlockArena<Node, kNodeBlock>::Position nodes;
        BlockArena<double, kArgBlock>::Position partials;
        BlockArena<double*, kArgBlock>::Position args;
    };

    Tape() = default;
    Tape(const Tape&) = delete;
    Tape& operator=(const Tape&) = delete;
    Tape(Tape&&) = delete;
    Tape& operator=(Tape&&) = delete;
    ~Tape() = default;

    Node* record(std::uint32_t n_args) {
        Node* node = nodes_.allocate(1);
        node->adjoint = 0.0;
        node->n = n_args;
        if (n_args > 0) {
            node->partials = partials_.allocate(n_args);
            node->args = args_.allocate(n_args);
        } else {
            node->partials = nullptr;
            node->args = nullptr;
        }
        return node;
    }

    [[nodiscard]] Mark position() const noexcept { return {nodes_.position(), partials_.position(), args_.position()}; }
    void rewind(const Mark& m) noexcept {
        nodes_.rewind(m.nodes);
        partials_.rewind(m.partials);
        args_.rewind(m.args);
    }
    void clear() noexcept {
        rewind(Mark{});
        mark_ = Mark{};
    }

    // Checkpoint used by Monte Carlo: inputs and per-pricing precomputations
    // sit below the mark; each path is recorded above it, swept back to the
    // mark, then discarded with rewind_to_mark().
    void set_mark() noexcept { mark_ = position(); }
    void rewind_to_mark() noexcept { rewind(mark_); }
    [[nodiscard]] const Mark& mark() const noexcept { return mark_; }

    void propagate_to_start() { propagate(Mark{}, position()); }
    void propagate_to_mark() { propagate(mark_, position()); }
    void propagate_mark_to_start() { propagate(Mark{}, mark_); }

    void reset_adjoints() {
        nodes_.reverse_each(Mark{}.nodes, nodes_.position(), [](Node& n) noexcept { n.adjoint = 0.0; });
    }

    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.count_between({}, nodes_.position()); }
    [[nodiscard]] std::size_t reserved_bytes() const noexcept {
        return nodes_.reserved_bytes() + partials_.reserved_bytes() + args_.reserved_bytes();
    }

    // The tape the calling thread records onto. Each thread owns its own
    // tape, so recording needs no synchronization.
    static Tape*& active() noexcept {
        thread_local Tape* tape = nullptr;
        return tape;
    }

private:
    void propagate(const Mark& from, const Mark& to) {
        nodes_.reverse_each(from.nodes, to.nodes, [](const Node& n) noexcept { n.propagate(); });
    }

    BlockArena<Node, kNodeBlock> nodes_;
    BlockArena<double, kArgBlock> partials_;
    BlockArena<double*, kArgBlock> args_;
    Mark mark_{};
};

// RAII: make `tape` the active tape for this thread for the scope's lifetime.
class ActiveTape {
public:
    explicit ActiveTape(Tape& tape) noexcept : previous_(Tape::active()) { Tape::active() = &tape; }
    ActiveTape(const ActiveTape&) = delete;
    ActiveTape& operator=(const ActiveTape&) = delete;
    ActiveTape(ActiveTape&&) = delete;
    ActiveTape& operator=(ActiveTape&&) = delete;
    ~ActiveTape() { Tape::active() = previous_; }

private:
    Tape* previous_;
};

}  // namespace greekstream::aad
