// Exact, read-only equality of integer address expressions. No range inference.
#ifndef BISHENGIR_L1PREFETCHPS_ADDRESS_H
#define BISHENGIR_L1PREFETCHPS_ADDRESS_H

#include "L1PrefetchPSCommon.h"
#include "llvm/ADT/APInt.h"

namespace mlir::hivm::l1prefetch {
// Linear algebra at the ORIGINAL width is bitvector algebra. APInt avoids both
// host overflow and an invalid mathematical-integer interpretation. Division,
// remainder, casts and runtime products remain typed atoms: no cancellation,
// cast motion, signed/floor conversion, or independently numbered affine locals.
// A schedule still proves the ownership theorem and movement execution domain.
class AddressEquivalence {
  struct Form {
    llvm::APInt constant;
    DenseMap<unsigned, llvm::APInt> terms;
    explicit Form(unsigned bits) : constant(bits, 0) {}
  };
  DenseMap<Value, Form> memo;
  SmallVector<Value> atoms;

  static unsigned width(Type type) {
    // A5 index lowering is 64-bit. Identities modulo 2^64 also hold when an
    // index is subsequently narrowed; casts themselves remain opaque.
    return type.isIndex() ? 64 : cast<IntegerType>(type).getWidth();
  }
  static bool same(const Form &a, const Form &b) {
    return a.constant == b.constant && a.terms == b.terms;
  }
  static void add(Form &a, const Form &b, llvm::APInt factor) {
    a.constant += b.constant * factor;
    for (auto &term : b.terms) {
      auto found = a.terms.find(term.first);
      auto coefficient = term.second * factor;
      if (found != a.terms.end())
        coefficient += found->second;
      if (coefficient.isZero())
        a.terms.erase(term.first);
      else
        a.terms[term.first] = coefficient;
    }
  }
  bool sameAtom(Value a, Value b) {
    if (a == b)
      return true;
    if (a.getType() != b.getType())
      return false;
    Operation *x = a.getDefiningOp(), *y = b.getDefiningOp();
    if (!x || !y || x->getName() != y->getName() ||
        x->getNumOperands() != y->getNumOperands())
      return false;
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
            arith::ExtUIOp, arith::TruncIOp, arith::DivSIOp, arith::DivUIOp,
            arith::RemSIOp, arith::RemUIOp>(x)) {
      // Same operation and same typed operand values: exact even for negative
      // operands. No claim that signed division is floor division is needed.
      for (auto operands : llvm::zip(x->getOperands(), y->getOperands()))
        if (!equal(std::get<0>(operands), std::get<1>(operands)))
          return false;
      return true;
    }
    auto mul = dyn_cast<arith::MulIOp>(x);
    auto other = dyn_cast<arith::MulIOp>(y);
    return mul && other && mul.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
           other.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
           ((equal(mul.getLhs(), other.getLhs()) && equal(mul.getRhs(), other.getRhs())) ||
            (equal(mul.getLhs(), other.getRhs()) && equal(mul.getRhs(), other.getLhs())));
  }
  Form form(Value value) {
    auto found = memo.find(value);
    if (found != memo.end())
      return found->second;
    Form result(width(value.getType()));
    bool expanded = false;
    if (auto op = value.getDefiningOp<arith::ConstantOp>();
        op && isa<IntegerAttr>(op.getValue())) {
      result.constant = cast<IntegerAttr>(op.getValue()).getValue().sextOrTrunc(width(value.getType()));
      expanded = true;
    } else if (auto op = value.getDefiningOp<arith::AddIOp>();
               op && op.getOverflowFlags() == arith::IntegerOverflowFlags::none) {
      result = form(op.getLhs());
      add(result, form(op.getRhs()), llvm::APInt(width(value.getType()), 1));
      expanded = true;
    } else if (auto op = value.getDefiningOp<arith::SubIOp>();
               op && op.getOverflowFlags() == arith::IntegerOverflowFlags::none) {
      result = form(op.getLhs());
      add(result, form(op.getRhs()), llvm::APInt::getAllOnes(width(value.getType())));
      expanded = true;
    } else if (auto op = value.getDefiningOp<arith::MulIOp>();
               op && width(value.getType()) <= 64 &&
               op.getOverflowFlags() == arith::IntegerOverflowFlags::none) {
      auto constant = getConstantIntValue(op.getLhs());
      Value variable = op.getRhs();
      if (!constant) {
        constant = getConstantIntValue(op.getRhs());
        variable = op.getLhs();
      }
      if (constant) {
        add(result, form(variable), llvm::APInt(width(value.getType()), *constant, true));
        expanded = true;
      }
    }
    if (!expanded) {
      // Intern children first. Structural atom comparison may recurse into
      // operands; topological numbering prevents comparing an ancestor while
      // one of its own operands is still being interned.
      if (auto *op = value.getDefiningOp())
        for (Value operand : op->getOperands())
          if (operand.getType().isIntOrIndex())
            (void)form(operand);
      unsigned id = 0;
      for (; id < atoms.size(); ++id)
        if (sameAtom(value, atoms[id]))
          break;
      if (id == atoms.size())
        atoms.push_back(value);
      result.terms.try_emplace(id, width(value.getType()), 1);
    }
    memo.try_emplace(value, result);
    return result;
  }
public:
  // False means unproved, never evidence of disequality. Callers retain their
  // named ownership rejection; no approximate or range-based equality exists.
  bool equal(Value a, Value b) {
    if (!a || !b || a.getType() != b.getType() || !a.getType().isIntOrIndex())
      return false;
    return a == b || same(form(a), form(b));
  }
  bool product(Value value, Value root, int64_t coefficient) {
    if (!value || !root || value.getType() != root.getType())
      return false;
    Form expected(width(value.getType()));
    add(expected, form(root), llvm::APInt(width(value.getType()), coefficient, true));
    return same(form(value), expected);
  }
  bool sum(Value value, Value a, Value b) {
    if (!value || !a || !b || value.getType() != a.getType() || a.getType() != b.getType())
      return false;
    Form expected = form(a);
    add(expected, form(b), llvm::APInt(width(value.getType()), 1));
    return same(form(value), expected);
  }
};
} // namespace mlir::hivm::l1prefetch
#endif
