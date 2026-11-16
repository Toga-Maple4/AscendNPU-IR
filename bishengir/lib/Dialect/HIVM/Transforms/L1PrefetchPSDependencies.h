// Shared dependency reproduction and execution-domain checks for New-L1.
#ifndef BISHENGIR_L1PREFETCHPS_DEPENDENCIES_H
#define BISHENGIR_L1PREFETCHPS_DEPENDENCIES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::hivm::l1prefetch {

// Produced only by normalization of an accepted, launch-validated schedule.
// This records an execution-domain premise, NOT ranges for the trip count or
// its factors. Materialization consumes it before ordinary lowering.
constexpr llvm::StringLiteral kNonemptyTileStream =
    "program_split.nonempty_tile_stream";

// Argument role recorded from KernelLayout, never inferred from argument count.
// Consumed by MIX blockification; no external contract or normal-path change.
constexpr llvm::StringLiteral kLaunchGridAxis = "program_split.launch_grid_axis";

inline bool isNonemptyTileStream(scf::ForOp loop) {
  if (loop->hasAttr(kNonemptyTileStream))
    return true;
  auto lower = getConstantIntValue(loop.getLowerBound());
  auto upper = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  return lower && upper && step && *step > 0 && *lower < *upper;
}

// This LLVM revision declares integer remainder unconditionally speculatable,
// although its operation contract makes a zero divisor undefined. Keep this
// correction local to New-L1 movement; do not change ordinary compiler behavior.
inline bool isSpeculatableDependency(Operation *op) {
  if (isa<arith::RemSIOp, arith::RemUIOp>(op)) {
    APInt divisor;
    return matchPattern(op->getOperand(1), m_ConstantInt(&divisor)) &&
           !divisor.isZero() &&
           (!isa<arith::RemSIOp>(op) || !divisor.isAllOnes());
  }
  return isSpeculatable(op);
}

// Roots come from the caller's verified ABI/mapping. External values must
// dominate the destination and are reused, not speculated. For cloned values,
// either the original block's execution condition is preserved by the caller,
// or MLIR must prove speculation safe. Purity alone establishes neither.
class DependencyProof {
public:
  DependencyProof(Operation *scope, ValueRange roots,
                  ArrayRef<Block *> preservedBlocks,
                  Operation *destination = nullptr,
                  ArrayRef<Operation *> metadata = {},
                  bool addressOnly = true)
      : scope(scope), roots(roots.begin(), roots.end()),
        preservedBlocks(preservedBlocks.begin(), preservedBlocks.end()),
        destination(destination), metadata(metadata.begin(), metadata.end()),
        addressOnly(addressOnly) {}

  // Sinking an operation may reuse invariant values already available at the
  // destination. Their definitions are neither moved nor cloned (an allocation,
  // for example, must not be reclassified as a reproducible pure operation).
  void reuseDominating(Value value) {
    if (destination && dominance.dominates(value, destination))
      roots.push_back(value);
  }

  bool check(Value value) {
    if (llvm::is_contained(roots, value))
      return true;
    Operation *op = value.getDefiningOp();
    if (!op)
      return false; // A loop-carried value is not an implicitly permitted root.
    if (!scope->isAncestor(op) && op != scope)
      return destination && dominance.dominates(value, destination);
    if (addressOnly &&
        !isa<IntegerType, IndexType, MemRefType>(value.getType()))
      return false;
    return checkOperation(op);
  }

  // Relocation checks operations, including zero-result and non-address ops.
  // Do not apply an address result-type/count restriction to this API.
  bool checkOperation(Operation *op) {
    if (checked.contains(op))
      return true;
    bool approvedRead = isa<memref::LoadOp>(op) &&
                        llvm::is_contained(metadata, op);
    if (op->getNumRegions() || op->getNumSuccessors() ||
        (!approvedRead && !isMemoryEffectFree(op)))
      return false;
    if (!llvm::is_contained(preservedBlocks, op->getBlock()) &&
        !isSpeculatableDependency(op))
      return false;
    for (Value operand : op->getOperands())
      if (!check(operand))
        return false;
    checked.insert(op);
    return true;
  }

private:
  Operation *scope;
  SmallVector<Value> roots;
  SmallVector<Block *> preservedBlocks;
  Operation *destination;
  SmallVector<Operation *> metadata;
  bool addressOnly;
  llvm::SmallPtrSet<Operation *, 32> checked;
  DominanceInfo dominance;
};

inline DependencyProof stagingDependencies(func::FuncOp func, Operation *scope,
                                           ValueRange mappedRoots,
                                           Operation *destination,
                                           ArrayRef<Block *> preservedBlocks,
                                           bool addressOnly = true) {
  SmallVector<Value> roots(mappedRoots.begin(), mappedRoots.end());
  // Physical ABI arguments are available values, not positivity assumptions.
  llvm::append_range(roots, func.getArguments());
  SmallVector<Operation *> metadata;
  func.walk([&](memref::LoadOp load) {
    if (load->hasAttr("program_split.metadata"))
      metadata.push_back(load);
  });
  return DependencyProof(scope, roots, preservedBlocks, destination, metadata,
                         addressOnly);
}

} // namespace mlir::hivm::l1prefetch
#endif
