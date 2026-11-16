//===- L1PrefetchMapPrograms.cpp - Whole-program MIX distribution ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "L1PrefetchPSDependencies.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/L1PrefetchPS.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

namespace mlir {
#define GEN_PASS_DEF_L1PREFETCHMAPPROGRAMS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
constexpr llvm::StringLiteral kLogicalBlockNum = "logical_block_num";

void collectDependencies(Value value, llvm::SmallPtrSetImpl<Operation *> &keep) {
  if (isa<BlockArgument>(value))
    return;
  Operation *def = value.getDefiningOp();
  if (!def || !keep.insert(def).second)
    return;
  for (Value operand : def->getOperands())
    collectDependencies(operand, keep);
}

// The launcher proves each recorded grid dimension positive and their product
// <= INT32_MAX. Only products within the actual logical-grid expression inherit
// this fact, with each axis used once. This is not inference from a generic
// positive product, nor from the positions of physical function arguments.
SmallVector<Value> positiveGridDivisors(func::FuncOp func, Value logicalGrid) {
  SmallVector<Value> axes(3), positive;
  for (BlockArgument arg : func.getArguments()) {
    auto axis = func.getArgAttrOfType<IntegerAttr>(
        arg.getArgNumber(), l1prefetch::kLaunchGridAxis);
    if (!axis)
      continue;
    if (axis.getInt() < 0 || axis.getInt() >= 3 ||
        axes[axis.getInt()] || !arg.getType().isInteger(32))
      return {};
    axes[axis.getInt()] = arg;
  }
  if (llvm::any_of(axes, [](Value value) { return !value; }))
    return {};
  std::function<unsigned(Value)> collect = [&](Value value) -> unsigned {
    auto found = llvm::find(axes, value);
    if (found != axes.end()) {
      positive.push_back(value);
      return 1u << (found - axes.begin());
    }
    auto mul = value.getDefiningOp<arith::MulIOp>();
    if (!mul || !value.getType().isInteger(32))
      return 0;
    unsigned lhs = collect(mul.getLhs()), rhs = collect(mul.getRhs());
    if (!lhs || !rhs || (lhs & rhs))
      return 0;
    positive.push_back(value);
    return lhs | rhs;
  };
  if (collect(logicalGrid) != 7)
    return {};
  return positive;
}

// A remapped valid ID is another original program. An odd grid additionally
// has an inactive peer: do not evaluate unsafe raw-ID-dependent operations for
// that peer. Keep safe ID de-linearization in place and sink the remaining
// slice under the existing payload predicate, without guarding synchronization.
// Normalization puts the entire payload in work; only its preceding dependency
// slice is scanned here, not arbitrary operations after the guard.
LogicalResult guardInactivePeer(scf::ForOp loop, scf::IfOp work, Value logical,
                                ArrayRef<Value> positiveDivisors) {
  llvm::SmallPtrSet<Operation *, 32> dependent, guarded;
  SmallVector<Operation *> ordered;
  dependent.insert(logical.getDefiningOp());
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (&op == work.getOperation())
      break;
    if (&op == logical.getDefiningOp() ||
        !llvm::any_of(op.getOperands(), [&](Value value) {
          return dependent.contains(value.getDefiningOp());
        }))
      continue;
    dependent.insert(&op);
    bool safe = l1prefetch::isSpeculatableDependency(&op) ||
        (isa<arith::DivSIOp, arith::RemSIOp>(op) &&
         llvm::is_contained(positiveDivisors, op.getOperand(1)));
    if (safe && !llvm::any_of(op.getOperands(), [&](Value value) {
          return guarded.contains(value.getDefiningOp());
        }))
      continue;
    guarded.insert(&op);
    ordered.push_back(&op);
  }
  // Accepted Flat predicates use only grid de-linearization and linear bounds;
  // Program predicates use the raw valid-ID check. Neither needs the unsafe
  // payload slice to decide participation. Verify that invariant before moving;
  // work itself (a condition use) is an escape, not a use inside its payload.
  for (Operation *op : ordered)
    for (Operation *user : op->getUsers())
      if (!guarded.contains(user) && !work->isProperAncestor(user))
        return loop.emitError("L1 pipeline error: unsafe resident dependency "
                              "escapes the valid-program payload");
  Operation *first = &work.getThenRegion().front().front();
  for (Operation *op : ordered)
    op->moveBefore(first);
  return success();
}

// Present one continuous tile stream to the common L1 materializer. Without a
// resident-pair extent, each trio owns one logical pair; leave that IR unchanged.
// When a trio owns several pairs, concatenate their normalized tile
// loops before materialization, so parity and lookahead cross pair boundaries.
LogicalResult normalizeResidentTileStream(scope::ScopeOp scope, Value selected,
                                         Value logicalBase, Value physicalId,
                                         Value halfGrid, Value logicalGrid, IntegerAttr pairStride,
                                         bool wholeProgram,
                                         ArrayRef<Value> positiveDivisors,
                                         IRRewriter &rewriter) {
  if (!pairStride)
    return success();
  if (pairStride.getInt() <= 0)
    return scope.emitError("program-split invalid resident pair stride");

  LoadOp load;
  scope.walk([&](LoadOp candidate) {
    if (candidate.getDst() == selected)
      load = candidate;
  });
  auto loop = load ? load->getParentOfType<scf::ForOp>() : scf::ForOp();
  // Canonicalization removes a one-iteration tile loop. Treat the remaining
  // accepted logical program as a one-tile stream, then use the same resident
  // concatenation below. This also covers statically single-tile programs.
  if (load && !loop) {
    rewriter.setInsertionPoint(scope.getRegion().front().getTerminator());
    Location loc = load.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    loop = rewriter.create<scf::ForOp>(loc, zero, one, one);
    SmallVector<Operation *> body;
    for (Operation *op = logicalBase.getDefiningOp()->getNextNode();
         op && op != zero.getDefiningOp(); op = op->getNextNode())
      body.push_back(op);
    for (Operation *op : body)
      op->moveBefore(loop.getBody()->getTerminator());
  }
  if (!loop || loop->getBlock() != &scope.getRegion().front() ||
      loop.getNumRegionIterArgs() || !loop.getInductionVar().getType().isInteger(32))
    return scope.emitError("program-split lost normalized resident tile loop");
  auto lower = loop.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
  auto step = loop.getStep().getDefiningOp<arith::ConstantIntOp>();
  if (!lower || lower.value() != 0 || !step || step.value() != 1)
    return scope.emitError("program-split requires zero-based unit-step tile stream");

  // Only the program-dependent address/metadata slice moves into the stream.
  // Invariant allocations, loop bounds and side-effecting setup/teardown stay
  // in place. These are accepted read-only metadata loads, not arbitrary loads.
  llvm::SmallPtrSet<Operation *, 32> dependent;
  SmallVector<Operation *> ordered;
  dependent.insert(logicalBase.getDefiningOp());
  auto dependencies = l1prefetch::stagingDependencies(
      scope->getParentOfType<func::FuncOp>(), scope,
      ValueRange{logicalBase}, loop, {&scope.getRegion().front()},
      /*addressOnly=*/false);
  for (Operation &op : scope.getRegion().front()) {
    if (&op == loop.getOperation())
      break;
    if (&op == logicalBase.getDefiningOp() ||
        !llvm::any_of(op.getOperands(), [&](Value value) {
          return dependent.contains(value.getDefiningOp());
        })) {
      for (Value result : op.getResults())
        dependencies.reuseDominating(result);
      continue;
    }
    // Valid resident IDs reproduce original executions. guardInactivePeer
    // below additionally protects unsafe arithmetic from the odd-grid peer.
    if (!dependencies.checkOperation(&op))
      return scope.emitError("L1 pipeline error: resident dependency proof was lost");
    dependent.insert(&op);
    ordered.push_back(&op);
  }
  if (dependent.contains(loop.getUpperBound().getDefiningOp()))
    return scope.emitError("program-split resident tile extent depends on program ID");
  for (Operation *op : ordered)
    for (Operation *user : op->getUsers())
      if (!dependent.contains(user) && !loop->isAncestor(user))
        return scope.emitError("program-split resident address escapes tile stream");

  Location loc = loop.getLoc();
  Value tilesPerPair = loop.getUpperBound();
  rewriter.setInsertionPoint(loop);
  Value stride = rewriter.create<arith::ConstantIntOp>(loc, pairStride.getInt(), 32);
  Value padding = rewriter.create<arith::ConstantIntOp>(loc, pairStride.getInt() - 1, 32);
  Value remaining = rewriter.create<arith::SubIOp>(loc, halfGrid, physicalId);
  Value rounded = rewriter.create<arith::AddIOp>(loc, remaining, padding);
  Value pairs = rewriter.create<arith::DivSIOp>(loc, rounded, stride);
  Value steps = rewriter.create<arith::MulIOp>(loc, pairs, tilesPerPair);
  bool nonempty = l1prefetch::isNonemptyTileStream(loop);
  loop.setUpperBound(steps);
  // physical_block_count launches min(halfGrid, pairStride) groups. Therefore
  // every launched group owns at least one pair; concatenation preserves entry.
  if (nonempty)
    loop->setAttr(l1prefetch::kNonemptyTileStream, rewriter.getUnitAttr());

  Operation *first = &loop.getBody()->front();
  rewriter.setInsertionPoint(first);
  Value flat = loop.getInductionVar();
  Value pair = rewriter.create<arith::DivSIOp>(loc, flat, tilesPerPair);
  Value tile = rewriter.create<arith::RemSIOp>(loc, flat, tilesPerPair);
  Value offset = rewriter.create<arith::MulIOp>(loc, pair, stride);
  Value wideOffset = rewriter.create<arith::ExtSIOp>(loc, logicalBase.getType(), offset);
  Value logical = rewriter.create<arith::AddIOp>(loc, logicalBase, wideOffset);
  for (Operation *op : ordered)
    op->moveBefore(first);
  flat.replaceUsesWithIf(tile, [&](OpOperand &use) {
    return use.getOwner() != pair.getDefiningOp() &&
           use.getOwner() != tile.getDefiningOp();
  });
  logicalBase.replaceUsesWithIf(logical, [&](OpOperand &use) {
    return use.getOwner() != logical.getDefiningOp() &&
           loop->isAncestor(use.getOwner());
  });
  // get_program_id lowering uses a remainder by the logical grid. An odd
  // grid has one inactive peer; guard its raw ID before that remainder can
  // turn it into a duplicate execution of logical program zero.
  if (!logicalGrid) return success();
  auto work = load->getParentOfType<scf::IfOp>();
  if (!work) return scope.emitError("program-split missing active work guard");
  rewriter.setInsertionPoint(work);
  Value wideGrid = rewriter.create<arith::ExtSIOp>(loc, logical.getType(), logicalGrid);
  Value inGrid = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, logical, wideGrid);
  // Whole-program normalization creates only XYZ grid checks here. A valid
  // raw ID implies all three after de-linearization: launch validation proves
  // positive dimensions and bounds the grid product/resident lookahead. Replace
  // those checks only once this stronger, pre-remainder predicate is available.
  // Flat streams also guard their data extent, which must remain independent.
  Value active = inGrid;
  if (!wholeProgram)
    active = rewriter.create<arith::AndIOp>(loc, inGrid, work.getCondition());
  work.getConditionMutable().assign(active);
  return guardInactivePeer(loop, work, logical, positiveDivisors);
}

struct L1PrefetchMapProgramsPass
    : impl::L1PrefetchMapProgramsBase<L1PrefetchMapProgramsPass> {
  using L1PrefetchMapProgramsBase::L1PrefetchMapProgramsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!hacc::utils::isDeviceEntry(func) ||
        !func->hasAttr("program_split.transformed"))
      return;

    auto launch = func->getAttrOfType<DictionaryAttr>("program_split.launch");
    if (!launch) {
      func.emitError("L1 pipeline error: missing transformed launch metadata");
      return signalPassFailure();
    }
    auto fixedLogical = launch.getAs<IntegerAttr>("logical_programs");
    auto fixedPhysical = launch.getAs<IntegerAttr>("physical_programs");
    if (fixedLogical ? (!fixedPhysical || fixedLogical.getInt() <= 0 ||
                        fixedLogical.getInt() % 2 ||
                        fixedPhysical.getInt() != fixedLogical.getInt() / 2)
                     : (!launch.getAs<IntegerAttr>("rows_arg") &&
                        !launch.getAs<IntegerAttr>("static_rows") &&
                        !launch.getAs<IntegerAttr>("jagged_copy") &&
                        !launch.contains("flat_tile") && !launch.contains("program_tile") &&
                        !launch.contains("persistent_runtime"))) {
      func.emitError("L1 pipeline error: inconsistent transformed launch metadata");
      return signalPassFailure();
    }
    if (launch.contains("persistent_runtime")) {
      auto mode = launch.getAs<IntegerAttr>("persistent_runtime");
      bool valid = mode && mode.getInt() == 1 && !fixedLogical &&
                   !launch.contains("pair_stride");
      for (StringRef name : {"logical_programs", "persistent_steps", "row_tasks"}) {
        auto arg = launch.getAs<IntegerAttr>((name + "_arg").str());
        auto literal = launch.getAs<IntegerAttr>(("static_" + name).str());
        auto grid = launch.getAs<IntegerAttr>((name + "_grid").str());
        valid &= bool(arg) + bool(literal) + bool(grid) == 1;
        valid &= !arg || arg.getInt() >= 2;
        valid &= !literal || literal.getInt() > 0;
        valid &= !grid || (grid.getInt() >= 0 && grid.getInt() < 3);
      }
      if (!valid) {
        func.emitError("L1 pipeline error: malformed runtime Persistent counts");
        return signalPassFailure();
      }
    }
    if (failed(verifyL1PrefetchPSRealizationRoles(func, launch)))
      return signalPassFailure();
    unsigned inputs = launch.getAs<DenseI32ArrayAttr>("input_args").size();
    SmallVector<Value> selectedInput(inputs);
    bool invalidSelection = false;
    unsigned selectedMarks = 0;
    func.walk([&](annotation::MarkOp mark) {
      if (auto id = mark->getAttrOfType<IntegerAttr>(
              "program_split.selected_input")) {
        if (!id.getType().isInteger(32) || id.getInt() < 0 ||
            id.getInt() >= inputs || selectedInput[id.getInt()])
          invalidSelection = true;
        else
          selectedInput[id.getInt()] = mark.getSrc();
        ++selectedMarks;
      }
    });
    for (unsigned id = 0; id < inputs; ++id) {
      unsigned selectedLoads = 0;
      if (selectedInput[id])
        func.walk([&](LoadOp load) {
          selectedLoads += load.getDst() == selectedInput[id];
        });
      if (invalidSelection || selectedMarks != inputs || selectedLoads != 1 ||
          !selectedInput[id].getDefiningOp<memref::AllocOp>()) {
        func.emitError("program-split lost selected input identity");
        return signalPassFailure();
      }
    }
    auto coreType = queryFuncCoreType(func);
    if (!coreType || *coreType != TFuncCoreType::MIX) {
      func.emitError("program-split requires a MIX function");
      return signalPassFailure();
    }

    Block &entry = func.getBody().front();
    annotation::MarkOp logicalMark;
    GetBlockIdxOp blockIdx;
    for (Operation &op : entry) {
      if (auto mark = dyn_cast<annotation::MarkOp>(op)) {
        if (mark->hasAttr(kLogicalBlockNum)) {
          if (logicalMark) {
            func.emitError("program-split found multiple logical grid marks");
            return signalPassFailure();
          }
          logicalMark = mark;
        }
      }
      if (auto idx = dyn_cast<GetBlockIdxOp>(op)) {
        if (blockIdx) {
          func.emitError("program-split found multiple block IDs");
          return signalPassFailure();
        }
        blockIdx = idx;
      }
    }
    if (!logicalMark || !blockIdx ||
        !logicalMark.getSrc().getType().isInteger(32)) {
      func.emitError("program-split lost logical grid or block ID");
      return signalPassFailure();
    }

    auto positiveDivisors = positiveGridDivisors(func, logicalMark.getSrc());
    for (BlockArgument arg : func.getArguments())
      func.removeArgAttr(arg.getArgNumber(), l1prefetch::kLaunchGridAxis);

    llvm::SmallPtrSet<Operation *, 8> keep;
    collectDependencies(logicalMark.getSrc(), keep);
    keep.insert(logicalMark);
    keep.insert(blockIdx);
    SmallVector<Operation *> body;
    for (Operation &op : entry)
      if (!isa<func::ReturnOp>(op) && !keep.contains(&op))
        body.push_back(&op);

    IRRewriter rewriter(func.getContext());
    rewriter.setInsertionPointAfter(blockIdx);
    Location loc = blockIdx.getLoc();
    Value physicalId = rewriter.create<arith::TruncIOp>(
        loc, rewriter.getI32Type(), blockIdx.getResult());
    Operation *hardwareIdCast = physicalId.getDefiningOp();

    rewriter.setInsertionPoint(entry.getTerminator());
    Value halfGrid;
    if (fixedLogical) {
      halfGrid = rewriter.create<arith::ConstantIntOp>(loc, fixedLogical.getInt() / 2, 32);
    } else {
      Value two = rewriter.create<arith::ConstantIntOp>(loc, 2, 32);
      Value logicalCount = logicalMark.getSrc();
      if (launch.contains("flat_tile") || launch.contains("program_tile")) {
        Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
        logicalCount = rewriter.create<arith::AddIOp>(loc, logicalCount, one);
      }
      halfGrid = rewriter.create<arith::DivSIOp>(loc, logicalCount, two);
    }
    halfGrid.getDefiningOp()->setAttr("program_split.physical_count", rewriter.getUnitAttr());

    // The accepted launch supplies an optional resident-pair extent. The
    // common normalizer below folds it into the tile loop, never around it.
    auto pairStride = launch.getAs<IntegerAttr>("pair_stride");
    if (launch.contains("jagged_copy") && !pairStride) {
      func.emitError("program-split missing jagged pair stride");
      return signalPassFailure();
    }
    physicalId.getDefiningOp()->setAttr("program_split.physical_id",
                                      rewriter.getUnitAttr());

    // Keep the complete logical program in one VECTOR scope. SplitMixKernel
    // removes this scope from the empty AIC companion and retains it in AIV.
    auto vectorScope =
        rewriter.create<scope::ScopeOp>(loc, TypeRange{});
    vectorScope.setNoInline(true);
    vectorScope->setAttr(
        TCoreTypeAttr::name,
        TCoreTypeAttr::get(rewriter.getContext(), TCoreType::VECTOR));
    vectorScope->setAttr("program_split.whole_program_vector",
                         rewriter.getUnitAttr());
    Block *scopeBody = rewriter.createBlock(&vectorScope.getRegion());

    rewriter.setInsertionPointToStart(scopeBody);
    Value subBlock = rewriter.create<GetSubBlockIdxOp>(loc).getResult();
    Value subBlockId = rewriter.create<arith::TruncIOp>(
        loc, rewriter.getI32Type(), subBlock);
    Value logicalOffset = rewriter.create<arith::MulIOp>(
        loc, subBlockId, halfGrid);
    Value logicalId32 = rewriter.create<arith::AddIOp>(
        loc, physicalId, logicalOffset);
    Value logicalId = rewriter.create<arith::ExtSIOp>(
        loc, rewriter.getI64Type(), logicalId32);
    logicalId.getDefiningOp()->setAttr("program_split.logical_id",
                                       rewriter.getUnitAttr());
    rewriter.replaceAllUsesExcept(blockIdx.getResult(), logicalId,
                                  hardwareIdCast);

    for (Operation *op : body)
      op->moveBefore(scopeBody, scopeBody->end());
    rewriter.setInsertionPointToEnd(scopeBody);
    rewriter.create<scope::ReturnOp>(loc);
    if (failed(normalizeResidentTileStream(vectorScope, selectedInput[0],
                                          logicalId, physicalId, halfGrid,
                                          (launch.contains("flat_tile") || launch.contains("program_tile")) ? logicalMark.getSrc() : Value{},
                                          pairStride, launch.contains("program_tile"), positiveDivisors, rewriter)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createL1PrefetchMapProgramsPass() {
  return std::make_unique<L1PrefetchMapProgramsPass>();
}
