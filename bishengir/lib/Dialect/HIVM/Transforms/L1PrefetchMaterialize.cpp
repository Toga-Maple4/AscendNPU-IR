//===- L1PrefetchMaterialize.cpp - Ping/pong L1 data path ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "L1PrefetchPSCommon.h"
#include "L1PrefetchPSDependencies.h"
#include "L1PrefetchPSReservation.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>

namespace mlir {
#define GEN_PASS_DEF_L1PREFETCHMATERIALIZE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
constexpr llvm::StringLiteral kSelectedInput =
    "program_split.selected_input";
constexpr llvm::StringLiteral kVectorProgram =
    "program_split.whole_program_vector";
constexpr llvm::StringLiteral kPhysicalId = "program_split.physical_id";
constexpr llvm::StringLiteral kLogicalId = "program_split.logical_id";

bool isInside(Operation *ancestor, Operation *op) {
  return ancestor == op || ancestor->isAncestor(op);
}

FailureOr<Value> cloneDependency(Value value, Operation *vectorScope,
                                 IRMapping &mapping, OpBuilder &builder) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  if (isa<BlockArgument>(value))
    return value;
  Operation *def = value.getDefiningOp();
  if (!def || !isInside(vectorScope, def))
    return value;
  for (Value operand : def->getOperands()) {
    FailureOr<Value> cloned =
        cloneDependency(operand, vectorScope, mapping, builder);
    if (failed(cloned))
      return failure();
    mapping.map(operand, *cloned);
  }
  Operation *clone = builder.clone(*def, mapping);
  for (auto [oldResult, newResult] : llvm::zip(def->getResults(),
                                               clone->getResults()))
    mapping.map(oldResult, newResult);
  Value mapped = mapping.lookupOrNull(value);
  if (!mapped)
    return failure();
  return mapped;
}

void markBufferEffects(OpBuilder &builder, Location loc, Value allocation) {
  auto mark = builder.create<annotation::MarkOp>(loc, allocation);
  // These marks reserve the cube-owned L1 banks. Do not retain them in
  // the vector half after SplitMixKernel removes all accesses to those banks.
  mark->setAttr(TCoreTypeAttr::name,
                TCoreTypeAttr::get(builder.getContext(), TCoreType::CUBE));
  mark->setAttr("effects", builder.getArrayAttr(
                               {builder.getStringAttr("write"),
                                builder.getStringAttr("read")}));
}

void bindBuffer(OpBuilder &builder, Location loc, Value tensor, Value l1) {
  builder.create<annotation::MarkOp>(
      loc, tensor, ValueRange{l1},
      builder.getArrayAttr(builder.getStringAttr("bind_buffer")));
}

struct L1PrefetchMaterializePass
    : impl::L1PrefetchMaterializeBase<L1PrefetchMaterializePass> {
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
    if (failed(verifyL1PrefetchPSRealizationRoles(func, launch)))
      return signalPassFailure();
    unsigned inputs = launch.getAs<DenseI32ArrayAttr>("input_args").size();
    bool independentConsumers = launch.contains("pair_stride");
    scope::ScopeOp vectorScope;
    SmallVector<annotation::MarkOp> selectedMark(inputs);
    SmallVector<memref::AllocOp> selectedAlloc(inputs);
    SmallVector<LoadOp> selectedLoads(inputs);
    SmallVector<bufferization::ToTensorOp> selectedTensor(inputs);
    bool invalidSelection = false;
    StoreOp vectorStore;
    Operation *physicalIdOp = nullptr;
    Operation *logicalIdOp = nullptr;
    Value runtimeHalfGrid;

    func.walk([&](Operation *op) {
      if (auto scope = dyn_cast<scope::ScopeOp>(op);
          scope && scope->hasAttr(kVectorProgram))
        vectorScope = scope;
      if (op->hasAttr("program_split.physical_count") && op->getNumResults() == 1)
        runtimeHalfGrid = op->getResult(0);
      if (op->hasAttr(kPhysicalId))
        physicalIdOp = op;
      if (op->hasAttr(kLogicalId))
        logicalIdOp = op;
      if (auto mark = dyn_cast<annotation::MarkOp>(op)) {
        if (auto id = mark->getAttrOfType<IntegerAttr>(kSelectedInput)) {
          if (!id.getType().isInteger(32) || id.getInt() < 0 ||
              id.getInt() >= inputs || selectedMark[id.getInt()])
            invalidSelection = true;
          else
            selectedMark[id.getInt()] = mark;
        }
      }
    });

    if (!runtimeHalfGrid || !vectorScope || invalidSelection ||
        llvm::any_of(selectedMark, [](auto mark) { return !mark; }) || !physicalIdOp || !logicalIdOp ||
        physicalIdOp->getNumResults() != 1 ||
        logicalIdOp->getNumResults() != 1) {
      func.emitError("L1 pipeline error: L1 materialization lost the whole-program selection");
      return signalPassFailure();
    }

    for (unsigned id = 0; id < inputs; ++id) {
      selectedAlloc[id] =
          selectedMark[id]
              ? selectedMark[id].getSrc().getDefiningOp<memref::AllocOp>()
              : memref::AllocOp();
      if (!selectedAlloc[id]) {
        func.emitError("L1 pipeline error: selected input is not an allocation");
        return signalPassFailure();
      }
      for (Operation *user : selectedAlloc[id].getResult().getUsers()) {
        if (auto load = dyn_cast<LoadOp>(user)) {
          if (load.getDst() != selectedAlloc[id] || selectedLoads[id]) {
            func.emitError("program-split selected input has duplicate or "
                           "unsupported loads");
            return signalPassFailure();
          }
          selectedLoads[id] = load;
        } else if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(user)) {
          if (selectedTensor[id]) {
            func.emitError(
                "program-split selected input has duplicate tensor views");
            return signalPassFailure();
          }
          selectedTensor[id] = toTensor;
        } else if (user != selectedMark[id].getOperation()) {
          func.emitError(
              "L1 pipeline error: selected input has an unsupported direct use");
          return signalPassFailure();
        }
      }
      if (!selectedLoads[id] || !selectedTensor[id] ||
          selectedLoads[id]->getParentOfType<scope::ScopeOp>() != vectorScope) {
        func.emitError("L1 pipeline error: selected load or tensor view is missing");
        return signalPassFailure();
      }
    }
    LoadOp selectedLoad = selectedLoads[0];

    auto loop = selectedLoad->getParentOfType<scf::ForOp>();
    auto condition = selectedLoad->getParentOfType<scf::IfOp>();
    unsigned stores = 0;
    if (condition)
      condition.walk([&](StoreOp store) {
        vectorStore = store;
        ++stores;
      });
    auto selectedType = dyn_cast<MemRefType>(selectedAlloc[0].getType());
    if (!loop || !condition || stores != 1 || !vectorStore || !selectedType ||
        (selectedType.getRank() != 1 && selectedType.getRank() != 2) || !selectedType.hasStaticShape() ||
        selectedType.getMemorySpace()) {
      func.emitError("L1 pipeline error: requires matched static rank-1 or rank-2 selected UB tiles");
      return signalPassFailure();
    }

    auto target = hacc::utils::getNPUTargetSpec(func->getParentOfType<ModuleOp>());
    auto reservation = l1prefetch::checkL1Reservation(
        selectedType, inputs, target ? *target : hacc::HACCTargetDeviceSpecInterface());
    if (!reservation.succeeded()) {
      func.emitError("L1 pipeline error: ") << reservation.message;
      return signalPassFailure();
    }

    bool flat = selectedType.getRank() == 1;
    SmallVector<int64_t> transferShape(selectedType.getShape());
    if (flat) transferShape.insert(transferShape.begin(), 1);
    if (!l1prefetch::validTile(MemRefType::get(transferShape,
            selectedType.getElementType()), inputs)) {
      func.emitError("L1 pipeline error: tile violates ND transfer descriptor bounds or row alignment");
      return signalPassFailure();
    }
    auto asTransferView = [&](OpBuilder &builder, Value value) -> Value {
      if (!flat) return value;
      auto type = cast<MemRefType>(value.getType());
      SmallVector<int64_t> shape{1, type.getDimSize(0)};
      return builder.create<memref::ExpandShapeOp>(value.getLoc(), shape, value,
          SmallVector<ReassociationIndices>{{0, 1}});
    };

    Value logicalId = logicalIdOp->getResult(0);
    Value physicalId = physicalIdOp->getResult(0);
    auto dependencies = l1prefetch::stagingDependencies(
        func, vectorScope, ValueRange{logicalId, loop.getInductionVar()}, loop,
        {&vectorScope.getRegion().front(), loop.getBody()});
    if (!dependencies.check(condition.getCondition())) {
      func.emitError("L1 pipeline error: cannot clone the selected load address slice");
      return signalPassFailure();
    }
    // A source is cloned only inside the copied payload condition. The loop
    // execution condition is preserved in the prologue and lookahead below.
    auto sourceDependencies = l1prefetch::stagingDependencies(
        func, vectorScope, ValueRange{logicalId, loop.getInductionVar()}, loop,
        {&vectorScope.getRegion().front(), loop.getBody(),
         &condition.getThenRegion().front()});
    for (unsigned id = 0; id < inputs; ++id)
      if (selectedAlloc[id].getType() != selectedType ||
          selectedLoads[id]->getParentOfType<scf::ForOp>() != loop ||
          selectedLoads[id]->getParentOfType<scf::IfOp>() != condition ||
          !sourceDependencies.check(selectedLoads[id].getSrc())) {
        func.emitError("program-split selected inputs lost their matched "
                       "tile/address contract");
        return signalPassFailure();
      }

    // Selection IDs follow the public ABI, which need not match source order.
    // Insert replacement allocations and delivery before every selected use.
    Operation *firstAllocation = selectedAlloc[0].getOperation();
    Operation *firstLoad = selectedLoads[0].getOperation();
    for (unsigned id = 1; id < inputs; ++id) {
      if (selectedAlloc[id]->getBlock() != firstAllocation->getBlock() ||
          selectedLoads[id]->getBlock() != firstLoad->getBlock()) {
        func.emitError("L1 pipeline error: selected inputs lost their common "
                       "allocation/load block");
        return signalPassFailure();
      }
      if (selectedAlloc[id]->isBeforeInBlock(firstAllocation))
        firstAllocation = selectedAlloc[id].getOperation();
      if (selectedLoads[id]->isBeforeInBlock(firstLoad))
        firstLoad = selectedLoads[id].getOperation();
    }

    IRRewriter rewriter(func.getContext());
    Location loc = selectedLoad.getLoc();
    // The selected load already carries the live input rectangle proved by
    // analysis. Only those rows can reach an observable output. Keep the full
    // physical width and allocation, but do not deliver unobserved rows to UB.
    SmallVector<OpFoldResult, 2> liveRows;
    bool boundedRows = false;
    for (unsigned id = 0; id < inputs; ++id) {
      rewriter.setInsertionPoint(selectedLoads[id]);
      // Padding-producing loads initialize the rest of the destination too;
      // retain those rows. PadNull is the normalized no-padding contract.
      OpFoldResult rows = rewriter.getIndexAttr(transferShape[0]);
      auto padMode = selectedLoads[id].getPadMode();
      if (!flat && padMode && padMode->getPadmode() == PadMode::PadNull)
        rows = memref::getMixedSize(
            rewriter, loc, selectedLoads[id].getSrc(), 0);
      liveRows.push_back(rows);
      boundedRows |= getConstantIntValue(rows) != transferShape[0];
    }

    auto l1Space = AddressSpaceAttr::get(func.getContext(), AddressSpace::L1);
    auto ubSpace = AddressSpaceAttr::get(func.getContext(), AddressSpace::UB);
    auto transferType = flat ? MemRefType::get(transferShape, selectedType.getElementType())
                             : selectedType;
    auto l1Type = MemRefType::get(transferShape,
                                  selectedType.getElementType(),
                                  transferType.getLayout(), l1Space);
    auto ubType = MemRefType::get(selectedType.getShape(),
                                  selectedType.getElementType(),
                                  selectedType.getLayout(), ubSpace);

    rewriter.setInsertionPoint(vectorScope);
    SmallVector<SmallVector<SmallVector<Value, 2>, 2>, 2> l1Tiles(inputs);
    for (unsigned id = 0; id < inputs; ++id) {
      l1Tiles[id].resize(2);
      for (int stream = 0; stream < 2; ++stream) {
        for (int bank = 0; bank < 2; ++bank) {
          Value tile = rewriter.create<memref::AllocOp>(loc, l1Type);
          markBufferEffects(rewriter, loc, tile);
          l1Tiles[id][stream].push_back(tile);
        }
      }
    }

    rewriter.setInsertionPoint(independentConsumers ? condition.getOperation() : firstAllocation);
    SmallVector<memref::AllocOp> ubAlloc(inputs);
    SmallVector<annotation::MarkOp> replacementMark(inputs);
    for (unsigned id = 0; id < inputs; ++id) {
      ubAlloc[id] = rewriter.create<memref::AllocOp>(loc, ubType);
      ubAlloc[id]->setAttrs(selectedAlloc[id]->getAttrs());
      replacementMark[id] =
          rewriter.create<annotation::MarkOp>(loc, ubAlloc[id]);
      replacementMark[id]->setAttrs(selectedMark[id]->getAttrs());
      Value ubCast = rewriter.create<memref::MemorySpaceCastOp>(
          loc, selectedType, ubAlloc[id]);
      selectedTensor[id]->setOperand(0, ubCast);
    }

    auto emitPrefetchPair = [&](OpBuilder &builder, Value iteration,
                                int bank) -> LogicalResult {
      for (int stream = 0; stream < 2; ++stream) {
        IRMapping mapping;
        mapping.map(loop.getInductionVar(), iteration);
        Value cubeLogical = physicalId;
        if (stream == 1) {
          Value offset = runtimeHalfGrid;
          cubeLogical = builder.create<arith::AddIOp>(
              loc, physicalId, offset);
        }
        if (cubeLogical.getType() != logicalId.getType())
          cubeLogical = builder.create<arith::ExtSIOp>(
              loc, logicalId.getType(), cubeLogical);
        mapping.map(logicalId, cubeLogical);

        FailureOr<Value> clonedCondition = cloneDependency(
            condition.getCondition(), vectorScope, mapping, builder);
        if (failed(clonedCondition))
          return failure();
        auto cubeIf = builder.create<scf::IfOp>(
            loc, *clonedCondition, /*withElseRegion=*/false);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&cubeIf.getThenRegion().front());
        for (unsigned id = 0; id < inputs; ++id) {
          FailureOr<Value> source = cloneDependency(
              selectedLoads[id].getSrc(), vectorScope, mapping, builder);
          if (failed(source))
            return failure();
          Value staging = builder.create<memref::AllocOp>(loc, transferType);
          mapping.map(selectedLoads[id].getSrc(), asTransferView(builder, *source));
          mapping.map(selectedAlloc[id].getResult(), staging);
          auto stagingLoad = cast<LoadOp>(builder.clone(*selectedLoads[id], mapping));
          // PadNull copies only the live source rectangle. Keep the full bank
          // allocation/binding, but give the load an equal-shaped destination;
          // canonicalization may make a partial row count statically known.
          auto padMode = stagingLoad.getPadMode();
          if (!flat && padMode && padMode->getPadmode() == PadMode::PadNull) {
            OpBuilder::InsertionGuard insertionGuard(builder);
            builder.setInsertionPoint(stagingLoad);
            SmallVector<OpFoldResult> offsets(transferShape.size(), builder.getIndexAttr(0));
            SmallVector<OpFoldResult> strides(transferShape.size(), builder.getIndexAttr(1));
            SmallVector<OpFoldResult> sizes;
            for (unsigned dim = 0; dim < transferShape.size(); ++dim)
              sizes.push_back(memref::getMixedSize(builder, loc, stagingLoad.getSrc(), dim));
            Value liveDestination = builder.create<memref::SubViewOp>(
                loc, staging, offsets, sizes, strides);
            stagingLoad.getDstMutable().assign(liveDestination);
          }
          stagingLoad->setAttr(kGroupedL1LoadAttrName, builder.getUnitAttr());
          stagingLoad.setTcoretypeAttr(
              TCoreTypeAttr::get(func.getContext(), TCoreType::CUBE));
          auto tensorType = RankedTensorType::get(transferShape, selectedType.getElementType());
          Value tensor = builder.create<bufferization::ToTensorOp>(
              loc, tensorType, staging, /*restrict=*/true, /*writable=*/true);
          bindBuffer(builder, loc, tensor, l1Tiles[id][stream][bank]);
        }
      }
      return success();
    };

    // Prologue: bank 0 holds iteration zero before either vector subcore
    // enters the persistent loop.
    rewriter.setInsertionPoint(loop);
    auto prologueScope = rewriter.create<scope::ScopeOp>(loc, TypeRange{});
    prologueScope.setNoInline(true);
    prologueScope->setAttr(
        TCoreTypeAttr::name,
        TCoreTypeAttr::get(func.getContext(), TCoreType::CUBE));
    prologueScope->setAttr("program_split.l1_prologue",
                           rewriter.getUnitAttr());
    Block *prologueBody = rewriter.createBlock(&prologueScope.getRegion());
    rewriter.setInsertionPointToStart(prologueBody);
    // Loop-body arithmetic is not executed on a zero-trip source loop. Only
    // a normalization proof (or constant bounds) permits omitting this guard.
    // hasNext already supplies this execution condition for lookahead.
    if (!l1prefetch::isNonemptyTileStream(loop)) {
      Value enters = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, loop.getLowerBound(), loop.getUpperBound());
      auto entered = rewriter.create<scf::IfOp>(loc, enters, false);
      rewriter.setInsertionPointToStart(&entered.getThenRegion().front());
    }
    if (failed(emitPrefetchPair(rewriter, loop.getLowerBound(), 0))) {
      func.emitError("L1 pipeline error: failed to clone the L1 prologue");
      return signalPassFailure();
    }
    rewriter.setInsertionPointToEnd(prologueBody);
    rewriter.create<scope::ReturnOp>(loc);

    // The parity calculation is loop-uniform and dominates both the guarded
    // current delivery and the next-iteration prefetch.
    rewriter.setInsertionPoint(condition);
    Value two = rewriter.create<arith::ConstantIntOp>(loc, 2, 32);
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value parity = rewriter.create<arith::RemSIOp>(
        loc, loop.getInductionVar(), two);
    Value useBankZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, parity, zero);

    // Keep delivery and opposite-bank prefetch in the same parity branch.
    // Separate parity checks hide their mutual exclusion from GraphSyncSolver
    // and introduce a false same-iteration MTE1 -> MTE2 dependency.
    rewriter.setInsertionPoint(independentConsumers ? condition.getOperation() : firstLoad);
    Value nextIteration = rewriter.create<arith::AddIOp>(
        loc, loop.getInductionVar(), loop.getStep());
    Value hasNext = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, nextIteration, loop.getUpperBound());
    auto selectCurrentBank = rewriter.create<scf::IfOp>(
        loc, useBankZero, /*withElseRegion=*/true);
    for (unsigned bank = 0; bank < 2; ++bank) {
      Region &branch = bank == 0 ? selectCurrentBank.getThenRegion()
                                : selectCurrentBank.getElseRegion();
      rewriter.setInsertionPointToStart(&branch.front());
      {
        // Persistent activation schedules advance monotonically through row
        // tasks. Jagged parity is outside the per-vector active condition, so
        // either consumer can independently require the next tile. Prefetch can
        // share the current-work guard; each stream still checks its next tile.
        // Issue the next input before current UB handoff waits. Only reuse the
        // opposite L1 bank; GraphSync owns its previous-reader dependency.
        // Restore the insertion point so delivery remains after this prefetch.
        OpBuilder::InsertionGuard prefetchGuard(rewriter);
        auto prefetchNext = rewriter.create<scf::IfOp>(
            loc, hasNext, /*withElseRegion=*/false);
        rewriter.setInsertionPointToStart(&prefetchNext.getThenRegion().front());
        if (failed(emitPrefetchPair(rewriter, nextIteration, 1 - bank))) {
          func.emitError("L1 pipeline error: failed to clone the next prefetch");
          return signalPassFailure();
        }
      }

      // Keep every input of one consumer together for its UB handoff.
      for (int stream = 0; stream < 2; ++stream) {
        OpBuilder::InsertionGuard deliveryGuard(rewriter);
        // A bounded source has positive row extent only while its original
        // load is active. Guard the payload per consumer; ConsumerSync places
        // handback/readiness outside this guard, including for empty consumers.
        IRMapping mapping;
        if (independentConsumers || boundedRows) {
          Value cubeLogical = physicalId;
          if (stream) cubeLogical = rewriter.create<arith::AddIOp>(loc, physicalId, runtimeHalfGrid);
          if (cubeLogical.getType() != logicalId.getType())
            cubeLogical = rewriter.create<arith::ExtSIOp>(loc, logicalId.getType(), cubeLogical);
          mapping.map(logicalId, cubeLogical);
          mapping.map(loop.getInductionVar(), loop.getInductionVar());
          auto active = cloneDependency(condition.getCondition(), vectorScope, mapping, rewriter);
          if (failed(active)) return signalPassFailure();
          auto delivery = rewriter.create<scf::IfOp>(loc, *active, false);
          rewriter.setInsertionPointToStart(&delivery.getThenRegion().front());
        }
        for (unsigned id = 0; id < inputs; ++id) {
          Value source = l1Tiles[id][stream][bank];
          Value target = asTransferView(rewriter, ubAlloc[id].getResult());
          OpFoldResult rows = liveRows[id];
          if (getConstantIntValue(rows) != transferShape[0]) {
            if (auto value = dyn_cast<Value>(rows)) {
              auto cloned = cloneDependency(value, vectorScope, mapping, rewriter);
              if (failed(cloned))
                return signalPassFailure();
              rows = *cloned;
            }
            SmallVector<OpFoldResult> offsets(2, rewriter.getIndexAttr(0));
            SmallVector<OpFoldResult> strides(2, rewriter.getIndexAttr(1));
            SmallVector<OpFoldResult> sizes{
                rows, rewriter.getIndexAttr(selectedType.getDimSize(1))};
            source = rewriter.create<memref::SubViewOp>(
                loc, source, offsets, sizes, strides);
            target = rewriter.create<memref::SubViewOp>(
                loc, target, offsets, sizes, strides);
          }
          auto copy = rewriter.create<L12UBOp>(loc, TypeRange{}, source, target);
          copy.setSubBlockidAttr(rewriter.getBoolAttr(stream != 0));
        }
      }
    }

    // Nested analysis proves for each input that only the valid output rectangle
    // observes its tensor, through same-shape floating-point elementwise math.
    // Unused L1/UB lanes therefore need no zero fill or vector mask. Keep the
    // original arithmetic untouched, including on full-tile launches.

    // The selected UB stays busy until the asynchronous MTE3 store has
    // completed. CrossCoreGSS resolves this stable id and emits the physical
    // MTE3 -> MTE1 handback before the next L1 -> UB overwrite.
    for (unsigned id = 0; id < inputs; ++id)
      replacementMark[id]->setAttr(kSyncMemoryReadId,
                                   rewriter.getI32IntegerAttr(id));
    SmallVector<int32_t> readIds;
    for (unsigned id = 0; id < inputs; ++id) readIds.push_back(id);
    vectorStore->setAttr(kSyncMemoryReadIds, rewriter.getDenseI32ArrayAttr(readIds));

    for (unsigned id = 0; id < inputs; ++id) {
      rewriter.eraseOp(selectedLoads[id]);
      rewriter.eraseOp(selectedMark[id]);
      rewriter.eraseOp(selectedAlloc[id]);
    }

    // CrossCoreGSS must see both sides in one persistent loop. Resolve the
    // temporary scopes while preserving the per-operation core ownership.
    auto inlineScope = [&](scope::ScopeOp scope) {
      Block &body = scope.getRegion().front();
      for (Operation &op : llvm::make_early_inc_range(
               llvm::make_range(body.begin(), std::prev(body.end()))))
        op.moveBefore(scope);
      rewriter.eraseOp(scope);
    };
    loop->removeAttr(l1prefetch::kNonemptyTileStream);
    inlineScope(prologueScope);
    inlineScope(vectorScope);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createL1PrefetchMaterializePass() {
  return std::make_unique<L1PrefetchMaterializePass>();
}
