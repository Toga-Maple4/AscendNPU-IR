// Whole-program staging proofs. Computation and output addressing stay intact.
#ifndef BISHENGIR_L1PREFETCHPS_PROGRAM_H
#define BISHENGIR_L1PREFETCHPS_PROGRAM_H

#include "L1PrefetchPSFacts.h"
#include "L1PrefetchPSDependencies.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::hivm::l1prefetch {

// The AIC may reproduce scalar address calculations, but never tensor
// computation or a value read from memory. Original arithmetic is unchanged.
inline bool programAddress(Value value, func::FuncOp func,
                           KernelLayout &layout) {
  SmallVector<Value> roots;
  for (BlockArgument arg : func.getArguments())
    if (arg.getArgNumber() >= layout.publicBegin() &&
        (isa<IntegerType, IndexType>(arg.getType()) ||
         (layout.isPublic(arg) && isa<MemRefType>(arg.getType()))))
      roots.push_back(arg);
  // These definitions are unconditional in the original logical program.
  // normalizeFlat keeps them inside its valid-program guard; materialization
  // reproduces that guard before the address. Runtime divisors need no invented
  // range: their original execution condition is retained.
  DependencyProof proof(func, roots, {&func.getBody().front()});
  return proof.check(value);
}

// Follow the dimension containing invalid rows, rather than recognizing a
// formula. Linalg indexing maps must preserve that parallel dimension in every
// result; reducing it would mix discarded values into observable outputs.
inline bool hasUnobservableRowPadding(bufferization::ToTensorOp input,
                                     Value liveRows, func::FuncOp func) {
  SmallVector<std::pair<Value, unsigned>> pending{{input.getResult(), 0}}, seen;
  bool reachesStore = false;
  while (!pending.empty()) {
    auto [value, dim] = pending.pop_back_val();
    if (llvm::is_contained(seen, std::make_pair(value, dim)))
      continue;
    seen.emplace_back(value, dim);
    auto sourceType = cast<RankedTensorType>(value.getType());
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (user->getParentOp() != func)
        return false;
      if (auto slice = dyn_cast<tensor::ExtractSliceOp>(user)) {
        if (slice.getSource() != value ||
            slice.getMixedSizes().size() != unsigned(sourceType.getRank()) ||
            slice.getMixedSizes()[dim] != OpFoldResult(liveRows) ||
            llvm::any_of(slice.getStaticOffsets(), [](int64_t x) { return x != 0; }) ||
            llvm::any_of(slice.getStaticStrides(), [](int64_t x) { return x != 1; }) ||
            slice.getResult().use_empty())
          return false;
        for (Operation *sink : slice.getResult().getUsers()) {
          auto store = dyn_cast<bufferization::MaterializeInDestinationOp>(sink);
          if (!store || store.getSource() != slice.getResult() ||
              store->getParentOp() != func)
            return false;
        }
        reachesStore = true;
        continue;
      }
      if (!isMemoryEffectFree(user))
        return false;
      if (auto linalg = dyn_cast<linalg::LinalgOp>(user)) {
        if (!linalg.hasPureTensorSemantics())
          return false;
        bool canSpeculate = true;
        user->walk([&](Operation *nested) {
          if (nested != user && !isa<linalg::YieldOp>(nested) &&
              !isSpeculatable(nested))
            canSpeculate = false;
        });
        if (!canSpeculate)
          return false;
        auto maps = linalg.getIndexingMapsArray();
        if (use.getOperandNumber() >= maps.size())
          return false;
        auto loopDim = dyn_cast<AffineDimExpr>(
            maps[use.getOperandNumber()].getResult(dim));
        if (!loopDim || linalg.getIteratorTypesArray()[loopDim.getPosition()] !=
                            utils::IteratorType::parallel)
          return false;
        for (auto result : llvm::enumerate(user->getResults())) {
          auto map = maps[linalg.getNumDpsInputs() + result.index()];
          auto found = llvm::find(map.getResults(), loopDim);
          if (found == map.getResults().end())
            return false;
          pending.emplace_back(result.value(), found - map.getResults().begin());
        }
        continue;
      }
      if (!user->hasTrait<OpTrait::Elementwise>() || user->getNumRegions() ||
          !isSpeculatable(user))
        return false;
      for (Value result : user->getResults()) {
        auto type = dyn_cast<RankedTensorType>(result.getType());
        if (!type || type.getShape() != sourceType.getShape())
          return false;
        pending.emplace_back(result, dim);
      }
    }
  }
  return reachesStore;
}

// A loopless program is kept whole: its stores and tensor operations are not
// remapped to input shapes. Selection is restricted by transport/lifetime
// capabilities, not by the identity of its arithmetic operations.
inline std::string analyzeProgramInputs(func::FuncOp func, KernelLayout &layout,
                                       EffectInventory &inventory) {
  if (!inventory.loops.empty() || !inventory.metadataLoads.empty())
    return "program_unmodelled_memory_dependency";
  // The current materializer has one completion boundary. Keep this physical
  // limitation explicit until relevant-reader completion supports more stores.
  if (inventory.stores.size() != 1)
    return "program_multiple_completion_stores";
  SmallVector<memref::CopyOp> selected;
  MemRefType tile;
  for (auto copy : inventory.copies) {
    auto dst = copy.getTarget().getDefiningOp<memref::SubViewOp>();
    auto alloc = dst ? dst.getSource().getDefiningOp<memref::AllocOp>()
                     : memref::AllocOp();
    if (!alloc || alloc.getType().getRank() != 2)
      continue;
    if (!tile)
      tile = alloc.getType();
    if (alloc.getType() == tile)
      selected.push_back(copy);
  }
  if (!validTile(tile, selected.size()))
    return "program_unsupported_selected_tile";
  AccessRoles roles;
  auto reason = deriveAccessRoles(func, layout, AccessSchedule::Program, selected,
                                 inventory.stores.front(), {}, {}, {}, tile, roles);
  if (!reason.empty())
    return reason;
  llvm::SmallPtrSet<Operation *, 16> accounted;
  SmallVector<scf::IfOp> padding;
  for (auto copy : selected) {
    auto src = copy.getSource().getDefiningOp<memref::SubViewOp>();
    auto dst = copy.getTarget().getDefiningOp<memref::SubViewOp>();
    auto base = src ? src.getSource().getDefiningOp<memref::ReinterpretCastOp>()
                    : memref::ReinterpretCastOp();
    auto alloc = dst.getSource().getDefiningOp<memref::AllocOp>();
    if (!src || !base || copy->getParentOp() != func ||
        base.getStaticSizes() != tile.getShape() ||
        base.getOffsets().size() != 1 ||
        !programAddress(base.getOffsets().front(), func, layout) ||
        cast<MemRefType>(base.getType()).getElementType() != tile.getElementType())
      return "program_unmodelled_access_dependency";
    auto strides = base.getMixedStrides();
    if (strides.size() != 2 || getConstantIntValue(strides[1]) != 1)
      return "program_unsupported_transfer_layout";
    auto sizes = src.getMixedSizes();
    if (sizes.size() != 2 || sizes != dst.getMixedSizes() ||
        getConstantIntValue(sizes[1]) != tile.getDimSize(1) ||
        src.getStaticOffsets() != ArrayRef<int64_t>{0, 0} ||
        dst.getStaticOffsets() != ArrayRef<int64_t>{0, 0} ||
        src.getStaticStrides() != ArrayRef<int64_t>{1, 1} ||
        dst.getStaticStrides() != ArrayRef<int64_t>{1, 1})
      return "program_unproved_live_rectangle";
    Value live = dyn_cast<Value>(sizes[0]);
    auto sub = live ? live.getDefiningOp<arith::SubIOp>() : arith::SubIOp();
    Value bound = live ? tileExtentBound(live) : Value();
    if (!sub || !bound || !programAddress(live, func, layout) ||
        !boundedTileExtent(live, sub.getRhs(), bound, tile.getDimSize(0)))
      return "program_unproved_live_rectangle";
    bufferization::ToTensorOp tensor;
    scf::IfOp fillGuard;
    for (Operation *user : alloc->getUsers()) {
      if (auto t = dyn_cast<bufferization::ToTensorOp>(user)) {
        if (tensor)
          return "program_escaping_selected_storage";
        tensor = t;
      } else if (auto fill = dyn_cast<linalg::FillOp>(user)) {
        auto guard = dyn_cast<scf::IfOp>(fill->getParentOp());
        if (fillGuard || !guard || guard->getParentOp() != func ||
            guard.getNumResults() || !guard.getElseRegion().empty() ||
            std::distance(guard.getThenRegion().front().begin(),
                          guard.getThenRegion().front().end()) != 2 ||
            !programAddress(guard.getCondition(), func, layout))
          return "program_unproved_padding";
        fillGuard = guard;
        accounted.insert(fill);
        accounted.insert(guard);
        accounted.insert(guard.getThenRegion().front().getTerminator());
      } else if (user != dst.getOperation())
        return "program_escaping_selected_storage";
    }
    if (!tensor || !fillGuard)
      return "program_observable_or_unproved_padding";
    // Removing initialization is safe only when the copy overwrites its live
    // rows before tensor computation reads them. A later fill would also change
    // live values; a tensor read before the copy observes the initialization.
    if (!initializationPrecedesRead(fillGuard, copy, tensor))
      return "program_unproved_initialization_order";
    if (!hasUnobservableRowPadding(tensor, live, func))
      return "program_observable_or_unproved_padding";
    padding.push_back(fillGuard);
    for (Operation *op : {copy.getOperation(), alloc.getOperation(),
                          tensor.getOperation()})
      accounted.insert(op);
  }
  // No unaccounted write/read may be crossed by prefetch. Pure tensor regions
  // are retained verbatim; local side loads will need their own effect proof.
  accounted.insert(inventory.stores.front());
  if (!hasOnlyAccountedEffects(func, accounted))
    return "program_unmodelled_memory_dependency";
  OpBuilder builder(func.getContext());
  if (failed(publishAcceptedFacts(func, {AccessSchedule::Program, roles, {}, {}, {}})))
    return "inconsistent_accepted_facts";
  func->setAttr("l1_prefetch_ps.program", builder.getUnitAttr());
  for (auto copy : selected) {
    copy->setAttr("l1_prefetch_ps.load", builder.getUnitAttr());
    auto alloc = copy.getTarget().getDefiningOp<memref::SubViewOp>().getSource();
    auto pointer = accessView(copy.getSource()).getSource();
    unsigned id = llvm::find(roles.inputs, pointer) - roles.inputs.begin();
    builder.setInsertionPointAfter(alloc.getDefiningOp());
    auto mark = builder.create<annotation::MarkOp>(func.getLoc(), alloc);
    mark->setAttr("program_split.selected_input", builder.getI32IntegerAttr(id));
  }
  for (auto guard : padding)
    guard->setAttr("l1_prefetch_ps.padding", builder.getUnitAttr());
  return "";
}
} // namespace mlir::hivm::l1prefetch
#endif
