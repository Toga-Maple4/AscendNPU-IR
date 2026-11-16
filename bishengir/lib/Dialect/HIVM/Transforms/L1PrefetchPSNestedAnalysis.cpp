/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Semantic proof for a bounded nested row/column program with selected inputs.
 * This pass only annotates. The accepted retry normalizes the proved loops;
 * the launcher enforces the runtime premises derived from the accepted IR.
 */
#include "L1PrefetchPSAddress.h"
#include "L1PrefetchPSFacts.h"
#include "L1PrefetchPSProgram.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Transforms/L1PrefetchPS.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::hivm::l1prefetch;

namespace {
bool zeroOffsetsUnitStrides(memref::SubViewOp view) {
  return llvm::all_of(view.getStaticOffsets(),
                      [](int64_t n) { return n == 0; }) &&
         llvm::all_of(view.getStaticStrides(),
                      [](int64_t n) { return n == 1; }) &&
         view.getOffsets().empty() && view.getStrides().empty();
}

bool tileView(memref::ReinterpretCastOp view, Value base, MemRefType tile,
              Value stride, Value offset) {
  if (!view || view.getSource() != base || view.getOffsets().size() != 1 ||
      view.getOffsets().front() != offset)
    return false;
  auto type = cast<MemRefType>(view.getType());
  auto strides = view.getMixedStrides();
  auto sizes = view.getStaticSizes();
  if (type.getShape() != tile.getShape() || sizes != tile.getShape() ||
      type.getElementType() != tile.getElementType() || strides.size() != 2 ||
      !isa<Attribute>(strides[1]) ||
      cast<IntegerAttr>(cast<Attribute>(strides[1])).getInt() != 1)
    return false;
  if (auto value = dyn_cast<Value>(strides[0]))
    return value == stride;
  return integer(stride) &&
         cast<IntegerAttr>(cast<Attribute>(strides[0])).getInt() ==
             *integer(stride);
}
// A loopless contiguous elementwise program is a one-tile stream. Derive
// pointer/scalar roles from the accesses, not a kernel name or argument order.
// The same padding proof and physical staging serve nested and flat programs.
std::string analyzeFlatInputs(
    func::FuncOp func, KernelLayout &layout, ArrayRef<scf::IfOp> ifs,
    ArrayRef<memref::CopyOp> copies, MutableArrayRef<memref::AllocOp> allocs,
    ArrayRef<memref::SubViewOp> views,
    ArrayRef<memref::ReinterpretCastOp> bases,
    ArrayRef<bufferization::ToTensorOp> tensors,
    MutableArrayRef<bufferization::MaterializeInDestinationOp> stores,
    ArrayRef<tensor::ExtractSliceOp> slices, bool disallowed,
    EffectInventory &inventory) {
  unsigned inputs = copies.size();
  if (disallowed || inputs < 1 || allocs.size() != inputs ||
      tensors.size() != inputs || ifs.size() != inputs ||
      views.size() != 2 * inputs + 1 || bases.size() != inputs + 1 ||
      stores.size() != 1 || slices.size() != 1)
    return "flat_unsupported_memory_or_control_flow";
  auto tile = allocs.front().getType();
  if (tile.getRank() != 1 || !tile.hasStaticShape() ||
      tile.getDimSize(0) <= 0 || !tile.getLayout().isIdentity() ||
      tile.getMemorySpace() ||
      !(tile.getElementType().isF16() || tile.getElementType().isF32() ||
        tile.getElementType().isBF16()))
    return "flat_unsupported_selected_tile";
  int64_t elements = tile.getDimSize(0);
  auto firstCopy = copies.front();
  auto first = firstCopy.getSource().getDefiningOp<memref::SubViewOp>();
  auto base = first
                  ? first.getSource().getDefiningOp<memref::ReinterpretCastOp>()
                  : memref::ReinterpretCastOp();
  if (!base || base.getOffsets().size() != 1 || first.getSizes().size() != 1)
    return "flat_unproved_input_partition";
  AddressEquivalence equality;
  Value offset = base.getOffsets().front(), size = first.getSizes().front();
  auto cast = offset.getDefiningOp<arith::IndexCastOp>();
  if (!cast || !equality.product(cast.getIn(), layout.programId(), elements))
    return "flat_unproved_program_ownership";
  AccessRoles roles;
  auto roleError = deriveAccessRoles(func, layout, AccessSchedule::Flat, copies,
                                     stores.front(), {}, {}, {}, tile, roles);
  if (!roleError.empty())
    return roleError;
  auto matchesView = [&](memref::SubViewOp view) {
    if (!view || !zeroOffsetsUnitStrides(view) || view.getSizes().size() != 1 ||
        view.getSizes().front() != size)
      return false;
    auto b = view.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    auto arg = b ? dyn_cast<BlockArgument>(b.getSource()) : BlockArgument();
    if (!arg || !layout.isPublic(arg) || b.getOffsets().size() != 1 ||
        !equality.equal(b.getOffsets().front(), offset) ||
        b.getStaticSizes() != tile.getShape() ||
        b.getStaticStrides().size() != 1 || b.getStaticStrides()[0] != 1)
      return false;
    auto ty = dyn_cast<MemRefType>(arg.getType());
    return ty && ty.getRank() == 1 && ty.isDynamicDim(0) &&
           ty.getElementType() == tile.getElementType();
  };
  auto out = stores.front().getDest().getDefiningOp<memref::SubViewOp>();
  auto slice = slices.front();
  if (!matchesView(out) || stores.front().getSource() != slice ||
      slice.getSizes().size() != 1 || slice.getSizes().front() != size ||
      slice.getStaticOffsets() != ArrayRef<int64_t>{0} ||
      slice.getStaticStrides() != ArrayRef<int64_t>{1} ||
      slice->getParentOp() != func || stores.front()->getParentOp() != func)
    return "flat_unproved_store_partition";
  SmallVector<Value> pointers;
  pointers.push_back(
      out.getSource().getDefiningOp<memref::ReinterpretCastOp>().getSource());
  for (auto copy : copies) {
    auto src = copy.getSource().getDefiningOp<memref::SubViewOp>();
    auto dst = copy.getTarget().getDefiningOp<memref::SubViewOp>();
    auto alloc = dst ? dst.getSource().getDefiningOp<memref::AllocOp>()
                     : memref::AllocOp();
    if (!matchesView(src) || !dst || !zeroOffsetsUnitStrides(dst) ||
        dst.getSizes().size() != 1 || dst.getSizes().front() != size ||
        !alloc || alloc.getType() != tile || copy->getParentOp() != func)
      return "flat_unproved_selected_input";
    Value ptr =
        src.getSource().getDefiningOp<memref::ReinterpretCastOp>().getSource();
    if (llvm::is_contained(pointers, ptr))
      return "flat_unproved_disjoint_pointers";
    pointers.push_back(ptr);
    bufferization::ToTensorOp tensor;
    scf::IfOp padding;
    for (Operation *user : alloc->getUsers()) {
      if (auto t = dyn_cast<bufferization::ToTensorOp>(user)) {
        if (tensor)
          return "flat_unproved_selected_input";
        tensor = t;
      } else if (auto fill = dyn_cast<linalg::FillOp>(user)) {
        if (padding || fill.getInputs().size() != 1 ||
            fill.getOutputs().size() != 1)
          return "flat_unproved_padding";
        padding = fill->getParentOfType<scf::IfOp>();
        auto c = fill.getInputs().front().getDefiningOp<arith::ConstantOp>();
        auto f = c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr();
        if (!f || !f.getValue().isZero() || !padding ||
            padding->getParentOp() != func || padding.getNumResults() ||
            !padding.getElseRegion().empty() ||
            std::distance(padding.getThenRegion().front().begin(),
                          padding.getThenRegion().front().end()) != 2)
          return "flat_unproved_padding";
      } else if (user != dst.getOperation())
        return "flat_unproved_selected_input";
    }
    if (!initializationPrecedesRead(padding, copy, tensor))
      return "flat_unproved_initialization_order";
    auto cmp = padding ? padding.getCondition().getDefiningOp<arith::CmpIOp>()
                       : arith::CmpIOp();
    if (!tensor || !cmp || cmp.getPredicate() != arith::CmpIPredicate::slt ||
        cmp.getLhs() != size || !isInt(cmp.getRhs(), elements) ||
        !hasUnobservablePadding(tensor, slice, stores.front(), func))
      return "flat_observable_or_unproved_padding";
  }
  if (!hasStagingEffects(func, inventory, ifs))
    return "unmodelled_memory_dependency";
  OpBuilder b(func.getContext());
  if (failed(publishAcceptedFacts(func, {AccessSchedule::Flat, roles, {}, {}, {}})))
    return "inconsistent_accepted_facts";
  func->setAttr("l1_prefetch_ps.flat", b.getUnitAttr());
  for (auto entry : llvm::enumerate(copies)) {
    auto copy = entry.value();
    copy->setAttr("l1_prefetch_ps.load", b.getUnitAttr());
    auto alloc =
        copy.getTarget().getDefiningOp<memref::SubViewOp>().getSource();
    b.setInsertionPointAfter(alloc.getDefiningOp());
    auto mark = b.create<annotation::MarkOp>(func.getLoc(), alloc);
    auto source = copy.getSource().getDefiningOp<memref::SubViewOp>();
    auto pointer = source.getSource().getDefiningOp<memref::ReinterpretCastOp>()
                       .getSource();
    unsigned id = llvm::find(roles.inputs, pointer) - roles.inputs.begin();
    mark->setAttr("program_split.selected_input", b.getI32IntegerAttr(id));
  }
  for (auto padding : ifs)
    padding->setAttr("l1_prefetch_ps.padding", b.getUnitAttr());
  return "";
}

// One ownership and live-rectangle proof for every selected input.
std::string analyzeNestedInputs(func::FuncOp func, KernelLayout &layout,
                                EffectInventory &inventory) {
  auto &loops = inventory.loops;
  auto &ifs = inventory.ifs;
  auto &copies = inventory.copies;
  auto &allocs = inventory.allocs;
  auto &views = inventory.views;
  auto &bases = inventory.bases;
  auto &tensors = inventory.tensors;
  auto &stores = inventory.stores;
  auto &slices = inventory.slices;
  unsigned inputs = copies.size();
  if (inventory.hasUnsupportedOperation() || !inputs ||
      loops.size() != 2 || ifs.size() != inputs || allocs.size() != inputs ||
      views.size() != 2 * inputs + 1 || bases.size() != inputs + 1 ||
      tensors.size() != inputs || stores.size() != 1 || slices.size() != 1)
    return "nested_unsupported_memory_or_control_flow";
  if (func.getFunctionType().getNumResults())
    return "nested_unsupported_kernel_arguments";
  auto tile = allocs.front().getType();
  if (tile.getRank() != 2 || !tile.hasStaticShape() ||
      llvm::any_of(allocs, [&](auto alloc) {
        return alloc.getType() != tile;
      }))
    return "nested_unsupported_selected_tile";
  int64_t bm = tile.getDimSize(0), bn = tile.getDimSize(1);
  auto outer = loops[0], inner = loops[1];
  if (inner->getParentOp() == func)
    std::swap(outer, inner);
  if (outer->getParentOp() != func || inner->getParentOp() != outer ||
      outer.getNumResults() || inner.getNumResults() ||
      outer.getLowerBound() != layout.programId() ||
      outer.getStep() != layout.gridDim() || !isInt(inner.getLowerBound(), 0) ||
      !isInt(inner.getStep(), bn))
    return "nested_unsupported_loop";
  AccessRoles accessRoles;
  auto roleError = deriveAccessRoles(func, layout, AccessSchedule::Nested, copies,
                                     stores.front(), outer, inner, {}, tile,
                                     accessRoles);
  if (!roleError.empty())
    return roleError;
  auto orderedInputs = accessRoles.inputs;
  llvm::sort(orderedInputs, [](BlockArgument a, BlockArgument b) {
    return a.getArgNumber() < b.getArgNumber();
  });
  SmallVector<BlockArgument> pointers(orderedInputs);
  pointers.push_back(accessRoles.output);
  for (auto pointer : pointers) {
    auto type = dyn_cast<MemRefType>(pointer.getType());
    if (!type || type.getRank() != 1 || !type.isDynamicDim(0) ||
        type.getElementType() != tile.getElementType() ||
        !(type.getElementType().isF16() || type.getElementType().isBF16() ||
          type.getElementType().isF32()))
      return "nested_unsupported_pointer_type";
  }
  if (!validTile(tile, inputs))
    return "nested_unsupported_selected_tile";
  Value rows = accessRoles.rows;
  Value columnBound = accessRoles.cols;
  auto staticRows = integer(rows);
  auto staticCols = integer(columnBound);
  bool staticColumns = staticCols.has_value();
  // The shared derivation proves the ceiling-division relation, including a
  // folded constant upper bound. Launch premises guard dynamic i32 overflow.
  if ((staticRows && (*staticRows <= 0 ||
                      *staticRows > std::numeric_limits<int32_t>::max() - bm)) ||
      (staticCols && (*staticCols <= 0 ||
                      *staticCols > std::numeric_limits<int32_t>::max() - bn)))
    return "nested_unproved_shape_bounds";

  auto out = stores.front().getDest().getDefiningOp<memref::SubViewOp>();
  auto slice = slices.front();
  if (!out || stores.front().getSource() != slice.getResult() ||
      !zeroOffsetsUnitStrides(out) ||
      llvm::any_of(slice.getStaticOffsets(), [](int64_t n) { return n != 0; }) ||
      llvm::any_of(slice.getStaticStrides(), [](int64_t n) { return n != 1; }) ||
      !slice.getOffsets().empty() || !slice.getStrides().empty())
    return "nested_unproved_mask";
  SmallVector<memref::AllocOp> ordered(inputs);
  for (auto copy : copies) {
    auto src = copy.getSource().getDefiningOp<memref::SubViewOp>();
    auto dst = copy.getTarget().getDefiningOp<memref::SubViewOp>();
    auto input = src ? src.getSource().getDefiningOp<memref::ReinterpretCastOp>()
                     : memref::ReinterpretCastOp();
    auto alloc = dst ? dst.getSource().getDefiningOp<memref::AllocOp>()
                     : memref::AllocOp();
    if (!src || !dst || !input || !alloc || alloc.getType() != tile ||
        src == dst || src == out || dst == out ||
        !zeroOffsetsUnitStrides(src) || !zeroOffsetsUnitStrides(dst) ||
        src.getSizes().size() != 2 ||
        !llvm::equal(src.getSizes(), dst.getSizes()) ||
        !llvm::equal(src.getSizes(), out.getSizes()) ||
        !llvm::equal(src.getSizes(), slice.getSizes()))
      return "nested_unproved_mask";
    auto position = llvm::find(orderedInputs, input.getSource());
    if (position == orderedInputs.end())
      return "nested_unproved_selected_input";
    unsigned id = std::distance(orderedInputs.begin(), position);
    if (ordered[id])
      return "nested_unproved_selected_input";
    ordered[id] = alloc;
    auto inputView = src.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    auto outputView = out.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    if (!inputView || !outputView || inputView.getOffsets().size() != 1)
      return "nested_unproved_store_partition";
    Value offset = inputView.getOffsets().front();
    auto add = offset.getDefiningOp<arith::AddIOp>();
    auto rowOffset =
        add ? add.getLhs().getDefiningOp<arith::MulIOp>() : arith::MulIOp();
    auto rowStart = rowOffset
                        ? rowOffset.getLhs().getDefiningOp<arith::IndexCastOp>()
                        : arith::IndexCastOp();
    Value stride = rowOffset ? rowOffset.getRhs() : Value();
    auto staticStride = stride ? integer(stride) : std::nullopt;
    auto strideCast = stride ? stride.getDefiningOp<arith::IndexCastOp>()
                             : arith::IndexCastOp();
    auto strideArg = strideCast ? dyn_cast<BlockArgument>(strideCast.getIn())
                                : BlockArgument();
    bool validStride =
        staticStride ? (*staticStride > 0 &&
                        *staticStride <= std::numeric_limits<int32_t>::max() &&
                        (!staticColumns || *staticCols <= *staticStride))
                     : (strideArg && layout.isPublic(strideArg));
    if (!add || !rowOffset || !rowStart || !validStride ||
        !mulOf(rowStart.getIn(), outer.getInductionVar(), bm) ||
        !castOf(add.getRhs(), inner.getInductionVar()) ||
        !tileView(inputView, orderedInputs[id], tile, stride, offset) ||
        !tileView(outputView, accessRoles.output, tile, stride, offset))
      return "nested_unproved_store_partition";
    if (!boundedTileExtent(src.getSizes()[0], rowStart.getResult(), rows, bm) ||
        !boundedTileExtent(src.getSizes()[1], add.getRhs(), columnBound, bn))
      return "nested_unproved_mask";

    scf::IfOp padding;
    bufferization::ToTensorOp tensor;
    for (Operation *user : alloc.getResult().getUsers()) {
      if (auto t = dyn_cast<bufferization::ToTensorOp>(user)) {
        if (tensor)
          return "nested_unproved_selected_input";
        tensor = t;
      } else if (auto fill = dyn_cast<linalg::FillOp>(user)) {
        if (!isa<scf::IfOp>(fill->getParentOp()))
          return "nested_unproved_selected_input";
        if (padding)
          return "nested_unproved_padding";
        padding = fill->getParentOfType<scf::IfOp>();
        auto zero = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
        auto value = zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr();
        if (!padding || fill.getOutputs().size() != 1 || !value ||
            !value.getValue().isZero() ||
            std::distance(padding.getThenRegion().front().begin(),
                          padding.getThenRegion().front().end()) != 2)
          return "nested_unproved_padding";
      } else if (user != dst.getOperation())
        return "nested_unproved_selected_input";
    }
    auto mask = padding ? padding.getCondition().getDefiningOp<arith::OrIOp>()
                        : arith::OrIOp();
    auto rc = mask ? mask.getLhs().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp();
    auto cc = mask ? mask.getRhs().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp();
    if (!tensor || !padding || !rc || !cc || padding.getNumResults() ||
        !padding.getElseRegion().empty() ||
        rc.getPredicate() != arith::CmpIPredicate::slt ||
        cc.getPredicate() != arith::CmpIPredicate::slt ||
        rc.getLhs() != src.getSizes()[0] || cc.getLhs() != src.getSizes()[1] ||
        !isInt(rc.getRhs(), bm) || !isInt(cc.getRhs(), bn))
      return "nested_unproved_padding";
    for (Operation *op : {copy.getOperation(), alloc.getOperation(),
                          tensor.getOperation(), padding.getOperation()})
      if (op->getParentOp() != inner)
        return "nested_unsupported_memory_or_control_flow";
    if (!initializationPrecedesRead(padding, copy, tensor))
      return "nested_unproved_initialization_order";
    // Full-tile mode proves padding unreachable from constant bounds. Partial
    // mode additionally needs elementwise independence for each selected input.
    if (nestedTileMode(accessRoles, tile) == NestedTileMode::Partial &&
        !hasUnobservablePadding(tensor, slice, stores.front(), inner))
      return "nested_unproved_padding_independence";
  }
  if (llvm::any_of(ordered, [](auto alloc) { return !alloc; }) ||
      stores.front()->getParentOp() != inner || slice->getParentOp() != inner)
    return "nested_unproved_selected_input";
  bool extraWrite = false;
  func.walk([&](linalg::FillOp fill) {
    for (Value output : fill.getOutputs())
      if (isa<MemRefType>(output.getType()) &&
          !llvm::is_contained(ifs, fill->getParentOp()))
        extraWrite = true;
  });
  if (extraWrite)
    return "nested_unproved_selected_input";
  if (!hasStagingEffects(func, inventory, ifs))
    return "unmodelled_memory_dependency";
  OpBuilder builder(func.getContext());
  if (failed(publishAcceptedFacts(func, {AccessSchedule::Nested, accessRoles, {}, {}, {}})))
    return "inconsistent_accepted_facts";
  outer->setAttr("l1_prefetch_ps.outer", builder.getUnitAttr());
  inner->setAttr("l1_prefetch_ps.inner", builder.getUnitAttr());
  for (auto copy : copies)
    copy->setAttr("l1_prefetch_ps.load", builder.getUnitAttr());
  for (auto padding : ifs)
    padding->setAttr("l1_prefetch_ps.padding", builder.getUnitAttr());
  for (auto view : views)
    view->setAttr("l1_prefetch_ps.full_tile", builder.getUnitAttr());
  slice->setAttr("l1_prefetch_ps.full_tile", builder.getUnitAttr());
  for (unsigned id = 0; id < inputs; ++id) {
    builder.setInsertionPointAfter(ordered[id]);
    auto mark = builder.create<annotation::MarkOp>(ordered[id].getLoc(),
                                                  ordered[id].getResult());
    mark->setAttr("program_split.selected_input", builder.getI32IntegerAttr(id));
  }
  return "";
}
} // namespace

std::string mlir::hivm::analyzeNestedTileCandidate(func::FuncOp func,
                                                 KernelLayout &layout,
                                                 EffectInventory &inventory) {
  if (inventory.loops.empty())
    if (llvm::any_of(inventory.allocs, [](auto alloc) {
          return alloc.getType().getRank() == 2;
        }))
      return analyzeProgramInputs(func, layout, inventory);
  if (inventory.loops.empty())
    return analyzeFlatInputs(func, layout, inventory.ifs, inventory.copies,
                             inventory.allocs, inventory.views, inventory.bases,
                             inventory.tensors, inventory.stores,
                             inventory.slices,
                             inventory.hasUnsupportedOperation(), inventory);
  return analyzeNestedInputs(func, layout, inventory);
}
