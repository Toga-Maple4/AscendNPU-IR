//===- L1PrefetchPSJaggedAnalysis.cpp - Jagged copy proof ----------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "L1PrefetchPSFacts.h"
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
#include <optional>

using namespace mlir;
using namespace mlir::hivm::l1prefetch;
namespace {
template <class Op> bool binary(Value v, Value a, Value b) {
  auto op = v.getDefiningOp<Op>();
  return op && op.getLhs() == a && op.getRhs() == b;
}
bool increment(Value v, Value base, int64_t n) {
  auto op = v.getDefiningOp<arith::AddIOp>();
  return op && op.getLhs() == base && isInt(op.getRhs(), n);
}
bool rectangle(memref::SubViewOp v) {
  return v && v.getOffsets().empty() && v.getStrides().empty() &&
         v.getStaticOffsets() == ArrayRef<int64_t>({0, 0}) &&
         v.getStaticStrides() == ArrayRef<int64_t>({1, 1});
}
bool metadata(memref::LoadOp load, Value offsets, Value index) {
  auto view = load.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
  if (!view || view.getSource() != offsets || load.getIndices().size() != 1 ||
      !isInt(load.getIndices()[0], 0) || view.getOffsets().size() != 1 ||
      !castOf(view.getOffsets()[0], index))
    return false;
  auto sizes = view.getMixedSizes(), strides = view.getMixedStrides();
  auto one = [](OpFoldResult v) {
    if (auto attr = dyn_cast<Attribute>(v))
      return cast<IntegerAttr>(attr).getInt() == 1;
    return isInt(cast<Value>(v), 1);
  };
  return sizes.size() == 1 && strides.size() == 1 && one(sizes[0]) &&
         one(strides[0]);
}
bool viewOf(memref::ReinterpretCastOp view, Value base, MemRefType tile,
            Value baseOffset, Value blockOffset, Value stride) {
  if (!view || view.getSource() != base || view.getOffsets().size() != 1 ||
      view.getSizes().size() || view.getStaticSizes() != tile.getShape())
    return false;
  auto add = view.getOffsets()[0].getDefiningOp<arith::AddIOp>();
  if (!add || !castOf(add.getLhs(), baseOffset) || add.getRhs() != blockOffset)
    return false;
  auto strides = view.getMixedStrides();
  if (strides.size() != 2 || !isa<Value>(strides[0]) ||
      cast<Value>(strides[0]) != stride)
    return false;
  if (auto attr = dyn_cast<Attribute>(strides[1]))
    return cast<IntegerAttr>(attr).getInt() == 1;
  return isInt(cast<Value>(strides[1]), 1);
}
} // namespace

// Prove the original one-sequence-per-program copy. Metadata is read-only under
// the all-pointer noalias caller contract. Monotone, in-bounds offsets remain
// the original kernel's input precondition; no host read of device metadata is
// added. Every side effect and loop recurrence is checked before attaching any
// marker.
std::string mlir::hivm::analyzeJaggedTileCandidate(func::FuncOp func,
                                                   KernelLayout &layout,
                                                   EffectInventory &inventory) {
  auto &loops = inventory.loops;
  auto &ifs = inventory.ifs;
  auto &metadataLoads = inventory.metadataLoads;
  auto &copies = inventory.copies;
  auto &allocs = inventory.allocs;
  auto &tensors = inventory.tensors;
  auto &stores = inventory.stores;
  auto &slices = inventory.slices;
  auto &fills = inventory.fills;
  bool unsupported = inventory.hasUnsupportedOperation();
  if (unsupported || loops.size() != 2 || ifs.size() != 1 ||
      metadataLoads.size() != 2 || copies.size() != 1 || allocs.size() != 1 ||
      tensors.size() != 1 || stores.size() != 1 || slices.size() != 1 ||
      fills.size() != 1)
    return "jagged_unsupported_memory_or_control_flow";
  auto outer = loops[0], inner = loops[1];
  if (inner->getParentOp() == func)
    std::swap(outer, inner);
  auto tile = cast<MemRefType>(allocs[0].getType());
  if (!validTile(tile, 1))
    return "jagged_unsupported_selected_tile";
  AccessRoles roles;
  auto roleError =
      deriveAccessRoles(func, layout, AccessSchedule::Jagged, copies, stores[0],
                        outer, inner, metadataLoads, tile, roles);
  if (!roleError.empty())
    return roleError;
  if (cast<MemRefType>(roles.inputs.front().getType()).getElementType() !=
          tile.getElementType() ||
      cast<MemRefType>(roles.output.getType()).getElementType() !=
          tile.getElementType())
    return "jagged_unsupported_selected_tile";
  if (!cast<MemRefType>(roles.offsets.getType()).getElementType().isInteger(64))
    return "jagged_unsupported_pointer_type";
  int64_t bm = tile.getDimSize(0), bn = tile.getDimSize(1);
  Value begin = metadataLoads[0], end = metadataLoads[1],
        pid = layout.programId();
  auto next =
      metadataLoads[1].getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
  auto nextCast = next && next.getOffsets().size() == 1
                      ? next.getOffsets()[0].getDefiningOp<arith::IndexCastOp>()
                      : arith::IndexCastOp();
  if (!metadata(metadataLoads[0], roles.offsets, pid) || !nextCast ||
      !increment(nextCast.getIn(), pid, 1) ||
      !metadata(metadataLoads[1], roles.offsets, nextCast.getIn()) ||
      metadataLoads[0]->getParentOp() != func ||
      metadataLoads[1]->getParentOp() != func)
    return "jagged_unproved_offsets";
  if (outer->getParentOp() != func || inner->getParentOp() != outer ||
      outer.getNumRegionIterArgs() != 1 || inner.getNumRegionIterArgs() != 2 ||
      !outer->use_empty() || !inner->use_empty() ||
      outer.getLowerBound() != begin || outer.getUpperBound() != end ||
      !isInt(outer.getStep(), bm) || !isInt(outer.getInitArgs()[0], 0) ||
      !isInt(inner.getLowerBound(), 0) || !isInt(inner.getStep(), bn) ||
      !isInt(inner.getInitArgs()[0], 0))
    return "jagged_unproved_loop";
  Value row = outer.getRegionIterArgs()[0], col = inner.getRegionIterArgs()[0];
  Value blockOffset = inner.getRegionIterArgs()[1];
  auto rowStride = inner.getInitArgs()[1].getDefiningOp<arith::MulIOp>();
  auto outerYield = cast<scf::YieldOp>(outer.getBody()->getTerminator());
  auto innerYield = cast<scf::YieldOp>(inner.getBody()->getTerminator());
  if (!rowStride || rowStride.getLhs() != row ||
      !castOf(rowStride.getRhs(), roles.cols) ||
      !increment(outerYield.getOperand(0), row, bm) ||
      !increment(innerYield.getOperand(0), col, bn) ||
      !increment(innerYield.getOperand(1), blockOffset, bn))
    return "jagged_unproved_loop";
  Value columns = inner.getUpperBound();
  if (!binary<arith::MinSIOp>(columns, roles.cols, roles.inputStride))
    return "jagged_unproved_bounds";
  auto copy = copies[0];
  auto src = copy.getSource().getDefiningOp<memref::SubViewOp>();
  auto dst = copy.getTarget().getDefiningOp<memref::SubViewOp>();
  auto out = stores[0].getDest().getDefiningOp<memref::SubViewOp>();
  if (!rectangle(src) || !rectangle(dst) || !rectangle(out) || src == dst ||
      src == out || dst == out || dst.getSource() != allocs[0] ||
      tensors[0].getMemref() != allocs[0] ||
      slices[0].getSource() != tensors[0] ||
      stores[0].getSource() != slices[0] || !tensors[0]->hasOneUse() ||
      !slices[0]->hasOneUse() || src.getSizes().size() != 2 ||
      !llvm::equal(src.getSizes(), dst.getSizes()) ||
      !llvm::equal(src.getSizes(), out.getSizes()) ||
      !llvm::equal(src.getSizes(), slices[0].getSizes()) ||
      !slices[0].getOffsets().empty() || !slices[0].getStrides().empty() ||
      slices[0].getStaticOffsets() != ArrayRef<int64_t>({0, 0}) ||
      slices[0].getStaticStrides() != ArrayRef<int64_t>({1, 1}))
    return "jagged_unproved_copy_rectangle";
  auto rowMax = src.getSizes()[0].getDefiningOp<arith::MaxSIOp>();
  // Find the row bound through the exact clamp; extent() verifies every edge.
  auto rowMin = rowMax ? rowMax.getLhs().getDefiningOp<arith::MinSIOp>()
                       : arith::MinSIOp();
  auto rowSub =
      rowMin ? rowMin.getLhs().getDefiningOp<arith::SubIOp>() : arith::SubIOp();
  auto rowEnd = rowSub ? rowSub.getLhs().getDefiningOp<arith::MinSIOp>()
                       : arith::MinSIOp();
  auto rowBound = rowEnd ? rowEnd.getRhs().getDefiningOp<arith::MaxSIOp>()
                         : arith::MaxSIOp();
  auto boundCast = rowBound
                       ? rowBound.getRhs().getDefiningOp<arith::IndexCastOp>()
                       : arith::IndexCastOp();
  auto bound = boundCast ? boundCast.getIn().getDefiningOp<arith::MinSIOp>()
                         : arith::MinSIOp();
  auto denseRows =
      bound ? bound.getLhs().getDefiningOp<arith::ExtSIOp>() : arith::ExtSIOp();
  if (!bound || !denseRows ||
      !binary<arith::DivSIOp>(denseRows.getIn(), roles.matrixStride,
                              roles.cols) ||
      !binary<arith::SubIOp>(bound.getRhs(), end, begin) ||
      !boundedTileExtent(src.getSizes()[0], row, bound, bm) ||
      !boundedTileExtent(src.getSizes()[1], col, columns, bn))
    return "jagged_unproved_bounds";
  auto inView = src.getSource().getDefiningOp<memref::ReinterpretCastOp>();
  auto outView = out.getSource().getDefiningOp<memref::ReinterpretCastOp>();
  auto inAdd = inView && inView.getOffsets().size() == 1
                   ? inView.getOffsets()[0].getDefiningOp<arith::AddIOp>()
                   : arith::AddIOp();
  auto inCast = inAdd ? inAdd.getLhs().getDefiningOp<arith::IndexCastOp>()
                      : arith::IndexCastOp();
  auto inBase =
      inCast ? inCast.getIn().getDefiningOp<arith::MulIOp>() : arith::MulIOp();
  auto stride64 = inBase ? inBase.getRhs().getDefiningOp<arith::ExtSIOp>()
                         : arith::ExtSIOp();
  auto outAdd = outView && outView.getOffsets().size() == 1
                    ? outView.getOffsets()[0].getDefiningOp<arith::AddIOp>()
                    : arith::AddIOp();
  auto outCast = outAdd ? outAdd.getLhs().getDefiningOp<arith::IndexCastOp>()
                        : arith::IndexCastOp();
  if (!inBase || inBase.getLhs() != begin || !stride64 ||
      stride64.getIn() != roles.inputStride || !outCast ||
      !binary<arith::MulIOp>(outCast.getIn(), pid, roles.matrixStride) ||
      !viewOf(inView, roles.inputs.front(), tile, inBase, blockOffset,
              rowStride.getRhs()) ||
      !viewOf(outView, roles.output, tile, outCast.getIn(), blockOffset,
              rowStride.getRhs()))
    return "jagged_unproved_address";
  auto fill = fills[0];
  auto zero = fill.getInputs().size() == 1
                  ? fill.getInputs()[0].getDefiningOp<arith::ConstantOp>()
                  : arith::ConstantOp();
  auto f = zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr();
  if (fill->getParentOp() != ifs[0] || !ifs[0].getElseRegion().empty() ||
      ifs[0].getNumResults() || fill.getOutputs().size() != 1 ||
      fill.getOutputs()[0] != allocs[0] || !f || !f.getValue().isZero())
    return "jagged_unproved_padding";
  for (Operation *op : {copy.getOperation(), allocs[0].getOperation(),
                        tensors[0].getOperation(), slices[0].getOperation(),
                        stores[0].getOperation(), ifs[0].getOperation()})
    if (op->getParentOp() != inner)
      return "jagged_unsupported_memory_or_control_flow";
  if (!allocs[0]->isBeforeInBlock(ifs[0]) || !ifs[0]->isBeforeInBlock(copy) ||
      !copy->isBeforeInBlock(tensors[0]))
    return "jagged_unproved_copy_order";
  for (Operation *user : allocs[0].getResult().getUsers())
    if (user != dst && user != tensors[0] && user != fill)
      return "jagged_unproved_selected_input";
  // Reject escaping views: the selected copy/store are the only payload
  // effects.
  if (!src->hasOneUse() || !dst->hasOneUse() || !out->hasOneUse())
    return "jagged_unproved_copy_rectangle";
  if (!hasStagingEffects(func, inventory, ifs, metadataLoads))
    return "unmodelled_memory_dependency";
  OpBuilder b(func.getContext());
  if (failed(publishAcceptedFacts(func, {AccessSchedule::Jagged, roles, {}, {}, {}})))
    return "inconsistent_accepted_facts";
  func->setAttr("l1_prefetch_ps.jagged", b.getUnitAttr());
  bound->setAttr("l1_prefetch_ps.rows", b.getUnitAttr());
  outer->setAttr("l1_prefetch_ps.outer", b.getUnitAttr());
  inner->setAttr("l1_prefetch_ps.inner", b.getUnitAttr());
  copy->setAttr("l1_prefetch_ps.load", b.getUnitAttr());
  ifs[0]->setAttr("l1_prefetch_ps.padding", b.getUnitAttr());
  for (auto load : metadataLoads)
    load->setAttr("l1_prefetch_ps.metadata", b.getUnitAttr());
  b.setInsertionPointAfter(allocs[0]);
  auto mark =
      b.create<annotation::MarkOp>(allocs[0].getLoc(), allocs[0].getResult());
  mark->setAttr("program_split.selected_input", b.getI32IntegerAttr(0));
  return "";
}
