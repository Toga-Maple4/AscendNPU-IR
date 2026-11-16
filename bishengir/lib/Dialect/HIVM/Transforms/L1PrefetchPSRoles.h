// Access-derived roles shared by analysis, transformation and contract writing.
#ifndef BISHENGIR_L1PREFETCHPS_ROLES_H
#define BISHENGIR_L1PREFETCHPS_ROLES_H

#include "L1PrefetchPSCommon.h"

namespace mlir::hivm::l1prefetch {

enum class AccessSchedule { Persistent, Nested, Flat, Jagged, Program };

// Analysis derives these operands and proves their meaning. AcceptedFacts
// records only those proved roles; consumers decode them on the digest-verified
// pristine clone. This transient view never survives normalization or cloning.
struct AccessRoles {
  SmallVector<BlockArgument> inputs;
  BlockArgument output, offsets;
  Value rows, cols, size, inputStride, matrixStride;
  OpFoldResult stride;
  SmallVector<OpFoldResult> inputStrides;
};

// Partial tiles are the general case, independent of input count. Elide live
// extents only when the access bounds themselves prove every tile is full;
// observed launch shapes must not become a new runtime divisibility premise.
enum class NestedTileMode { Partial, Full };
inline NestedTileMode nestedTileMode(const AccessRoles &roles, MemRefType tile) {
  auto rows = integer(roles.rows), cols = integer(roles.cols);
  return rows && cols && *rows > 0 && *cols > 0 &&
                 *rows % tile.getDimSize(0) == 0 &&
                 *cols % tile.getDimSize(1) == 0
             ? NestedTileMode::Full
             : NestedTileMode::Partial;
}

inline Value roleValue(Value value) {
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return cast.getIn();
  return value;
}

inline memref::ReinterpretCastOp accessView(Value value) {
  if (auto slice = value.getDefiningOp<memref::SubViewOp>())
    value = slice.getSource();
  return value.getDefiningOp<memref::ReinterpretCastOp>();
}

inline bool scalarRole(Value value, KernelLayout &layout) {
  return value && ((layout.isPublic(value) && value.getType().isInteger(32)) ||
                   integer(value).has_value());
}

inline std::string
deriveAccessRoles(func::FuncOp func, KernelLayout &layout,
                  AccessSchedule schedule, ArrayRef<memref::CopyOp> loads,
                  bufferization::MaterializeInDestinationOp store,
                  scf::ForOp outer, scf::ForOp inner,
                  ArrayRef<memref::LoadOp> metadataLoads, MemRefType tile,
                  AccessRoles &roles) {
  if (loads.empty() || !store || !tile)
    return "unmodelled_access_dependency";
  SmallVector<memref::ReinterpretCastOp> inputViews;
  auto pointer = [&](memref::ReinterpretCastOp view) -> BlockArgument {
    if (!view || !layout.isPublic(view.getSource()))
      return {};
    auto arg = cast<BlockArgument>(view.getSource());
    auto type = dyn_cast<MemRefType>(arg.getType());
    if (!type || type.getRank() != 1 || !type.isDynamicDim(0))
      return {};
    return arg;
  };
  for (auto load : loads) {
    auto view = accessView(load.getSource());
    auto arg = pointer(view);
    if (!arg)
      return "unmodelled_access_dependency";
    if (llvm::is_contained(roles.inputs, arg))
      return schedule == AccessSchedule::Nested
                 ? "nested_unproved_selected_input"
             : schedule == AccessSchedule::Flat
                 ? "flat_unproved_disjoint_pointers"
                 : "unproved_pointer_overlap";
    inputViews.push_back(view);
    roles.inputs.push_back(arg);
  }
  auto outputView = accessView(store.getDest());
  roles.output = pointer(outputView);
  if (!roles.output)
    return "unmodelled_access_dependency";
  if (llvm::is_contained(roles.inputs, roles.output))
    return schedule == AccessSchedule::Nested
               ? "nested_unproved_store_partition"
           : schedule == AccessSchedule::Jagged ? "jagged_unproved_address"
           : schedule == AccessSchedule::Flat
               ? "flat_unproved_disjoint_pointers"
               : "unproved_pointer_overlap";
  // Stable selection identity is independent of source load order.
  llvm::sort(roles.inputs, [](BlockArgument a, BlockArgument b) {
    return a.getArgNumber() < b.getArgNumber();
  });
  if (schedule == AccessSchedule::Program) {
    for (BlockArgument input : roles.inputs) {
      auto view = *llvm::find_if(inputViews, [&](auto v) {
        return v.getSource() == input;
      });
      auto strides = view.getMixedStrides();
      if (strides.size() != 2)
        return "unmodelled_access_dependency";
      OpFoldResult stride = strides.front();
      if (auto value = dyn_cast<Value>(stride)) {
        value = roleValue(value);
        if (!scalarRole(value, layout))
          return "unmodelled_access_dependency";
        stride = value;
      }
      roles.inputStrides.push_back(stride);
    }
    return "";
  }
  if (schedule == AccessSchedule::Flat) {
    auto firstLoad = loads.front();
    auto slice = firstLoad.getSource().getDefiningOp<memref::SubViewOp>();
    auto view = inputViews.front();
    if (!slice || slice.getSizes().size() != 1 || view.getOffsets().size() != 1)
      return "flat_unproved_mask";
    auto extent = extentArgument(func, layout, slice.getSizes().front(),
                                 view.getOffsets().front(), tile.getDimSize(0));
    if (extent.kind == ExtentMatchKind::Ambiguous)
      return "flat_ambiguous_extent";
    if (extent.kind != ExtentMatchKind::Unique)
      return "flat_unproved_mask";
    roles.size = extent.argument;
    return "";
  }
  auto strides = inputViews.front().getMixedStrides();
  if (strides.size() != 2)
    return "unmodelled_access_dependency";
  roles.stride = strides.front();
  if (auto value = dyn_cast<Value>(roles.stride)) {
    Value scalar = roleValue(value);
    if (!scalarRole(scalar, layout))
      return schedule == AccessSchedule::Nested
                 ? "nested_unproved_store_partition"
                 : "unmodelled_access_dependency";
    roles.stride = scalar;
  }
  if (schedule == AccessSchedule::Persistent)
    return "";
  if (!outer || !inner)
    return "unmodelled_access_dependency";
  if (schedule == AccessSchedule::Nested) {
    roles.rows = ceilDivBound(outer.getUpperBound(), tile.getDimSize(0));
    if (!roles.rows) {
      auto load = loads.front();
      auto source = load.getSource().getDefiningOp<memref::SubViewOp>();
      if (!source || source.getSizes().size() != 2)
        return "nested_unproved_shape_bounds";
      Value bound = tileExtentBound(source.getSizes()[0]);
      auto rows = bound ? integer(bound) : std::nullopt;
      auto tasks = integer(outer.getUpperBound());
      int64_t bm = tile.getDimSize(0);
      if (!rows || !tasks || *rows <= 0 ||
          *rows > std::numeric_limits<int32_t>::max() - bm ||
          *tasks != (*rows + bm - 1) / bm)
        return "nested_unproved_shape_bounds";
      roles.rows = bound;
    }
    roles.cols = inner.getUpperBound();
    if (!scalarRole(roles.rows, layout) ||
        !scalarRole(roles.cols, layout))
      return "nested_unproved_shape_bounds";
    return "";
  }
  if (metadataLoads.empty())
    return "unmodelled_access_dependency";
  auto firstMetadata = metadataLoads.front();
  roles.offsets = pointer(accessView(firstMetadata.getMemRef()));
  if (!roles.offsets)
    return "unmodelled_access_dependency";
  if (roles.offsets == roles.output ||
      llvm::is_contained(roles.inputs, roles.offsets))
    return "unproved_pointer_overlap";
  auto columns = inner.getUpperBound().getDefiningOp<arith::MinSIOp>();
  if (!columns)
    return "jagged_unproved_bounds";
  roles.cols = columns.getLhs();
  roles.inputStride = columns.getRhs();
  auto view = outputView;
  auto add = view.getOffsets().size() == 1
                 ? view.getOffsets().front().getDefiningOp<arith::AddIOp>()
                 : arith::AddIOp();
  auto mul = add ? roleValue(add.getLhs()).getDefiningOp<arith::MulIOp>()
                 : arith::MulIOp();
  if (!mul || mul.getLhs() != layout.programId())
    return "unmodelled_access_dependency";
  roles.matrixStride = mul.getRhs();
  if (!scalarRole(roles.cols, layout) ||
      !scalarRole(roles.inputStride, layout) ||
      !scalarRole(roles.matrixStride, layout))
    return "unmodelled_access_dependency";
  return "";
}

} // namespace mlir::hivm::l1prefetch
#endif
