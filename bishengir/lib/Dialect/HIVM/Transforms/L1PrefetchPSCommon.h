// Shared ABI, operation inventory and proofs for program-split candidates.
#ifndef BISHENGIR_L1PREFETCHPS_COMMON_H
#define BISHENGIR_L1PREFETCHPS_COMMON_H

#include "bishengir/Dialect/HIVM/Transforms/L1PrefetchPS.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>
#include <optional>

namespace mlir::hivm::l1prefetch {
// TTAdapter owns this ABI: two hidden byte pointers, public arguments, then
// grid XYZ and logical program-ID XYZ. All schedules use this one layout.
class KernelLayout {
public:
  static std::optional<KernelLayout> get(func::FuncOp func) {
    if (func.getNumArguments() < kHiddenArguments + kLaunchArguments ||
        func.getFunctionType().getNumResults())
      return std::nullopt;
    for (unsigned i = 0; i < kHiddenArguments; ++i) {
      auto type = dyn_cast<MemRefType>(func.getArgument(i).getType());
      if (!type || type.getRank() != 1 || !type.isDynamicDim(0) ||
          !type.getElementType().isInteger(8))
        return std::nullopt;
    }
    KernelLayout layout(func);
    for (unsigned i = layout.publicEnd(); i < func.getNumArguments(); ++i)
      if (!func.getArgument(i).getType().isInteger(32))
        return std::nullopt;
    return layout;
  }

  BlockArgument gridDim(unsigned axis = 0) {
    assert(axis < 3);
    return func.getArgument(publicEnd() + axis);
  }
  BlockArgument programId(unsigned axis = 0) {
    assert(axis < 3);
    return func.getArgument(publicEnd() + 3 + axis);
  }
  unsigned publicBegin() { return kHiddenArguments; }
  unsigned publicEnd() { return func.getNumArguments() - kLaunchArguments; }
  bool isPublic(Value value) {
    auto arg = dyn_cast<BlockArgument>(value);
    return arg && arg.getOwner() == &func.front() &&
           arg.getArgNumber() >= publicBegin() &&
           arg.getArgNumber() < publicEnd();
  }

private:
  explicit KernelLayout(func::FuncOp func) : func(func) {}
  static constexpr unsigned kHiddenArguments = 2;
  static constexpr unsigned kLaunchArguments = 6;
  func::FuncOp func;
};

// Inventory is structural evidence, not acceptance. Collect once at the common
// front; ownership, effects and padding still require their corresponding
// proof.
// Maple4 LinalgToHFusion maps __hmf_tanhf to the same HFusion tanh as
// math.tanh. Accept only its scalar f32 declaration and exact pointwise map;
// readnone alone does not establish the semantics of an arbitrary call.
inline bool isMappedTanh(linalg::MapOp map) {
  if (!map || map.getInputs().size() != 1 || map->getNumResults() != 1 ||
      !map.getMapper().hasOneBlock())
    return false;
  auto tensor = dyn_cast<RankedTensorType>(map.getInputs()[0].getType());
  if (!tensor || !tensor.getElementType().isF32() ||
      map.getInit().getType() != tensor ||
      map->getResult(0).getType() != tensor)
    return false;
  auto &body = map.getMapper().front();
  if (body.getNumArguments() != 1 || body.getOperations().size() != 2)
    return false;
  auto call = dyn_cast<func::CallOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!call || call.getCallee() != "__hmf_tanhf" ||
      call.getNumOperands() != 1 || call.getNumResults() != 1 ||
      call.getOperand(0) != body.getArgument(0) ||
      !call.getResult(0).getType().isF32() || !yield ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != call.getResult(0))
    return false;
  auto decl = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  return decl && decl.isExternal() && decl.isPrivate() &&
         decl->hasAttrOfType<UnitAttr>("llvm.readnone") &&
         decl.getFunctionType().getNumInputs() == 1 &&
         decl.getFunctionType().getInput(0).isF32() &&
         decl.getFunctionType().getNumResults() == 1 &&
         decl.getFunctionType().getResult(0).isF32();
}


// The semantic exemption belongs to this exact mapped operation, not to an
// arbitrary call with a readnone declaration.
inline bool isMappedTanhCall(Operation *op) {
  auto call = dyn_cast<func::CallOp>(op);
  if (!call)
    return false;
  auto map = dyn_cast<linalg::MapOp>(call->getParentOp());
  return isMappedTanh(map) && &map.getMapper().front().front() == op;
}

struct EffectInventory {
  explicit EffectInventory(func::FuncOp func) {
    func.walk([&](Operation *op) {
      if (op == func.getOperation())
        return;
      unsupportedOperation |= !supportedOperation(op);
      if (auto v = dyn_cast<scf::ForOp>(op))
        loops.push_back(v);
      if (auto v = dyn_cast<scf::IfOp>(op))
        ifs.push_back(v);
      if (auto v = dyn_cast<memref::CopyOp>(op))
        copies.push_back(v);
      if (auto v = dyn_cast<memref::AllocOp>(op))
        allocs.push_back(v);
      if (auto v = dyn_cast<memref::LoadOp>(op))
        metadataLoads.push_back(v);
      if (auto v = dyn_cast<memref::SubViewOp>(op))
        views.push_back(v);
      if (auto v = dyn_cast<memref::ReinterpretCastOp>(op))
        bases.push_back(v);
      if (auto v = dyn_cast<bufferization::ToTensorOp>(op))
        tensors.push_back(v);
      if (auto v = dyn_cast<bufferization::MaterializeInDestinationOp>(op))
        stores.push_back(v);
      if (auto v = dyn_cast<tensor::ExtractSliceOp>(op))
        slices.push_back(v);
      if (auto v = dyn_cast<linalg::FillOp>(op))
        fills.push_back(v);
    });
  }
  SmallVector<scf::ForOp> loops;
  SmallVector<scf::IfOp> ifs;
  SmallVector<memref::CopyOp> copies;
  SmallVector<memref::AllocOp> allocs;
  SmallVector<memref::LoadOp> metadataLoads;
  SmallVector<memref::SubViewOp> views;
  SmallVector<memref::ReinterpretCastOp> bases;
  SmallVector<bufferization::ToTensorOp> tensors;
  SmallVector<bufferization::MaterializeInDestinationOp> stores;
  SmallVector<tensor::ExtractSliceOp> slices;
  SmallVector<linalg::FillOp> fills;
  bool hasUnsupportedOperation() const { return unsupportedOperation; }

private:
  // Structural operations are checked by the existing ownership proofs and the
  // accounted-effects walk. Arithmetic is classified by interfaces, not names.
  static bool supportedOperation(Operation *op) {
    if (isa<func::CallOp>(op))
      return isMappedTanhCall(op);
    if (isa<func::ReturnOp, scf::ForOp, scf::IfOp, scf::YieldOp,
            memref::AllocOp, memref::CopyOp, memref::LoadOp,
            bufferization::ToTensorOp,
            bufferization::MaterializeInDestinationOp, linalg::YieldOp>(op))
      return true;
    if (isa<linalg::LinalgOp>(op))
      return true; // Buffer effects still require explicit accounting.
    return op->getNumRegions() == 0 && op->getNumSuccessors() == 0 &&
           isMemoryEffectFree(op);
  }
  bool unsupportedOperation = false;
};

// Account operation identities proved by a recognizer; a type or a metadata tag
// alone never authorizes a memory effect. Containers are traversed recursively.
inline bool hasOnlyAccountedEffects(
    func::FuncOp func, const llvm::SmallPtrSetImpl<Operation *> &accounted) {
  bool valid = true;
  func.walk([&](Operation *op) {
    if (op == func || accounted.contains(op) || isMappedTanhCall(op) ||
        isa<func::ReturnOp, scf::YieldOp, linalg::YieldOp>(op))
      return;
    if (!isMemoryEffectFree(op) ||
        (op->getNumRegions() && !isa<linalg::LinalgOp>(op)))
      valid = false;
  });
  return valid;
}

inline bool hasStagingEffects(func::FuncOp func, EffectInventory &inventory,
                              ArrayRef<scf::IfOp> padding,
                              ArrayRef<memref::LoadOp> metadata = {}) {
  llvm::SmallPtrSet<Operation *, 32> accounted;
  // These containers have already passed this schedule's exact structural
  // proof. Accounting for a container does not skip walking its body.
  for (auto op : inventory.loops)
    accounted.insert(op);
  for (auto op : inventory.ifs)
    accounted.insert(op);
  for (auto op : inventory.copies)
    accounted.insert(op);
  for (auto op : inventory.allocs)
    accounted.insert(op);
  for (auto op : inventory.tensors)
    accounted.insert(op);
  for (auto op : inventory.stores)
    accounted.insert(op);
  for (auto op : metadata)
    accounted.insert(op);
  for (auto fill : inventory.fills) {
    if (llvm::is_contained(padding, fill->getParentOp()) &&
        llvm::all_of(fill.getOutputs(), [&](Value output) {
          return llvm::any_of(inventory.allocs, [&](auto alloc) {
            return output == alloc.getResult();
          });
        }))
      accounted.insert(fill);
  }
  return hasOnlyAccountedEffects(func, accounted);
}

inline bool initializationPrecedesRead(scf::IfOp fill, memref::CopyOp copy,
                                      bufferization::ToTensorOp tensor) {
  return fill && tensor && fill->getBlock() == copy->getBlock() &&
         tensor->getBlock() == copy->getBlock() &&
         fill->isBeforeInBlock(copy) && copy->isBeforeInBlock(tensor);
}

inline std::optional<int64_t> integer(Value value) {
  auto c = value.getDefiningOp<arith::ConstantOp>();
  auto a = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr();
  return a ? std::optional<int64_t>(a.getInt()) : std::nullopt;
}
inline bool isInt(Value value, int64_t expected) {
  return integer(value) == expected;
}
inline bool castOf(Value value, Value source) {
  auto cast = value.getDefiningOp<arith::IndexCastOp>();
  return cast && cast.getIn() == source;
}
inline bool mulOf(Value value, Value source, int64_t factor) {
  auto mul = value.getDefiningOp<arith::MulIOp>();
  return mul && ((mul.getLhs() == source && isInt(mul.getRhs(), factor)) ||
                 (mul.getRhs() == source && isInt(mul.getLhs(), factor)));
}
// Exact lowered mask: min(start+tile,max(start,bound))-start, optionally
// surrounded by redundant min(tile)/max(0). All copy/store slices must use
// these SAME sizes. Nested two-input normalization still requires full tiles;
// flat normalization preserves the proved live extent for either input count.
inline bool boundedTileExtent(Value size, Value start, Value bound,
                              int64_t tile) {
  auto max0 = size.getDefiningOp<arith::MaxSIOp>();
  if (max0) {
    if (!isInt(max0.getRhs(), 0))
      return false;
    auto minTile = max0.getLhs().getDefiningOp<arith::MinSIOp>();
    if (!minTile || !isInt(minTile.getRhs(), tile))
      return false;
    size = minTile.getLhs();
  }
  auto sub = size.getDefiningOp<arith::SubIOp>();
  if (!sub || sub.getRhs() != start)
    return false;
  auto minEnd = sub.getLhs().getDefiningOp<arith::MinSIOp>();
  if (!minEnd)
    return false;
  auto end = minEnd.getLhs().getDefiningOp<arith::AddIOp>();
  auto maxBound = minEnd.getRhs().getDefiningOp<arith::MaxSIOp>();
  return end && end.getLhs() == start && isInt(end.getRhs(), tile) &&
         maxBound && maxBound.getLhs() == start &&
         (castOf(maxBound.getRhs(), bound) ||
          (integer(bound) && integer(maxBound.getRhs()) == integer(bound)));
}

// Recover the bound operand of the same clamp that boundedTileExtent proves.
// This is used when constant folding removed the row-loop ceiling division;
// the live rectangle still identifies the exact row bound, not just tile count.
inline Value tileExtentBound(Value size) {
  if (auto max0 = size.getDefiningOp<arith::MaxSIOp>()) {
    if (!isInt(max0.getRhs(), 0))
      return {};
    auto minTile = max0.getLhs().getDefiningOp<arith::MinSIOp>();
    if (!minTile)
      return {};
    size = minTile.getLhs();
  }
  auto sub = size.getDefiningOp<arith::SubIOp>();
  auto end = sub ? sub.getLhs().getDefiningOp<arith::MinSIOp>()
                 : arith::MinSIOp();
  auto bound = end ? end.getRhs().getDefiningOp<arith::MaxSIOp>()
                   : arith::MaxSIOp();
  if (!bound)
    return {};
  Value value = bound.getRhs();
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    value = cast.getIn();
  return value;
}

// Padding is unobservable only when the selected tensor reaches the already
// proved bounded output slice through same-shape, pure floating-point
// elementwise operations. Reject any cross-element, integer-conversion or
// escaping use rather than assuming that the source padding can be discarded
// for arbitrary kernels.
inline bool hasUnobservablePadding(
    bufferization::ToTensorOp input, tensor::ExtractSliceOp output,
    bufferization::MaterializeInDestinationOp store, Operation *parent) {
  auto tile = cast<RankedTensorType>(input.getResult().getType());
  SmallVector<Value> pending{input.getResult()};
  llvm::SmallPtrSet<Operation *, 16> seen;
  bool reachesOutput = false;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (user == output.getOperation()) {
        if (output.getSource() != value ||
            !llvm::all_of(output.getResult().getUsers(), [&](Operation *use) {
              return use == store.getOperation();
            }))
          return false;
        reachesOutput = true;
        continue;
      }
      if (!seen.insert(user).second)
        continue;
      bool mappedTanh = isMappedTanh(dyn_cast<linalg::MapOp>(user));
      if (user->getParentOp() != parent ||
          user->getNumResults() != 1 || !isMemoryEffectFree(user) ||
          (!mappedTanh && (user->getNumRegions() ||
                          !user->hasTrait<OpTrait::Elementwise>() ||
                          !isSpeculatable(user))))
        return false;
      for (Value operand : user->getOperands()) {
        auto type = dyn_cast<RankedTensorType>(operand.getType());
        if (!type || type.getShape() != tile.getShape() ||
            !isa<FloatType>(type.getElementType()))
          return false;
      }
      auto result = dyn_cast<RankedTensorType>(user->getResult(0).getType());
      if (!result || result.getShape() != tile.getShape() ||
          !isa<FloatType>(result.getElementType()))
        return false;
      pending.push_back(user->getResult(0));
    }
  }
  return reachesOutput;
}

// Recover a launch-visible extent from the same clamp proof used by ownership
// analysis. Arithmetic-only or unused scalar arguments cannot steal this role.
enum class ExtentMatchKind { Missing, Unique, Ambiguous };
struct ExtentMatch {
  ExtentMatchKind kind;
  BlockArgument argument;
};
inline ExtentMatch extentArgument(func::FuncOp func, KernelLayout &layout,
                                  Value size, Value start, int64_t tile) {
  BlockArgument bound;
  for (unsigned i = layout.publicBegin(); i < layout.publicEnd(); ++i) {
    auto arg = func.getArgument(i);
    if (!arg.getType().isInteger(32) ||
        !boundedTileExtent(size, start, arg, tile))
      continue;
    if (bound)
      return {ExtentMatchKind::Ambiguous, {}};
    bound = arg;
  }
  return {bound ? ExtentMatchKind::Unique : ExtentMatchKind::Missing, bound};
}

// Recover the bound of the exact positive-i32 ceiling division emitted by
// TTAdapter. This helper checks shape, not overflow; runtime premises retain
// responsibility for arithmetic bounds.
inline Value ceilDivBound(Value value, int64_t tile) {
  auto div = value.getDefiningOp<arith::DivSIOp>();
  auto add =
      div ? div.getLhs().getDefiningOp<arith::AddIOp>() : arith::AddIOp();
  if (!div || !isInt(div.getRhs(), tile) || !add ||
      !isInt(add.getRhs(), tile - 1))
    return {};
  return add.getLhs();
}

// ND L1->UB encodes a 12-bit burst count and a 16-bit burst length in
// 32-byte blocks. A contiguous physical row must contain whole blocks; row
// counts need no alignment. Physical capacity belongs to PlanMemory.
inline bool validTile(MemRefType tile, unsigned inputs) {
  if (!tile || !tile.hasStaticShape() || tile.getRank() != 2 || !inputs ||
      !(tile.getElementType().isF16() || tile.getElementType().isBF16() ||
        tile.getElementType().isF32()) ||
      !tile.getLayout().isIdentity() || tile.getMemorySpace())
    return false;
  int64_t rows = tile.getDimSize(0), cols = tile.getDimSize(1);
  if (rows <= 0 || cols <= 0 || cols % (256 / tile.getElementTypeBitWidth()) ||
      rows > 4095 ||
      cols / (256 / tile.getElementTypeBitWidth()) >
          std::numeric_limits<uint16_t>::max())
    return false;
  return rows <= std::numeric_limits<int64_t>::max() /
                     tile.getElementTypeBitWidth() / kL1BanksPerInput / inputs /
                     cols;
}

} // namespace mlir::hivm::l1prefetch

namespace mlir::hivm {

// Internal ownership proofs share the already validated ABI and inventory.
std::string analyzeNestedTileCandidate(func::FuncOp func,
                                       l1prefetch::KernelLayout &layout,
                                       l1prefetch::EffectInventory &inventory);
std::string analyzeJaggedTileCandidate(func::FuncOp func,
                                       l1prefetch::KernelLayout &layout,
                                       l1prefetch::EffectInventory &inventory);
} // namespace mlir::hivm

#endif
