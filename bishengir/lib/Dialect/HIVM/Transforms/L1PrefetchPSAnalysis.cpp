/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "L1PrefetchPSRoles.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/L1PrefetchPS.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <limits>
#include <optional>

#define DEBUG_TYPE "l1-prefetch-ps-analysis"

namespace mlir {
#define GEN_PASS_DEF_L1PREFETCHPSANALYSIS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir
using namespace mlir;

namespace {
struct CandidateResult {
  bool accepted = false;
  std::string reason;
};
} // namespace

namespace {

bool matchRound(Value value, Value iv, int64_t columns) {
  auto div = value.getDefiningOp<arith::DivSIOp>();
  return div && div.getLhs() == iv &&
         hivm::l1prefetch::isInt(div.getRhs(), columns);
}

bool matchColumn(Value value, Value iv, int64_t columns) {
  auto rem = value.getDefiningOp<arith::RemSIOp>();
  return rem && rem.getLhs() == iv &&
         hivm::l1prefetch::isInt(rem.getRhs(), columns);
}

bool matchRowTask(Value value, Value logicalId, Value iv,
                  int64_t &logicalPrograms, int64_t columns) {
  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;
  Value other = add.getLhs() == logicalId   ? add.getRhs()
                : add.getRhs() == logicalId ? add.getLhs()
                                            : Value();
  if (!other)
    return false;
  auto mul = other.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;
  Value round;
  std::optional<int64_t> count = hivm::l1prefetch::integer(mul.getLhs());
  if (count)
    round = mul.getRhs();
  else {
    count = hivm::l1prefetch::integer(mul.getRhs());
    round = mul.getLhs();
  }
  if (!count || !matchRound(round, iv, columns))
    return false;
  logicalPrograms = *count;
  return true;
}

bool matchRowOffset(Value value, Value rowTask, int64_t rows, int64_t stride) {
  auto mul = value.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;
  Value castValue = hivm::l1prefetch::isInt(mul.getLhs(), stride) ? mul.getRhs()
                    : hivm::l1prefetch::isInt(mul.getRhs(), stride)
                        ? mul.getLhs()
                        : Value();
  auto cast = castValue ? castValue.getDefiningOp<arith::IndexCastOp>()
                        : arith::IndexCastOp();
  if (!cast)
    return false;
  return hivm::l1prefetch::mulOf(cast.getIn(), rowTask, rows);
}

bool matchColumnOffset(Value value, Value iv, int64_t width, int64_t columns) {
  auto cast = value.getDefiningOp<arith::IndexCastOp>();
  if (!cast)
    return false;
  auto mul = cast.getIn().getDefiningOp<arith::MulIOp>();
  return mul && ((hivm::l1prefetch::isInt(mul.getLhs(), width) &&
                  matchColumn(mul.getRhs(), iv, columns)) ||
                 (hivm::l1prefetch::isInt(mul.getRhs(), width) &&
                  matchColumn(mul.getLhs(), iv, columns)));
}

bool matchOffset(Value value, Value rowTask, Value iv, int64_t rows,
                 int64_t width, int64_t stride) {
  auto add = value.getDefiningOp<arith::AddIOp>();
  return add && ((matchRowOffset(add.getLhs(), rowTask, rows, stride) &&
                  matchColumnOffset(add.getRhs(), iv, width, stride / width)) ||
                 (matchRowOffset(add.getRhs(), rowTask, rows, stride) &&
                  matchColumnOffset(add.getLhs(), iv, width, stride / width)));
}

bool expectedTileView(Operation *view, Value base, MemRefType tile,
                      int64_t stride, Value &offset) {
  if (!view || view->getName().getStringRef() != "memref.reinterpret_cast" ||
      view->getNumOperands() != 2 || view->getOperand(0) != base ||
      view->getNumResults() != 1)
    return false;
  auto type = dyn_cast<MemRefType>(view->getResult(0).getType());
  if (!type || type.getShape() != tile.getShape() ||
      type.getElementType() != tile.getElementType())
    return false;
  SmallVector<int64_t> strides;
  int64_t staticOffset;
  if (failed(getStridesAndOffset(type, strides, staticOffset)) ||
      strides.size() != 2 || strides[0] != stride || strides[1] != 1 ||
      !ShapedType::isDynamic(staticOffset))
    return false;
  offset = view->getOperand(1);
  return true;
}

} // namespace

static CandidateResult analyzeAndAttachCandidate(ModuleOp module) {
  auto func = hivm::getL1PrefetchPSEntry(module);
  if (!func)
    return {false, "multiple_or_missing_functions"};
  auto mixMode = func->getAttrOfType<StringAttr>("mix_mode");
  auto parallelMode = func->getAttrOfType<StringAttr>("parallel_mode");
  if (!mixMode || mixMode.getValue() != "aiv" || !parallelMode ||
      parallelMode.getValue() != "simd")
    return {false, "not_pure_vector"};
  // Every schedule moves future loads ahead of current stores. This is a caller
  // premise, independent of the loop shape or number of selected inputs.
  if (!func->hasAttrOfType<UnitAttr>("ascend.memory.pointer_args_noalias"))
    return {false, "missing_pointer_noalias_contract"};
  for (NamedAttribute attr : func->getAttrs())
    if (attr.getName().strref().starts_with("program_split."))
      return {false, "preexisting_program_split_plan"};
  auto layout = hivm::l1prefetch::KernelLayout::get(func);
  if (!layout)
    return {false, "unsupported_kernel_abi"};
  hivm::l1prefetch::EffectInventory inventory(func);
  if (!inventory.metadataLoads.empty() || inventory.loops.size() != 1) {
    std::string reason =
        !inventory.metadataLoads.empty()
            ? hivm::analyzeJaggedTileCandidate(func, *layout, inventory)
            : hivm::analyzeNestedTileCandidate(func, *layout, inventory);
    return {reason.empty(), reason.empty() ? "accepted" : reason};
  }
  auto &loops = inventory.loops;
  auto &ifs = inventory.ifs;
  auto &copies = inventory.copies;
  auto &allocs = inventory.allocs;
  auto &stores = inventory.stores;
  auto &toTensors = inventory.tensors;
  unsigned reinterpretCount = inventory.bases.size();
  bool disallowed =
      !inventory.supports(hivm::l1prefetch::OperationProfile::Persistent);
  if (disallowed || loops.size() != 1 || ifs.size() != 1 ||
      copies.size() != 1 || allocs.size() != 1 || stores.size() != 1 ||
      reinterpretCount != 2 || toTensors.size() != 1)
    return {false, "unsupported_memory_or_control_flow"};

  auto selectedType = dyn_cast<MemRefType>(allocs.front().getType());
  if (!hivm::l1prefetch::validTile(selectedType, 1))
    return {false, "unsupported_selected_tile"};
  auto sourceType = dyn_cast<MemRefType>(copies.front().getSource().getType());
  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset;
  if (!sourceType ||
      failed(getStridesAndOffset(sourceType, sourceStrides, sourceOffset)) ||
      sourceStrides.size() != 2 || sourceStrides[0] <= 0 ||
      sourceStrides[0] > std::numeric_limits<int32_t>::max() ||
      sourceStrides[0] % selectedType.getDimSize(1))
    return {false, "unproved_store_partition"};
  int64_t stride = sourceStrides[0];
  int64_t columns = stride / selectedType.getDimSize(1);
  scf::ForOp loop = loops.front();
  scf::IfOp ifOp = ifs.front();
  std::optional<int64_t> persistentSteps =
      hivm::l1prefetch::integer(loop.getUpperBound());
  if (loop->getParentOp() != func || ifOp->getParentOp() != loop ||
      !ifOp.getElseRegion().empty() || loop.getNumResults() != 0 ||
      ifOp.getNumResults() != 0 ||
      !hivm::l1prefetch::isInt(loop.getLowerBound(), 0) || !persistentSteps ||
      *persistentSteps <= 0 ||
      *persistentSteps > std::numeric_limits<int32_t>::max() ||
      !hivm::l1prefetch::isInt(loop.getStep(), 1))
    return {false, "unsupported_persistent_loop"};
  auto cmp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  std::optional<int64_t> rowTasks =
      cmp ? hivm::l1prefetch::integer(cmp.getRhs()) : std::nullopt;
  int64_t logicalPrograms = 0;
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt || !rowTasks ||
      *rowTasks <= 0 || *rowTasks > std::numeric_limits<int32_t>::max() ||
      !matchRowTask(cmp.getLhs(), layout->programId(), loop.getInductionVar(),
                    logicalPrograms, columns))
    return {false, "unproved_worker_disjointness"};
  if (logicalPrograms <= 0 || logicalPrograms % 2 != 0 ||
      logicalPrograms > std::numeric_limits<int32_t>::max())
    return {false, "unsupported_logical_grid"};
  Value rowTask = cmp.getLhs();
  auto copy = copies.front();
  auto alloc = allocs.front();
  Operation *store = stores.front().getOperation();
  if (copy->getParentOp() != ifOp || alloc->getParentOp() != ifOp ||
      store->getParentOp() != ifOp || copy.getTarget() != alloc.getResult() ||
      toTensors.front()->getNumOperands() != 1 ||
      toTensors.front()->getOperand(0) != alloc.getResult() ||
      store->getNumOperands() != 2)
    return {false, "unproved_selected_input_or_store"};
  hivm::l1prefetch::AccessRoles roles;
  auto roleError = hivm::l1prefetch::deriveAccessRoles(
      func, *layout, hivm::l1prefetch::AccessSchedule::Persistent, copies,
      stores.front(), loop, {}, {}, selectedType, roles);
  if (!roleError.empty())
    return {false, roleError};
  for (auto pointer : {roles.inputs.front(), roles.output}) {
    auto type = cast<MemRefType>(pointer.getType());
    if (type.getElementType() != selectedType.getElementType())
      return {false, "unsupported_pointer_type"};
  }
  Value inputOffset, outputOffset;
  if (!expectedTileView(copy.getSource().getDefiningOp(), roles.inputs.front(),
                        selectedType, stride, inputOffset) ||
      !expectedTileView(store->getOperand(1).getDefiningOp(), roles.output,
                        selectedType, stride, outputOffset) ||
      inputOffset != outputOffset ||
      !matchOffset(inputOffset, rowTask, loop.getInductionVar(),
                   selectedType.getDimSize(0), selectedType.getDimSize(1),
                   stride))
    return {false, "unproved_store_partition"};

  // With q in [0,N), each row task is q + N*floor(step/columns).
  // Distinct (q,round) pairs have distinct rows; step%columns selects
  // disjoint tile-width intervals in a row. The one store uses exactly
  // that address. The caller's opt-in supplies the input/output no-alias
  // contract, so moving complete q programs onto the same AIV preserves all
  // defined inter-program dependencies.
  OpBuilder builder(module.getContext());
  loop->setAttr("l1_prefetch_ps.outer", builder.getUnitAttr());
  // These select proven relationships; their numeric facts remain in SSA.
  cmp->setAttr("l1_prefetch_ps.active", builder.getUnitAttr());
  copy->setAttr("l1_prefetch_ps.load", builder.getUnitAttr());
  builder.setInsertionPointAfter(alloc);
  auto mark =
      builder.create<annotation::MarkOp>(alloc.getLoc(), alloc.getResult());
  mark->setAttr("program_split.selected_input", builder.getI32IntegerAttr(0));
  return {true, "accepted"};
}

namespace {
struct L1PrefetchPSAnalysisPass
    : impl::L1PrefetchPSAnalysisBase<L1PrefetchPSAnalysisPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Reanalysis replaces its annotations, never inherits a prior acceptance.
    SmallVector<annotation::MarkOp> marks;
    module.walk([&](Operation *op) {
      SmallVector<StringAttr> erase;
      for (NamedAttribute attr : op->getAttrs())
        if (attr.getName().strref().starts_with("l1_prefetch_ps."))
          erase.push_back(attr.getName());
      for (auto name : erase)
        op->removeAttr(name);
      if (auto mark = dyn_cast<annotation::MarkOp>(op);
          mark && mark->hasAttr("program_split.selected_input"))
        marks.push_back(mark);
    });
    for (auto mark : marks)
      mark.erase();
    CandidateResult result;
    if (!hacc::utils::isAscend950(module))
      result = {false, "unsupported_target"};
    else
      result = analyzeAndAttachCandidate(module);
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (result.accepted) {
        func->setAttr("l1_prefetch_ps.accepted",
                      UnitAttr::get(module.getContext()));
        func->setAttr("l1_prefetch_ps.proof",
                      StringAttr::get(module.getContext(),
                                      hivm::l1PrefetchPSFingerprint(func)));
      } else {
        func->setAttr("l1_prefetch_ps.reject_reason",
                      StringAttr::get(module.getContext(), result.reason));
      }
    }
  }
};
} // namespace
std::unique_ptr<Pass> mlir::hivm::createL1PrefetchPSAnalysisPass() {
  return std::make_unique<L1PrefetchPSAnalysisPass>();
}
