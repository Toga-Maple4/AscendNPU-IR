/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "L1PrefetchPSFacts.h"
#include "L1PrefetchPSAddress.h"
#include "bishengir/Dialect/Analysis/VFFusion/CostModelInfo/CostModelInfo.h"
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
#include "llvm/Support/JSON.h"
#include <limits>
#include <optional>

#define DEBUG_TYPE "l1-prefetch-analyze"

namespace mlir {
#define GEN_PASS_DEF_L1PREFETCHANALYZE
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

bool matchRound(Value value, Value iv, int64_t columns,
                hivm::l1prefetch::AddressEquivalence &equality) {
  auto div = value.getDefiningOp<arith::DivSIOp>();
  return div && equality.equal(div.getLhs(), iv) &&
         hivm::l1prefetch::isInt(div.getRhs(), columns);
}

bool matchColumn(Value value, Value iv, int64_t columns,
                 hivm::l1prefetch::AddressEquivalence &equality) {
  auto rem = value.getDefiningOp<arith::RemSIOp>();
  return rem && equality.equal(rem.getLhs(), iv) &&
         hivm::l1prefetch::isInt(rem.getRhs(), columns);
}

// Search only the original arithmetic DAG for the schedule's N*round atom.
// Equality accepts reassociation, but N must still be an actual proven scalar
// source. This is not inference of a new schedule or reconstruction of N.
SmallVector<Value> addressValues(Value root) {
  SmallVector<Value> values{root};
  llvm::SmallDenseSet<Value> seen{root};
  for (unsigned i = 0; i < values.size(); ++i)
    if (auto *op = values[i].getDefiningOp();
        op && isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op))
      for (Value operand : op->getOperands())
        if (seen.insert(operand).second)
          values.push_back(operand);
  return values;
}

bool matchRowTask(Value value, Value logicalId, Value iv,
                  Value &logicalPrograms, int64_t columns,
                  hivm::l1prefetch::AddressEquivalence &equality) {
  for (Value candidate : addressValues(value)) {
    auto mul = candidate.getDefiningOp<arith::MulIOp>();
    if (!mul || !equality.sum(value, logicalId, candidate))
      continue;
    if (matchRound(mul.getLhs(), iv, columns, equality))
      logicalPrograms = mul.getRhs();
    else if (matchRound(mul.getRhs(), iv, columns, equality))
      logicalPrograms = mul.getLhs();
    else
      continue;
    return true;
  }
  return false;
}

bool matchRowOffset(Value value, Value rowTask, int64_t rows, int64_t stride,
                    hivm::l1prefetch::AddressEquivalence &equality) {
  for (Value candidate : addressValues(value))
    if (auto cast = candidate.getDefiningOp<arith::IndexCastOp>();
        cast && equality.product(value, candidate, stride) &&
        equality.product(cast.getIn(), rowTask, rows))
      return true;
  return false;
}

bool matchColumnOffset(Value value, Value iv, int64_t width, int64_t columns,
                       hivm::l1prefetch::AddressEquivalence &equality) {
  auto cast = value.getDefiningOp<arith::IndexCastOp>();
  if (!cast)
    return false;
  for (Value candidate : addressValues(cast.getIn()))
    if (matchColumn(candidate, iv, columns, equality) &&
        equality.product(cast.getIn(), candidate, width))
      return true;
  return false;
}

bool matchOffset(Value value, Value rowTask, Value iv, int64_t rows,
                 int64_t width, int64_t stride,
                 hivm::l1prefetch::AddressEquivalence &equality) {
  auto values = addressValues(value);
  for (Value row : values)
    if (matchRowOffset(row, rowTask, rows, stride, equality))
      for (Value column : values)
        if (matchColumnOffset(column, iv, width, stride / width, equality) &&
            equality.sum(value, row, column))
          return true;
  return false;
}

bool expectedTileView(Operation *op, Value base, MemRefType tile,
                      int64_t stride, Value &offset) {
  auto view = dyn_cast_or_null<memref::ReinterpretCastOp>(op);
  if (!view || view.getSource() != base || view.getOffsets().size() != 1 ||
      view.getType().getShape() != tile.getShape() ||
      view.getType().getElementType() != tile.getElementType())
    return false;
  auto sizes = view.getMixedSizes(), strides = view.getMixedStrides();
  if (sizes.size() != 2 || strides.size() != 2 ||
      getConstantIntValue(sizes[0]) != tile.getDimSize(0) ||
      getConstantIntValue(sizes[1]) != tile.getDimSize(1) ||
      getConstantIntValue(strides[0]) != stride ||
      getConstantIntValue(strides[1]) != 1)
    return false;
  offset = view.getOffsets().front();
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
      inventory.hasUnsupportedOperation();
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
  auto launchCount = [&](Value value) {
    return hivm::l1prefetch::persistentCountSource(value, *layout);
  };
  hivm::l1prefetch::AddressEquivalence equality;
  if (loop->getParentOp() != func || ifOp->getParentOp() != loop ||
      !ifOp.getElseRegion().empty() || loop.getNumResults() != 0 ||
      ifOp.getNumResults() != 0 ||
      !hivm::l1prefetch::isInt(loop.getLowerBound(), 0) ||
      !launchCount(loop.getUpperBound()) ||
      !hivm::l1prefetch::isInt(loop.getStep(), 1))
    return {false, "unsupported_persistent_loop"};
  auto cmp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  Value logicalPrograms;
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt ||
      !launchCount(cmp.getRhs()) ||
      !matchRowTask(cmp.getLhs(), layout->programId(), loop.getInductionVar(),
                    logicalPrograms, columns, equality))
    return {false, "unproved_worker_disjointness"};
  auto logicalConstant = hivm::l1prefetch::integer(logicalPrograms);
  if (!launchCount(logicalPrograms) ||
      (logicalConstant && *logicalConstant % 2 != 0))
    return {false, "unsupported_logical_grid"};
  // Separate correctness tightening for the fixed Persistent domain. Individual
  // i32 counts do not bound their products. Mirror valid_persistent_tiles using
  // division guards, so even the checker never evaluates an overflowing product.
  auto steps = hivm::l1prefetch::integer(loop.getUpperBound());
  auto tasks = hivm::l1prefetch::integer(cmp.getRhs());
  if (logicalConstant && steps && tasks) {
    int64_t n = *logicalConstant, round = (*steps - 1) / columns;
    constexpr int64_t max32 = std::numeric_limits<int32_t>::max();
    if (round > (max32 - (n - 1)) / n)
      return {false, "persistent_row_task_overflow"};
    int64_t last = std::min(*tasks - 1, n - 1 + n * round);
    int64_t bm = selectedType.getDimSize(0), bn = selectedType.getDimSize(1);
    if (last > max32 / bm)
      return {false, "persistent_row_offset_overflow"};
    int64_t column = std::min(columns - 1, *steps - 1 - (last / n) * columns);
    int64_t tail = column * bn + bn - 1;
    int64_t lastRow = last * bm + bm - 1;
    int64_t maxElement = std::numeric_limits<int64_t>::max() /
                         (selectedType.getElementTypeBitWidth() / 8);
    if (lastRow > (maxElement - tail) / stride)
      return {false, "persistent_byte_offset_overflow"};
  }
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
      !equality.equal(inputOffset, outputOffset) ||
      !matchOffset(inputOffset, rowTask, loop.getInductionVar(),
                   selectedType.getDimSize(0), selectedType.getDimSize(1),
                   stride, equality))
    return {false, "unproved_store_partition"};

  if (!hivm::l1prefetch::hasStagingEffects(func, inventory, {}))
    return {false, "unmodelled_memory_dependency"};

  // With q in [0,N), each row task is q + N*floor(step/columns).
  // Distinct (q,round) pairs have distinct rows; step%columns selects
  // disjoint tile-width intervals in a row. The one store uses exactly
  // that address. The caller's opt-in supplies the input/output no-alias
  // contract, so moving complete q programs onto the same AIV preserves all
  // defined inter-program dependencies.
  if (failed(hivm::l1prefetch::publishAcceptedFacts(
      func, {hivm::l1prefetch::AccessSchedule::Persistent, roles, logicalPrograms,
             loop.getUpperBound(), cmp.getRhs()})))
    return {false, "inconsistent_accepted_facts"};
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

// Read-only issue-work estimate, not a cycle estimate or a semantic proof.
hivm::L1PrefetchPSProfitability evaluateProfitability(func::FuncOp func) {
  hivm::L1PrefetchPSProfitability facts;
  auto &score = facts.score;
  auto &reason = facts.reason;
  auto &inputCount = facts.inputs;
  reason = "missing_contract";
  auto evaluate = [&]() {
    auto result = hivm::getL1PrefetchPSExternalContract(func);
    if (failed(result))
      return;
    auto guard = result->getAs<DictionaryAttr>("guard");
    auto roles = guard.getAs<DictionaryAttr>("roles");
    auto inputs = roles.getAs<ArrayAttr>("inputs");
    inputCount = inputs.size();
    double inputBytes = 0;
    func.walk([&](annotation::MarkOp mark) {
      if (!mark->hasAttr("program_split.selected_input"))
        return;
      auto type = cast<MemRefType>(mark->getOperand(0).getType());
      inputBytes += static_cast<double>(type.getNumElements()) *
                    type.getElementTypeBitWidth() / 8;
    });
    if (inputBytes <= 0) {
      reason = "missing_selected_storage";
      return;
    }

    // Count the tensor arithmetic feeding the sole proved output. Constant
    // fills and tensor views are not arithmetic. Unknown operations/costs are
    // unknown, never the VF model's generic fallback cost.
    SmallVector<Value> worklist;
    Block *computeBlock = nullptr;
    func.walk([&](bufferization::MaterializeInDestinationOp store) {
      worklist.push_back(store.getSource());
      computeBlock = store->getBlock();
    });
    llvm::SmallPtrSet<Operation *, 32> seen;
    static const auto costs = analysis::CostModelInfo().getConfigMap(false);
    double work = 0;
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      Operation *op = value.getDefiningOp();
      if (!op) {
        reason = "unknown_compute_value";
        return;
      }
      if (!seen.insert(op).second)
        continue;
      if (isa<bufferization::ToTensorOp>(op))
        continue;
      if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op)) {
        worklist.push_back(slice.getSource());
        continue;
      }
      if (isa<arith::ConstantOp, linalg::FillOp>(op))
        continue;
      if (op->getBlock() != computeBlock) {
        reason = "unknown_compute_multiplicity";
        return;
      }
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || !type.hasStaticShape() || op->getNumResults() != 1) {
        reason = "unknown_compute_shape";
        return;
      }
      auto info = op->getRegisteredInfo();
      auto found = info ? costs.find(info->getTypeID()) : costs.end();
      auto element = type.getElementType();
      auto costType = element.isF32() ? analysis::TypeKind::F32
                      : element.isF16() ? analysis::TypeKind::F16
                                        : analysis::TypeKind::Unknown;
      std::optional<int> issueInterval;
      if (found != costs.end() && found->second.count(costType))
        issueInterval = found->second.at(costType).execInterval;
      // Local cost proxy: price FP32->BF16 like FP32->FP16 (currently
      // one issue unit). This does not assign costs to native BF16 arithmetic.
      if (auto trunc = dyn_cast<arith::TruncFOp>(op);
          trunc && element.isBF16() &&
          cast<RankedTensorType>(trunc.getIn().getType())
              .getElementType()
              .isF32() &&
          found != costs.end() && found->second.count(analysis::TypeKind::F16))
        issueInterval = found->second.at(analysis::TypeKind::F16).execInterval;
      if (isa<math::TanhOp>(op) && element.isF32()) {
        // Mirror buildTanhApproximation: clamp, multiply by two, exp,
        // add -1, add +1, divide. Reuse its seven primitive FP32 costs
        // (currently 1+1+1+4+1+1+4=13), without changing VF fusion's table.
        int total = 0;
        for (TypeID id : {TypeID::get<arith::MinimumFOp>(),
                          TypeID::get<arith::MaximumFOp>(),
                          TypeID::get<arith::MulFOp>(),
                          TypeID::get<math::ExpOp>(),
                          TypeID::get<arith::AddFOp>(),
                          TypeID::get<arith::AddFOp>(),
                          TypeID::get<arith::DivFOp>()}) {
          auto primitive = costs.find(id);
          if (primitive == costs.end() ||
              !primitive->second.count(analysis::TypeKind::F32)) {
            reason = "unknown_operation_cost";
            return;
          }
          total += primitive->second.at(analysis::TypeKind::F32).execInterval;
        }
        issueInterval = total;
      }
      if (!issueInterval) {
        reason = "unknown_operation_cost";
        return;
      }
      unsigned width = element.getIntOrFloatBitWidth();
      for (Value operand : op->getOperands()) {
        if (auto operandType = dyn_cast<RankedTensorType>(operand.getType())) {
          if (operandType.getShape() != type.getShape() ||
              !isa<FloatType>(operandType.getElementType())) {
            reason = "unknown_compute_shape";
            return;
          }
          width = std::max(width, operandType.getElementTypeBitWidth());
          worklist.push_back(operand);
        }
      }
      // Normalize work to a 256-byte A5 vector register. Counting conversions
      // at the wider type avoids treating FP16->FP32 work as one FP16 vector.
      work += static_cast<double>(type.getNumElements()) * width / 2048 *
              *issueInterval;
    }
    score = work / (inputBytes / 1024);

    reason = facts.filtered() ? "compute_score_below_40"
                              : "compute_score_at_least_40";
    // Deferred considerations (not gates): the former score-64 threshold,
    // multiple inputs, 32-byte alignment/fragmentation, metadata-dependent trip
    // counts and baseline overlap. Pure copies score zero naturally; there is
    // no independent copy-kernel rejection. Revisit only with separate evidence.
  };
  evaluate();
  return facts;
}

void reportProfitability(func::FuncOp func) {
  const auto facts = evaluateProfitability(func);
  llvm::errs() << "[program-split-profitability] ";
  llvm::json::OStream json(llvm::errs());
  json.object([&] {
    json.attribute("version", 2);
    json.attribute("threshold", 40);
    json.attribute("function", func.getSymName());
    json.attribute("decision", facts.decision());
    json.attribute("reason", facts.reason);
    json.attribute("inputs", facts.inputs);
    if (facts.score)
      json.attribute("compute_score", *facts.score);
    else
      json.attribute("compute_score", nullptr);
  });
  llvm::errs() << "\n";
}

struct L1PrefetchAnalyzePass
    : impl::L1PrefetchAnalyzeBase<L1PrefetchAnalyzePass> {
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
        reportProfitability(func);
      } else {
        func->setAttr("l1_prefetch_ps.reject_reason",
                      StringAttr::get(module.getContext(), result.reason));
      }
    }
  }
};
} // namespace
std::unique_ptr<Pass> mlir::hivm::createL1PrefetchAnalyzePass() {
  return std::make_unique<L1PrefetchAnalyzePass>();
}

mlir::hivm::L1PrefetchPSProfitability
mlir::hivm::getL1PrefetchPSProfitability(func::FuncOp func) {
  return evaluateProfitability(func);
}
