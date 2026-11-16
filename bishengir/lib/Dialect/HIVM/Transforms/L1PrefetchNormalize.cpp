//===- L1PrefetchNormalize.cpp - Consume accepted structured IR --------===//
#include "L1PrefetchPSFacts.h"
#include "L1PrefetchPSDependencies.h"
#include "L1PrefetchPSReservation.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/L1PrefetchPS.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DEF_L1PREFETCHNORMALIZE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir
using namespace mlir;

LogicalResult mlir::hivm::verifyL1PrefetchPSRealizationRoles(
    Operation *anchor, DictionaryAttr contract, func::FuncOp originalABI) {
  auto inputs = contract ? contract.getAs<DenseI32ArrayAttr>("input_args")
                         : DenseI32ArrayAttr();
  auto output = contract ? contract.getAs<IntegerAttr>("output_arg")
                         : IntegerAttr();
  if (!inputs || inputs.empty() || !output)
    return anchor->emitError(
        "L1 pipeline error: missing explicit input/output roles");
  if (!output.getType().isInteger(32) || output.getInt() < 0)
    return anchor->emitError(
        "L1 pipeline error: output_arg must be a nonnegative i32 index");
  SmallVector<int32_t> indices;
  for (int32_t index : inputs.asArrayRef()) {
    if (index < 0)
      return anchor->emitError(
          "L1 pipeline error: input_args contains a negative index");
    if (llvm::is_contained(indices, index))
      return anchor->emitError(
          "L1 pipeline error: input_args contains duplicate indices");
    if (index == output.getInt())
      return anchor->emitError(
          "L1 pipeline error: input/output pointer roles overlap");
    indices.push_back(index);
  }
  if (!originalABI)
    return success();
  auto layout = l1prefetch::KernelLayout::get(originalABI);
  if (!layout)
    return anchor->emitError(
        "L1 pipeline error: invalid original TTAdapter ABI for pointer roles");
  indices.push_back(output.getInt());
  for (int32_t index : indices) {
    if (uint64_t(index) >= originalABI.getNumArguments() ||
        !layout->isPublic(originalABI.getArgument(index)) ||
        !isa<MemRefType>(originalABI.getArgument(index).getType()))
      return anchor->emitError(
          "L1 pipeline error: pointer role does not name an original public pointer");
  }
  return success();
}

namespace {
constexpr StringLiteral kProof = "l1_prefetch_ps.proof";
// These objects only view the current accepted IR. They are never retained
// across passes or clones and perform no semantic eligibility analysis.
using hivm::l1prefetch::AccessSchedule;
struct Selection {
  scf::ForOp outer, inner;
  SmallVector<memref::CopyOp, 2> loads;
  bufferization::MaterializeInDestinationOp store;
  SmallVector<memref::LoadOp> metadata;
  arith::CmpIOp active;
  AccessSchedule schedule;
  MemRefType tile;
  Value rowBound;
  BlockArgument flatBound;
  hivm::l1prefetch::AccessRoles roles;
  Value logicalPrograms, persistentSteps, rowTasks;
  bool runtimePersistent = false;
};

LogicalResult handoffFailure(std::string &reason, const Twine &message) {
  reason = message.str();
  return failure();
}

AccessSchedule inferSchedule(func::FuncOp func, const Selection &selection) {
  if (func->hasAttr("l1_prefetch_ps.program"))
    return AccessSchedule::Program;
  if (func->hasAttr("l1_prefetch_ps.flat"))
    return AccessSchedule::Flat;
  if (func->hasAttr("l1_prefetch_ps.jagged"))
    return AccessSchedule::Jagged;
  return selection.inner ? AccessSchedule::Nested : AccessSchedule::Persistent;
}

LogicalResult collectSelection(func::FuncOp func, Selection &selection,
                               std::string &reason) {
  unsigned outer = 0, inner = 0, loads = 0, active = 0, stores = 0;
  auto walk = func.walk([&](Operation *op) -> WalkResult {
    if (op->hasAttr("l1_prefetch_ps.outer")) {
      ++outer;
      selection.outer = dyn_cast<scf::ForOp>(op);
      if (!selection.outer) {
        reason = "outer marker must name an scf.for";
        return WalkResult::interrupt();
      }
    }
    if (op->hasAttr("l1_prefetch_ps.inner")) {
      ++inner;
      selection.inner = dyn_cast<scf::ForOp>(op);
      if (!selection.inner) {
        reason = "inner marker must name an scf.for";
        return WalkResult::interrupt();
      }
    }
    if (op->hasAttr("l1_prefetch_ps.load")) {
      ++loads;
      auto copy = dyn_cast<memref::CopyOp>(op);
      if (!copy) {
        reason = "load marker must name a memref.copy";
        return WalkResult::interrupt();
      }
      selection.loads.push_back(copy);
    }
    if (op->hasAttr("l1_prefetch_ps.active")) {
      ++active;
      selection.active = dyn_cast<arith::CmpIOp>(op);
      if (!selection.active) {
        reason = "active marker must name an arith.cmpi";
        return WalkResult::interrupt();
      }
    }
    if (auto store = dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
      ++stores;
      selection.store = store;
    }
    if (auto load = dyn_cast<memref::LoadOp>(op))
      selection.metadata.push_back(load);
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return failure(); // The marker type diagnostic is already recorded.

  selection.schedule = inferSchedule(func, selection);
  bool loopless = selection.schedule == AccessSchedule::Flat ||
                  selection.schedule == AccessSchedule::Program;
  if (outer != (loopless ? 0u : 1u))
    return handoffFailure(reason, "unexpected outer marker count");
  if (!loads || (!inner && !loopless && loads > 1))
    return handoffFailure(reason, "unexpected selected load count");
  if (inner > 1)
    return handoffFailure(reason, "multiple inner markers");
  if (active != ((inner || loopless) ? 0u : 1u))
    return handoffFailure(reason, "unexpected active marker count");
  if (stores != 1)
    return handoffFailure(reason, "expected one completion store");
  for (auto copy : selection.loads) {
    Value target = copy.getTarget();
    if (auto slice = target.getDefiningOp<memref::SubViewOp>())
      target = slice.getSource();
    auto type = dyn_cast<MemRefType>(target.getType());
    if (!type || (selection.tile && selection.tile != type))
      return handoffFailure(
          reason, "selected buffers must share a memref tile type");
    selection.tile = type;
  }
  return success();
}

LogicalResult checkLoopNesting(func::FuncOp func, Selection &selection,
                              std::string &reason) {
  bool jagged = selection.schedule == AccessSchedule::Jagged;
  if (selection.outer) {
    if (selection.outer->getParentOp() != func)
      return handoffFailure(
          reason, "outer loop must be directly inside the function");
    bool validResults = jagged ? selection.outer.getNumRegionIterArgs() == 1
                               : selection.outer.getNumResults() == 0;
    if (!validResults)
      return handoffFailure(reason, "unexpected outer loop result count");
  }
  if (selection.inner) {
    if (selection.inner->getParentOp() != selection.outer)
      return handoffFailure(
          reason, "inner loop must be directly inside the outer loop");
    bool validResults = jagged ? selection.inner.getNumRegionIterArgs() == 2
                               : selection.inner.getNumResults() == 0;
    if (!validResults)
      return handoffFailure(reason, "unexpected inner loop result count");
  }
  return success();
}

// A work/padding guard contributes no values and has no alternative payload.
bool isThenOnlyGuard(scf::IfOp guard, Operation *parent) {
  return guard && guard->getParentOp() == parent && !guard.getNumResults() &&
         guard.getElseRegion().empty();
}

bool isFillOnlyPaddingGuard(Operation *op, Operation *payload) {
  auto guard = dyn_cast<scf::IfOp>(op);
  if (!isThenOnlyGuard(guard, payload))
    return false;
  auto &body = guard.getThenRegion().front();
  return std::distance(body.begin(), body.end()) == 2 &&
         isa<linalg::FillOp>(body.front()) && isa<scf::YieldOp>(body.back());
}

LogicalResult checkPayload(func::FuncOp func, Selection &selection,
                           const hivm::l1prefetch::AcceptedFacts &facts,
                           std::string &reason) {
  bool loopless = selection.schedule == AccessSchedule::Flat ||
                  selection.schedule == AccessSchedule::Program;
  Operation *payload =
      loopless ? func.getOperation() : selection.inner.getOperation();
  if (selection.schedule == AccessSchedule::Persistent) {
    auto work = dyn_cast<scf::IfOp>(selection.store->getParentOp());
    if (!isThenOnlyGuard(work, selection.outer) ||
        work.getCondition() != selection.active.getResult())
      return handoffFailure(
          reason, "persistent work guard must use the active marker");
    payload = work;
    if (facts.persistentSteps != selection.outer.getUpperBound() ||
        facts.rowTasks != selection.active.getRhs())
      return handoffFailure(
          reason, "persistent counts disagree with loop bound or active test");
    selection.logicalPrograms = facts.logicalPrograms;
    selection.persistentSteps = facts.persistentSteps;
    selection.rowTasks = facts.rowTasks;
    selection.runtimePersistent = !hivm::l1prefetch::integer(facts.logicalPrograms) ||
                                  !hivm::l1prefetch::integer(facts.persistentSteps) ||
                                  !hivm::l1prefetch::integer(facts.rowTasks);
  }
  auto outputView = hivm::l1prefetch::accessView(selection.store.getDest());
  if (!outputView || selection.store->getParentOp() != payload ||
      outputView.getSource() != selection.roles.output)
    return handoffFailure(
        reason, "completion store must use the output role in the payload");
  SmallVector<Value> pointers;
  for (auto copy : selection.loads) {
    auto view = hivm::l1prefetch::accessView(copy.getSource());
    if (!view || copy->getParentOp() != payload ||
        !llvm::is_contained(selection.roles.inputs, view.getSource()) ||
        llvm::is_contained(pointers, view.getSource()))
      return handoffFailure(
          reason, "selected copies must use distinct input roles in the payload");
    pointers.push_back(view.getSource());
    Value target = copy.getTarget();
    if (auto sub = target.getDefiningOp<memref::SubViewOp>())
      target = sub.getSource();
    auto alloc = target.getDefiningOp<memref::AllocOp>();
    if (!alloc || alloc->getParentOp() != payload)
      return handoffFailure(
          reason, "selected allocation must be local to the payload");
    for (auto *user : alloc->getUsers())
      if (auto tensor = dyn_cast<bufferization::ToTensorOp>(user))
        if (tensor->getParentOp() != payload)
          return handoffFailure(
              reason, "selected tensor view must be local to the payload");
  }
  if (auto slice = selection.store.getSource().getDefiningOp<tensor::ExtractSliceOp>();
      slice && slice->getParentOp() != payload)
    return handoffFailure(reason, "output slice must be local to the payload");
  if (selection.schedule == AccessSchedule::Jagged)
    for (auto load : selection.metadata)
      if (load->getParentOp() != func)
        return handoffFailure(
            reason, "jagged metadata load must be directly inside the function");
  if (pointers.size() != selection.roles.inputs.size())
    return handoffFailure(
        reason, "selected copies do not cover every input role");
  auto padding = func.walk([&](Operation *op) -> WalkResult {
    if (op->hasAttr("l1_prefetch_ps.padding") && !isFillOnlyPaddingGuard(op, payload))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (padding.wasInterrupted())
    return handoffFailure(
        reason, "padding guard must contain only a fill and yield in the payload");
  return success();
}

FailureOr<Selection> readSelection(func::FuncOp func,
                                   hivm::l1prefetch::KernelLayout &layout,
                                   std::string &reason) {
  Selection selection;
  if (failed(collectSelection(func, selection, reason)))
    return failure();
  auto facts = hivm::l1prefetch::decodeAcceptedFacts(func, layout, &reason);
  if (failed(facts))
    return failure();
  if (facts->schedule != selection.schedule)
    return handoffFailure(
        reason, "accepted schedule disagrees with selection markers");
  selection.roles = facts->roles;
  // Physical handoff checks consume the proof; they do not recognize eligibility.
  if (failed(checkLoopNesting(func, selection, reason)) ||
      failed(checkPayload(func, selection, *facts, reason)))
    return failure();
  if (selection.schedule == AccessSchedule::Flat) {
    selection.flatBound = dyn_cast<BlockArgument>(selection.roles.size);
    if (!selection.flatBound)
      return handoffFailure(reason, "flat extent must name a block argument");
  } else if (selection.schedule == AccessSchedule::Nested) {
    selection.rowBound = selection.roles.rows;
  }
  return selection;
}

} // namespace

std::string mlir::hivm::l1PrefetchPSFingerprint(func::FuncOp func) {
  // OperationFingerPrint includes Operation*/Value addresses, so it is not
  // clone-stable. Hash canonical local-scope assembly (without locations)
  // instead. Only the digest itself is excluded; all selections and executable
  // IR bind it.
  auto prior = func->removeAttr(kProof);
  std::string text;
  llvm::raw_string_ostream stream(text);
  func->getParentOfType<ModuleOp>()->getAttrDictionary().print(stream);
  func.print(stream, OpPrintingFlags().useLocalScope().printGenericOpForm());
  // The meaning of an external call also depends on its declaration. Bind that
  // context so a changed type, purity attribute or body invalidates acceptance.
  for (auto other : func->getParentOfType<ModuleOp>().getOps<func::FuncOp>())
    if (other != func)
      other.print(stream,
                  OpPrintingFlags().useLocalScope().printGenericOpForm());
  if (prior)
    func->setAttr(kProof, prior);
  llvm::MD5 hash;
  hash.update(text);
  llvm::MD5::MD5Result result;
  hash.final(result);
  SmallString<32> digest;
  llvm::MD5::stringifyResult(result, digest);
  return digest.str().str();
}

namespace {
FailureOr<Selection> readAcceptedSelection(func::FuncOp func) {
  auto layout = hivm::l1prefetch::KernelLayout::get(func);
  auto proof = func->getAttrOfType<StringAttr>(kProof);
  std::string reason;
  if (!func->hasAttrOfType<UnitAttr>("l1_prefetch_ps.accepted"))
    reason = "missing acceptance marker";
  else if (!proof)
    reason = "missing proof fingerprint";
  else if (proof.getValue() != hivm::l1PrefetchPSFingerprint(func))
    reason = "proof fingerprint does not match the current IR";
  else if (!layout)
    reason = "invalid TTAdapter argument layout";
  else {
    auto selected = readSelection(func, *layout, reason);
    if (succeeded(selected))
      return selected;
  }
  func.emitError("L1 pipeline error: missing, stale or inconsistent accepted "
                 "analysis metadata: ")
      << reason;
  return failure();
}
} // namespace
LogicalResult mlir::hivm::verifyL1PrefetchPSAcceptance(func::FuncOp func) {
  return success(succeeded(readAcceptedSelection(func)));
}

namespace {
// These encoders transport proven facts; they do not derive new launch policy.
bool encodeScalarField(Builder &builder, NamedAttrList &attrs,
                       StringRef argumentKey, StringRef constantKey,
                       OpFoldResult source) {
  auto setI32 = [&](StringRef key, int64_t value) {
    attrs.set(key, builder.getI32IntegerAttr(value));
  };
  if (auto value = dyn_cast<Value>(source)) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      setI32(argumentKey, arg.getArgNumber());
      return true;
    }
    auto constant = hivm::l1prefetch::integer(value);
    if (!constant)
      return false;
    setI32(constantKey, *constant);
  } else {
    auto integer = dyn_cast<IntegerAttr>(cast<Attribute>(source));
    if (!integer)
      return false;
    setI32(constantKey, integer.getInt());
  }
  return true;
}

FailureOr<IntegerAttr> residentPairStride(func::FuncOp func) {
  auto target = hacc::utils::getTargetDevice(func->getParentOfType<ModuleOp>());
  if (!target)
    return failure();
  auto spec = hacc::utils::getNPUTargetSpecAttr(func.getContext(), *target,
                                              func.getLoc());
  auto cores = spec.getSpecForIdentifierEnum(hacc::DeviceSpec::CUBE_CORE_COUNT);
  if (!cores || !isa<IntegerAttr>(cores.getValue()))
    return failure();
  return cast<IntegerAttr>(cores.getValue());
}

LogicalResult emitPersistentFields(func::FuncOp func, const Selection &selection,
                                    Builder &builder, NamedAttrList &attrs) {
  auto setI32 = [&](StringRef key, int64_t value) {
    attrs.set(key, builder.getI32IntegerAttr(value));
  };
  if (!selection.runtimePersistent) {
    auto logicalPrograms = *hivm::l1prefetch::integer(selection.logicalPrograms);
    setI32("logical_programs", logicalPrograms);
    setI32("physical_programs", logicalPrograms / 2);
    setI32("persistent_steps", *hivm::l1prefetch::integer(selection.persistentSteps));
    setI32("row_tasks", *hivm::l1prefetch::integer(selection.rowTasks));
    return success();
  }
  setI32("persistent_runtime", 1);
  auto layout = hivm::l1prefetch::KernelLayout::get(func);
  auto encodeCount = [&](StringRef name, Value value) {
    if (layout)
      for (unsigned axis = 0; axis < 3; ++axis)
        if (value == layout->gridDim(axis)) {
          setI32((name + "_grid").str(), axis);
          return true;
        }
    return encodeScalarField(builder, attrs, (name + "_arg").str(),
                             ("static_" + name).str(), value);
  };
  if (!encodeCount("logical_programs", selection.logicalPrograms) ||
      !encodeCount("persistent_steps", selection.persistentSteps) ||
      !encodeCount("row_tasks", selection.rowTasks))
    return failure();
  // Persistent's partition theorem requires a static stride/column count.
  auto stride = getConstantIntValue(selection.roles.stride);
  if (!stride)
    return failure();
  setI32("row_stride", *stride);
  return success();
}

FailureOr<DictionaryAttr> getRealizationContract(func::FuncOp func,
                                                const Selection &selection) {
  Builder builder(func.getContext());
  NamedAttrList attrs;
  auto setI32 = [&](StringRef key, int64_t value) {
    attrs.set(key, builder.getI32IntegerAttr(value));
  };
  auto encodeScalar = [&](StringRef argumentKey, StringRef constantKey,
                          OpFoldResult source) {
    return encodeScalarField(builder, attrs, argumentKey, constantKey, source);
  };
  // Every realization carries explicit pointers in public ABI (bank) order.
  SmallVector<int32_t> inputs;
  for (auto arg : selection.roles.inputs)
    inputs.push_back(arg.getArgNumber());
  attrs.set("input_args", builder.getDenseI32ArrayAttr(inputs));
  setI32("output_arg", selection.roles.output.getArgNumber());
  if (failed(hivm::verifyL1PrefetchPSRealizationRoles(
          func, attrs.getDictionary(func.getContext()), func)))
    return failure();

  bool flat = selection.schedule == AccessSchedule::Flat;
  bool program = selection.schedule == AccessSchedule::Program;
  bool jagged = selection.schedule == AccessSchedule::Jagged;
  bool fixed = selection.schedule == AccessSchedule::Persistent &&
               !selection.runtimePersistent;
  if (!fixed) {
    if (flat)
      setI32("tile_elements", selection.tile.getDimSize(0));
    else {
      setI32("tile_rows", selection.tile.getDimSize(0));
      setI32("tile_cols", selection.tile.getDimSize(1));
    }
    setI32("element_bits", selection.tile.getElementTypeBitWidth());
  }
  if (flat || program || jagged) {
    auto stride = residentPairStride(func);
    if (failed(stride))
      return failure();
    setI32("pair_stride", stride->getInt());
  }

  switch (selection.schedule) {
  case AccessSchedule::Program:
    setI32("program_tile", 1);
    break;
  case AccessSchedule::Flat:
    setI32("flat_tile", 1);
    setI32("size_arg", selection.flatBound.getArgNumber());
    break;
  case AccessSchedule::Jagged:
    setI32("jagged_copy", 1);
    setI32("offsets_arg", selection.roles.offsets.getArgNumber());
    if (!encodeScalar("input_stride_arg", "static_input_stride",
                      selection.roles.inputStride) ||
        !encodeScalar("stride_arg", "row_stride", selection.roles.cols) ||
        !encodeScalar("matrix_stride_arg", "static_matrix_stride",
                      selection.roles.matrixStride))
      return failure();
    break;
  case AccessSchedule::Persistent:
    if (failed(emitPersistentFields(func, selection, builder, attrs)))
      return failure();
    break;
  case AccessSchedule::Nested:
    if (hivm::l1prefetch::nestedTileMode(selection.roles, selection.tile) ==
        hivm::l1prefetch::NestedTileMode::Partial)
      setI32("partial_tiles", 1);
    if (!encodeScalar("rows_arg", "static_rows", selection.rowBound) ||
        !encodeScalar("cols_arg", "static_cols", selection.roles.cols) ||
        !encodeScalar("stride_arg", "row_stride", selection.roles.stride))
      return failure();
    break;
  }
  return attrs.getDictionary(func.getContext());
}
} // namespace

FailureOr<DictionaryAttr>
mlir::hivm::getL1PrefetchPSLaunchContract(
    func::FuncOp func, hacc::HACCTargetDeviceSpecInterface targetSpec,
    std::string &rejectReason) {
  rejectReason.clear();
  auto selection = readAcceptedSelection(func);
  if (failed(selection))
    return failure();
  auto contract = getRealizationContract(func, *selection);
  if (failed(contract))
    return failure();
  // The driver supplies the same requested-target spec that AppendDeviceSpec
  // installs before physical planning. Do not trust a stale spec in input IR.
  auto reservation = l1prefetch::checkL1Reservation(
      selection->tile, selection->roles.inputs.size(), targetSpec);
  if (!reservation.succeeded()) {
    rejectReason = reservation.reason.str();
    return failure();
  }
  return contract;
}

FailureOr<DictionaryAttr>
mlir::hivm::getL1PrefetchPSExternalContract(func::FuncOp func) {
  auto selected = readAcceptedSelection(func);
  if (failed(selected))
    return failure();
  auto &selection = *selected;
  Builder b(func.getContext());
  auto pointerRole = [&](BlockArgument arg) -> DictionaryAttr {
    auto element = cast<MemRefType>(arg.getType()).getElementType();
    std::string type;
    if (element.isBF16())
      type = "*bf16";
    else if (element.isF16())
      type = "*fp16";
    else if (element.isF32())
      type = "*fp32";
    else if (auto integer = dyn_cast<IntegerType>(element))
      type = "*i" + std::to_string(integer.getWidth());
    else
      return {};
    return b.getDictionaryAttr(
        {b.getNamedAttr("arg", b.getI32IntegerAttr(arg.getArgNumber())),
         b.getNamedAttr("type", b.getStringAttr(type))});
  };
  auto scalarRole = [&](OpFoldResult value) -> DictionaryAttr {
    Attribute source;
    StringRef key;
    if (auto v = dyn_cast<Value>(value)) {
      if (auto arg = dyn_cast<BlockArgument>(v)) {
        auto layout = l1prefetch::KernelLayout::get(func);
        if (selection.runtimePersistent && layout)
          for (unsigned axis = 0; axis < 3; ++axis)
            if (v == layout->gridDim(axis))
              return b.getDictionaryAttr({
                  b.getNamedAttr("grid", b.getStringAttr(StringRef("xyz").substr(axis, 1))),
                  b.getNamedAttr("type", b.getStringAttr("i32"))});
        key = "arg";
        source = b.getI32IntegerAttr(arg.getArgNumber());
      } else if (auto constant = l1prefetch::integer(v)) {
        key = "const";
        source = b.getI64IntegerAttr(*constant);
      }
    } else if (auto attr = dyn_cast<Attribute>(value)) {
      key = "const";
      source = attr;
    }
    if (!source)
      return {};
    return b.getDictionaryAttr({b.getNamedAttr(key, source),
                                b.getNamedAttr("type", b.getStringAttr("i32"))});
  };
  NamedAttrList roles, launch, guard, tile;
  SmallVector<Attribute> inputs;
  for (auto input : selection.roles.inputs) {
    auto role = pointerRole(input);
    if (!role)
      return failure();
    inputs.push_back(role);
  }
  roles.set("inputs", b.getArrayAttr(inputs));
  auto output = pointerRole(selection.roles.output);
  if (!output)
    return failure();
  roles.set("output", output);
  auto addScalar = [&](StringRef name, OpFoldResult value) -> bool {
    auto role = scalarRole(value);
    if (!role)
      return false;
    roles.set(name, role);
    return true;
  };
  const bool flat = func->hasAttr("l1_prefetch_ps.flat");
  const bool program = func->hasAttr("l1_prefetch_ps.program");
  const bool jagged = func->hasAttr("l1_prefetch_ps.jagged");
  const bool persistent = !flat && !program && !jagged && !selection.inner;
  const bool fixed = persistent && !selection.runtimePersistent;
  StringRef distribution = fixed ? "fixed"
                                 : flat || program || jagged ? "resident_pairs" : "half_grid";
  launch.set("distribution", b.getStringAttr(distribution));
  launch.set("allow_odd_grid", b.getBoolAttr(flat || program));
  if (program)
    launch.set("logical_grid", b.getStringAttr("xyz"));
  if (fixed) {
    launch.set("logical_programs", b.getI32IntegerAttr(
        *l1prefetch::integer(selection.logicalPrograms)));
  } else if (flat || program || jagged) {
    auto stride = residentPairStride(func);
    if (failed(stride))
      return failure();
    launch.set("resident_stride", *stride);
  }
  if (program) {
    SmallVector<Attribute> strides;
    for (auto stride : selection.roles.inputStrides) {
      auto role = scalarRole(stride);
      if (!role)
        return failure();
      strides.push_back(role);
    }
    roles.set("input_strides", b.getArrayAttr(strides));
  } else if (flat) {
    if (!addScalar("size", selection.roles.size))
      return failure();
  } else if (jagged) {
    auto offsets = pointerRole(selection.roles.offsets);
    if (!offsets || !addScalar("input_stride", selection.roles.inputStride) ||
        !addScalar("stride", selection.roles.cols) ||
        !addScalar("matrix_stride", selection.roles.matrixStride))
      return failure();
    roles.set("offsets", offsets);
  } else if (selection.runtimePersistent) {
    if (!addScalar("logical_programs", selection.logicalPrograms) ||
        !addScalar("persistent_steps", selection.persistentSteps) ||
        !addScalar("row_tasks", selection.rowTasks) ||
        !addScalar("stride", selection.roles.stride))
      return failure();
  } else if (!fixed) {
    if (!addScalar("rows", selection.roles.rows) ||
        !addScalar("cols", selection.roles.cols) ||
        !addScalar("stride", selection.roles.stride))
      return failure();
  }
  tile.set("rows", b.getI64IntegerAttr(flat ? 1 : selection.tile.getDimSize(0)));
  tile.set("cols", b.getI64IntegerAttr(selection.tile.getDimSize(flat ? 0 : 1)));
  tile.set("element_bits", b.getI32IntegerAttr(selection.tile.getElementTypeBitWidth()));
  StringRef kind = selection.runtimePersistent ? "persistent_tiles" : fixed ? "fixed_programs"
                         : program ? "program_tiles" : flat ? "flat_tiles"
                                : jagged ? "metadata_bounded_tiles" : "tiles_2d";
  guard.set("kind", b.getStringAttr(kind));
  bool partial = flat || program || jagged ||
                 (!persistent && l1prefetch::nestedTileMode(selection.roles, selection.tile) ==
                                l1prefetch::NestedTileMode::Partial);
  guard.set("tile_mode", b.getStringAttr(partial ? "partial" : "full"));
  guard.set("tile", tile.getDictionary(func.getContext()));
  guard.set("roles", roles.getDictionary(func.getContext()));
  return b.getDictionaryAttr(
      {b.getNamedAttr("launch", launch.getDictionary(func.getContext())),
       b.getNamedAttr("guard", guard.getDictionary(func.getContext()))});
}

namespace {
// Normalize a loopless logical program to the same zero-based tile stream
// consumed by blockification. Resident normalization concatenates these
// one-tile programs; the materializer owns the shared prefetch/buffering
// schedule.
void normalizeFlat(func::FuncOp func, hivm::l1prefetch::KernelLayout &layout,
                   Selection selection, DictionaryAttr contract) {
  IRRewriter b(func.getContext());
  Location loc = func.getLoc();
  SmallVector<Operation *> original;
  for (Operation &op : func.getBody().front().without_terminator())
    if (!isa<arith::ConstantOp, memref::AllocOp, annotation::MarkOp>(op))
      original.push_back(&op);
  b.setInsertionPoint(func.getBody().front().getTerminator());
  Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = b.create<arith::ConstantIntOp>(loc, 1, 32);
  auto loop = b.create<scf::ForOp>(loc, zero, one, one);
  b.setInsertionPointToStart(loop.getBody());
  Value active;
  if (contract.contains("program_tile")) {
    for (unsigned axis = 0; axis < 3; ++axis) {
      Value inGrid = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, layout.programId(axis), layout.gridDim(axis));
      active = active ? b.create<arith::AndIOp>(loc, active, inGrid) : inGrid;
    }
  } else {
    Value pid = layout.programId(), grid = layout.gridDim();
    Value width = b.create<arith::ConstantIntOp>(
        loc, contract.getAs<IntegerAttr>("tile_elements").getInt(), 64);
    Value widePid = b.create<arith::ExtSIOp>(loc, b.getI64Type(), pid);
    Value offset = b.create<arith::MulIOp>(loc, widePid, width);
    Value inGrid = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, pid, grid);
    Value wideSize = b.create<arith::ExtSIOp>(loc, b.getI64Type(), selection.roles.size);
    Value inRange = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, offset, wideSize);
    active = b.create<arith::AndIOp>(loc, inGrid, inRange);
  }
  auto work = b.create<scf::IfOp>(loc, active, false);
  b.setInsertionPointToStart(&work.getThenRegion().front());
  IRMapping mapping;
  for (Operation *op : original) {
    if (op->hasAttr("l1_prefetch_ps.padding"))
      continue;
    if (op->hasAttr("l1_prefetch_ps.load")) {
      auto copy = cast<memref::CopyOp>(op);
      auto dst = copy.getTarget().getDefiningOp<memref::SubViewOp>();
      b.create<hivm::LoadOp>(
          loc, TypeRange{}, mapping.lookupOrDefault(copy.getSource()),
          dst.getSource(),
          hivm::PadModeAttr::get(func.getContext(), hivm::PadMode::PadNull),
          Value{});
    } else {
      bool selectedView = llvm::any_of(selection.loads, [&](memref::CopyOp copy) {
        return copy.getTarget().getDefiningOp() == op;
      });
      if (!selectedView)
        b.clone(*op, mapping);
    }
  }
  for (Operation *op : llvm::reverse(original))
    b.eraseOp(op);
}

// All rounds are bounded by the dense destination, independently of the two
// sequence lengths. Empty/truncated rows retain the original output padding.
// Payload predicates can differ; synchronization participation must not.
void normalizeJagged(func::FuncOp func, Selection selection) {
  IRRewriter b(func.getContext());
  Location loc = selection.outer.getLoc();
  Value rows;
  func.walk([&](arith::MinSIOp op) {
    if (op->hasAttr("l1_prefetch_ps.rows"))
      rows = op;
  });
  func.walk([&](memref::LoadOp op) {
    if (op->hasAttr("l1_prefetch_ps.metadata"))
      op->setAttr("program_split.metadata", b.getUnitAttr());
  });
  b.setInsertionPoint(selection.outer);
  Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = b.create<arith::ConstantIntOp>(loc, 1, 32);
  auto type = selection.tile;
  Value bm = b.create<arith::ConstantIntOp>(loc, type.getDimSize(0), 32);
  Value bn = b.create<arith::ConstantIntOp>(loc, type.getDimSize(1), 32);
  Value rowPad =
      b.create<arith::ConstantIntOp>(loc, type.getDimSize(0) - 1, 32);
  Value colPad =
      b.create<arith::ConstantIntOp>(loc, type.getDimSize(1) - 1, 32);
  Value denseRows =
      b.create<arith::DivSIOp>(loc, selection.roles.matrixStride, selection.roles.cols);
  Value columns = b.create<arith::DivSIOp>(
      loc, b.create<arith::AddIOp>(loc, selection.inner.getUpperBound(), colPad), bn);
  Value rowTiles = b.create<arith::DivSIOp>(
      loc, b.create<arith::AddIOp>(loc, denseRows, rowPad), bm);
  Value steps = b.create<arith::MulIOp>(loc, rowTiles, columns);
  auto loop = b.create<scf::ForOp>(loc, zero, steps, one);
  // The corresponding schema-2 launcher proves positive row/column task
  // counts and their non-overflowing product (tiles_2d / metadata_bounded_tiles).
  // Record only loop entry; do not infer ranges for arbitrary SSA factors.
  loop->setAttr(hivm::l1prefetch::kNonemptyTileStream, b.getUnitAttr());
  b.setInsertionPointToStart(loop.getBody());
  Value row = b.create<arith::MulIOp>(
      loc, bm, b.create<arith::DivSIOp>(loc, loop.getInductionVar(), columns));
  Value col = b.create<arith::MulIOp>(
      loc, bn, b.create<arith::RemSIOp>(loc, loop.getInductionVar(), columns));
  Value row64 = b.create<arith::ExtSIOp>(loc, b.getI64Type(), row);
  Value active =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, row64, rows);
  auto work = b.create<scf::IfOp>(loc, active, false);
  b.setInsertionPointToStart(&work.getThenRegion().front());
  IRMapping mapping;
  Value rowIndex = b.create<arith::IndexCastOp>(loc, b.getIndexType(), row);
  Value colIndex = b.create<arith::IndexCastOp>(loc, b.getIndexType(), col);
  mapping.map(selection.outer.getRegionIterArgs()[0], rowIndex);
  mapping.map(selection.outer.getInductionVar(),
              b.create<arith::AddIOp>(loc, selection.outer.getLowerBound(), row64));
  mapping.map(selection.inner.getInductionVar(), col);
  mapping.map(selection.inner.getRegionIterArgs()[0], colIndex);
  for (Operation &op : selection.outer.getBody()->without_terminator())
    if (&op != selection.inner.getOperation())
      b.clone(op, mapping);
  mapping.map(
      selection.inner.getRegionIterArgs()[1],
      b.create<arith::AddIOp>(
          loc, mapping.lookupOrDefault(selection.inner.getInitArgs()[1]), colIndex));
  auto load = selection.loads.front();
  auto dst = load.getTarget().getDefiningOp<memref::SubViewOp>();
  for (Operation &op : selection.inner.getBody()->without_terminator()) {
    if (op.hasAttr("l1_prefetch_ps.padding") || &op == dst.getOperation())
      continue;
    if (&op == load.getOperation()) {
      b.create<hivm::LoadOp>(
          loc, TypeRange{}, mapping.lookupOrDefault(load.getSource()),
          mapping.lookupOrDefault(dst.getSource()),
          hivm::PadModeAttr::get(func.getContext(), hivm::PadMode::PadNull),
          Value{});
    } else
      b.clone(op, mapping);
  }
  b.eraseOp(selection.outer);
}

void normalizeNested(func::FuncOp func, Selection selection) {
  IRRewriter b(func.getContext());
  Location loc = selection.outer.getLoc();
  b.setInsertionPoint(selection.outer);
  auto tile = selection.tile;
  Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = b.create<arith::ConstantIntOp>(loc, 1, 32);
  Value bm = b.create<arith::ConstantIntOp>(loc, tile.getDimSize(0), 32);
  Value bn = b.create<arith::ConstantIntOp>(loc, tile.getDimSize(1), 32);
  Value grid = selection.outer.getStep(), pid = selection.outer.getLowerBound();
  bool partialTiles = hivm::l1prefetch::nestedTileMode(selection.roles, selection.tile) ==
                      hivm::l1prefetch::NestedTileMode::Partial;
  Value rowBound = selection.rowBound;
  if (auto constant = hivm::l1prefetch::integer(rowBound))
    rowBound = b.create<arith::ConstantIntOp>(loc, *constant, 32);
  Value tasks = partialTiles ? selection.outer.getUpperBound()
                             : b.create<arith::DivSIOp>(loc, rowBound, bm);
  Value columnEnd = selection.inner.getUpperBound();
  if (partialTiles)
    columnEnd = b.create<arith::SubIOp>(
        loc, b.create<arith::AddIOp>(loc, columnEnd, bn), one);
  Value columns = b.create<arith::DivSIOp>(loc, columnEnd, bn);
  Value rounded = b.create<arith::SubIOp>(
      loc, b.create<arith::AddIOp>(loc, tasks, grid), one);
  Value rounds = b.create<arith::DivSIOp>(loc, rounded, grid);
  Value steps = b.create<arith::MulIOp>(loc, rounds, columns);
  auto loop = b.create<scf::ForOp>(loc, zero, steps, one);
  // The corresponding schema-2 launcher proves positive row/column task
  // counts and their non-overflowing product (tiles_2d / metadata_bounded_tiles).
  // Record only loop entry; do not infer ranges for arbitrary SSA factors.
  loop->setAttr(hivm::l1prefetch::kNonemptyTileStream, b.getUnitAttr());
  b.setInsertionPointToStart(loop.getBody());
  Value round = b.create<arith::DivSIOp>(loc, loop.getInductionVar(), columns);
  Value column = b.create<arith::RemSIOp>(loc, loop.getInductionVar(), columns);
  Value row = b.create<arith::AddIOp>(
      loc, pid, b.create<arith::MulIOp>(loc, round, grid));
  Value offset = b.create<arith::MulIOp>(loc, column, bn);
  Value active =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, row, tasks);
  auto work = b.create<scf::IfOp>(loc, active, false);
  b.setInsertionPointToStart(&work.getThenRegion().front());
  IRMapping mapping;
  mapping.map(selection.outer.getInductionVar(), row);
  mapping.map(selection.inner.getInductionVar(), offset);
  for (Operation &op : selection.outer.getBody()->without_terminator())
    if (&op != selection.inner.getOperation())
      b.clone(op, mapping);
  for (Operation &op : selection.inner.getBody()->without_terminator()) {
    if (op.hasAttr("l1_prefetch_ps.padding"))
      continue;
    if (partialTiles && llvm::any_of(selection.loads, [&](auto load) {
          return &op == load.getTarget().getDefiningOp();
        }))
      continue;
    if (!partialTiles && op.hasAttr("l1_prefetch_ps.full_tile")) {
      // Constant access bounds prove that this live extent is the full tile.
      mapping.map(op.getResult(0), mapping.lookupOrDefault(op.getOperand(0)));
      continue;
    }
    auto selected = llvm::find_if(selection.loads, [&](auto load) {
      return &op == load.getOperation();
    });
    if (partialTiles && selected != selection.loads.end()) {
      // Preserve the proved GM extent. The allocated tile remains the physical
      // destination. Per-input padding independence proves that unused edges
      // are unobservable; no fill is needed on cube-side L1 or vector-side UB.
      auto dst = selected->getTarget().getDefiningOp<memref::SubViewOp>();
      b.create<hivm::LoadOp>(
          loc, TypeRange{},
          mapping.lookupOrDefault(selected->getSource()),
          mapping.lookupOrDefault(dst.getSource()),
          hivm::PadModeAttr::get(func.getContext(), hivm::PadMode::PadNull),
          Value{});
      continue;
    }
    b.clone(op, mapping);
  }
  b.eraseOp(selection.outer);
}

struct L1PrefetchNormalizePass
    : impl::L1PrefetchNormalizeBase<L1PrefetchNormalizePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto func = hivm::getL1PrefetchPSEntry(module);
    if (!func) {
      module.emitError("L1 pipeline error: expected one accepted entry");
      return signalPassFailure();
    }
    auto selection = readAcceptedSelection(func);
    if (failed(selection))
      return signalPassFailure();
    auto contract = getRealizationContract(func, *selection);
    if (failed(contract))
      return signalPassFailure();
    auto layout = hivm::l1prefetch::KernelLayout::get(func);
    if (!layout)
      return signalPassFailure();
    Selection accepted = *selection;
    // Every accepted launcher checks positive grid dimensions and a bounded
    // grid product. Preserve their verified ABI identities through ID lowering.
    for (unsigned axis = 0; axis < 3; ++axis)
      func.setArgAttr(layout->gridDim(axis).getArgNumber(),
                      hivm::l1prefetch::kLaunchGridAxis,
                      IntegerAttr::get(IntegerType::get(module.getContext(), 32), axis));
    if (func->hasAttr("l1_prefetch_ps.flat") ||
        func->hasAttr("l1_prefetch_ps.program"))
      normalizeFlat(func, *layout, accepted, *contract);
    else if (func->hasAttr("l1_prefetch_ps.jagged"))
      normalizeJagged(func, accepted);
    else if (accepted.inner)
      normalizeNested(func, accepted);
    // Retire proof annotations; normal lowering changes their premises. The
    // selected-input value mark remains until physical staging consumes it.
    func.walk([&](Operation *op) {
      SmallVector<StringAttr> names;
      for (NamedAttribute attr : op->getAttrs())
        if (attr.getName().strref().starts_with("l1_prefetch_ps."))
          names.push_back(attr.getName());
      for (auto name : names)
        op->removeAttr(name);
    });
    func->setAttr("program_split.transformed",
                  UnitAttr::get(module.getContext()));
    func->setAttr("program_split.launch", *contract);
    func->setAttr(hivm::TFuncCoreTypeAttr::name,
                  hivm::TFuncCoreTypeAttr::get(module.getContext(),
                                               hivm::TFuncCoreType::MIX));
    func->setAttr("mix_mode", StringAttr::get(module.getContext(), "mix"));
    module->setAttr("hivm.disable_auto_tile_and_bind_subblock",
                    UnitAttr::get(module.getContext()));
  }
};
} // namespace
std::unique_ptr<Pass> mlir::hivm::createL1PrefetchNormalizePass() {
  return std::make_unique<L1PrefetchNormalizePass>();
}
