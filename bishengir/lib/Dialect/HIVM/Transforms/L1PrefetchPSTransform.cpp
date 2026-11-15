//===- L1PrefetchPSTransform.cpp - Consume accepted structured IR --------===//
#include "L1PrefetchPSRoles.h"
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
#define GEN_PASS_DEF_L1PREFETCHPSTRANSFORM
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
struct Selection {
  scf::ForOp outer, inner;
  SmallVector<memref::CopyOp, 2> loads;
  arith::CmpIOp active;
  MemRefType tile;
  Value rowBound;
  BlockArgument flatBound;
  hivm::l1prefetch::AccessRoles roles;
  int64_t logicalPrograms = 0, persistentSteps = 0, rowTasks = 0;
};
FailureOr<Selection> readSelection(func::FuncOp func,
                                   hivm::l1prefetch::KernelLayout &layout) {
  Selection s;
  unsigned outer = 0, inner = 0, load = 0, active = 0;
  bool malformed = false;
  func.walk([&](Operation *op) {
    if (op->hasAttr("l1_prefetch_ps.outer")) {
      ++outer;
      s.outer = dyn_cast<scf::ForOp>(op);
      malformed |= !s.outer;
    }
    if (op->hasAttr("l1_prefetch_ps.inner")) {
      ++inner;
      s.inner = dyn_cast<scf::ForOp>(op);
      malformed |= !s.inner;
    }
    if (op->hasAttr("l1_prefetch_ps.load")) {
      ++load;
      auto copy = dyn_cast<memref::CopyOp>(op);
      malformed |= !copy;
      if (copy)
        s.loads.push_back(copy);
    }
    if (op->hasAttr("l1_prefetch_ps.active")) {
      ++active;
      s.active = dyn_cast<arith::CmpIOp>(op);
      malformed |= !s.active;
    }
  });
  bool flat = func->hasAttr("l1_prefetch_ps.flat");
  if (malformed || outer != (flat ? 0u : 1u) || load < 1 ||
      (!inner && !flat && load > 1u) || inner > 1 ||
      active != ((inner || flat) ? 0 : 1))
    return failure();
  for (auto loadOp : s.loads) {
    Value target = loadOp.getTarget();
    if (auto slice = target.getDefiningOp<memref::SubViewOp>())
      target = slice.getSource();
    auto type = dyn_cast<MemRefType>(target.getType());
    if (!type || (s.tile && s.tile != type))
      return failure();
    s.tile = type;
  }
  bufferization::MaterializeInDestinationOp store;
  SmallVector<memref::LoadOp> metadata;
  unsigned stores = 0;
  func.walk([&](bufferization::MaterializeInDestinationOp op) {
    ++stores;
    store = op;
  });
  func.walk([&](memref::LoadOp op) { metadata.push_back(op); });
  if (stores != 1)
    return failure();
  using hivm::l1prefetch::AccessSchedule;
  auto schedule = flat ? AccessSchedule::Flat
                  : func->hasAttr("l1_prefetch_ps.jagged")
                      ? AccessSchedule::Jagged
                  : s.inner ? AccessSchedule::Nested
                            : AccessSchedule::Persistent;
  if (!hivm::l1prefetch::deriveAccessRoles(func, layout, schedule, s.loads,
                                           store, s.outer, s.inner, metadata,
                                           s.tile, s.roles)
           .empty())
    return failure();
  if (flat) {
    s.flatBound = dyn_cast<BlockArgument>(s.roles.size);
    if (!s.flatBound)
      return failure();
  } else if (schedule == AccessSchedule::Nested) {
    s.rowBound = s.roles.rows;
  } else if (!flat && !s.inner) {
    auto row = s.active.getLhs().getDefiningOp<arith::AddIOp>();
    if (!row)
      return failure();
    Value pid = layout.programId();
    Value other = row.getLhs() == pid ? row.getRhs() : row.getLhs();
    auto mul = other.getDefiningOp<arith::MulIOp>();
    if (!mul)
      return failure();
    auto logical = hivm::l1prefetch::integer(mul.getLhs());
    if (!logical)
      logical = hivm::l1prefetch::integer(mul.getRhs());
    auto steps = hivm::l1prefetch::integer(s.outer.getUpperBound());
    auto tasks = hivm::l1prefetch::integer(s.active.getRhs());
    if (!logical || !steps || !tasks)
      return failure();
    s.logicalPrograms = *logical;
    s.persistentSteps = *steps;
    s.rowTasks = *tasks;
  }
  return s;
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

LogicalResult mlir::hivm::verifyL1PrefetchPSAcceptance(func::FuncOp func) {
  auto layout = l1prefetch::KernelLayout::get(func);
  auto proof = func->getAttrOfType<StringAttr>(kProof);
  if (!func->hasAttrOfType<UnitAttr>("l1_prefetch_ps.accepted") || !proof ||
      proof.getValue() != l1PrefetchPSFingerprint(func) || !layout ||
      failed(readSelection(func, *layout)))
    return func.emitError("L1 pipeline error: missing, stale or inconsistent "
                          "accepted analysis metadata");
  return success();
}

namespace {
FailureOr<DictionaryAttr> getRealizationContract(func::FuncOp func) {
  using namespace mlir::hivm;
  if (failed(verifyL1PrefetchPSAcceptance(func)))
    return failure();
  auto layout = l1prefetch::KernelLayout::get(func);
  if (!layout)
    return failure();
  auto selection = readSelection(func, *layout);
  if (failed(selection))
    return failure();
  Selection s = *selection;
  Builder b(func.getContext());
  NamedAttrList attrs;
  auto put = [&](StringRef name, int64_t value) {
    attrs.set(name, b.getI32IntegerAttr(value));
  };
  // Every realization carries the same explicit input-list and output roles.
  // Selection identity, including bank order, follows public ABI order.
  SmallVector<int32_t> inputs;
  for (auto arg : s.roles.inputs)
    inputs.push_back(arg.getArgNumber());
  attrs.set("input_args", b.getDenseI32ArrayAttr(inputs));
  put("output_arg", s.roles.output.getArgNumber());
  // The access proof already establishes these facts. Validate the emitted
  // representation too, before it crosses the pristine/lowered ABI boundary.
  if (failed(verifyL1PrefetchPSRealizationRoles(
          func, attrs.getDictionary(func.getContext()), func)))
    return failure();
  auto scalar = [&](StringRef argumentKey, StringRef constantKey,
                    OpFoldResult value) -> bool {
    if (auto v = dyn_cast<Value>(value)) {
      if (auto arg = dyn_cast<BlockArgument>(v)) {
        put(argumentKey, arg.getArgNumber());
        return true;
      }
      auto constant = l1prefetch::integer(v);
      if (!constant)
        return false;
      put(constantKey, *constant);
    } else {
      auto integer = dyn_cast<IntegerAttr>(cast<Attribute>(value));
      if (!integer)
        return false;
      put(constantKey, integer.getInt());
    }
    return true;
  };
  if (func->hasAttr("l1_prefetch_ps.flat")) {
    auto type = s.tile;
    put("flat_tile", 1);
    put("tile_elements", type.getDimSize(0));
    put("element_bits", type.getElementTypeBitWidth());
    put("size_arg", s.flatBound.getArgNumber());
    auto target =
        hacc::utils::getTargetDevice(func->getParentOfType<ModuleOp>());
    if (!target)
      return failure();
    auto spec = hacc::utils::getNPUTargetSpecAttr(func.getContext(), *target,
                                                  func.getLoc());
    auto cores =
        spec.getSpecForIdentifierEnum(hacc::DeviceSpec::CUBE_CORE_COUNT);
    if (!cores || !isa<IntegerAttr>(cores.getValue()))
      return failure();
    put("pair_stride", cast<IntegerAttr>(cores.getValue()).getInt());
    return attrs.getDictionary(func.getContext());
  }
  if (func->hasAttr("l1_prefetch_ps.jagged")) {
    auto target =
        hacc::utils::getTargetDevice(func->getParentOfType<ModuleOp>());
    if (!target)
      return failure();
    auto spec = hacc::utils::getNPUTargetSpecAttr(func.getContext(), *target,
                                                  func.getLoc());
    auto cores =
        spec.getSpecForIdentifierEnum(hacc::DeviceSpec::CUBE_CORE_COUNT);
    if (!cores || !isa<IntegerAttr>(cores.getValue()))
      return failure();
    put("pair_stride", cast<IntegerAttr>(cores.getValue()).getInt());
    put("jagged_copy", 1);
    put("tile_rows", s.tile.getDimSize(0));
    put("tile_cols", s.tile.getDimSize(1));
    put("element_bits", s.tile.getElementTypeBitWidth());
    put("offsets_arg", s.roles.offsets.getArgNumber());
    if (!scalar("input_stride_arg", "static_input_stride", s.roles.inputStride) ||
        !scalar("stride_arg", "row_stride", s.roles.cols) ||
        !scalar("matrix_stride_arg", "static_matrix_stride", s.roles.matrixStride))
      return failure();
    return attrs.getDictionary(func.getContext());
  }
  if (!s.inner) {
    put("logical_programs", s.logicalPrograms);
    put("physical_programs", s.logicalPrograms / 2);
    put("persistent_steps", s.persistentSteps);
    put("row_tasks", s.rowTasks);
  } else {
    MemRefType tile = s.tile;
    put("tile_rows", tile.getDimSize(0));
    put("tile_cols", tile.getDimSize(1));
    put("element_bits", tile.getElementTypeBitWidth());
    if (l1prefetch::nestedTileMode(s.roles, s.tile) ==
        l1prefetch::NestedTileMode::Partial)
      put("partial_tiles", 1);
    if (!scalar("rows_arg", "static_rows", s.rowBound) ||
        !scalar("cols_arg", "static_cols", s.roles.cols) ||
        !scalar("stride_arg", "row_stride", s.roles.stride))
      return failure();
  }
  return attrs.getDictionary(func.getContext());
}
} // namespace

FailureOr<DictionaryAttr>
mlir::hivm::getL1PrefetchPSLaunchContract(
    func::FuncOp func, hacc::HACCTargetDeviceSpecInterface targetSpec,
    std::string &rejectReason) {
  rejectReason.clear();
  auto contract = getRealizationContract(func);
  if (failed(contract))
    return failure();
  auto layout = l1prefetch::KernelLayout::get(func);
  if (!layout)
    return failure();
  auto selection = readSelection(func, *layout);
  if (failed(selection))
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
  if (failed(verifyL1PrefetchPSAcceptance(func)))
    return failure();
  auto layout = l1prefetch::KernelLayout::get(func);
  if (!layout)
    return failure();
  auto selected = readSelection(func, *layout);
  if (failed(selected))
    return failure();
  auto &s = *selected;
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
  for (auto input : s.roles.inputs) {
    auto role = pointerRole(input);
    if (!role)
      return failure();
    inputs.push_back(role);
  }
  roles.set("inputs", b.getArrayAttr(inputs));
  auto output = pointerRole(s.roles.output);
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
  const bool jagged = func->hasAttr("l1_prefetch_ps.jagged");
  const bool fixed = !flat && !jagged && !s.inner;
  StringRef distribution = fixed ? "fixed"
                                 : flat || jagged ? "resident_pairs" : "half_grid";
  launch.set("distribution", b.getStringAttr(distribution));
  launch.set("allow_odd_grid", b.getBoolAttr(flat));
  if (fixed) {
    launch.set("logical_programs", b.getI32IntegerAttr(s.logicalPrograms));
  } else if (flat || jagged) {
    auto target = hacc::utils::getTargetDevice(func->getParentOfType<ModuleOp>());
    if (!target)
      return failure();
    auto spec = hacc::utils::getNPUTargetSpecAttr(func.getContext(), *target,
                                                func.getLoc());
    auto cores = spec.getSpecForIdentifierEnum(hacc::DeviceSpec::CUBE_CORE_COUNT);
    if (!cores || !isa<IntegerAttr>(cores.getValue()))
      return failure();
    launch.set("resident_stride", cores.getValue());
  }
  if (flat) {
    if (!addScalar("size", s.roles.size))
      return failure();
  } else if (jagged) {
    auto offsets = pointerRole(s.roles.offsets);
    if (!offsets || !addScalar("input_stride", s.roles.inputStride) ||
        !addScalar("stride", s.roles.cols) ||
        !addScalar("matrix_stride", s.roles.matrixStride))
      return failure();
    roles.set("offsets", offsets);
  } else if (!fixed) {
    if (!addScalar("rows", s.roles.rows) || !addScalar("cols", s.roles.cols) ||
        !addScalar("stride", s.roles.stride))
      return failure();
  }
  tile.set("rows", b.getI64IntegerAttr(flat ? 1 : s.tile.getDimSize(0)));
  tile.set("cols", b.getI64IntegerAttr(s.tile.getDimSize(flat ? 0 : 1)));
  tile.set("element_bits", b.getI32IntegerAttr(s.tile.getElementTypeBitWidth()));
  StringRef kind = fixed ? "fixed_programs"
                         : flat ? "flat_tiles"
                                : jagged ? "metadata_bounded_tiles" : "tiles_2d";
  guard.set("kind", b.getStringAttr(kind));
  bool partial = flat || jagged ||
                 (!fixed && l1prefetch::nestedTileMode(s.roles, s.tile) ==
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
                   Selection s, DictionaryAttr contract) {
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
  Value pid = layout.programId();
  Value grid = layout.gridDim();
  Value width = b.create<arith::ConstantIntOp>(
      loc, contract.getAs<IntegerAttr>("tile_elements").getInt(), 64);
  Value widePid = b.create<arith::ExtSIOp>(loc, b.getI64Type(), pid);
  Value offset = b.create<arith::MulIOp>(loc, widePid, width);
  Value size = s.roles.size;
  Value inGrid =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, pid, grid);
  Value wideSize = b.create<arith::ExtSIOp>(loc, b.getI64Type(), size);
  Value inRange =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, offset, wideSize);
  Value active = b.create<arith::AndIOp>(loc, inGrid, inRange);
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
      bool selectedView = llvm::any_of(s.loads, [&](memref::CopyOp copy) {
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
void normalizeJagged(func::FuncOp func, Selection s) {
  IRRewriter b(func.getContext());
  Location loc = s.outer.getLoc();
  Value rows;
  func.walk([&](arith::MinSIOp op) {
    if (op->hasAttr("l1_prefetch_ps.rows"))
      rows = op;
  });
  func.walk([&](memref::LoadOp op) {
    if (op->hasAttr("l1_prefetch_ps.metadata"))
      op->setAttr("program_split.metadata", b.getUnitAttr());
  });
  b.setInsertionPoint(s.outer);
  Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = b.create<arith::ConstantIntOp>(loc, 1, 32);
  auto type = s.tile;
  Value bm = b.create<arith::ConstantIntOp>(loc, type.getDimSize(0), 32);
  Value bn = b.create<arith::ConstantIntOp>(loc, type.getDimSize(1), 32);
  Value rowPad =
      b.create<arith::ConstantIntOp>(loc, type.getDimSize(0) - 1, 32);
  Value colPad =
      b.create<arith::ConstantIntOp>(loc, type.getDimSize(1) - 1, 32);
  Value denseRows =
      b.create<arith::DivSIOp>(loc, s.roles.matrixStride, s.roles.cols);
  Value columns = b.create<arith::DivSIOp>(
      loc, b.create<arith::AddIOp>(loc, s.inner.getUpperBound(), colPad), bn);
  Value rowTiles = b.create<arith::DivSIOp>(
      loc, b.create<arith::AddIOp>(loc, denseRows, rowPad), bm);
  Value steps = b.create<arith::MulIOp>(loc, rowTiles, columns);
  auto loop = b.create<scf::ForOp>(loc, zero, steps, one);
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
  mapping.map(s.outer.getRegionIterArgs()[0], rowIndex);
  mapping.map(s.outer.getInductionVar(),
              b.create<arith::AddIOp>(loc, s.outer.getLowerBound(), row64));
  mapping.map(s.inner.getInductionVar(), col);
  mapping.map(s.inner.getRegionIterArgs()[0], colIndex);
  for (Operation &op : s.outer.getBody()->without_terminator())
    if (&op != s.inner.getOperation())
      b.clone(op, mapping);
  mapping.map(
      s.inner.getRegionIterArgs()[1],
      b.create<arith::AddIOp>(
          loc, mapping.lookupOrDefault(s.inner.getInitArgs()[1]), colIndex));
  auto load = s.loads.front();
  auto dst = load.getTarget().getDefiningOp<memref::SubViewOp>();
  for (Operation &op : s.inner.getBody()->without_terminator()) {
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
  b.eraseOp(s.outer);
}

void normalizeNested(func::FuncOp func, Selection s) {
  IRRewriter b(func.getContext());
  Location loc = s.outer.getLoc();
  b.setInsertionPoint(s.outer);
  auto tile = s.tile;
  Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = b.create<arith::ConstantIntOp>(loc, 1, 32);
  Value bm = b.create<arith::ConstantIntOp>(loc, tile.getDimSize(0), 32);
  Value bn = b.create<arith::ConstantIntOp>(loc, tile.getDimSize(1), 32);
  Value grid = s.outer.getStep(), pid = s.outer.getLowerBound();
  bool partialTiles = hivm::l1prefetch::nestedTileMode(s.roles, s.tile) ==
                      hivm::l1prefetch::NestedTileMode::Partial;
  Value rowBound = s.rowBound;
  if (auto constant = hivm::l1prefetch::integer(rowBound))
    rowBound = b.create<arith::ConstantIntOp>(loc, *constant, 32);
  Value tasks = partialTiles ? s.outer.getUpperBound()
                             : b.create<arith::DivSIOp>(loc, rowBound, bm);
  Value columnEnd = s.inner.getUpperBound();
  if (partialTiles)
    columnEnd = b.create<arith::SubIOp>(
        loc, b.create<arith::AddIOp>(loc, columnEnd, bn), one);
  Value columns = b.create<arith::DivSIOp>(loc, columnEnd, bn);
  Value rounded = b.create<arith::SubIOp>(
      loc, b.create<arith::AddIOp>(loc, tasks, grid), one);
  Value rounds = b.create<arith::DivSIOp>(loc, rounded, grid);
  Value steps = b.create<arith::MulIOp>(loc, rounds, columns);
  auto loop = b.create<scf::ForOp>(loc, zero, steps, one);
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
  mapping.map(s.outer.getInductionVar(), row);
  mapping.map(s.inner.getInductionVar(), offset);
  for (Operation &op : s.outer.getBody()->without_terminator())
    if (&op != s.inner.getOperation())
      b.clone(op, mapping);
  for (Operation &op : s.inner.getBody()->without_terminator()) {
    if (op.hasAttr("l1_prefetch_ps.padding"))
      continue;
    if (partialTiles && llvm::any_of(s.loads, [&](auto load) {
          return &op == load.getTarget().getDefiningOp();
        }))
      continue;
    if (!partialTiles && op.hasAttr("l1_prefetch_ps.full_tile")) {
      // Constant access bounds prove that this live extent is the full tile.
      mapping.map(op.getResult(0), mapping.lookupOrDefault(op.getOperand(0)));
      continue;
    }
    auto selected = llvm::find_if(s.loads, [&](auto load) {
      return &op == load.getOperation();
    });
    if (partialTiles && selected != s.loads.end()) {
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
  b.eraseOp(s.outer);
}

struct L1PrefetchPSTransformPass
    : impl::L1PrefetchPSTransformBase<L1PrefetchPSTransformPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto func = hivm::getL1PrefetchPSEntry(module);
    if (!func) {
      module.emitError("L1 pipeline error: expected one accepted entry");
      return signalPassFailure();
    }
    auto contract = getRealizationContract(func);
    if (failed(contract))
      return signalPassFailure();
    auto layout = hivm::l1prefetch::KernelLayout::get(func);
    if (!layout)
      return signalPassFailure();
    auto selection = readSelection(func, *layout);
    if (failed(selection))
      return signalPassFailure();
    Selection s = *selection;
    if (func->hasAttr("l1_prefetch_ps.flat"))
      normalizeFlat(func, *layout, s, *contract);
    else if (func->hasAttr("l1_prefetch_ps.jagged"))
      normalizeJagged(func, s);
    else if (s.inner)
      normalizeNested(func, s);
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
std::unique_ptr<Pass> mlir::hivm::createL1PrefetchPSTransformPass() {
  return std::make_unique<L1PrefetchPSTransformPass>();
}
