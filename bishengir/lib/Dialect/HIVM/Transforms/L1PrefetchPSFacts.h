// Compact proof results on the accepted, digest-verified pristine function.
#ifndef BISHENGIR_L1PREFETCHPS_FACTS_H
#define BISHENGIR_L1PREFETCHPS_FACTS_H

#include "L1PrefetchPSRoles.h"
#include "llvm/ADT/StringMap.h"
#include <limits>

namespace mlir::hivm::l1prefetch {
constexpr StringLiteral kAcceptedFacts = "l1_prefetch_ps.facts";
constexpr StringLiteral kValueRoles = "l1_prefetch_ps.value_roles";

struct AcceptedFacts {
  AccessSchedule schedule;
  AccessRoles roles;
  Value logicalPrograms, persistentSteps, rowTasks;
};

// The producer and reader share the field set; decoding still independently
// checks types and sources after clone/print/parse.
inline SmallVector<StringRef> acceptedFactFields(AccessSchedule schedule) {
  SmallVector<StringRef> fields{"version", "schedule", "inputs", "output"};
  switch (schedule) {
  case AccessSchedule::Persistent:
    fields.append({"stride", "logical_programs", "persistent_steps", "row_tasks"});
    break;
  case AccessSchedule::Nested:
    fields.append({"rows", "cols", "stride"});
    break;
  case AccessSchedule::Flat:
    fields.push_back("size");
    break;
  case AccessSchedule::Jagged:
    fields.append({"offsets", "cols", "input_stride", "matrix_stride", "stride"});
    break;
  case AccessSchedule::Program:
    fields.push_back("input_strides");
    break;
  }
  return fields;
}

// A source-category check, not proof that a recorded N equals the address
// coefficient. Analysis establishes that relationship before publication.
inline bool persistentCountSource(Value value, KernelLayout &layout) {
  if (!value || !value.getType().isInteger(32))
    return false;
  if (auto constant = integer(value))
    return *constant > 0 && *constant <= std::numeric_limits<int32_t>::max();
  return layout.isPublic(value) || value == layout.gridDim(0) ||
         value == layout.gridDim(1) || value == layout.gridDim(2);
}

// Called only after the schedule's complete proof. References do not introduce
// SSA uses or new constants; clone/print/parse preserve the original Values.
inline LogicalResult publishAcceptedFacts(func::FuncOp func,
                                          const AcceptedFacts &facts) {
  SmallVector<std::pair<StringRef, OpFoldResult>> values{
      {"output", facts.roles.output}, {"offsets", facts.roles.offsets},
      {"rows", facts.roles.rows}, {"cols", facts.roles.cols},
      {"size", facts.roles.size}, {"stride", facts.roles.stride},
      {"input_stride", facts.roles.inputStride},
      {"matrix_stride", facts.roles.matrixStride},
      {"logical_programs", facts.logicalPrograms},
      {"persistent_steps", facts.persistentSteps}, {"row_tasks", facts.rowTasks}};
  auto required = acceptedFactFields(facts.schedule);
  SmallVector<StringRef> present{"version", "schedule", "inputs"};
  for (auto [name, value] : values)
    if (value)
      present.push_back(name);
  if (!facts.roles.inputStrides.empty())
    present.push_back("input_strides");
  // Check everything before encode() attaches any result IDs. An erroneous
  // producer must never leave a partially published analysis record.
  if (facts.roles.inputs.empty() || present.size() != required.size() ||
      llvm::any_of(required, [&](StringRef name) {
        return !llvm::is_contained(present, name);
      }) || (facts.schedule == AccessSchedule::Program &&
             facts.roles.inputStrides.size() != facts.roles.inputs.size()))
    return failure();
  if (facts.schedule == AccessSchedule::Persistent) {
    auto layout = KernelLayout::get(func);
    if (!layout || !persistentCountSource(facts.logicalPrograms, *layout) ||
        !persistentCountSource(facts.persistentSteps, *layout) ||
        !persistentCountSource(facts.rowTasks, *layout) ||
        !getConstantIntValue(facts.roles.stride))
      return failure();
  }
  Builder b(func.getContext());
  NamedAttrList fields;
  unsigned next = 0;
  DenseMap<Value, Attribute> references;
  auto encode = [&](OpFoldResult source) -> Attribute {
    if (auto literal = dyn_cast<Attribute>(source))
      return b.getDictionaryAttr({b.getNamedAttr("literal", literal)});
    Value value = cast<Value>(source);
    if (auto arg = dyn_cast<BlockArgument>(value))
      return b.getDictionaryAttr({b.getNamedAttr(
          "arg", b.getI32IntegerAttr(arg.getArgNumber()))});
    auto found = references.find(value);
    if (found != references.end())
      return found->second;
    auto result = cast<OpResult>(value);
    auto id = std::to_string(next++);
    Operation *op = result.getOwner();
    NamedAttrList ids(op->getAttrOfType<DictionaryAttr>(kValueRoles));
    ids.set(id, b.getI32IntegerAttr(result.getResultNumber()));
    op->setAttr(kValueRoles, ids.getDictionary(func.getContext()));
    auto ref = b.getDictionaryAttr({b.getNamedAttr("result", b.getStringAttr(id))});
    references[value] = ref;
    return ref;
  };
  fields.set("version", b.getI32IntegerAttr(1));
  fields.set("schedule", b.getI32IntegerAttr(int(facts.schedule)));
  SmallVector<int32_t> inputs;
  for (auto input : facts.roles.inputs)
    inputs.push_back(input.getArgNumber());
  fields.set("inputs", b.getDenseI32ArrayAttr(inputs));
  for (auto [name, value] : values)
    if (value)
      fields.set(name, encode(value));
  if (!facts.roles.inputStrides.empty()) {
    SmallVector<Attribute> strides;
    for (auto stride : facts.roles.inputStrides)
      strides.push_back(encode(stride));
    fields.set("input_strides", b.getArrayAttr(strides));
  }
  func->setAttr(kAcceptedFacts, fields.getDictionary(func.getContext()));
  return success();
}

inline FailureOr<AcceptedFacts> decodeAcceptedFacts(func::FuncOp func,
                                                    KernelLayout &layout) {
  auto fields = func->getAttrOfType<DictionaryAttr>(kAcceptedFacts);
  if (!fields)
    return failure();
  auto version = fields.getAs<IntegerAttr>("version");
  auto kind = fields.getAs<IntegerAttr>("schedule");
  auto inputs = fields.getAs<DenseI32ArrayAttr>("inputs");
  if (!version || !version.getType().isInteger(32) || version.getInt() != 1 ||
      !kind || !kind.getType().isInteger(32) || kind.getInt() < 0 ||
      kind.getInt() > int(AccessSchedule::Program) || !inputs || inputs.empty())
    return failure();
  AcceptedFacts facts{AccessSchedule(kind.getInt()), {}, {}, {}, {}};
  llvm::StringMap<Value> results;
  bool malformed = false;
  func.walk([&](Operation *op) {
    auto attr = op->getAttr(kValueRoles);
    if (!attr)
      return;
    auto ids = dyn_cast<DictionaryAttr>(attr);
    if (!ids) {
      malformed = true;
      return;
    }
    for (auto id : ids) {
      auto index = dyn_cast<IntegerAttr>(id.getValue());
      if (!index || !index.getType().isInteger(32) || index.getInt() < 0 ||
          uint64_t(index.getInt()) >= op->getNumResults() ||
          !results.try_emplace(id.getName().strref(),
                               op->getResult(index.getInt())).second)
        malformed = true;
    }
  });
  llvm::SmallDenseSet<StringRef> used;
  auto decode = [&](Attribute attr) -> OpFoldResult {
    auto ref = dyn_cast_or_null<DictionaryAttr>(attr);
    if (!ref || ref.size() != 1)
      return {};
    if (auto arg = ref.getAs<IntegerAttr>("arg")) {
      if (!arg.getType().isInteger(32) || arg.getInt() < 0 ||
          uint64_t(arg.getInt()) >= func.getNumArguments())
        return {};
      return func.getArgument(arg.getInt());
    }
    if (auto literal = ref.getAs<IntegerAttr>("literal"))
      return literal;
    if (auto result = ref.getAs<StringAttr>("result")) {
      auto found = results.find(result.getValue());
      if (found != results.end()) {
        used.insert(result.getValue());
        return found->second;
      }
    }
    return {};
  };
  auto value = [&](StringRef name) -> Value {
    return dyn_cast_if_present<Value>(decode(fields.get(name)));
  };
  auto pointer = [&](Value v) -> BlockArgument {
    auto arg = dyn_cast_or_null<BlockArgument>(v);
    auto type = arg ? dyn_cast<MemRefType>(arg.getType()) : MemRefType();
    return type && type.getRank() == 1 && type.isDynamicDim(0) && layout.isPublic(v)
               ? arg : BlockArgument();
  };
  int32_t previous = -1;
  for (int32_t index : inputs.asArrayRef()) {
    if (index <= previous || index < 0 || unsigned(index) >= func.getNumArguments())
      return failure();
    auto arg = pointer(func.getArgument(index));
    if (!arg)
      return failure();
    facts.roles.inputs.push_back(arg);
    previous = index;
  }
  facts.roles.output = pointer(value("output"));
  if (!facts.roles.output || llvm::is_contained(facts.roles.inputs, facts.roles.output))
    return failure();
  auto required = acceptedFactFields(facts.schedule);
  auto scalar = [&](StringRef name, Value &dst) {
    dst = value(name);
    malformed |= !dst || !dst.getType().isInteger(32);
  };
  auto stride = [&](StringRef name, OpFoldResult &dst) {
    dst = decode(fields.get(name));
    if (auto v = dyn_cast_if_present<Value>(dst))
      malformed |= !v.getType().isIntOrIndex();
    else if (auto a = dyn_cast_if_present<Attribute>(dst))
      malformed |= !isa<IntegerAttr>(a);
    else
      malformed = true;
  };
  switch (facts.schedule) {
  case AccessSchedule::Persistent:
    stride("stride", facts.roles.stride);
    scalar("logical_programs", facts.logicalPrograms);
    scalar("persistent_steps", facts.persistentSteps);
    scalar("row_tasks", facts.rowTasks);
    malformed |= !persistentCountSource(facts.logicalPrograms, layout) ||
                 !persistentCountSource(facts.persistentSteps, layout) ||
                 !persistentCountSource(facts.rowTasks, layout) ||
                 !getConstantIntValue(facts.roles.stride);
    break;
  case AccessSchedule::Nested:
    scalar("rows", facts.roles.rows);
    scalar("cols", facts.roles.cols);
    stride("stride", facts.roles.stride);
    break;
  case AccessSchedule::Flat:
    scalar("size", facts.roles.size);
    break;
  case AccessSchedule::Jagged:
    facts.roles.offsets = pointer(value("offsets"));
    malformed |= !facts.roles.offsets || facts.roles.offsets == facts.roles.output ||
                 llvm::is_contained(facts.roles.inputs, facts.roles.offsets);
    scalar("cols", facts.roles.cols);
    scalar("input_stride", facts.roles.inputStride);
    scalar("matrix_stride", facts.roles.matrixStride);
    stride("stride", facts.roles.stride);
    break;
  case AccessSchedule::Program:
    if (auto strides = fields.getAs<ArrayAttr>("input_strides")) {
      for (auto source : strides) {
        auto ref = decode(source);
        if (!ref || (isa<Value>(ref) && !cast<Value>(ref).getType().isIntOrIndex()) ||
            (isa<Attribute>(ref) && !isa<IntegerAttr>(cast<Attribute>(ref))))
          return failure();
        facts.roles.inputStrides.push_back(ref);
      }
    }
    malformed |= facts.roles.inputStrides.size() != facts.roles.inputs.size();
    break;
  }
  if (malformed || used.size() != results.size() || fields.size() != required.size())
    return failure();
  for (auto name : required)
    if (!fields.contains(name))
      return failure();
  return facts;
}
} // namespace mlir::hivm::l1prefetch
#endif
