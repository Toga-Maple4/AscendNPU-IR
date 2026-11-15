// Planner-owned selected-storage capacity facts and their internal encoding.
#ifndef BISHENGIR_HIVM_SELECTED_CAPACITY_H
#define BISHENGIR_HIVM_SELECTED_CAPACITY_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Builders.h"
#include <limits>
#include <optional>

namespace mlir::hivm {
struct SelectedBufferCapacity {
  uint64_t alignedSlotBits;
  uint32_t slots;
  uint64_t capacityBits;
  AddressSpace space;
  bool exceedsCapacity() const {
    return slots && alignedSlotBits > capacityBits / slots;
  }
};

// A conservative lower bound: sum only when all reservations must coexist.
// Otherwise the largest reservation is a safe bound; never sum reusable storage.
struct SelectedSetCapacity {
  SmallVector<SelectedBufferCapacity, 2> inputs;
  uint64_t lowerBoundBits = 0;
  bool mandatoryConflict = true;

  bool deriveLowerBound() {
    if (inputs.empty()) return false;
    uint64_t sum = 0, maximum = 0;
    for (auto input : inputs) {
      if (!input.slots || !input.alignedSlotBits || !input.capacityBits ||
          input.capacityBits != inputs.front().capacityBits ||
          input.space != inputs.front().space ||
          input.alignedSlotBits > uint64_t(INT64_MAX) / input.slots)
        return false;
      uint64_t total = input.alignedSlotBits * input.slots;
      if (sum > uint64_t(INT64_MAX) - total) return false;
      sum += total;
      maximum = std::max(maximum, total);
    }
    lowerBoundBits = mandatoryConflict ? sum : maximum;
    return true;
  }
  bool exceedsCapacity() const {
    return !inputs.empty() && lowerBoundBits > inputs.front().capacityBits;
  }
  bool hasMultipleSlots() const {
    return llvm::any_of(inputs, [](auto input) { return input.slots >= 2; });
  }

  static std::optional<SelectedSetCapacity> decode(DictionaryAttr result,
                                                   unsigned expectedInputs) {
    if (!result) return std::nullopt;
    auto version = result.getAs<IntegerAttr>("version");
    auto ids = result.getAs<DenseI32ArrayAttr>("selected_ids");
    auto bits = result.getAs<DenseI64ArrayAttr>("aligned_slot_bits");
    auto slots = result.getAs<DenseI32ArrayAttr>("slots");
    auto bound = result.getAs<IntegerAttr>("lower_bound_bits");
    auto capacity = result.getAs<IntegerAttr>("capacity_bits");
    auto conflict = result.getAs<BoolAttr>("mandatory_conflict");
    auto space = result.getAs<StringAttr>("space");
    if (!version || version.getInt() != 2 || !ids || ids.size() != expectedInputs ||
        !bits || bits.size() != expectedInputs || !slots || slots.size() != expectedInputs ||
        !bound || !capacity || capacity.getInt() <= 0 || !conflict ||
        !space || space.getValue() != "UB") return std::nullopt;
    SelectedSetCapacity value;
    value.mandatoryConflict = conflict.getValue();
    for (unsigned i = 0; i < expectedInputs; ++i) {
      if (ids[i] != int32_t(i) || bits[i] <= 0 || slots[i] <= 0) return std::nullopt;
      value.inputs.push_back({uint64_t(bits[i]), uint32_t(slots[i]),
                              uint64_t(capacity.getInt()), AddressSpace::UB});
    }
    if (!value.deriveLowerBound() || bound.getInt() < 0 ||
        uint64_t(bound.getInt()) != value.lowerBoundBits) return std::nullopt;
    return value;
  }
};
} // namespace mlir::hivm
#endif
