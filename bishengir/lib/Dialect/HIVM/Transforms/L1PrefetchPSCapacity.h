// Program-split capacity reporting adapter; MemPlan owns all physical facts.
#ifndef BISHENGIR_L1PREFETCHPS_CAPACITY_H
#define BISHENGIR_L1PREFETCHPS_CAPACITY_H
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Transforms/SelectedCapacity.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::hivm {
class SelectedCapacityRequest {
  func::FuncOp function;
  SmallVector<Value> inputs;
  bool requested = false, valid = true;
public:
  SelectedCapacityRequest(func::FuncOp function, bool local) : function(function) {
    if (!local) return;
    DenseMap<int32_t, Value> marks;
    function.walk([&](annotation::MarkOp mark) {
      auto attr = mark->getAttr("program_split.selected_input");
      if (!attr) return;
      requested = true;
      auto id = dyn_cast<IntegerAttr>(attr);
      if (!id || !id.getType().isInteger(32) || id.getInt() < 0 ||
          !marks.try_emplace(id.getInt(), mark.getSrc()).second) valid = false;
    });
    for (unsigned i = 0; i < marks.size(); ++i) {
      auto found = marks.find(i);
      if (found == marks.end()) { valid = false; break; }
      inputs.push_back(found->second);
    }
  }
  template <typename Planner>
  void publish(const Planner &plan, bool succeeded) {
    if (!requested) return;
    auto capacity = valid ? plan.getSelectedSetCapacity(inputs) : std::nullopt;
    bool excess = capacity && capacity->hasMultipleSlots() && capacity->exceedsCapacity();
    Builder b(function.getContext());
    NamedAttrList fields;
    fields.set("version", b.getI32IntegerAttr(2));
    fields.set("kind", b.getStringAttr(succeeded ? "ordinary_plan_succeeded" :
        excess ? "proven_selected_set_capacity_excess" : "unknown_or_other_failure"));
    SmallVector<int32_t> ids;
    for (unsigned i = 0; i < inputs.size(); ++i) ids.push_back(i);
    fields.set("selected_ids", b.getDenseI32ArrayAttr(ids));
    if (capacity) {
      SmallVector<int64_t> bits;
      SmallVector<int32_t> slots;
      for (auto input : capacity->inputs) {
        bits.push_back(input.alignedSlotBits);
        slots.push_back(input.slots);
      }
      fields.set("aligned_slot_bits", b.getDenseI64ArrayAttr(bits));
      fields.set("slots", b.getDenseI32ArrayAttr(slots));
      fields.set("capacity_bits", b.getI64IntegerAttr(capacity->inputs.front().capacityBits));
      fields.set("lower_bound_bits", b.getI64IntegerAttr(capacity->lowerBoundBits));
      fields.set("mandatory_conflict", b.getBoolAttr(capacity->mandatoryConflict));
      fields.set("space", b.getStringAttr("UB"));
    }
    function->setAttr("program_split.memory_result", fields.getDictionary(function.getContext()));
  }
};
} // namespace mlir::hivm
#endif
