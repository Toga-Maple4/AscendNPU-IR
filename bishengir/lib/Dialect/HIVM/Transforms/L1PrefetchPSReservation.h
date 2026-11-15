// Physical reservation shared by retry feasibility and L1 materialization.
#ifndef BISHENGIR_L1PREFETCHPS_RESERVATION_H
#define BISHENGIR_L1PREFETCHPS_RESERVATION_H

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/L1PrefetchPS.h"
#include "mlir/IR/BuiltinTypes.h"
#include <limits>

namespace mlir::hivm::l1prefetch {
struct ReservationCheck {
  StringRef reason;
  StringRef message;
  bool succeeded() const { return reason.empty(); }
};

// Both callers use this exact calculation. Given identical tile, input count
// and target spec, early success implies the materializer's reservation check
// succeeds. This does not predict UB planning or other downstream constraints.
inline ReservationCheck checkL1Reservation(
    MemRefType tile, unsigned inputs, hacc::HACCTargetDeviceSpecInterface spec) {
  if (!inputs || !tile || !tile.hasStaticShape())
    return {"invalid_l1_reservation", "invalid selected tile reservation"};
  uint64_t tileBits = tile.getElementTypeBitWidth();
  for (int64_t extent : tile.getShape()) {
    if (extent <= 0 || tileBits > uint64_t(INT64_MAX) / extent)
      return {"l1_reservation_overflow", "tile reservation size overflows"};
    tileBits *= extent;
  }
  if (!spec)
    return {"missing_l1_target_spec", "missing NPU target specification"};
  auto capacityEntry = spec.getSpecForIdentifierEnum(hacc::DeviceSpec::L1_SIZE);
  auto alignEntry = spec.getSpecForIdentifierEnum(hacc::DeviceSpec::L1_ALIGN_SIZE);
  auto capacityAttr = capacityEntry
                          ? dyn_cast<IntegerAttr>(capacityEntry.getValue())
                          : IntegerAttr();
  auto alignAttr = alignEntry ? dyn_cast<IntegerAttr>(alignEntry.getValue())
                             : IntegerAttr();
  if (!capacityAttr || !alignAttr)
    return {"missing_l1_capacity_spec",
            "missing or non-integer target L1 capacity/alignment"};
  int64_t capacity = capacityAttr.getInt(), align = alignAttr.getInt();
  if (capacity <= 0 || align <= 0 ||
      tileBits > uint64_t(INT64_MAX - align + 1))
    return {"invalid_l1_capacity_spec", "invalid target reservation capacity"};
  uint64_t alignedBits = (tileBits + align - 1) / align * align;
  if (alignedBits > uint64_t(capacity) /
                        (uint64_t(kL1BanksPerInput) * inputs))
    return {"l1_reservation_exceeds_capacity",
            "ping/pong reservation exceeds target L1 capacity"};
  return {};
}
} // namespace mlir::hivm::l1prefetch

#endif
