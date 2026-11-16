//===- L1PrefetchPS.h - Program-split boundary ---*- C++ -*-===//
#ifndef BISHENGIR_HIVM_L1PREFETCHPS_H
#define BISHENGIR_HIVM_L1PREFETCHPS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>
#include <string>

namespace mlir::hacc {
class HACCTargetDeviceSpecInterface;
}

namespace mlir::hivm {

// Two consumer streams, each with two ping/pong banks.
constexpr unsigned kL1BanksPerInput = 4;

// Library declarations are not kernel entries. Additional definitions remain
// ambiguous and are rejected by every consumer of the accepted entry.
inline func::FuncOp getL1PrefetchPSEntry(ModuleOp module) {
  func::FuncOp entry;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;
    if (entry)
      return {};
    entry = func;
  }
  return entry;
}

// Policy is separate from semantic acceptance and physical capacity. Unknown
// scores do not reject; exactly 40 passes. No runtime information is consulted.
struct L1PrefetchPSProfitability {
  std::optional<double> score;
  StringRef reason = "not_analyzed";
  unsigned inputs = 0;
  bool filtered() const { return score && *score < 40; }
  StringRef decision() const {
    return !score ? "unknown" : filtered() ? "filtered" : "passed";
  }
};
L1PrefetchPSProfitability getL1PrefetchPSProfitability(func::FuncOp func);

// Integrity only: binds the accepted selection to its structured executable IR.
// This is not semantic recognition, nor a serialized transformation plan.
std::string l1PrefetchPSFingerprint(func::FuncOp func);
LogicalResult verifyL1PrefetchPSAcceptance(func::FuncOp func);
// Runtime ABI facts must leave the compiler. Derive them from accepted pristine
// IR, rather than duplicating them in the inter-pass proof annotations.
// A nonempty rejection reason denotes an unsupported interface or reservation. An
// empty reason on failure denotes an invalid compiler handoff and is fatal.
FailureOr<DictionaryAttr> getL1PrefetchPSLaunchContract(
    func::FuncOp func, hacc::HACCTargetDeviceSpecInterface targetSpec,
    std::string &rejectReason);

// Validate internal pointer-role metadata, independently of semantic eligibility.
// The producer supplies its original TTAdapter function for public-index/type
// checks. Lowered consumers omit it: outlining/hidden-argument insertion may
// change the current signature, while role indices retain their original ABI.
LogicalResult verifyL1PrefetchPSRealizationRoles(
    Operation *anchor, DictionaryAttr contract, func::FuncOp originalABI = {});

// External schema facts, separate from the unchanged physical realization ABI.
FailureOr<DictionaryAttr> getL1PrefetchPSExternalContract(func::FuncOp func);

} // namespace mlir::hivm

#endif
