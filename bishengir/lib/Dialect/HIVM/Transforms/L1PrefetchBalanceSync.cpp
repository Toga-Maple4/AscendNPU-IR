//===- L1PrefetchBalanceSync.cpp - Balance split L1 sync ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_L1PREFETCHBALANCESYNC
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

bool containsSelectedOperation(scf::IfOp ifOp, TFuncCoreType coreType) {
  bool found = false;
  ifOp.getThenRegion().walk([&](Operation *op) {
    if (coreType == TFuncCoreType::AIC && isa<L12UBOp>(op))
      found = true;
    if (coreType == TFuncCoreType::AIV && isa<StoreOp>(op) &&
        (op->hasAttr("hivm.sync_memory_read_id") ||
         op->hasAttr("hivm.sync_memory_read_ids")))
      found = true;
  });
  return found;
}

bool containsSelectedAllocation(scf::IfOp ifOp) {
  bool found = false;
  ifOp.getThenRegion().walk([&](annotation::MarkOp mark) {
    if (mark->hasAttr("hivm.sync_memory_read_id"))
      found = true;
  });
  return found;
}

bool isSyncOnly(scf::IfOp ifOp) {
  bool sawSync = false;
  for (Operation &op : ifOp.getThenRegion().front().without_terminator()) {
    if (!isa<SyncBlockSetOp, SyncBlockWaitOp>(op))
      return false;
    sawSync = true;
  }
  return sawSync &&
         (!ifOp.elseBlock() ||
          ifOp.getElseRegion().front().without_terminator().empty());
}

struct L1PrefetchBalanceSyncPass
    : impl::L1PrefetchBalanceSyncBase<L1PrefetchBalanceSyncPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func->hasAttr("program_split.transformed"))
      return;
    auto coreType = queryFuncCoreType(func);
    if (!coreType || (coreType.value() != TFuncCoreType::AIC &&
                      coreType.value() != TFuncCoreType::AIV))
      return;

    SmallVector<scf::IfOp> syncIfs;
    func.walk([&](scf::IfOp ifOp) {
      if (isSyncOnly(ifOp))
        syncIfs.push_back(ifOp);
    });

    for (scf::IfOp syncIf : syncIfs) {
      scf::IfOp workIf;
      for (Operation *prev = syncIf->getPrevNode(); prev;
           prev = prev->getPrevNode()) {
        auto candidate = dyn_cast<scf::IfOp>(prev);
        if (candidate && containsSelectedOperation(candidate, coreType.value())) {
          workIf = candidate;
          break;
        }
      }
      if (!workIf) {
        syncIf.emitError("program-split L1 sync lost its guarded work");
        return signalPassFailure();
      }

      // Preserve every solver dependency. Static event IDs can be reused and
      // do not identify a protocol. Redundancy must be proved in the shared
      // solver before splitting the participating cores, never inferred here.
      SmallVector<Operation *> waits;
      SmallVector<Operation *> sets;
      for (Operation &op : syncIf.getThenRegion().front().without_terminator()) {
        if (isa<SyncBlockWaitOp>(op))
          waits.push_back(&op);
        else
          sets.push_back(&op);
      }
      Operation *waitAnchor = workIf;
      if (coreType.value() == TFuncCoreType::AIV) {
        for (Operation *prev = workIf->getPrevNode(); prev;
             prev = prev->getPrevNode()) {
          auto candidate = dyn_cast<scf::IfOp>(prev);
          if (candidate && containsSelectedAllocation(candidate)) {
            waitAnchor = candidate;
            break;
          }
        }
      }
      for (Operation *wait : waits)
        wait->moveBefore(waitAnchor);
      Operation *insertAfter = workIf;
      for (Operation *set : sets) {
        set->moveAfter(insertAfter);
        insertAfter = set;
      }
      syncIf.erase();
    }

    // Maple4 may retain a mixed conditional containing shared UB storage.
    // Its solver events remain inside the work guard rather than a separate
    // sync-only sibling. Balance the same events for inactive workers without
    // splitting the shared allocation or changing any dependency/event ID.
    SmallVector<scf::IfOp> guardedWork;
    func.walk([&](scf::IfOp ifOp) {
      if (isa<scf::ForOp>(ifOp->getParentOp()) && !ifOp.elseBlock() &&
          ifOp.getNumResults() == 0 &&
          containsSelectedOperation(ifOp, coreType.value()))
        guardedWork.push_back(ifOp);
    });
    for (scf::IfOp workIf : guardedWork) {
      SmallVector<Operation *> waits, sets;
      for (Operation &op : workIf.getThenRegion().front().without_terminator()) {
        if (isa<SyncBlockWaitOp>(op))
          waits.push_back(&op);
        else if (isa<SyncBlockSetOp>(op))
          sets.push_back(&op);
      }
      for (Operation *wait : waits)
        wait->moveBefore(workIf);
      Operation *insertAfter = workIf;
      for (Operation *set : sets) {
        set->moveAfter(insertAfter);
        insertAfter = set;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createL1PrefetchBalanceSyncPass() {
  return std::make_unique<L1PrefetchBalanceSyncPass>();
}
