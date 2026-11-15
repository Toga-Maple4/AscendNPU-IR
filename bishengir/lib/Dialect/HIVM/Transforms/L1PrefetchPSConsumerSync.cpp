//===- L1PrefetchPSConsumerSync.cpp - Refine paired UB handoff --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

namespace mlir {
#define GEN_PASS_DEF_PROGRAMSPLITL1CONSUMERSYNC
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

// Refine only the materializer's two parity branches. Event allocation and
// loop-carried L1 protection remain owned by GraphSyncSolver. Running after
// PlanMemory prevents a one-consumer wait from proving pair-wide liveness.
struct Delivery {
  Operation *first;
  Operation *last;
};

struct Handoff {
  scf::IfOp work;
  SyncBlockWaitOp release;
  SmallVector<SyncBlockSetOp> ready;
  SmallVector<Delivery> deliveries;
  bool needsInactiveBranch = false;
};

bool containsDelivery(Region &region) {
  bool found = false;
  region.walk([&](L12UBOp) { found = true; });
  return found;
}

// A delivery may be unconditional or guarded by the consumer's payload extent.
// Put both endpoints outside that guard so an empty consumer still participates.
Operation *deliveryAnchor(L12UBOp copy, Block &bank) {
  if (copy->getBlock() == &bank)
    return copy;
  auto guard = dyn_cast<scf::IfOp>(copy->getParentOp());
  if (guard && guard->getBlock() == &bank && !guard.elseBlock())
    return guard;
  return nullptr;
}

LogicalResult collectHandoff(scf::ForOp loop, Handoff &handoff) {
  SmallVector<SyncBlockWaitOp> releases;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (auto wait = dyn_cast<SyncBlockWaitOp>(op)) {
      if (wait.getTpipeAttr().getPipe() == PIPE::PIPE_MTE3 &&
          wait.getPipeAttr().getPipe() == PIPE::PIPE_MTE1)
        releases.push_back(wait);
    }
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (containsDelivery(branch.getThenRegion()) ||
          containsDelivery(branch.getElseRegion())) {
        if (handoff.work)
          return loop.emitError("L1 consumer sync: multiple delivery regions");
        handoff.work = branch;
      }
    }
    if (auto signal = dyn_cast<SyncBlockSetOp>(op)) {
      if (signal.getTpipeAttr().getPipe() == PIPE::PIPE_MTE1)
        handoff.ready.push_back(signal);
    }
  }
  auto work = handoff.work;
  if (releases.size() != 1 || !work || handoff.ready.empty())
    return loop.emitError(
        "L1 consumer sync: missing paired release/readiness events");
  handoff.release = releases.front();

  // Consume the solver's dependency list; do not infer readiness from a kernel
  // kind or manufacture a V dependency for a consumer that only stores.
  auto validFlag = [](auto op) {
    auto flag = op.getStaticFlagId();
    return op.getTcoreTypeAttr().getTcoretype() == TCoreType::CUBE && flag &&
           flag->getInt() >= 0 &&
           flag->getInt() < util::INTRA_BLOCK_FLAG_ID_OFFSET;
  };
  if (!validFlag(handoff.release) ||
      !handoff.release->isBeforeInBlock(work))
    return loop.emitError("L1 consumer sync: invalid release event or ordering");
  SmallVector<PIPE> pipes;
  SmallVector<int64_t> flags{handoff.release.getStaticFlagId()->getInt()};
  for (auto ready : handoff.ready) {
    auto pipe = ready.getPipeAttr().getPipe();
    if (!validFlag(ready) || !work->isBeforeInBlock(ready) ||
        (pipe != PIPE::PIPE_V && pipe != PIPE::PIPE_MTE3) ||
        llvm::is_contained(pipes, pipe) ||
        llvm::is_contained(flags, ready.getStaticFlagId()->getInt()))
      return loop.emitError("L1 consumer sync: invalid readiness events or ordering");
    pipes.push_back(pipe);
    flags.push_back(ready.getStaticFlagId()->getInt());
  }

  SmallVector<scf::IfOp> bankChoices;
  work.walk([&](scf::IfOp branch) {
    if (branch.elseBlock() && containsDelivery(branch.getThenRegion()) &&
        containsDelivery(branch.getElseRegion()))
      bankChoices.push_back(branch);
  });
  if (bankChoices.size() != 1)
    return work.emitError("L1 consumer sync: expected one ping/pong delivery branch");
  auto parity = bankChoices.front();
  handoff.needsInactiveBranch = parity != work;
  if (handoff.needsInactiveBranch &&
      (work.elseBlock() || parity->getParentOp() != work))
    return work.emitError("L1 consumer sync: unsupported outer delivery guard");

  unsigned totalCopies = 0, bankCopies = 0;
  for (Region *region : {&parity.getThenRegion(), &parity.getElseRegion()}) {
    SmallVector<L12UBOp> copies;
    region->walk([&](L12UBOp copy) { copies.push_back(copy); });
    if ((copies.empty() || copies.size() % 2) ||
        (bankCopies && bankCopies != copies.size()) ||
        llvm::any_of(llvm::enumerate(copies), [&](auto item) {
          return item.value().getSubBlockid() !=
                 (item.index() >= copies.size() / 2);
        }))
      return parity.emitError(
          "L1 consumer sync: deliveries must group both consumers");
    bankCopies = copies.size();
    totalCopies += bankCopies;
    unsigned inputs = bankCopies / 2;
    Operation *previous = nullptr;
    for (unsigned stream = 0; stream < 2; ++stream) {
      Operation *first = deliveryAnchor(copies[stream * inputs], region->front());
      Operation *last = deliveryAnchor(copies[(stream + 1) * inputs - 1], region->front());
      if (!first || !last ||
          (first != last && (isa<scf::IfOp>(first) || isa<scf::IfOp>(last))) ||
          (first != last && !first->isBeforeInBlock(last)) ||
          (previous && (previous == first || !previous->isBeforeInBlock(first))))
        return parity.emitError("L1 consumer sync: unsupported consumer delivery guards");
      handoff.deliveries.push_back({first, last});
      previous = last;
    }
  }
  unsigned workCopies = 0;
  work.walk([&](L12UBOp) { ++workCopies; });
  if (workCopies != totalCopies)
    return work.emitError("L1 consumer sync: delivery outside the bank branches");
  return success();
}

void refineHandoff(Handoff &handoff) {
  OpBuilder builder(handoff.work.getContext());
  // Shared A5 mapping with ordinary sync lowering. These are existing
  // allocated cross-core events, not new local flags or new dependencies.
  auto waitFor = [&](unsigned stream) {
    builder.create<WaitIntraBlockImmInstrOp>(
        handoff.release.getLoc(), static_cast<uint64_t>(PIPE::PIPE_MTE1),
        util::intraBlockFlagForSubcore(
            handoff.release.getStaticFlagId()->getInt(), stream));
  };
  auto publishFor = [&](unsigned stream) {
    for (auto signal : handoff.ready)
      builder.create<SetIntraBlockImmInstrOp>(
          signal.getLoc(), static_cast<uint64_t>(PIPE::PIPE_MTE1),
          util::intraBlockFlagForSubcore(
              signal.getStaticFlagId()->getInt(), stream));
  };
  for (auto entry : llvm::enumerate(handoff.deliveries)) {
    builder.setInsertionPoint(entry.value().first);
    waitFor(entry.index() % 2);
    builder.setInsertionPointAfter(entry.value().last);
    publishFor(entry.index() % 2);
  }
  if (handoff.needsInactiveBranch) {
    Block *inactive = new Block();
    handoff.work.getElseRegion().push_back(inactive);
    builder.setInsertionPointToStart(inactive);
    for (unsigned stream = 0; stream < 2; ++stream) {
      waitFor(stream);
      publishFor(stream);
    }
    builder.create<scf::YieldOp>(handoff.work.getLoc());
  }
  handoff.release.erase();
  for (auto signal : handoff.ready)
    signal.erase();
}

struct ProgramSplitL1ConsumerSyncPass
    : impl::ProgramSplitL1ConsumerSyncBase<ProgramSplitL1ConsumerSyncPass> {
  void runOnOperation() override {
    auto func = getOperation();
    if (!func->hasAttr("program_split.transformed") ||
        queryFuncCoreType(func) != TFuncCoreType::AIC ||
        !hacc::utils::isRegBasedArch(func->getParentOfType<ModuleOp>()))
      return;

    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp loop) {
      bool hasCopies = false;
      loop.walk([&](L12UBOp copy) {
        hasCopies |= copy->getParentOfType<scf::ForOp>() == loop;
      });
      if (hasCopies)
        loops.push_back(loop);
    });
    // Check the whole function before inserting or erasing any endpoint.
    SmallVector<Handoff> handoffs(loops.size());
    for (auto entry : llvm::enumerate(loops))
      if (failed(collectHandoff(entry.value(), handoffs[entry.index()])))
        return signalPassFailure();
    for (auto &handoff : handoffs)
      refineHandoff(handoff);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createProgramSplitL1ConsumerSyncPass() {
  return std::make_unique<ProgramSplitL1ConsumerSyncPass>();
}
