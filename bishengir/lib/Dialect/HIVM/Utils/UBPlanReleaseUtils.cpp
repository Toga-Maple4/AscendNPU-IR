//===- UBPlanReleaseUtils.cpp - PlanMemory UB release hints ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/UBPlanReleaseUtils.h"

#include "bishengir/Dialect/Utils/Util.h"

namespace mlir {
namespace hivm {

bool isUBPlanReleaseMark(annotation::MarkOp markOp) {
  return markOp->hasAttr(UBPlanReleaseAttr::name);
}

std::optional<Value> traceUBAllocRoot(Value memref) {
  auto maybeAlloc = utils::tracebackMemRefToAlloc(memref);
  if (!maybeAlloc.has_value())
    return std::nullopt;
  return maybeAlloc.value().getResult();
}

bool isUBPlanReleaseOp(Operation *op) {
  if (auto markOp = dyn_cast<annotation::MarkOp>(op))
    return isUBPlanReleaseMark(markOp);
  if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
    auto addrSpace = getHIVMAddressSpace(deallocOp.getMemref().getType());
    return addrSpace == hivm::AddressSpace::UB;
  }
  return false;
}

std::optional<Value> getUBPlanReleaseMemref(Operation *op) {
  if (auto markOp = dyn_cast<annotation::MarkOp>(op))
    return markOp.getSrc();
  if (auto deallocOp = dyn_cast<memref::DeallocOp>(op))
    return deallocOp.getMemref();
  return std::nullopt;
}

annotation::MarkOp createUBPlanReleaseMark(OpBuilder &builder, Location loc,
                                           Value memref) {
  auto maybeRoot = traceUBAllocRoot(memref);
  Value marked = maybeRoot.has_value() ? maybeRoot.value() : memref;
  auto markOp = builder.create<annotation::MarkOp>(loc, marked);
  markOp->setAttr(UBPlanReleaseAttr::name, builder.getUnitAttr());
  return markOp;
}

void ensureUBPlanReleaseMarkBefore(OpBuilder &builder, Operation *insertBefore,
                                   Value memref) {
  auto maybeRoot = traceUBAllocRoot(memref);
  if (!maybeRoot.has_value())
    return;
  Value root = maybeRoot.value();
  Block *block = insertBefore->getBlock();
  annotation::MarkOp existing = nullptr;
  for (Operation &op : block->getOperations()) {
    if (&op == insertBefore)
      break;
    auto markOp = dyn_cast<annotation::MarkOp>(&op);
    if (!markOp || !isUBPlanReleaseMark(markOp))
      continue;
    auto otherRoot = traceUBAllocRoot(markOp.getSrc());
    if (otherRoot.has_value() && otherRoot.value() == root) {
      existing = markOp;
      break;
    }
  }
  if (existing) {
    if (insertBefore->isBeforeInBlock(existing))
      existing->moveBefore(insertBefore);
    return;
  }
  builder.setInsertionPoint(insertBefore);
  createUBPlanReleaseMark(builder, insertBefore->getLoc(), root);
}

void eraseUBPlanReleaseMarks(Operation *root) {
  SmallVector<annotation::MarkOp> toErase;
  root->walk([&](annotation::MarkOp markOp) {
    if (isUBPlanReleaseMark(markOp))
      toErase.push_back(markOp);
  });
  for (annotation::MarkOp markOp : toErase)
    markOp.erase();
}

} // namespace hivm
} // namespace mlir
