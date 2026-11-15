//===- LowerUBPlanRelease.cpp - memref.dealloc -> ub_plan_release mark ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/UBPlanReleaseUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_LOWERUBPLANRELEASE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct LowerUBPlanReleasePass
    : public impl::LowerUBPlanReleaseBase<LowerUBPlanReleasePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      SmallVector<memref::DeallocOp> deallocs;
      funcOp.walk([&](memref::DeallocOp op) { deallocs.push_back(op); });
      for (memref::DeallocOp deallocOp : deallocs) {
        if (hivm::getHIVMAddressSpace(deallocOp.getMemref().getType()) !=
            hivm::AddressSpace::UB)
          continue;
        OpBuilder builder(deallocOp);
        hivm::createUBPlanReleaseMark(builder, deallocOp.getLoc(),
                                      deallocOp.getMemref());
        deallocOp.erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createLowerUBPlanReleasePass() {
  return std::make_unique<LowerUBPlanReleasePass>();
}
