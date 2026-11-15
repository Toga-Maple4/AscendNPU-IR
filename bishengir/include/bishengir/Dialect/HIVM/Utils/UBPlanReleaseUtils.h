//===- UBPlanReleaseUtils.h - PlanMemory UB release hints --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_UTILS_UBPLANRELEASEUTILS_H
#define BISHENGIR_DIALECT_HIVM_UTILS_UBPLANRELEASEUTILS_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace hivm {

bool isUBPlanReleaseMark(annotation::MarkOp markOp);

/// memref.dealloc (legacy) or annotation.mark {hivm.ub_plan_release}.
bool isUBPlanReleaseOp(Operation *op);

std::optional<Value> getUBPlanReleaseMemref(Operation *op);

std::optional<Value> traceUBAllocRoot(Value memref);

annotation::MarkOp createUBPlanReleaseMark(OpBuilder &builder, Location loc,
                                           Value memref);

/// Place (or move) a plan-release mark immediately before `insertBefore`.
void ensureUBPlanReleaseMarkBefore(OpBuilder &builder, Operation *insertBefore,
                                   Value memref);

void eraseUBPlanReleaseMarks(Operation *root);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_UTILS_UBPLANRELEASEUTILS_H
