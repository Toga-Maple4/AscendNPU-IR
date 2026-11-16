// Direct tests of the private handoff, independent of the digest checkpoint.
#include "L1PrefetchPSAddress.h"
#include "L1PrefetchPSFacts.h"
#include "bishengir/Dialect/HIVM/Transforms/L1PrefetchPS.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::hivm::l1prefetch;

namespace {
class L1ProofTest : public testing::Test {
protected:
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  func::FuncOp func;
  void SetUp() override {
    context.loadDialect<arith::ArithDialect, func::FuncDialect,
                        memref::MemRefDialect>();
    module = parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @kernel(%lock: memref<?xi8>, %workspace: memref<?xi8>,
                          %input: memref<?xf32>, %output: memref<?xf32>,
                          %n: i32, %s: i32, %t: i32,
                          %gx: i32, %gy: i32, %gz: i32,
                          %px: i32, %py: i32, %pz: i32)
            attributes {SyncBlockLockArgIdx = 0 : i64,
                        WorkspaceArgIdx = 1 : i64} {
          %stride = arith.constant 8192 : index
          %count = arith.constant 4 : i32
          return
        }
      })mlir", &context);
    ASSERT_TRUE(module);
    func = *module->getOps<func::FuncOp>().begin();
    ASSERT_TRUE(KernelLayout::get(func));
  }
  AcceptedFacts persistent() {
    AcceptedFacts facts{AccessSchedule::Persistent, {}, func.getArgument(4),
                        func.getArgument(5), func.getArgument(6)};
    facts.roles.inputs.push_back(func.getArgument(2));
    facts.roles.output = func.getArgument(3);
    facts.roles.stride = func.front().front().getResult(0);
    return facts;
  }
  void replaceField(StringRef name, Attribute value) {
    NamedAttrList fields(func->getAttrOfType<DictionaryAttr>(kAcceptedFacts));
    fields.set(name, value);
    func->setAttr(kAcceptedFacts, fields.getDictionary(&context));
  }
};

TEST_F(L1ProofTest, PublisherRejectsMissingAndUnexpectedBeforeMutation) {
  auto facts = persistent();
  facts.roles.rows = func.getArgument(4); // Unexpected for Persistent.
  EXPECT_TRUE(failed(publishAcceptedFacts(func, facts)));
  EXPECT_FALSE(func->hasAttr(kAcceptedFacts));
  EXPECT_FALSE(func.front().front().hasAttr(kValueRoles));
  facts.roles.rows = {};
  facts.rowTasks = {};
  EXPECT_TRUE(failed(publishAcceptedFacts(func, facts)));
  EXPECT_FALSE(func->hasAttr(kAcceptedFacts));
  EXPECT_FALSE(func.front().front().hasAttr(kValueRoles));
}

TEST_F(L1ProofTest, ReaderRejectsMalformedFieldSetWithoutRelyingOnDigest) {
  auto facts = persistent();
  auto layout = *KernelLayout::get(func);
  ASSERT_TRUE(succeeded(publishAcceptedFacts(func, facts)));
  ASSERT_TRUE(succeeded(decodeAcceptedFacts(func, layout)));
  Builder b(&context);
  replaceField("rows", b.getDictionaryAttr({b.getNamedAttr("arg", b.getI32IntegerAttr(4))}));
  EXPECT_TRUE(failed(decodeAcceptedFacts(func, layout)));
  ASSERT_TRUE(succeeded(publishAcceptedFacts(func, facts)));
  NamedAttrList fields(func->getAttrOfType<DictionaryAttr>(kAcceptedFacts));
  fields.erase("row_tasks");
  func->setAttr(kAcceptedFacts, fields.getDictionary(&context));
  EXPECT_TRUE(failed(decodeAcceptedFacts(func, layout)));
}

TEST_F(L1ProofTest, CountsAllowPublicGridConstantButNotProgramId) {
  auto layout = *KernelLayout::get(func);
  Builder b(&context);
  for (unsigned index : {4u, 7u, 8u, 9u, 10u, 11u, 12u}) {
    auto facts = persistent();
    facts.logicalPrograms = func.getArgument(index);
    bool allowed = index < 10;
    EXPECT_EQ(succeeded(publishAcceptedFacts(func, facts)), allowed);
    ASSERT_TRUE(succeeded(publishAcceptedFacts(func, persistent())));
    for (StringRef field : {"logical_programs", "persistent_steps", "row_tasks"}) {
      replaceField(field, b.getDictionaryAttr({b.getNamedAttr("arg", b.getI32IntegerAttr(index))}));
      EXPECT_EQ(succeeded(decodeAcceptedFacts(func, layout)), allowed);
      ASSERT_TRUE(succeeded(publishAcceptedFacts(func, persistent())));
    }
  }
  auto facts = persistent();
  facts.logicalPrograms = std::next(func.front().begin())->getResult(0);
  facts.persistentSteps = facts.rowTasks = facts.logicalPrograms;
  ASSERT_TRUE(succeeded(publishAcceptedFacts(func, facts)));
  EXPECT_TRUE(succeeded(decodeAcceptedFacts(func, layout)));
  // Shared result IDs are valid: every ID must be used, not used exactly once.
  EXPECT_EQ(std::next(func.front().begin())->getAttrOfType<DictionaryAttr>(kValueRoles).size(), 1u);
  facts.roles.stride = func.getArgument(4);
  EXPECT_TRUE(failed(publishAcceptedFacts(func, facts)));
  replaceField("stride", b.getDictionaryAttr({b.getNamedAttr("arg", b.getI32IntegerAttr(4))}));
  EXPECT_TRUE(failed(decodeAcceptedFacts(func, layout)));
}

TEST_F(L1ProofTest, FactFailuresExplainTheBrokenObligation) {
  auto layout = *KernelLayout::get(func);
  Builder builder(&context);
  auto expectFailure = [&](StringRef expected) {
    std::string reason;
    EXPECT_TRUE(failed(decodeAcceptedFacts(func, layout, &reason)));
    EXPECT_NE(reason.find(expected.str()), std::string::npos) << reason;
  };
  expectFailure("missing accepted-facts dictionary");
  ASSERT_TRUE(succeeded(publishAcceptedFacts(func, persistent())));
  replaceField("logical_programs", builder.getDictionaryAttr(
      {builder.getNamedAttr("arg", builder.getI32IntegerAttr(10))}));
  expectFailure("persistent counts");
  ASSERT_TRUE(succeeded(publishAcceptedFacts(func, persistent())));
  replaceField("row_tasks", builder.getDictionaryAttr(
      {builder.getNamedAttr("result", builder.getStringAttr("missing"))}));
  expectFailure("row_tasks must reference an i32 value");
  ASSERT_TRUE(succeeded(publishAcceptedFacts(func, persistent())));
  replaceField("extra", builder.getUnitAttr());
  expectFailure("field count disagrees with the schedule");
  ASSERT_TRUE(succeeded(publishAcceptedFacts(func, persistent())));
  // An unreferenced result ID is different from a duplicate role reference.
  std::next(func.front().begin())->setAttr(kValueRoles, builder.getDictionaryAttr(
      {builder.getNamedAttr("unused", builder.getI32IntegerAttr(0))}));
  expectFailure("unreferenced value-role ID");
}

TEST_F(L1ProofTest, HandoffDiagnosticsSurviveTheDigestCheckpoint) {
  context.loadDialect<bufferization::BufferizationDialect, scf::SCFDialect,
                      tensor::TensorDialect>();
  module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @flat(%lock: memref<?xi8>, %workspace: memref<?xi8>,
                      %input: memref<?xf32>, %output: memref<?xf32>, %size: i32,
                      %gx: i32, %gy: i32, %gz: i32,
                      %px: i32, %py: i32, %pz: i32)
          attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64} {
        %source = memref.reinterpret_cast %input to offset: [0], sizes: [8], strides: [1]
          : memref<?xf32> to memref<8xf32>
        %dest = memref.reinterpret_cast %output to offset: [0], sizes: [8], strides: [1]
          : memref<?xf32> to memref<8xf32>
        %buffer = memref.alloc() : memref<8xf32>
        memref.copy %source, %buffer : memref<8xf32> to memref<8xf32>
        %tensor = bufferization.to_tensor %buffer restrict writable : memref<8xf32>
        bufferization.materialize_in_destination %tensor in writable %dest
          : (tensor<8xf32>, memref<8xf32>) -> ()
        return
      }
    })mlir", &context);
  ASSERT_TRUE(module);
  func = *module->getOps<func::FuncOp>().begin();
  Builder builder(&context);
  auto unit = builder.getUnitAttr();
  auto copy = *func.getOps<memref::CopyOp>().begin();
  auto alloc = *func.getOps<memref::AllocOp>().begin();
  auto tensor = *func.getOps<bufferization::ToTensorOp>().begin();
  copy->setAttr("l1_prefetch_ps.load", unit);
  func->setAttr("l1_prefetch_ps.flat", unit);
  func->setAttr("l1_prefetch_ps.accepted", unit);
  AcceptedFacts facts{AccessSchedule::Flat, {}, {}, {}, {}};
  facts.roles.inputs.push_back(func.getArgument(2));
  facts.roles.output = func.getArgument(3);
  facts.roles.size = func.getArgument(4);
  ASSERT_TRUE(succeeded(publishAcceptedFacts(func, facts)));
  // Test the handoff boundary directly, not semantic recognition. Rebind after
  // deliberate damage so each inner check is exercised past the digest gate.
  auto bindProof = [&] {
    func->setAttr("l1_prefetch_ps.proof",
        builder.getStringAttr(mlir::hivm::l1PrefetchPSFingerprint(func)));
  };
  bindProof();
  ASSERT_TRUE(succeeded(mlir::hivm::verifyL1PrefetchPSAcceptance(func)));
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &message) {
    llvm::raw_string_ostream stream(diagnostic);
    message.print(stream);
    return success();
  });
  auto expectFailure = [&](StringRef expected) {
    bindProof();
    diagnostic.clear();
    EXPECT_TRUE(failed(mlir::hivm::verifyL1PrefetchPSAcceptance(func)));
    EXPECT_NE(diagnostic.find("L1 pipeline error:"), std::string::npos);
    EXPECT_NE(diagnostic.find(expected.str()), std::string::npos) << diagnostic;
    EXPECT_FALSE(func->hasAttr("program_split.transformed"));
  };
  alloc->setAttr("l1_prefetch_ps.load", unit);
  expectFailure("load marker must name a memref.copy");
  alloc->removeAttr("l1_prefetch_ps.load");
  copy->removeAttr("l1_prefetch_ps.load");
  expectFailure("unexpected selected load count");
  copy->setAttr("l1_prefetch_ps.load", unit);
  alloc->setAttr("l1_prefetch_ps.padding", unit);
  expectFailure("padding guard must contain only a fill and yield");
  alloc->removeAttr("l1_prefetch_ps.padding");
  // Moving the tensor view under an otherwise valid guard breaks parentage.
  OpBuilder opBuilder(&context);
  opBuilder.setInsertionPoint(tensor);
  Value condition = opBuilder.create<arith::ConstantIntOp>(func.getLoc(), 1, 1);
  auto guard = opBuilder.create<scf::IfOp>(func.getLoc(), condition, false);
  tensor->moveBefore(guard.thenBlock()->getTerminator());
  expectFailure("selected tensor view must be local to the payload");
  tensor->moveBefore(guard);
  guard.erase();
  bindProof();
  EXPECT_TRUE(succeeded(mlir::hivm::verifyL1PrefetchPSAcceptance(func)));
}

TEST_F(L1ProofTest, SharedAddressDag) {
  OpBuilder b(&context);
  b.setInsertionPoint(func.front().getTerminator());
  auto loc = func.getLoc();
  AddressEquivalence equality;
  Value left = func.getArgument(4), right = left;
  Value step = func.getArgument(5);
  // Repeated queries over shared diamonds exercise one context's memoized DAG.
  // No wall-time threshold or coverage cap is part of the compiler contract.
  for (unsigned i = 0; i < 256; ++i) {
    Value sum = b.create<arith::AddIOp>(loc, left, step);
    Value delta = b.create<arith::SubIOp>(loc, sum, left);
    left = b.create<arith::AddIOp>(loc, left, delta);
    right = b.create<arith::AddIOp>(loc, step, right);
    ASSERT_TRUE(equality.equal(left, right));
    EXPECT_TRUE(equality.equal(delta, step));
  }
  Value wrong = b.create<arith::SubIOp>(loc, right, step);
  EXPECT_FALSE(equality.equal(left, wrong));
}
} // namespace
