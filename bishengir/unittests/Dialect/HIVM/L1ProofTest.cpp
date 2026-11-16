// Direct tests of the private handoff, independent of the digest checkpoint.
#include "L1PrefetchPSAddress.h"
#include "L1PrefetchPSFacts.h"
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
