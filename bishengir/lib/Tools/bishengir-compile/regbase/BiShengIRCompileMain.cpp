//===- BiShengIRCompileMain.cpp - RegBase compile orchestration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RegBase (A5) pipeline orchestration with retry/fallback logic.
// Migrated from AscendNPU-IR-Dev BishengIRCompileMain.cpp.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/L1PrefetchPS.h"
#include "bishengir/Dialect/HIVM/Transforms/SelectedCapacity.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Tools/RetriablePassManager/RetriablePassManager.h"
#include "llvm/Support/JSON.h"
#include "bishengir/Tools/hivmc/Config.h"
#include "bishengir/Tools/hivmc/HIVMC.h"
#include "bishengir/Tools/hivmc/HIVMCA5.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Version/Version.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/regbase/PassPipeline.h"
#include "bishengir/Tools/bishengir-compile/regbase/Utility.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VersionTuple.h"
#include <functional>
#include <set>

#define DEBUG_TYPE "bishengir-compile-regbase"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

namespace {

using PipelineBuilder = std::function<void(mlir::PassManager &,
                                           const BiShengIRCompileMainConfig &)>;

enum class CompileFlow {
  Mixed,
  PureSimt,
  Simd,
};

CompileFlow getCompileFlow(const BiShengIRCompileMainConfig &config) {
  if (config.getEnableSimdSimtMixCompile())
    return CompileFlow::Mixed;
  if (config.getPureSimt())
    return CompileFlow::PureSimt;
  return CompileFlow::Simd;
}

/// Get the lib directory path (../lib relative to bishengir-compile
/// executable). Returns canonical absolute path without ".." or ".".
static std::string getLibDirFromExecutable(StringRef executablePath) {
  if (executablePath.empty() ||
      (!executablePath.contains('/') && !executablePath.contains('\\')))
    return "";
  llvm::SmallString<256> absPath(executablePath);
  if (llvm::sys::fs::make_absolute(absPath))
    return "";
  llvm::SmallString<256> realPath;
  if (!llvm::sys::fs::real_path(absPath, realPath))
    absPath = realPath;
  llvm::sys::path::remove_filename(absPath);
  llvm::sys::path::append(absPath, "..", "lib");
  llvm::sys::path::remove_dots(absPath, /*remove_dot_dot=*/true);
  return std::string(absPath.str());
}

/// Add bitcode path attributes to ModuleOp from ../lib/*.bc files.
/// Paths are canonical (no ".." or ".") before being stored in attributes.
static void addBitcodeAttrsToModule(ModuleOp module, StringRef executablePath,
                                    const BiShengIRCompileMainConfig &config) {
  std::string libDir = getLibDirFromExecutable(executablePath);
  MLIRContext *ctx = module->getContext();
  ctx->loadDialect<mlir::hivm::HIVMDialect>();

  auto addIfExists = [&](const char *filename, llvm::StringRef attrName,
                         auto createAttr) {
    llvm::SmallString<256> bcPath(libDir);
    llvm::sys::path::append(bcPath, filename);
    if (!llvm::sys::fs::exists(bcPath))
      return;
    llvm::SmallString<256> canonicalPath;
    if (llvm::sys::fs::real_path(bcPath, canonicalPath))
      return;
    module->setAttr(
        attrName,
        createAttr(ctx, mlir::StringAttr::get(ctx, canonicalPath.str().str())));
  };

  if (mlir::hacc::utils::isAscend950(config.getTarget())) {
    addIfExists("meta_op.aic.c310.bc", mlir::hivm::AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIC_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.aiv.c310.bc", mlir::hivm::AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIV_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.mix.aic.c310.bc",
                mlir::hivm::MIX_AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIC_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.mix.aiv.c310.bc",
                mlir::hivm::MIX_AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIV_BITCODEAttr::get(c, s);
                });
  } else {
    addIfExists("meta_op.aic.c220.bc", mlir::hivm::AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIC_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.aiv.c220.bc", mlir::hivm::AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIV_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.mix.aic.c220.bc",
                mlir::hivm::MIX_AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIC_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.mix.aiv.c220.bc",
                mlir::hivm::MIX_AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIV_BITCODEAttr::get(c, s);
                });
  }
  addIfExists("host.bc", mlir::hivm::HOST_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::HOST_BITCODEAttr::get(c, s);
              });
}

using MixedModules = std::pair<ModuleOp, SmallVector<ModuleOp, 2>>;

/// Run a pipeline on a module using the regbase builder and A3's runPipeline
/// utility.
static LogicalResult runPipelineRegBase(ModuleOp mod,
                                        const PipelineBuilder &buildPipeline,
                                        BiShengIRCompileMainConfig &config,
                                        const std::string &pipelineName) {
  // Bind the config parameter into the builder to match A3's runPipeline
  // signature.
  auto boundPipeline =
      std::bind(buildPipeline, std::placeholders::_1, std::cref(config));
  return bishengir::runPipeline(mod, boundPipeline, config, pipelineName);
}

//   Scope::ScopeOp SIMT detection logic needed.
static MixedModules getMixedModules(ModuleOp topMod) {
  MixedModules res;
  res.first = nullptr;
  for (auto subMod : topMod.getOps<ModuleOp>()) {
    if (subMod->hasAttr(hacc::SIMTModuleAttr::name)) {
      res.second.push_back(subMod);
    } else {
      assert(!res.first && "only one main module is allowed");
      res.first = subMod;
    }
  };
  // if no main module, return the top module
  if (!res.first)
    res.first = topMod;

  return res;
}

static bool runModulePipeline(ModuleOp module, const PipelineBuilder &builder,
                              BiShengIRCompileMainConfig &config,
                              StringRef pipelineName) {
  return succeeded(
      runPipelineRegBase(module, builder, config, pipelineName.str()));
}

static bool runModulePipelines(ArrayRef<ModuleOp> modules,
                               const PipelineBuilder &builder,
                               BiShengIRCompileMainConfig &config,
                               StringRef pipelineName) {
  bool success = true;
  for (auto module : modules) {
    // Stop this pipeline on each module after earlier pipeline fails.
    success =
        success && runModulePipeline(module, builder, config, pipelineName);
  }
  return success;
}

static bool runMixedPipelines(ModuleOp mixedModule,
                              BiShengIRCompileMainConfig &config) {
  if (!runModulePipeline(mixedModule, regbase::buildBiShengHIRPipeline, config,
                         "BiShengHIR")) {
    return false;
  }

  auto [mainMod, simtMods] = getMixedModules(mixedModule);
  if (!runModulePipelines(simtMods, regbase::buildBiShengTTIRPipeline, config,
                          "BiShengSIMT")) {
    return false;
  }

  if (!runModulePipeline(mixedModule, regbase::buildBiShengHIRFinishPipeline,
                         config, "BishengHIR")) {
    return false;
  }

  if (!runModulePipeline(mainMod, regbase::buildFinalHIVMPipelines, config,
                         "buildFinalHIVMPipelines")) {
    return false;
  }
  return runModulePipeline(mainMod, regbase::buildBiShengHIRAVEToLLVMPipeline,
                           config, "BiShengSIMD");
}

LogicalResult handleSaveTemps(ModuleOp module, BiShengIRCompileMainConfig& config) {
  TempDirectoriesStore tempDirsStore;
  std::string inputFile = "module.hivm.opt.mlir";
  std::string outputFile = config.getOutputFile();
  std::unique_ptr<llvm::ToolOutputFile> inputFileHandler;

  // Handle --save-temps=<directory> option to store module.hivm.opt.mlir
  if (!config.getSaveTemps().empty()) {
    llvm::SmallString<256> saveTempsDir(config.getSaveTemps());
    if (llvm::sys::fs::make_absolute(saveTempsDir)) {
      llvm::errs() << "[ERROR] Failed to get absolute path for save-temps.\n";
      return failure();
    }
    if (!llvm::sys::fs::exists(saveTempsDir))
      if (auto ec = llvm::sys::fs::create_directories(saveTempsDir)) {
        llvm::errs() << "[ERROR] Failed to create save-temps directory: " << saveTempsDir << "\n";
        return failure();
      }
    llvm::sys::path::append(saveTempsDir, inputFile);
    std::string errorMessage;
    inputFileHandler = mlir::openOutputFile(saveTempsDir, &errorMessage);
    if (!inputFileHandler) {
      llvm::errs() << "[ERROR] Failed to open save-temps file: " << errorMessage << "\n";
      return failure();
    }
    inputFileHandler->keep();
  } else {
    inputFileHandler = getTempFile(inputFile, tempDirsStore);
    if (!inputFileHandler) {
      llvm::dbgs() << "[ERROR] Failed to create temporary input file needed to run hivmc compile.\n";
      return failure();
    }
  }
  inputFile = inputFileHandler->outputFilename();

  module.print(inputFileHandler->os(), mlir::OpPrintingFlags().enableDebugInfo(
                                         config.getEnableSanitizer() ||
                                         config.getEnableDebugInfo()));
  inputFileHandler->os().flush();
  return success();
}

HIVMCMainConfig HIVMCFromBiShengIRConfig(BiShengIRCompileMainConfig& config) {
    HIVMCMainConfig hivmcConfig;

    /// TODO: add flag difference logic for hivmc and bishengir flags in Options.td
    /// These flags are set manually due to mismatch in
    /// HIVMCConfigCLOptions constructor and Options.td
    ///
    /// Options.td is used both by bishengir pipeline and hivmc pipeline,
    /// So the flags which are both in bishengir pipeline and hivmc pipeline are just
    /// copied from the BiShengIRCompileMainConfig through dispatch table
    /// (which is generated in CompileOptionsGen.cpp)
    /// P.S : the same problem is also on A3 pipeline, check the translation there

    /// A5-specific
    hivmcConfig.autoVectorizeV2(false);
    hivmcConfig.targetBackend(config.getTarget());
    ///

    hivmcConfig.limitAutoMultiBufferForLocalBuffer(true);
    hivmcConfig.deterministicComputing(true);
    hivmcConfig.simtOptimizationMode(1900101);
    hivmcConfig.enableAutoCVBalance(true);
    hivmcConfig.setUseDPX(true);
    hivmcConfig.onlyRunHIVMPipeline(false);

    hivmcConfig.appendBishengOptions(config.getAppendBishengOptions());
    hivmcConfig.compileTriton(config.getEnableTritonKernelCompile());
    hivmcConfig.compileTritonDialect(config.getEnableTritonIRCompile());
    hivmcConfig.enableSimdSimtMixCompile(config.getEnableSimdSimtMixCompile());
    hivmcConfig.enableSIMTOnly(config.getPureSimt());
    hivmcConfig.enableSanitizer(config.getEnableSanitizer());
    hivmcConfig.enableSIMTFastDiv(config.getEnableSIMTFastDiv());
    hivmcConfig.enableDebugVariables(config.getEnableDebugVariables());
    hivmcConfig.enableDebugInfo(config.getEnableDebugInfo());
    hivmcConfig.saveTemps(config.getSaveTemps());
    hivmcConfig.injectBarrierAllSync(config.getEnableHIVMInjectBarrierAllSync());
    hivmcConfig.setExtraDeviceBCPaths(config.getLinkAicoreBitcode());
    hivmcConfig.setDisableFMA(config.getDisableFMA());
    hivmcConfig.setSaveLinkedIR(config.getSaveLinkedIR());
    hivmcConfig.setNumWarps(config.getNumWarps());
    hivmcConfig.setThreadsPerWarp(config.getThreadsPerWarp());
    hivmcConfig.setSharedDynamicSize(config.getSharedMemDynamicSize());
    hivmcConfig.simtStackLimit(config.getSimtStackLimit());
    hivmcConfig.tritonMetadataOutput(config.getTritonMetadataOutput());
    hivmcConfig.disableDecomposeReduction(config.getDisableDecomposeReduction());
    hivmcConfig.disableReorderInstruction(config.getDisableReorderInstruction());

    if (hivmcConfig.getTritonGridDim().size() > 3) {
        report_fatal_error(
            "Invalid --simt-triton-grid: at most 3 elements allowed x,y,z\n");
    }
    hivmcConfig.setOutputFile(config.getOutputFile());

    StringTmpPath path(hivmcConfig.outputFile());

    // TODO: investigate if this check is redundant
    llvm::cantFail(llvm::errorCodeToError(hivmcCanonicalizePath(path)));
    hivmcConfig.setOutputFile(path.str().str());
    return hivmcConfig;
}

} // namespace

static LogicalResult
runOrdinaryRegBasePipeline(ModuleOp mod, BiShengIRCompileMainConfig &config) {
  if (failed(checkInOutOptionsValidity(config))) {
    return failure();
  }

  // Overflow tracking: UB gets a Vector-only multi-buffer fallback; L1/L0C
  // overflow still uses the coarse enable-auto-multi-buffer switch.
  bool hasUboverflow = false;
  MLIRContext *ctx = mod->getContext();
  mlir::DiagnosticEngine &diagEngine = ctx->getDiagEngine();
  std::vector<Diagnostic> collectedDiagnostics;
  // Collect diagnostics and emit them afterwards because we have tuning
  // mechanism.
  auto handlerID = diagEngine.registerHandler([&](Diagnostic &diag) {
    // VF fusion may cause ub overflow. in this case, it will fallback to allop
    // fused to decrease ub occupation
    // Todo: use Enum to standardize the format of error message printing
    if (diag.getSeverity() == mlir::DiagnosticSeverity::Error) {
      std::string errMsg;
      llvm::raw_string_ostream errStream(errMsg);
      errStream << diag;
      const std::string &msg = errStream.str();
      if (msg.find("ub overflow") != std::string::npos) {
        hasUboverflow = true;
      }
    }
    collectedDiagnostics.emplace_back(std::move(diag));
  });

  bool hirCompileSuccess = false;
  int tryTimes = 2;
  // triton compile has nothing to do with HFusion auto schedule, so we don't
  // need to tune for it.
  //
  // TODO: refactor this ad-hoc retry loop into a dedicated retryPassManager
  // so each fallback policy is composable and explicit. Planned policies:
  //   - OpFusion retry policy: bump tiling-max-counter each attempt, up to 5
  //     retries (the current default branch).
  //   - AutoBlockify retry policy: progressively disable hoisting and then
  //     multi-buffer.
  //   - MultiBuffer retry policy: disable auto-multi-buffer.
  // Once the retryPassManager exists, the tryTimes / nested-if logic below
  // should be replaced by composing those policies.
  if (config.getEnableTuningMode()) {
    tryTimes = 1;
  }
  CompileFlow compileFlow = getCompileFlow(config);
  for (int i = 0; i < tryTimes; i++) {
    LDBG("Attempt number: " << i << " with max buffer count tuning delta: "
                            << config.getHfusionMaxBufferCountTuning());
    ModuleOp hirCompileMode = mod.clone();
    bool success = true;
    hasUboverflow = false;

    if (compileFlow == CompileFlow::Mixed) {
      success = runMixedPipelines(hirCompileMode, config);
    } else if (compileFlow == CompileFlow::PureSimt) {
      success =
          runModulePipeline(hirCompileMode, regbase::buildBiShengTTIRPipeline,
                            config, "BiShengSIMT");
    } else {
      // Standard HIR compile path (no SIMD/SIMT split)
      success =
          runModulePipeline(hirCompileMode, regbase::buildBiShengHIRPipeline,
                            config, "BiShengHIR");
      // Stop final HIVM lowering after earlier pipeline fails.
      success = success && runModulePipeline(hirCompileMode,
                                             regbase::buildFinalHIVMPipelines,
                                             config, "buildFinalHIVMPipelines");
    }
    if (!success) {
      if (hasUboverflow &&
          config.getVfFusionMode() == mlir::analysis::FusionMode::MaxParallel) {
        LDBG("ub overflow detected at attempt "
             << (i + 1) << "/" << tryTimes << ", fallback with all-op mode");
        config.setVfFusionMode(mlir::analysis::FusionMode::AllOp);
        // increase max buffers by 2 in HFusion auto schedule
        config.increaseHfusionMaxBufferCountTuning(2);
        collectedDiagnostics.clear();
        continue;
      }
      LDBG("can't retry at attempt " << (i + 1) << "/" << tryTimes
                                     << ", no fallback");
      break;
    }

    if (compileFlow == CompileFlow::Simd) {
      // Stop SIMD lowering after earlier pipeline fails.
      success = runModulePipeline(hirCompileMode,
                                  regbase::buildBiShengHIRAVEToLLVMPipeline,
                                  config, "BiShengSIMD");
    }


    addBitcodeAttrsToModule(hirCompileMode, config.getExecutablePath(), config);

    auto savedTemp = handleSaveTemps(hirCompileMode, config);
    if (failed(savedTemp)) {
      return failure();
    }

    auto hivmcConfig = HIVMCFromBiShengIRConfig(config);
    if (success && succeeded(runHIVMCCompileA5(hirCompileMode, hivmcConfig))) {
      hirCompileSuccess = true;
      mod = hirCompileMode.clone();
      break;
    }

    // increase max buffers by 2 in HFusion auto schedule
    config.increaseHfusionMaxBufferCountTuning(2);
  }

  // Restore to the default handler.
  diagEngine.eraseHandler(handlerID);

  if (!hirCompileSuccess) {
    for (auto &diag : llvm::reverse(collectedDiagnostics)) {
      diagEngine.emit(std::move(diag));
    }
    return failure();
  }

  if (config.shouldEnableCPURunner()) {
    auto fileHandle = mlir::openOutputFile(config.getOutputFile());
    assert(fileHandle != nullptr);
    fileHandle->os() << mod << '\n';
    fileHandle->keep();
    return success();
  }

  return success();
}

// New-L1 uses the active A5 lowering stages. The unrequested ordinary path
// below retains Maple4's existing orchestration and fallback behavior.
static LogicalResult
runProgramSplitRegBasePipeline(ModuleOp mod, BiShengIRCompileMainConfig config) {
  MLIRContext *ctx = mod->getContext();
  auto &diagEngine = ctx->getDiagEngine();
  std::vector<std::unique_ptr<Diagnostic>> collectedDiagnostics;
  // Ordinary fallback tuning must not change the separate retry configuration.
  const BiShengIRCompileMainConfig requestedConfig = config;
  const bool requestProgramSplit = config.getEnableProgramSplitMix();
  const bool effectiveAutoMultiBuffer = config.getEnableAutoMultiBuffer();
  // The first attempt always uses the ordinary kernel schedule. The option
  // selects the separate blockifier only on a proved capacity retry.
  if (requestProgramSplit)
    config.setEnableProgramSplitMix(false);
  const std::string splitResultPath =
      config.getOutputFile() + ".program_split.json";
  if (requestProgramSplit)
    (void)llvm::sys::fs::remove(splitResultPath);
  bool acceptedProgramSplit = false;
  bool candidateAccepted = false;
  OwningOpRef<ModuleOp> candidateModule;
  DictionaryAttr candidateRuntimeContract;
  DictionaryAttr candidateExternalContract;
  StringAttr candidateProof;
  std::string candidateReason = "not_requested";
  const std::string programSplitTarget =
      hacc::stringifyTargetDeviceEnum(config.getTarget()).str();
  const bool supportedProgramSplitTarget =
      hacc::utils::isAscend950(programSplitTarget);
  if (requestProgramSplit) {
    if (!effectiveAutoMultiBuffer) {
      candidateReason = "multibuffer_off";
    } else if (!supportedProgramSplitTarget) {
      candidateReason = "unsupported_target";
    } else {
      candidateModule = OwningOpRef<ModuleOp>(mod.clone());
      if (failed(runPipeline(*candidateModule,
          [](mlir::PassManager &pm) {
            pm.addPass(hivm::createL1PrefetchPSAnalysisPass());
          }, config, "L1-semantic-analysis")))
        return failure();
      auto entry = hivm::getL1PrefetchPSEntry(*candidateModule);
      if (!entry) {
        candidateReason = "multiple_or_missing_functions";
      } else {
        candidateAccepted = entry->hasAttr("l1_prefetch_ps.accepted");
        if (candidateAccepted) {
          std::string interfaceReason;
          auto targetSpec = hacc::utils::getNPUTargetSpecAttr(
              entry.getContext(), requestedConfig.getTarget(), entry.getLoc());
          auto contract = hivm::getL1PrefetchPSLaunchContract(
              entry, targetSpec, interfaceReason);
          if (failed(contract)) {
            if (interfaceReason.empty()) return failure();
            if (interfaceReason != "l1_reservation_exceeds_capacity") {
              entry.emitError("L1 pipeline error: ") << interfaceReason;
              return failure();
            }
            candidateAccepted = false;
            candidateReason = interfaceReason;
          } else {
            candidateRuntimeContract = *contract;
            auto external = hivm::getL1PrefetchPSExternalContract(entry);
            if (failed(external)) {
              entry.emitError("could not derive the accepted external launch contract");
              return failure();
            }
            candidateExternalContract = *external;
            candidateProof = entry->getAttrOfType<StringAttr>("l1_prefetch_ps.proof");
            candidateReason = "accepted";
          }
        } else {
          auto reason = entry->getAttrOfType<StringAttr>("l1_prefetch_ps.reject_reason");
          if (!reason) {
            entry.emitError("L1 pipeline error: semantic analysis produced no outcome");
            return failure();
          }
          candidateReason = reason.getValue().str();
        }
      }
    }
  }

  if (requestProgramSplit && !candidateAccepted) {
    llvm::errs() << "[program-split] reject: " << candidateReason << "\n";
    if (candidateReason == "preexisting_program_split_plan") {
      mod.emitError("L1 pipeline error: source-supplied split plan is not trusted");
      return failure();
    }
  }

  auto writeProgramSplitResult = [&]() -> LogicalResult {
    if (requestProgramSplit) {
      std::error_code ec;
      llvm::raw_fd_ostream result(splitResultPath, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        mod.emitError("could not write program-split compile result");
        return failure();
      }
      llvm::json::OStream json(result);
      json.object([&] {
        json.attribute("schema_version", 2);
        json.attribute("kind", acceptedProgramSplit ? "split_mix_half_grid" : "ordinary");
        if (acceptedProgramSplit) {
          // One serializer for all runtime contract kinds. Physical realization
          // continues to consume its internal dictionary, never this sidecar.
          std::function<void(mlir::Attribute)> writeValue = [&](mlir::Attribute value) {
            if (auto boolean = dyn_cast<BoolAttr>(value))
              json.value(boolean.getValue());
            else if (auto integer = dyn_cast<IntegerAttr>(value))
              json.value(integer.getInt());
            else if (auto string = dyn_cast<StringAttr>(value))
              json.value(string.getValue());
            else if (auto array = dyn_cast<ArrayAttr>(value)) {
              json.array([&] {
                for (mlir::Attribute item : array)
                  writeValue(item);
              });
            } else if (auto dictionary = dyn_cast<DictionaryAttr>(value)) {
              json.object([&] {
                for (NamedAttribute item : dictionary) {
                  json.attributeBegin(item.getName().strref());
                  writeValue(item.getValue());
                  json.attributeEnd();
                }
              });
            } else {
              llvm::report_fatal_error(
                  "unsupported external program-split attribute");
            }
          };
          for (NamedAttribute item : candidateExternalContract) {
            json.attributeBegin(item.getName().strref());
            writeValue(item.getValue());
            json.attributeEnd();
          }
          json.attribute("target", programSplitTarget);
        } else {
          json.attribute("reason", candidateAccepted ? "ordinary_compiled" : candidateReason);
        }
        json.attributeObject("compile_options", [&] {
          json.attribute("enable_vf_fusion", config.getEnableVFFusion());
          json.attribute("disable_vf_reachable_check", config.getDisableVFReachableCheck());
          json.attribute("hfusion_max_buffer_count_tuning", config.getHfusionMaxBufferCountTuning());
          json.attribute("enable_auto_multi_buffer", config.getEnableAutoMultiBuffer());
        });
      });
      result << "\n";
      result.close();
      if (result.has_error()) {
        mod.emitError("could not finish program-split compile result");
        return failure();
      }
    }
    return success();
  };


  auto handlerID = diagEngine.registerHandler([&](Diagnostic &diag) {
    collectedDiagnostics.push_back(std::make_unique<Diagnostic>(std::move(diag)));
  });
  auto runHIRAttempt = [&](ModuleOp attempt, BiShengIRCompileMainConfig &options) {
    switch (getCompileFlow(options)) {
    case CompileFlow::Mixed:
      return runMixedPipelines(attempt, options);
    case CompileFlow::PureSimt:
      return runModulePipeline(attempt, regbase::buildBiShengTTIRPipeline,
                               options, "BiShengSIMT");
    case CompileFlow::Simd:
      return runModulePipeline(attempt, regbase::buildBiShengHIRPipeline,
                               options, "BiShengHIR") &&
             runModulePipeline(attempt, regbase::buildFinalHIVMPipelines,
                               options, "buildFinalHIVMPipelines");
    }
    llvm_unreachable("unknown compile flow");
  };
  // A capacity-only failure is recoverable. Verifier, contract, lowering and
  // tool errors must not be hidden by switching to a different schedule.
  auto capacityOnlyFailure = [&](size_t start) {
    bool capacity = false;
    for (size_t i = start; i < collectedDiagnostics.size(); ++i) {
      const Diagnostic &diag = *collectedDiagnostics[i];
      if (diag.getSeverity() != mlir::DiagnosticSeverity::Error)
        continue;
      std::string message = diag.str();
      StringRef text(message);
      if (text == "L1 pipeline error: ping/pong reservation exceeds target L1 capacity") {
        capacity = true;
        continue;
      }
      if ((text.starts_with("ub overflow, requires ") ||
           text.starts_with("cbuf overflow, requires ") ||
           text.starts_with("l1 overflow, requires ")) &&
          text.contains(" bits available!")) {
        capacity = true;
        continue;
      }
      // RetriablePassManager adds this wrapper after the originating error.
      if (text.starts_with("Failed to run ") && text.contains(" pipeline"))
        continue;
      return false;
    }
    return capacity;
  };
  auto installResult = [&](OwningOpRef<ModuleOp> &attempt) {
    (*attempt)->removeAttr("hivm.defer_memory_fallback");
    mod.erase();
    mod = attempt.release();
    collectedDiagnostics.clear();
  };

  // true: compiled; false: an understood rejection permits ordinary fallback;
  // failure: an unexpected compiler error that must remain visible.
  auto tryBeforeMemoryFallback = [&]() -> FailureOr<bool> {
    llvm::errs() << "[program-split] trying ordinary MB before memory fallback\n";
    OwningOpRef<ModuleOp> ordinary(mod.clone());
    (*ordinary)->setAttr("hivm.defer_memory_fallback", UnitAttr::get(ctx));
    ordinary->walk([](func::FuncOp func) {
      func->removeAttr("program_split.memory_result");
    });
    if (runHIRAttempt(*ordinary, config)) {
      installResult(ordinary);
      return true;
    }
    if (!capacityOnlyFailure(0))
      return failure();
    if (!candidateAccepted)
      return false;

    // Selection marks live only in a diagnostic clone: ordinary compilation
    // is not changed by their effects on canonicalization or use counts.
    size_t probeStart = collectedDiagnostics.size();
    OwningOpRef<ModuleOp> probe(candidateModule->clone());
    (*probe)->setAttr("hivm.defer_memory_fallback", UnitAttr::get(ctx));
    bool probeSuccess = runHIRAttempt(*probe, config);
    if (!probeSuccess && !capacityOnlyFailure(probeStart))
      return failure();
    bool provenSelectedExcess = false;
    for (auto func : probe->getOps<func::FuncOp>()) {
      auto result = func->getAttrOfType<DictionaryAttr>("program_split.memory_result");
      auto proof = func->getAttrOfType<StringAttr>("l1_prefetch_ps.proof");
      if (!result || proof != candidateProof)
        continue;
      if (failed(hivm::verifyL1PrefetchPSRealizationRoles(func, candidateRuntimeContract)))
        return failure();
      auto inputs = candidateRuntimeContract.getAs<DenseI32ArrayAttr>("input_args");
      auto capacity = hivm::SelectedSetCapacity::decode(result, inputs.size());
      auto kind = result.getAs<StringAttr>("kind");
      if (kind && kind.getValue() == "proven_selected_set_capacity_excess" &&
          capacity && capacity->hasMultipleSlots() && capacity->exceedsCapacity())
        provenSelectedExcess = true;
    }
    collectedDiagnostics.resize(probeStart);
    if (!provenSelectedExcess)
      return false;

    llvm::errs() << "[program-split] trying New-L1 after selected-capacity proof\n";
    BiShengIRCompileMainConfig splitConfig = requestedConfig;
    splitConfig.setEnableProgramSplitMix(true);
    splitConfig.setEnableAutoMultiBuffer(false);
    splitConfig.setEnableMixedCV(true);
    splitConfig.setEnableAutoBindSubBlock(false);
    RetriablePassManager splitPm(splitConfig, ctx);
    OwningOpRef<ModuleOp> split(candidateModule->clone());
    const size_t splitStart = collectedDiagnostics.size();
    // A transform failure means an invalid/stale proof, not a capacity reject.
    if (failed(splitPm.runOnce(*split, [](mlir::PassManager &pm) {
          pm.addPass(hivm::createL1PrefetchPSTransformPass());
        }, "L1-transform")))
      return failure();
    // The acceptance fingerprint binds module attributes: add the physical
    // retry budget only after the transform has consumed the unchanged proof.
    (*split)->setAttr("hivm.defer_memory_fallback",
                     IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 20));
    if (!runHIRAttempt(*split, splitConfig)) {
      if (!capacityOnlyFailure(splitStart))
        return failure();
      llvm::errs() << "[program-split] physical capacity rejection; "
                      "continuing ordinary memory fallback\n";
      return false;
    }
    installResult(split);
    config = splitConfig;
    acceptedProgramSplit = true;
    return true;
  };


  FailureOr<bool> initial = false;
  if (effectiveAutoMultiBuffer && supportedProgramSplitTarget)
    initial = tryBeforeMemoryFallback();
  diagEngine.eraseHandler(handlerID);
  if (failed(initial)) {
    for (auto &diag : llvm::reverse(collectedDiagnostics))
      diagEngine.emit(std::move(*diag));
    return failure();
  }
  if (!*initial) {
    // No probe or rejected split IR is reused by the ordinary fallback.
    if (effectiveAutoMultiBuffer && supportedProgramSplitTarget) {
      config.setEnableAutoMultiBuffer(false);
      RetriablePassManager::emitFallbackNote(
          "capacity rejection after New-L1 opportunity",
          "enable-auto-multi-buffer", "false");
    }
    if (failed(runOrdinaryRegBasePipeline(mod, config)))
      return failure();
    return writeProgramSplitResult();
  }

  // Complete the accepted attempt using the same A5 code-generation stages
  // and bitcode selection as ordinary compilation. Emit no sidecar on failure.
  if (getCompileFlow(config) == CompileFlow::Simd &&
      !runModulePipeline(mod, regbase::buildBiShengHIRAVEToLLVMPipeline,
                         config, "BiShengSIMD"))
    return failure();
  addBitcodeAttrsToModule(mod, config.getExecutablePath(), config);
  if (failed(handleSaveTemps(mod, config)))
    return failure();
  auto hivmcConfig = HIVMCFromBiShengIRConfig(config);
  if (failed(runHIVMCCompileA5(mod, hivmcConfig)))
    return failure();
  if (config.shouldEnableCPURunner()) {
    auto fileHandle = mlir::openOutputFile(config.getOutputFile());
    if (!fileHandle)
      return failure();
    fileHandle->os() << mod << '\n';
    fileHandle->keep();
  }
  return writeProgramSplitResult();
}

LogicalResult
bishengir::regbase::runRegBasePipeline(ModuleOp mod,
                                     BiShengIRCompileMainConfig config) {
  if (config.getEnableProgramSplitMix())
    return runProgramSplitRegBasePipeline(mod, config);
  return runOrdinaryRegBasePipeline(mod, config);
}
