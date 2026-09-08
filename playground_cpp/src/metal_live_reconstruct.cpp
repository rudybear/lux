#include "metal_live_reconstruct.h"
#include "metal_context.h"
#include "metal_scene_manager.h"

MetalLiveReconstruct::~MetalLiveReconstruct() {
    if (unetOutputBuffer_) unetOutputBuffer_->release();
    if (reconOutputTex_) reconOutputTex_->release();
}

void MetalLiveReconstruct::init(MetalContext& ctx, const GaussianSplatData& proxySceneData,
                                 const InitParams& p) {
    proxyW_ = p.proxyW;
    proxyH_ = p.proxyH;
    targetW_ = p.targetW;
    targetH_ = p.targetH;

    // Proxy-res splat render (DLSS attachments: colour + packed aux
    // motion/depth + optional foreground_coverage) -- the same luxc-compiled
    // pipeline/backend the network was trained against.
    splatRProxy_ = std::make_unique<MetalSplatLuxcRenderer>();
    splatRProxy_->init(ctx, proxySceneData, p.shaderBase, proxyW_, proxyH_);
    splatRProxy_->setSortSchedule(p.sortEveryNFrames, p.sortViewThresholdDeg);

    netInput_.init(ctx, p.textureNpyPath, p.bgSphereNpyPath, proxyW_, proxyH_, p.paramStride, p.hiddenChannels);
    if (splatRProxy_->hasForegroundCoverage()) {
        netInput_.setFgSource(NetInputAssembly::kFgSourceFgTexture);
    }
    netW_ = netInput_.getNetW();
    netH_ = netInput_.getNetH();

    unet_.init(ctx, p.unetWeightsBinPath, p.unetLayersTxtPath, netW_, netH_);
    unetOutputBuffer_ = ctx.newBuffer(
        static_cast<size_t>(netW_) * netH_ * unet_.getOutChannels() * sizeof(uint16_t),
        MTL::ResourceStorageModeShared);

    // nBlend isn't stored directly in the manifest -- derive it from
    // out_channels = sp*sp*K*K + nBlend*sp*sp + hidden (same derivation
    // SplatView.mm's -loadSceneAndInitRenderer used to do inline).
    uint32_t sp = unet_.getUpscale() * unet_.getParamStride();
    uint32_t kk = unet_.getK() * unet_.getK();
    nBlend_ = (unet_.getOutChannels() - sp * sp * kk - unet_.getHiddenChannels()) / (sp * sp);

    reconstruct_.init(ctx, p.textureNpyPath, p.bgSphereNpyPath, p.memoryHeadNpzPath, targetW_, targetH_, proxyW_,
                       proxyH_, netW_, netH_, unet_.getK(), unet_.getHiddenChannels(), nBlend_,
                       targetW_ / proxyW_, unet_.getParamStride());

    auto* reconDesc =
        MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, targetW_, targetH_, false);
    reconDesc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    reconDesc->setStorageMode(MTL::StorageModePrivate);
    reconOutputTex_ = ctx.newTexture(reconDesc);
}

void MetalLiveReconstruct::applyCameraAndJitter(const FrameInputs& in) {
    splatRProxy_->updateCameraExplicit(in.proxyCur.eye, in.proxyCur.viewGl, in.proxyCur.proj, in.proxyCur.fx,
                                        in.proxyCur.fy);
    splatRProxy_->setPreviousCameraExplicit(in.proxyPrev.viewGl, in.proxyPrev.proj);
    splatRProxy_->setJitter(in.jitterProxyX, in.jitterProxyY);
}

double MetalLiveReconstruct::gpuMsAcrossPossibleSplit(MTL::CommandBuffer* before, MTL::CommandBuffer* after) {
    double ms = (after->GPUEndTime() - after->GPUStartTime()) * 1000.0;
    if (before != after) {
        before->waitUntilCompleted();
        ms += (before->GPUEndTime() - before->GPUStartTime()) * 1000.0;
    }
    return ms;
}

MTL::CommandBuffer* MetalLiveReconstruct::encodeFrame(MetalContext& ctx, MTL::CommandBuffer* cmdBuf,
                                                       const FrameInputs& in, uint32_t frameInFlightIndex) {
    applyCameraAndJitter(in);

    if (splatRProxy_->hasMotion()) {
        // Folds this frame's morph evaluation into the SAME shared command
        // buffer (setMorphTime's sharedCmdBuf overload) instead of a
        // separate blocking round trip -- see that method's own comment.
        splatRProxy_->setMorphTime(in.morphTimeCur, cmdBuf);
        // Guarded internally by firstMvFrame_: a real (blocking, 2-round-trip)
        // cost only on the very first frame this renderer instance ever
        // runs; a no-op every frame after (render()/encodeFrame()'s own GPU
        // ping-pong already keeps prevPosBuffer_ current from then on).
        splatRProxy_->seedPreviousMorphTime(in.morphTimePrev);
    }

    // Proxy render: preprocess + sort + draw, fused into `cmdBuf`.
    splatRProxy_->encodeFrame(ctx, cmdBuf, frameInFlightIndex);

    // Reconstruct step 0 (must run BEFORE net input assembly/UNet this
    // frame): warp+downsample last frame's raw hidden state into this
    // frame's net-res hidden_in, using THIS frame's just-rendered aux/motion
    // texture (backward warp from last frame -> this frame).
    bool isFirstReconFrame = netInput_.isNextFrameFirst();
    if (isFirstReconFrame) {
        // Task A warm start: right after a resetHistory() (or on this
        // instance's very first frame ever), seed prevColor_'s ping-pong
        // slots from THIS frame's own just-rendered proxy colour rather
        // than leaving them undefined GPU memory -- see
        // LiveReconstructPass::seedHistoryFromProxy()'s own doc comment for
        // why this does NOT change this particular frame's own output
        // (disocc is forced to 1 here regardless, matching
        // mobiledlss.train.train.rollout's own t==0 convention, which zeroes
        // the history blend weight either way).
        reconstruct_.seedHistoryFromProxy(ctx, cmdBuf, splatRProxy_->getOutputTexture());
    }
    reconstruct_.prepareHiddenInput(ctx, cmdBuf, splatRProxy_->getAuxTexture(), isFirstReconFrame,
                                     reconstruct_.getHiddenInputBuffer());

    // Net input assembly (26-ch NHWC input from this frame's proxy DLSS
    // attachments, hidden_in from the step above).
    netInput_.run(ctx, cmdBuf, splatRProxy_->getOutputTexture(), splatRProxy_->getAuxTexture(),
                  splatRProxy_->hasForegroundCoverage() ? splatRProxy_->getFgTexture() : nullptr,
                  reconstruct_.getHiddenInputBuffer(), in.proxyCur.eye, in.proxyCur.r, in.proxyCur.u,
                  in.proxyCur.f, in.proxyCur.fx, in.proxyCur.fy, 0.5f * static_cast<float>(proxyW_),
                  0.5f * static_cast<float>(proxyH_), in.jitterProxyX, in.jitterProxyY);

    // Capture depth/first-frame state BEFORE the UNet/reconstruct calls
    // below (netInput_.run() already flipped its own ping-pong index at the
    // end of the call above -- these two accessors already account for
    // that, see their own comments).
    MTL::Texture* curDepth = netInput_.getDepthWrittenThisFrame();
    MTL::Texture* prevDepth = netInput_.getDepthFromPreviousFrame();
    bool wasFirst = netInput_.wasFirstFrame();

    // UNet -- may return a different (MPSGraph-swapped) command buffer; all
    // subsequent encoding in this call, and the caller's own trailing work,
    // MUST use the returned pointer.
    MTL::CommandBuffer* cur = unet_.encode(ctx, cmdBuf, netInput_.getOutputBuffer(), unetOutputBuffer_);

    // Reconstruct-with-memory -> getReconOutputTexture().
    reconstruct_.run(ctx, cur, splatRProxy_->getOutputTexture(), splatRProxy_->getAuxTexture(), curDepth, prevDepth,
                      wasFirst, unetOutputBuffer_, reconOutputTex_, in.targetCur.eye, in.targetCur.r,
                      in.targetCur.u, in.targetCur.f, in.targetCur.fx, in.targetCur.fy,
                      0.5f * static_cast<float>(targetW_), 0.5f * static_cast<float>(targetH_), in.jitterTargetX,
                      in.jitterTargetY);

    return cur;
}

MetalLiveReconstruct::ProfiledResult MetalLiveReconstruct::encodeFrameProfiled(MetalContext& ctx,
                                                                                const FrameInputs& in) {
    ProfiledResult result;
    applyCameraAndJitter(in);

    if (splatRProxy_->hasMotion()) {
        // No shared cmdBuf here -- renderProfiled-style diagnostic path
        // pays its own CPU<->GPU round trips deliberately (see class
        // comment); a plain blocking setMorphTime() is consistent with that.
        splatRProxy_->setMorphTime(in.morphTimeCur);
        splatRProxy_->seedPreviousMorphTime(in.morphTimePrev);
    }

    // Stage 1: proxy render, own command buffer -- render() already reports
    // real fused GPU time via getLastGpuTotalMs().
    splatRProxy_->render(ctx);
    result.proxyGpuMs = splatRProxy_->getLastGpuTotalMs();
    result.numCommandBuffers += 1;

    // Stage 2: reconstruct's hidden warp/downsample + net input assembly,
    // fused into one command buffer (these two are tightly coupled --
    // splitting them further wouldn't isolate anything meaningful).
    {
        auto* stageCmdBuf = ctx.beginCommandBuffer();
        bool isFirstReconFrame = netInput_.isNextFrameFirst();
        reconstruct_.prepareHiddenInput(ctx, stageCmdBuf, splatRProxy_->getAuxTexture(), isFirstReconFrame,
                                         reconstruct_.getHiddenInputBuffer());
        netInput_.run(ctx, stageCmdBuf, splatRProxy_->getOutputTexture(), splatRProxy_->getAuxTexture(),
                      splatRProxy_->hasForegroundCoverage() ? splatRProxy_->getFgTexture() : nullptr,
                      reconstruct_.getHiddenInputBuffer(), in.proxyCur.eye, in.proxyCur.r, in.proxyCur.u,
                      in.proxyCur.f, in.proxyCur.fx, in.proxyCur.fy, 0.5f * static_cast<float>(proxyW_),
                      0.5f * static_cast<float>(proxyH_), in.jitterProxyX, in.jitterProxyY);
        stageCmdBuf->commit();
        stageCmdBuf->waitUntilCompleted();
        result.inputGpuMs = (stageCmdBuf->GPUEndTime() - stageCmdBuf->GPUStartTime()) * 1000.0;
        result.numCommandBuffers += 1;
    }

    MTL::Texture* curDepth = netInput_.getDepthWrittenThisFrame();
    MTL::Texture* prevDepth = netInput_.getDepthFromPreviousFrame();
    bool wasFirst = netInput_.wasFirstFrame();

    // Stage 3: UNet, own command buffer.
    {
        auto* stageCmdBuf = ctx.beginCommandBuffer();
        MTL::CommandBuffer* cur = unet_.encode(ctx, stageCmdBuf, netInput_.getOutputBuffer(), unetOutputBuffer_);
        cur->commit();
        cur->waitUntilCompleted();
        // See gpuMsAcrossPossibleSplit()'s own comment: MPSGraph may have
        // already committed `stageCmdBuf` internally (commitAndContinue)
        // before swapping to `cur` -- reading `cur`'s own timestamps alone
        // would silently drop most (sometimes ~all) of the real UNet GPU
        // cost.
        result.netGpuMs = gpuMsAcrossPossibleSplit(stageCmdBuf, cur);
        result.numCommandBuffers += (stageCmdBuf != cur) ? 2 : 1;
        cur->release();  // balances encode()'s extra retain -- see its own doc comment.
    }

    // Stage 4: reconstruct, own command buffer.
    {
        auto* stageCmdBuf = ctx.beginCommandBuffer();
        reconstruct_.run(ctx, stageCmdBuf, splatRProxy_->getOutputTexture(), splatRProxy_->getAuxTexture(),
                          curDepth, prevDepth, wasFirst, unetOutputBuffer_, reconOutputTex_, in.targetCur.eye,
                          in.targetCur.r, in.targetCur.u, in.targetCur.f, in.targetCur.fx, in.targetCur.fy,
                          0.5f * static_cast<float>(targetW_), 0.5f * static_cast<float>(targetH_),
                          in.jitterTargetX, in.jitterTargetY);
        stageCmdBuf->commit();
        stageCmdBuf->waitUntilCompleted();
        result.reconGpuMs = (stageCmdBuf->GPUEndTime() - stageCmdBuf->GPUStartTime()) * 1000.0;
        result.numCommandBuffers += 1;
    }

    result.totalGpuMs = result.proxyGpuMs + result.inputGpuMs + result.netGpuMs + result.reconGpuMs;
    return result;
}
