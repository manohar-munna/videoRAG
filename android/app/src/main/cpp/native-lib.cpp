// JNI bridge for com.cctv.videorag.llm.OnDeviceVLM
//
// Rewritten against current llama.cpp + libmtmd. The previous implementation
// targeted a May-2024 vendored snapshot and could never load a Qwen2-VL GGUF:
// that build had no qwen2vl architecture, no M-RoPE and no qwen2vl_merger
// projector, so llama_load_model_from_file() returned nullptr, nativeHandle
// stayed 0, and every call site in Kotlin silently fell through to a template.
//
// Two structural fixes beyond the API migration:
//   1. Images are no longer evaluated at position 0 ahead of the prompt.
//      mtmd_tokenize() interleaves text and image chunks in template order.
//   2. Greedy argmax over the whole vocab is replaced by a sampler chain.

#include <jni.h>
#include <string>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <android/log.h>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define TAG "VideoRAG_Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

struct NativeVLMContext {
    llama_model   * model     = nullptr;
    llama_context * ctx_llama = nullptr;
    mtmd_context  * ctx_mtmd  = nullptr;
    llama_sampler * sampler   = nullptr;
    std::string     model_path;
    int             n_threads = 5;
    bool            has_vision = false;
};

// Route llama/ggml logging into logcat instead of stderr, which is discarded on Android.
void log_to_logcat(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (!text) return;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: LOGE("%s", text); break;
        case GGML_LOG_LEVEL_WARN:  LOGW("%s", text); break;
        default:                   LOGI("%s", text); break;
    }
}

std::string jstring_to_std(JNIEnv * env, jstring s) {
    if (!s) return {};
    const char * c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeInitWithFiles(
        JNIEnv * env, jobject /*thiz*/,
        jstring modelPath, jstring mmprojPath, jint layersToOffload) {

    const std::string m_path = jstring_to_std(env, modelPath);
    const std::string p_path = jstring_to_std(env, mmprojPath);

    llama_log_set(log_to_logcat, nullptr);
    mtmd_helper_log_set(log_to_logcat, nullptr);

    llama_backend_init();

    LOGI("Loading model: %s", m_path.c_str());

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = layersToOffload;

    llama_model * model = llama_model_load_from_file(m_path.c_str(), mparams);
    if (!model) {
        // Reaches here if the architecture is unknown to this llama.cpp build.
        LOGE("llama_model_load_from_file failed for %s", m_path.c_str());
        return 0L;
    }

    llama_context_params cparams = llama_context_default_params();
    // Sizing note: image tokens scale with SOURCE resolution, not with anything we
    // choose here. VideoFrameDecoder emits frames at the video's native size, so a
    // 1280x720 clip yields ~1125 tokens per frame - five of those is 5,625 and
    // overflows a 4096 context. Frames are now downscaled at decode AND capped via
    // image_max_tokens below; this ceiling is the third layer of protection.
    cparams.n_ctx   = 8192;
    cparams.n_batch = 2048;   // image chunks are submitted in one batch
    cparams.n_threads       = 5;
    cparams.n_threads_batch = 5;

    llama_context * lctx = llama_init_from_model(model, cparams);
    if (!lctx) {
        LOGE("llama_init_from_model failed");
        llama_model_free(model);
        return 0L;
    }

    auto * ctx = new NativeVLMContext();
    ctx->model      = model;
    ctx->ctx_llama  = lctx;
    ctx->model_path = m_path;
    ctx->n_threads  = 5;

    if (!p_path.empty() && access(p_path.c_str(), R_OK) == 0) {
        LOGI("Loading multimodal projector: %s", p_path.c_str());
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu        = (layersToOffload > 0);
        mp.print_timings  = true;
        mp.n_threads      = ctx->n_threads;
        mp.media_marker   = mtmd_default_marker();
        // Hard ceiling on tokens per image. Encode cost scales with token count:
        // measured ~20.3 s at 264 tokens (640x360) on an SD8Gen2, so an uncapped
        // 720p frame at ~1125 tokens costs ~87 s. Upstream warns Qwen-VL wants >=1024
        // tokens for grounding accuracy, so this is an explicit speed/accuracy trade
        // and deliberately sits at the top of the affordable range rather than the
        // bottom of the accurate one.
        mp.image_max_tokens = 512;

        ctx->ctx_mtmd = mtmd_init_from_file(p_path.c_str(), model, mp);
        if (!ctx->ctx_mtmd) {
            LOGE("mtmd_init_from_file failed — continuing text-only");
        } else {
            ctx->has_vision = mtmd_support_vision(ctx->ctx_mtmd);
            LOGI("Projector loaded (vision=%d)", (int) ctx->has_vision);
        }
    } else {
        LOGW("No readable mmproj at '%s' — text-only mode", p_path.c_str());
    }

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    ctx->sampler = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(ctx->sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(ctx->sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(ctx->sampler, llama_sampler_init_temp(0.2f));
    llama_sampler_chain_add(ctx->sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    LOGI("VLM initialised (handle=%p)", (void *) ctx);
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeInit(
        JNIEnv * env, jobject thiz, jstring modelDir, jint layersToOffload) {

    const std::string dir = jstring_to_std(env, modelDir);
    const std::string m   = dir + "/Qwen2-VL-2B-Instruct-Q4_K_M.gguf";
    // Q8_0 is preferred: measured ~20.3s/frame vs >=72s for f16 on an SD8Gen2,
    // 710MB vs 1.33GB resident, and no measurable loss of vision quality.
    const std::string q8  = dir + "/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf";
    const std::string f16 = dir + "/mmproj-Qwen2-VL-2B-Instruct-f16.gguf";
    const std::string p   = (access(q8.c_str(), R_OK) == 0) ? q8 : f16;

    jstring jm = env->NewStringUTF(m.c_str());
    jstring jp = env->NewStringUTF(p.c_str());
    jlong h = Java_com_cctv_videorag_llm_OnDeviceVLM_nativeInitWithFiles(env, thiz, jm, jp, layersToOffload);
    env->DeleteLocalRef(jm);
    env->DeleteLocalRef(jp);
    return h;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeGenerate(
        JNIEnv * env, jobject /*thiz*/,
        jlong handle, jstring prompt, jobjectArray imagePaths) {

    auto * ctx = reinterpret_cast<NativeVLMContext *>(handle);
    if (!ctx || !ctx->ctx_llama) {
        return env->NewStringUTF("Error: native VLM engine is not loaded.");
    }

    std::string prompt_str = jstring_to_std(env, prompt);
    const jsize n_images = imagePaths ? env->GetArrayLength(imagePaths) : 0;

    // Start each generation from a clean KV cache so calls do not accumulate.
    llama_memory_clear(llama_get_memory(ctx->ctx_llama), true);

    std::vector<mtmd_bitmap *> bitmaps;
    if (ctx->ctx_mtmd && ctx->has_vision) {
        for (jsize i = 0; i < n_images; ++i) {
            auto jp = (jstring) env->GetObjectArrayElement(imagePaths, i);
            const std::string path = jstring_to_std(env, jp);
            env->DeleteLocalRef(jp);
            if (path.empty() || access(path.c_str(), R_OK) != 0) {
                LOGW("Skipping unreadable image: %s", path.c_str());
                continue;
            }
            auto wrap = mtmd_helper_bitmap_init_from_file(
                ctx->ctx_mtmd, path.c_str(), false, mtmd_helper_init_opt_default());
            if (!wrap.bitmap) {
                LOGW("Failed to decode image: %s", path.c_str());
                continue;
            }
            bitmaps.push_back(wrap.bitmap);
        }
    }

    // mtmd places each image where a marker appears. If the caller supplied no
    // markers, prepend one per decoded image so ordering stays deterministic.
    const std::string marker = mtmd_default_marker();
    if (!bitmaps.empty() && prompt_str.find(marker) == std::string::npos) {
        std::string prefix;
        for (size_t i = 0; i < bitmaps.size(); ++i) prefix += marker + "\n";
        prompt_str = prefix + prompt_str;
    }

    mtmd_input_text text{};
    text.text          = prompt_str.c_str();
    text.text_len      = prompt_str.size();
    text.add_special   = false;  // caller supplies the full chat template
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const int32_t rc = mtmd_tokenize(ctx->ctx_mtmd, chunks, &text,
                                     (const mtmd_bitmap **) bitmaps.data(), bitmaps.size());
    for (auto * b : bitmaps) mtmd_bitmap_free(b);

    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        LOGE("mtmd_tokenize failed: %d", rc);
        return env->NewStringUTF("Error: failed to tokenize multimodal prompt.");
    }

    llama_pos n_past = 0;
    const int32_t eval_rc = mtmd_helper_eval_chunks(
        ctx->ctx_mtmd, ctx->ctx_llama, chunks,
        /*n_past*/ 0, /*seq_id*/ 0, /*n_batch*/ 2048,
        /*logits_last*/ true, &n_past);
    mtmd_input_chunks_free(chunks);

    if (eval_rc != 0) {
        LOGE("mtmd_helper_eval_chunks failed: %d", eval_rc);
        return env->NewStringUTF("Error: multimodal forward pass failed.");
    }

    const llama_vocab * vocab = llama_model_get_vocab(ctx->model);
    std::ostringstream oss;
    const int max_new_tokens = 256;

    for (int i = 0; i < max_new_tokens; ++i) {
        const llama_token tok = llama_sampler_sample(ctx->sampler, ctx->ctx_llama, -1);
        llama_sampler_accept(ctx->sampler, tok);

        if (llama_vocab_is_eog(vocab, tok)) break;

        char piece[256];
        const int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        if (n > 0) oss.write(piece, n);

        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&tok), 1);
        if (llama_decode(ctx->ctx_llama, batch) != 0) {
            LOGW("llama_decode failed at token %d — stopping", i);
            break;
        }
        n_past++;
    }

    std::string out = oss.str();
    if (out.empty()) out = "Error: model produced no output.";
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeGetModelInfo(
        JNIEnv * env, jobject /*thiz*/, jlong handle) {

    auto * ctx = reinterpret_cast<NativeVLMContext *>(handle);
    if (!ctx || !ctx->ctx_llama) return env->NewStringUTF("Not initialized");

    const size_t slash = ctx->model_path.find_last_of('/');
    const std::string name = (slash == std::string::npos)
        ? ctx->model_path : ctx->model_path.substr(slash + 1);

    char buf[256];
    snprintf(buf, sizeof(buf), "%s (arm64, %d threads, vision=%s)",
             name.c_str(), ctx->n_threads, ctx->has_vision ? "yes" : "no");
    return env->NewStringUTF(buf);
}

extern "C" JNIEXPORT void JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeClose(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {

    auto * ctx = reinterpret_cast<NativeVLMContext *>(handle);
    if (!ctx) return;
    LOGI("Releasing native VLM context");
    if (ctx->sampler)   llama_sampler_free(ctx->sampler);
    if (ctx->ctx_mtmd)  mtmd_free(ctx->ctx_mtmd);
    if (ctx->ctx_llama) llama_free(ctx->ctx_llama);
    if (ctx->model)     llama_model_free(ctx->model);
    delete ctx;
}
