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
#include <map>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <ctime>
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

    // Vision-encoder output, keyed by mtmd's per-image id (a SHA-256 of the pixels).
    //
    // Encoding one keyframe costs ~13-17 s and dominates query time. Follow-up questions
    // retrieve largely the same frames as the question before them, so without this the
    // same pixels are re-encoded on every turn. Keyed by content hash rather than file
    // path, so an identical frame reached by a different route still hits.
    std::map<std::string, std::vector<float>> embd_cache;
    size_t embd_cache_bytes = 0;

    // Last generate() breakdown, read back by nativeGetLastGenStats(). Generation is the
    // largest phase of a query and there is no other way to see inside it on devices
    // whose ROM discards this app's logcat - vivo FuntouchOS does exactly that.
    int  last_gen_tokens    = 0;
    long last_prefill_ms    = 0;
    long last_gen_ms        = 0;
    bool last_hit_token_cap = false;
    int  last_cache_hits    = 0;
    int  last_cache_misses  = 0;
};

static long now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

// Roughly 1.8 MB per 640x360 keyframe (299 tokens x 1536 dims x 4 bytes). 192 MB holds
// a hundred or so frames, which is far more than one conversation revisits.
static const size_t EMBD_CACHE_MAX_BYTES = 192ull * 1024 * 1024;

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
        // Hard ceiling on tokens per image. Encode cost scales with token count, and
        // encoding dominates query latency: on a Vivo I2304 five frames cost ~130 s of
        // a ~150 s query.
        //
        // Raised back to 512 after 256 cost recall: asked for "people wearing pink
        // colour costumes" the model answered "not visible" on frames where several
        // people in vivid pink fill the centre of shot. 640x360 keyframes come out at
        // ~299 tokens naturally, so 512 does not bind and 256 was actively truncating
        // detail - for roughly a 10% saving, since the frames were already under the
        // old cap. Upstream warns Qwen-VL prefers >=1024 tokens for grounding, so this
        // ceiling exists to bound pathological inputs, not to trim normal ones.
        mp.image_max_tokens = 512;

        // NOT setting image_min_tokens, despite upstream clip.cpp warning that Qwen-VL
        // "require[s] at minimum 1024 image tokens to function correctly on grounding
        // tasks" (ggml-org/llama.cpp#16842). Measured on the API-34 emulator, raising
        // frames from 299 to 1032 tokens cost:
        //   encode  18.3 s -> 58 s per frame
        //   decode   8.6 s -> 28 s per frame
        //   generate  fast -> 461 s (every sampled token attends over 5,160 image
        //                            tokens instead of 1,495)
        //   total    ~3 min -> ~15 min for one question
        // Encode scales linearly with tokens, but generation scales with the whole
        // context, so the cost lands three times over.
        //
        // It bought nothing, because resolution was not the binding constraint. The
        // retrieved frames were near-tied in CLIP score (0.222/0.219/0.212/0.210/0.209
        // for "what vehicles are visible" - a 0.013 spread, i.e. noise), so the model
        // was being handed arbitrary frames. Encoding the wrong frame more sharply does
        // not make it the right frame. Fix retrieval first; revisit this afterwards,
        // and if it comes back, pair it with a lower MAX_FRAMES_TO_ANALYSE so the
        // frames x tokens product stays near today's 1,495.

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

    const long t_prefill_start = now_ms();
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const int32_t rc = mtmd_tokenize(ctx->ctx_mtmd, chunks, &text,
                                     (const mtmd_bitmap **) bitmaps.data(), bitmaps.size());
    for (auto * b : bitmaps) mtmd_bitmap_free(b);

    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        LOGE("mtmd_tokenize failed: %d", rc);
        return env->NewStringUTF("Error: failed to tokenize multimodal prompt.");
    }

    // Walk the chunks ourselves rather than calling mtmd_helper_eval_chunks, so an image
    // already encoded on a previous turn can be decoded straight from cache. Text chunks
    // still go through the helper.
    llama_pos n_past = 0;
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    const int32_t n_embd  = llama_model_n_embd(ctx->model);
    int32_t eval_rc = 0;
    int cache_hits = 0, cache_misses = 0;

    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        const bool is_last = (i + 1 == n_chunks);

        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            eval_rc = mtmd_helper_eval_chunk_single(
                ctx->ctx_mtmd, ctx->ctx_llama, chunk,
                n_past, /*seq_id*/ 0, /*n_batch*/ 2048, is_last, &n_past);
            if (eval_rc != 0) break;
            continue;
        }

        const char * id = mtmd_input_chunk_get_id(chunk);
        const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk);
        const size_t need = n_tok * (size_t) n_embd;
        const std::string key = id ? id : "";

        auto it = key.empty() ? ctx->embd_cache.end() : ctx->embd_cache.find(key);
        if (it == ctx->embd_cache.end()) {
            cache_misses++;
            if (mtmd_encode_chunk(ctx->ctx_mtmd, chunk) != 0) {
                LOGE("mtmd_encode_chunk failed at chunk %zu", i);
                eval_rc = 1; break;
            }
            const float * out = mtmd_get_output_embd(ctx->ctx_mtmd);
            if (!out) { LOGE("no output embeddings"); eval_rc = 1; break; }

            if (!key.empty() && ctx->embd_cache_bytes + need * sizeof(float) <= EMBD_CACHE_MAX_BYTES) {
                ctx->embd_cache.emplace(key, std::vector<float>(out, out + need));
                ctx->embd_cache_bytes += need * sizeof(float);
                it = ctx->embd_cache.find(key);
            } else {
                // cache full (or unkeyed): decode straight from the encoder output
                eval_rc = mtmd_helper_decode_image_chunk(
                    ctx->ctx_mtmd, ctx->ctx_llama, chunk,
                    const_cast<float *>(out), n_past, /*seq_id*/ 0, /*n_batch*/ 2048,
                    &n_past, nullptr, nullptr);
                if (eval_rc != 0) break;
                continue;
            }
        } else {
            cache_hits++;
        }

        eval_rc = mtmd_helper_decode_image_chunk(
            ctx->ctx_mtmd, ctx->ctx_llama, chunk,
            it->second.data(), n_past, /*seq_id*/ 0, /*n_batch*/ 2048,
            &n_past, nullptr, nullptr);
        if (eval_rc != 0) break;
    }
    mtmd_input_chunks_free(chunks);

    ctx->last_prefill_ms   = now_ms() - t_prefill_start;
    ctx->last_cache_hits   = cache_hits;
    ctx->last_cache_misses = cache_misses;
    LOGI("image encode cache: %d hit, %d miss (%zu entries, %.1f MB)",
         cache_hits, cache_misses, ctx->embd_cache.size(),
         ctx->embd_cache_bytes / 1e6);

    if (eval_rc != 0) {
        LOGE("multimodal eval failed: %d", eval_rc);
        return env->NewStringUTF("Error: multimodal forward pass failed.");
    }

    const llama_vocab * vocab = llama_model_get_vocab(ctx->model);
    std::ostringstream oss;
    // a per-frame narration over 5 keyframes plus a summary needs more room
    // than a single verdict sentence
    const int max_new_tokens = 400;

    const long t_gen_start = now_ms();
    long t_bucket = t_gen_start;
    int  generated = 0;
    bool hit_cap   = true;          // cleared by the end-of-generation break below
    for (int i = 0; i < max_new_tokens; ++i) {
        const llama_token tok = llama_sampler_sample(ctx->sampler, ctx->ctx_llama, -1);
        llama_sampler_accept(ctx->sampler, tok);

        if (llama_vocab_is_eog(vocab, tok)) { hit_cap = false; break; }

        char piece[256];
        const int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        if (n > 0) oss.write(piece, n);

        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&tok), 1);
        if (llama_decode(ctx->ctx_llama, batch) != 0) {
            LOGW("llama_decode failed at token %d — stopping", i);
            break;
        }
        n_past++;
        generated++;
        // Per-token cost must stay flat. If it climbs with position the KV cache is not
        // being reused and every token is re-reading the whole context - which is what a
        // 10x-too-slow sampler would look like from outside.
        if (generated % 10 == 0) {
            const long now = now_ms();
            LOGI("gen tokens %d-%d: %ld ms (%.1f ms/tok, n_past=%d)",
                 generated - 10, generated, now - t_bucket, (now - t_bucket) / 10.0, n_past);
            t_bucket = now;
        }
    }

    ctx->last_gen_tokens    = generated;
    ctx->last_gen_ms        = now_ms() - t_gen_start;
    ctx->last_hit_token_cap = hit_cap;
    LOGI("generation: %d tokens in %ld ms (%.2f tok/s), cap_hit=%d, prefill %ld ms",
         generated, ctx->last_gen_ms,
         ctx->last_gen_ms > 0 ? 1000.0 * generated / ctx->last_gen_ms : 0.0,
         (int) hit_cap, ctx->last_prefill_ms);

    std::string out = oss.str();
    if (out.empty()) out = "Error: model produced no output.";
    return env->NewStringUTF(out.c_str());
}

/**
 * Breakdown of the most recent generate(), as "key=value" pairs.
 *
 * Query latency is dominated by generation - about 58 s of a 117 s query on an SD8Gen2 -
 * yet a 2B model at Q4 should sample far faster than the ~1.2 tok/s that implies. Without
 * a token count and a rate there is no way to tell a slow sampler from a model quietly
 * emitting 400 tokens and having most of them trimmed. Exposed through JNI rather than a
 * log line because vivo's ROM discards this app's logcat entirely.
 */
extern "C" JNIEXPORT jstring JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeGetLastGenStats(
        JNIEnv * env, jobject /*thiz*/, jlong handle) {

    auto * ctx = reinterpret_cast<NativeVLMContext *>(handle);
    if (!ctx) return env->NewStringUTF("");

    char buf[320];
    snprintf(buf, sizeof(buf),
             "gen_tokens=%d gen_ms=%ld tok_per_s=%.2f cap_hit=%d prefill_ms=%ld "
             "cache_hits=%d cache_misses=%d",
             ctx->last_gen_tokens, ctx->last_gen_ms,
             ctx->last_gen_ms > 0 ? 1000.0 * ctx->last_gen_tokens / ctx->last_gen_ms : 0.0,
             (int) ctx->last_hit_token_cap, ctx->last_prefill_ms,
             ctx->last_cache_hits, ctx->last_cache_misses);
    return env->NewStringUTF(buf);
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
