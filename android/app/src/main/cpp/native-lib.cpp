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
#include <unordered_map>
#include <list>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <ctime>
#include <cctype>
#include <cstdio>
#include <algorithm>
#include <sys/stat.h>
#include <dirent.h>
#include <atomic>
#include <android/log.h>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define TAG "VideoRAG_Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

// Roughly 1.8 MB per 640x360 keyframe (299 tokens x 1536 dims x 4 bytes). 192 MB holds
// a hundred or so frames, which is far more than one conversation revisits.
static const size_t EMBD_CACHE_MAX_BYTES = 192ull * 1024 * 1024;

struct NativeVLMContext {
    llama_model   * model     = nullptr;
    llama_context * ctx_llama = nullptr;
    mtmd_context  * ctx_mtmd  = nullptr;
    llama_sampler * sampler   = nullptr;
    std::string     model_path;
    int             n_threads = 5;
    bool            has_vision = false;
    std::atomic<bool> abort_requested{false};

    // Vision-encoder output, keyed by mtmd's per-image id (a SHA-256 of the pixels).
    //
    // Encoding one keyframe costs ~13-17 s and dominates query time. Follow-up questions
    // retrieve largely the same frames as the question before them, so without this the
    // same pixels are re-encoded on every turn. Keyed by content hash rather than file
    // path, so an identical frame reached by a different route still hits.
    // Employs LRU eviction so memory stays strictly capped under EMBD_CACHE_MAX_BYTES.
    struct EmbdEntry {
        std::vector<float> data;
        std::list<std::string>::iterator lru_it;
    };
    std::unordered_map<std::string, EmbdEntry> embd_cache;
    std::list<std::string> embd_lru;
    size_t embd_cache_bytes = 0;

    void evict_embd_cache(size_t need_bytes, size_t max_bytes) {
        while (!embd_lru.empty() && (embd_cache_bytes + need_bytes > max_bytes)) {
            const std::string oldest_key = embd_lru.back();
            embd_lru.pop_back();
            auto it = embd_cache.find(oldest_key);
            if (it != embd_cache.end()) {
                embd_cache_bytes -= it->second.data.size() * sizeof(float);
                embd_cache.erase(it);
            }
        }
    }

    // Last generate() breakdown, read back by nativeGetLastGenStats(). Generation is the
    // largest phase of a query and there is no other way to see inside it on devices
    // whose ROM discards this app's logcat - vivo FuntouchOS does exactly that.
    int  last_gen_tokens    = 0;
    long last_prefill_ms    = 0;
    long last_gen_ms        = 0;
    bool last_hit_token_cap = false;
    int  last_cache_hits    = 0;
    int  last_cache_misses  = 0;
    int  last_disk_hits     = 0;

    // Disk tier for embd_cache, set via nativeSetCacheDir(). Empty = disabled.
    std::string cache_dir;
    // Basename of the loaded projector; namespaces the disk cache, because the cache key
    // is a hash of the PIXELS - swap the projector (2B -> 3B, Q8 -> Q4) and the same key
    // must not resolve to the other model's embeddings.
    std::string proj_name;
};

static long now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

// ── disk tier for the vision-encode cache ────────────────────────────────────
//
// Measured on a Vivo I2304: prefill is 115.8 s of a 128 s query (90%), nearly all of it
// clip_encode at ~17-19 s per frame, while generation runs at 18.8 tok/s and costs ~5 s.
// The in-memory cache already skips re-encodes within a process, but the process does not
// live long on a phone, and losing it throws away ~98 s of work per five frames. Raw
// float dumps on disk keyed by mtmd's pixel hash survive restarts; loading one back costs
// ~10 ms against the ~18 s it saves.

// ~210 entries at 1.8 MB per 299-token frame; enough for every keyframe of a 13-minute
// video plus the query crops, with room to spare. Oldest-by-mtime beyond that.
static const size_t EMBD_DISK_MAX_BYTES = 384ull * 1024 * 1024;

// mtmd ids are hex hashes; anything else stays out of filesystem paths.
static bool embd_key_safe(const std::string & k) {
    if (k.empty() || k.size() > 120) return false;
    for (char c : k) if (!isalnum((unsigned char) c) && c != '_' && c != '-') return false;
    return true;
}

static bool load_embd_file(const std::string & dir, const std::string & key,
                           size_t need_floats, std::vector<float> & out) {
    const std::string path = dir + "/" + key + ".bin";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz != (long) (need_floats * sizeof(float))) {
        // Wrong shape for this projector/prompt - stale beyond use, so reclaim it.
        fclose(f);
        remove(path.c_str());
        return false;
    }
    out.resize(need_floats);
    const bool ok = fread(out.data(), sizeof(float), need_floats, f) == need_floats;
    fclose(f);
    return ok;
}

static void evict_embd_dir(const std::string & dir, size_t max_bytes) {
    DIR * d = opendir(dir.c_str());
    if (!d) return;
    struct Ent { time_t mt; size_t sz; std::string path; };
    std::vector<Ent> ents;
    size_t total = 0;
    while (dirent * e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        const std::string path = dir + "/" + e->d_name;
        struct stat st{};
        if (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            ents.push_back({st.st_mtime, (size_t) st.st_size, path});
            total += (size_t) st.st_size;
        }
    }
    closedir(d);
    if (total <= max_bytes) return;
    std::sort(ents.begin(), ents.end(),
              [](const Ent & a, const Ent & b) { return a.mt < b.mt; });
    for (const auto & en : ents) {
        if (total <= max_bytes) break;
        if (remove(en.path.c_str()) == 0) total -= en.sz;
    }
}

static void save_embd_file(const std::string & dir, const std::string & key,
                           const float * data, size_t need_floats) {
    // tmp + rename, so a crash mid-write can never leave a half file that later loads.
    const std::string tmp = dir + "/." + key + ".tmp";
    const std::string fin = dir + "/" + key + ".bin";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    const bool ok = fwrite(data, sizeof(float), need_floats, f) == need_floats;
    fclose(f);
    if (ok) rename(tmp.c_str(), fin.c_str());
    else    remove(tmp.c_str());

    static int save_counter = 0;
    if (++save_counter % 16 == 0) {
        evict_embd_dir(dir, EMBD_DISK_MAX_BYTES);
    }
}

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
    // Sizing note: each inference call processes a single keyframe + prompt (~300-500 tokens).
    // An 8192 context wastes ~1.2 GB of mobile RAM on unneeded KV cache, leading to Android
    // LMK process kills. 2048 tokens provides ample headroom while saving ~900 MB RAM.
    cparams.n_ctx   = 2048;
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
        const size_t psl = p_path.find_last_of('/');
        ctx->proj_name = (psl == std::string::npos) ? p_path : p_path.substr(psl + 1);
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

    ctx->abort_requested.store(false);

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

    if (!ctx->ctx_mtmd) {
        LOGE("ctx_mtmd is null, cannot perform multimodal tokenization");
        return env->NewStringUTF("Error: vision projector (mmproj) is not loaded.");
    }

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
    int cache_hits = 0, cache_misses = 0, disk_hits = 0;

    for (size_t i = 0; i < n_chunks; ++i) {
        if (ctx->abort_requested.load()) {
            LOGI("nativeGenerate aborted by user during prefill");
            eval_rc = 2;
            break;
        }
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
        const size_t need_bytes = need * sizeof(float);
        const std::string key = id ? id : "";

        auto it = key.empty() ? ctx->embd_cache.end() : ctx->embd_cache.find(key);
        if (it != ctx->embd_cache.end()) {
            cache_hits++;
            ctx->embd_lru.erase(it->second.lru_it);
            ctx->embd_lru.push_front(key);
            it->second.lru_it = ctx->embd_lru.begin();
        } else if (!ctx->cache_dir.empty() && embd_key_safe(key)) {
            // Second tier: an encode a previous process already paid for. Loading one
            // back costs ~10 ms against the ~18 s it took to compute.
            std::vector<float> from_disk;
            if (load_embd_file(ctx->cache_dir, key, need, from_disk)) {
                disk_hits++;
                ctx->evict_embd_cache(need_bytes, EMBD_CACHE_MAX_BYTES);
                if (ctx->embd_cache_bytes + need_bytes <= EMBD_CACHE_MAX_BYTES) {
                    ctx->embd_lru.push_front(key);
                    NativeVLMContext::EmbdEntry entry{std::move(from_disk), ctx->embd_lru.begin()};
                    it = ctx->embd_cache.emplace(key, std::move(entry)).first;
                    ctx->embd_cache_bytes += need_bytes;
                } else {
                    eval_rc = mtmd_helper_decode_image_chunk(
                        ctx->ctx_mtmd, ctx->ctx_llama, chunk,
                        from_disk.data(), n_past, /*seq_id*/ 0, /*n_batch*/ 2048,
                        &n_past, nullptr, nullptr);
                    if (eval_rc != 0) break;
                    continue;
                }
            }
        }
        if (it == ctx->embd_cache.end()) {
            cache_misses++;
            if (mtmd_encode_chunk(ctx->ctx_mtmd, chunk) != 0) {
                LOGE("mtmd_encode_chunk failed at chunk %zu", i);
                eval_rc = 1; break;
            }
            const float * out = mtmd_get_output_embd(ctx->ctx_mtmd);
            if (!out) { LOGE("no output embeddings"); eval_rc = 1; break; }

            // Persist immediately: this file is what turns the next process's ~116 s
            // prefill into ~20 s.
            if (!ctx->cache_dir.empty() && embd_key_safe(key)) {
                save_embd_file(ctx->cache_dir, key, out, need);
            }

            if (!key.empty()) {
                ctx->evict_embd_cache(need_bytes, EMBD_CACHE_MAX_BYTES);
            }
            if (!key.empty() && ctx->embd_cache_bytes + need_bytes <= EMBD_CACHE_MAX_BYTES) {
                ctx->embd_lru.push_front(key);
                NativeVLMContext::EmbdEntry entry{std::vector<float>(out, out + need), ctx->embd_lru.begin()};
                it = ctx->embd_cache.emplace(key, std::move(entry)).first;
                ctx->embd_cache_bytes += need_bytes;
            } else {
                // cache full (or unkeyed): decode straight from the encoder output
                eval_rc = mtmd_helper_decode_image_chunk(
                    ctx->ctx_mtmd, ctx->ctx_llama, chunk,
                    const_cast<float *>(out), n_past, /*seq_id*/ 0, /*n_batch*/ 2048,
                    &n_past, nullptr, nullptr);
                if (eval_rc != 0) break;
                continue;
            }
        }

        eval_rc = mtmd_helper_decode_image_chunk(
            ctx->ctx_mtmd, ctx->ctx_llama, chunk,
            it->second.data.data(), n_past, /*seq_id*/ 0, /*n_batch*/ 2048,
            &n_past, nullptr, nullptr);
        if (eval_rc != 0) break;
    }
    mtmd_input_chunks_free(chunks);

    ctx->last_prefill_ms   = now_ms() - t_prefill_start;
    ctx->last_cache_hits   = cache_hits;
    ctx->last_cache_misses = cache_misses;
    ctx->last_disk_hits    = disk_hits;
    LOGI("image encode cache: %d mem hit, %d disk hit, %d miss (%zu entries, %.1f MB)",
         cache_hits, disk_hits, cache_misses, ctx->embd_cache.size(),
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
        if (ctx->abort_requested.load()) {
            LOGI("nativeGenerate aborted by user at token %d", i);
            hit_cap = false;
            break;
        }
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
 * Point the vision-encode cache at a persistent directory.
 *
 * Kotlin passes the app's cacheDir - private, permission-free, and cleaned by the OS
 * under storage pressure, which is the right lifecycle for a pure performance artifact.
 * The projector's basename becomes a subdirectory, so weights can be swapped without
 * ever replaying another model's embeddings.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeSetCacheDir(
        JNIEnv * env, jobject /*thiz*/, jlong handle, jstring dir) {

    auto * ctx = reinterpret_cast<NativeVLMContext *>(handle);
    if (!ctx) return;
    const std::string base = jstring_to_std(env, dir);
    if (base.empty()) return;
    mkdir(base.c_str(), 0700);

    std::string tag = ctx->proj_name.empty() ? "textonly" : ctx->proj_name;
    for (auto & c : tag) {
        if (!isalnum((unsigned char) c) && c != '_' && c != '-') c = '_';
    }
    ctx->cache_dir = base + "/" + tag;
    mkdir(ctx->cache_dir.c_str(), 0700);
    LOGI("embd disk cache: %s", ctx->cache_dir.c_str());
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
             "cache_hits=%d disk_hits=%d cache_misses=%d",
             ctx->last_gen_tokens, ctx->last_gen_ms,
             ctx->last_gen_ms > 0 ? 1000.0 * ctx->last_gen_tokens / ctx->last_gen_ms : 0.0,
             (int) ctx->last_hit_token_cap, ctx->last_prefill_ms,
             ctx->last_cache_hits, ctx->last_disk_hits, ctx->last_cache_misses);
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

extern "C" JNIEXPORT void JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeAbort(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {

    auto * ctx = reinterpret_cast<NativeVLMContext *>(handle);
    if (!ctx) return;
    LOGI("nativeAbort called, aborting inference");
    ctx->abort_requested.store(true);
}
