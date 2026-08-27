#include <jni.h>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <android/log.h>

#include "llama.h"
#include "clip.h"
#include "llava.h"

#define TAG "VideoRAG_Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Build info definitions for common.cpp
int LLAMA_BUILD_NUMBER = 2800;
char const *LLAMA_COMMIT = "b2800";
char const *LLAMA_COMPILER = "Clang NDK";
char const *LLAMA_BUILD_TARGET = "Android ARM64";

struct NativeVLMContext {
    llama_model * model = nullptr;
    llama_context * ctx_llama = nullptr;
    clip_ctx * ctx_clip = nullptr;
    std::string model_path;
    std::string mmproj_path;
    int n_threads = 4;
    int n_ctx = 2048;
    bool is_initialized = false;
};

static void llama_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}

static void llama_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeInitWithFiles(
    JNIEnv *env,
    jobject /* this */,
    jstring modelPath,
    jstring mmprojPath,
    jint layersToOffload
) {
    const char *nativeModelPath = env->GetStringUTFChars(modelPath, nullptr);
    const char *nativeMmprojPath = env->GetStringUTFChars(mmprojPath, nullptr);

    std::string mPath(nativeModelPath ? nativeModelPath : "");
    std::string projPath(nativeMmprojPath ? nativeMmprojPath : "");

    env->ReleaseStringUTFChars(modelPath, nativeModelPath);
    env->ReleaseStringUTFChars(mmprojPath, nativeMmprojPath);

    LOGI("Loading REAL On-Device GGUF model: %s", mPath.c_str());

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = layersToOffload;

    llama_model * model = llama_load_model_from_file(mPath.c_str(), mparams);
    if (!model) {
        LOGE("Failed to load llama model from: %s", mPath.c_str());
        return 0L;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_threads = 4;
    cparams.n_batch = 512;

    llama_context * ctx_llama = llama_new_context_with_model(model, cparams);
    if (!ctx_llama) {
        LOGE("Failed to create llama context from model: %s", mPath.c_str());
        llama_free_model(model);
        return 0L;
    }

    clip_ctx * ctx_clip = nullptr;
    if (!projPath.empty() && access(projPath.c_str(), R_OK) == 0) {
        LOGI("Loading Vision Projector (clip): %s", projPath.c_str());
        ctx_clip = clip_model_load(projPath.c_str(), 1);
        if (!ctx_clip) {
            LOGW("Failed to load clip vision projector from: %s", projPath.c_str());
        }
    }

    auto *ctx = new NativeVLMContext();
    ctx->model = model;
    ctx->ctx_llama = ctx_llama;
    ctx->ctx_clip = ctx_clip;
    ctx->model_path = mPath;
    ctx->mmproj_path = projPath;
    ctx->n_threads = 4;
    ctx->n_ctx = 2048;
    ctx->is_initialized = true;

    LOGI("Real On-Device VLM successfully initialized in RAM! (handle=%p)", ctx);
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeInit(
    JNIEnv *env,
    jobject thiz,
    jstring modelDir,
    jint layersToOffload
) {
    const char *nativeModelDir = env->GetStringUTFChars(modelDir, nullptr);
    std::string baseDir(nativeModelDir ? nativeModelDir : "");
    env->ReleaseStringUTFChars(modelDir, nativeModelDir);

    std::string modelFile = baseDir + "/Qwen2-VL-2B-Instruct-Q4_K_M.gguf";
    std::string mmprojFile = baseDir + "/mmproj-Qwen2-VL-2B-Instruct-f16.gguf";

    jstring jModel = env->NewStringUTF(modelFile.c_str());
    jstring jProj = env->NewStringUTF(mmprojFile.c_str());
    jlong handle = Java_com_cctv_videorag_llm_OnDeviceVLM_nativeInitWithFiles(env, thiz, jModel, jProj, layersToOffload);
    env->DeleteLocalRef(jModel);
    env->DeleteLocalRef(jProj);
    return handle;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeGenerate(
    JNIEnv *env,
    jobject /* this */,
    jlong handle,
    jstring prompt,
    jobjectArray imagePaths
) {
    auto *ctx = reinterpret_cast<NativeVLMContext *>(handle);
    if (!ctx || !ctx->is_initialized || !ctx->ctx_llama) {
        return env->NewStringUTF("Error: Native VLM engine is not loaded.");
    }

    const char *nativePrompt = env->GetStringUTFChars(prompt, nullptr);
    std::string promptStr(nativePrompt ? nativePrompt : "");
    if (nativePrompt) env->ReleaseStringUTFChars(prompt, nativePrompt);

    int numImages = env->GetArrayLength(imagePaths);
    int n_past = 0;

    // 1. Process image keyframes with multimodal projector if present
    if (ctx->ctx_clip && numImages > 0) {
        for (int i = 0; i < numImages && i < 2; ++i) {
            auto jPath = (jstring)env->GetObjectArrayElement(imagePaths, i);
            const char *nativePath = env->GetStringUTFChars(jPath, nullptr);
            if (nativePath && access(nativePath, R_OK) == 0) {
                llava_image_embed * embed = llava_image_embed_make_with_filename(ctx->ctx_clip, ctx->n_threads, nativePath);
                if (embed) {
                    llava_eval_image_embed(ctx->ctx_llama, embed, 512, &n_past);
                    llava_image_embed_free(embed);
                }
            }
            if (nativePath) env->ReleaseStringUTFChars(jPath, nativePath);
            env->DeleteLocalRef(jPath);
        }
    }

    // 2. Tokenize prompt
    std::vector<llama_token> tokens;
    tokens.resize(promptStr.size() + 16);
    int n_tokens = llama_tokenize(llama_get_model(ctx->ctx_llama), promptStr.c_str(), promptStr.length(), tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(llama_get_model(ctx->ctx_llama), promptStr.c_str(), promptStr.length(), tokens.data(), tokens.size(), true, false);
    }
    tokens.resize(std::max(0, n_tokens));

    // 3. Evaluate prompt tokens in batch
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < (int)tokens.size(); ++i) {
        llama_batch_add(batch, tokens[i], n_past++, { 0 }, i == (int)tokens.size() - 1);
        if (batch.n_tokens >= 512 || i == (int)tokens.size() - 1) {
            if (llama_decode(ctx->ctx_llama, batch) != 0) {
                LOGE("llama_decode failed during prompt processing");
                llama_batch_free(batch);
                return env->NewStringUTF("Error: Neural forward pass failed.");
            }
            batch.n_tokens = 0;
        }
    }

    // 4. Autoregressive token generation
    std::ostringstream oss;
    int max_new_tokens = 256;
    for (int i = 0; i < max_new_tokens; ++i) {
        auto * logits = llama_get_logits_ith(ctx->ctx_llama, batch.n_tokens - 1);
        int n_vocab = llama_n_vocab(llama_get_model(ctx->ctx_llama));

        // Greedy token selection
        llama_token new_token = 0;
        float max_logit = -1e9f;
        for (int v = 0; v < n_vocab; ++v) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                new_token = v;
            }
        }

        if (new_token == llama_token_eos(llama_get_model(ctx->ctx_llama))) {
            break;
        }

        char piece[128];
        int n_piece = llama_token_to_piece(llama_get_model(ctx->ctx_llama), new_token, piece, sizeof(piece), false);
        if (n_piece > 0) {
            oss.write(piece, n_piece);
        }

        llama_batch_clear(batch);
        llama_batch_add(batch, new_token, n_past++, { 0 }, true);
        if (llama_decode(ctx->ctx_llama, batch) != 0) {
            break;
        }
    }
    llama_batch_free(batch);

    std::string response = oss.str();
    if (response.empty()) {
        response = "Neural inference completed.";
    }

    return env->NewStringUTF(response.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeGetModelInfo(
    JNIEnv *env,
    jobject /* this */,
    jlong handle
) {
    auto *ctx = reinterpret_cast<NativeVLMContext *>(handle);
    if (!ctx || !ctx->is_initialized) {
        return env->NewStringUTF("Not Initialized");
    }

    std::string modelName = ctx->model_path.substr(ctx->model_path.find_last_of('/') + 1);
    std::ostringstream oss;
    oss << modelName << " (ARM64 NEON, 4 threads)";
    return env->NewStringUTF(oss.str().c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_cctv_videorag_llm_OnDeviceVLM_nativeClose(
    JNIEnv *env,
    jobject /* this */,
    jlong handle
) {
    auto *ctx = reinterpret_cast<NativeVLMContext *>(handle);
    if (ctx) {
        LOGI("Releasing native VLM context...");
        if (ctx->ctx_clip) {
            clip_free(ctx->ctx_clip);
        }
        if (ctx->ctx_llama) {
            llama_free(ctx->ctx_llama);
        }
        if (ctx->model) {
            llama_free_model(ctx->model);
        }
        delete ctx;
    }
}
