// engine_core.h — pure C++ engine core. NO pybind11 dependency here; all
// GIL handling happens in binding.cpp. See design spec §4.1/§4.2.
#pragma once

#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "engine_lifecycle.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class EngineCore {
public:
    // verbosity/n_threads mirror common_params::verbosity and
    // common_params::cpuparams.n_threads (common/common.h) -- same names,
    // same defaults (3 = LOG_LEVEL_INFO, -1 = auto-detect thread count).
    // n_ctx_seq is the per-slot context length (llama.cpp's internal
    // n_ctx_seq -- context for a single sequence), NOT the total n_ctx.
    // 0 means "the model's training context length". Total n_ctx is derived
    // internally as n_ctx_seq * n_parallel (kv_unified=false, so each slot
    // gets a private n_ctx_seq-cell stream). See constructor comment.
    EngineCore(const std::string & model_path, const std::string & mmproj_path,
               int n_ctx_seq, int n_gpu_layers, int n_parallel,
               int32_t verbosity, int32_t n_threads);
    ~EngineCore();

    EngineCore(const EngineCore &) = delete;
    EngineCore & operator=(const EngineCore &) = delete;

    struct Timings {
        int32_t prompt_n            = 0;
        double  prompt_ms           = 0.0;
        double  prompt_per_second   = 0.0;
        int32_t predicted_n         = 0;
        double  predicted_ms        = 0.0;
        double  predicted_per_second = 0.0;
    };

    // Result of a single non-streaming generate() call.
    //
    // Error handling contract (design spec §4.2 point 3):
    //   - Errors detected *before* the task is posted (malformed body JSON,
    //     bad image_url, invalid sampling field) are raised as regular C++
    //     exceptions (std::exception) that propagate OUT of generate()/
    //     generate_stream() uncaught. The binding layer catches these.
    //   - Errors detected *after* posting (server-side send_error, e.g.
    //     image decode failure discovered on the loop thread, or
    //     context-size-exceeded discovered during slot processing) arrive
    //     asynchronously via the normal result channel and are surfaced via
    //     is_error/error_json on this struct — there is no C++ call stack to
    //     throw up at that point, so a raw exception is not possible.
    struct GenerateResult {
        bool        is_error = false;
        std::string error_json;   // valid only if is_error; JSON string, see server-common.cpp format_error_response()

        std::string content;
        std::string finish_reason;   // "stop" | "length"; valid only if !is_error
        int32_t     tokens_evaluated = 0;
        int32_t     tokens_predicted = 0;
        Timings     timings;
    };

    // Blocks the calling thread until the completion is fully generated.
    // May throw std::exception for pre-post errors (see contract above).
    GenerateResult generate(const std::string & body_json);

    // One chunk of a streaming generation.
    struct Chunk {
        bool        is_error = false;
        std::string error_json;

        std::string delta;
        bool        is_final = false;
        // valid only when is_final (and !is_error):
        std::string finish_reason;
        int32_t     tokens_evaluated = 0;
        int32_t     tokens_predicted = 0;
        Timings     timings;
    };

    // Owns the one server_response_reader for this stream's entire
    // lifetime. See the correctness note above this class — do NOT change
    // rd_ to a by-value server_response_reader member.
    class StreamHandle {
    public:
        explicit StreamHandle(std::unique_ptr<server_response_reader> rd, const EngineLifecycle & lifecycle);
        // Blocks until the next token (or the final chunk, or an error) is
        // ready. May throw std::exception for pre-post errors, but in
        // practice those all happen inside generate_stream() before a
        // StreamHandle is ever constructed, so next_chunk() itself should
        // only ever report async errors via Chunk::is_error.
        Chunk next_chunk();

    private:
        std::unique_ptr<server_response_reader> rd_;
        const EngineLifecycle & lifecycle_;
        bool done_ = false;
    };

    // May throw std::exception for pre-post errors (see contract above).
    StreamHandle generate_stream(const std::string & body_json);

    // The model's chat-template end token (e.g. "<|im_end|>"), as detokenized
    // by llama.cpp from the vocab's EOS token id. Empty string if the model
    // has no EOS token. Used by the Python Engine layer to auto-fill
    // SamplingParams.stop when the caller doesn't set it -- see design note
    // on `special=true` in engine_core.cpp: it preserves structured tokens
    // like <|box_start|> in generated text, but that also means this EOS
    // token gets rendered as literal text instead of being silently dropped.
    const std::string & eos_token_str() const { return meta_->eos_token_str; }

    // 暴露真实后台循环状态；不读取或修改上游私有任务。
    void check_health() const { lifecycle_.check_health(); }

private:
    server_context                        ctx_;
    common_params                         params_;
    std::unique_ptr<server_context_meta>  meta_;
    std::thread                           loop_thread_;
    bool                                  started_ = false;
    EngineLifecycle                       lifecycle_;

    // Guards oaicompat_chat_params_parse() + eval_llama_cmpl_schema() +
    // post_task() for both generate() and generate_stream(). Fixes the
    // "concurrent generate() calls corrupt each other's prompt" bug found
    // during technical validation (Tier2 #3). Deliberately does NOT guard
    // the blocking wait for a result — multiple slots must be able to decode
    // in parallel. See design spec §4.2 point 1.
    std::mutex parse_mu_;
};
