// engine_core.cpp
#include "engine_core.h"

#include "server-common.h"
#include "server-schema.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"
#include "gguf.h"
#include "backend_policy.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#endif
// load_packaged_backends() locates this extension's own module file so it
// can pass an explicit bin/ directory to ggml_backend_load_all_from_path()
// -- see that function's comment for why. Linux/macOS use dladdr(); Windows
// uses GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS) +
// GetModuleFileNameW() on a stable anchor function's address.

using json = common_json;

namespace {

// Maps llama-server's stop_type string (from server-task.cpp
// stop_type_to_str(): "eos"/"word"/"limit"/"none") onto the two
// OpenAI-style finish_reason values this library exposes. "eos"/"word" mean
// the model stopped itself (matches oaicompat's own to_json_oaicompat()
// mapping); "limit" means max_tokens was hit. Any other/unrecognized value
// (including "none", which should not occur for a genuinely-finished
// result) defaults to "stop" rather than an empty string, so the Python-side
// Literal["stop", "length"] contract is never violated.
std::string map_finish_reason(const std::string & stop_type_str) {
    if (stop_type_str == "limit") return "length";
    return "stop";
}

// Reads the model's training context length ({arch}.context_length) straight
// from the GGUF metadata, without loading weights. Needed because we default
// n_ctx_seq to n_ctx_train, but n_ctx (and thus the KV cache) must be sized
// *before* the model finishes loading -- at which point llama.cpp's own
// llama_model_n_ctx_train() isn't available yet. Metadata-only parsing
// (no_alloc, ctx=NULL) is cheap. Returns 0 if the file/key can't be read, so
// the caller can fall back to llama.cpp's own default handling.
int read_n_ctx_train_from_gguf(const std::string & model_path) {
    gguf_init_params gp{ /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), gp);
    if (ctx == nullptr) return 0;

    int result = 0;
    const int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    if (arch_id >= 0 && gguf_get_kv_type(ctx, arch_id) == GGUF_TYPE_STRING) {
        const std::string key = std::string(gguf_get_val_str(ctx, arch_id)) + ".context_length";
        const int64_t ctx_id = gguf_find_key(ctx, key.c_str());
        if (ctx_id >= 0 && gguf_get_kv_type(ctx, ctx_id) == GGUF_TYPE_UINT32) {
            result = (int) gguf_get_val_u32(ctx, ctx_id);
        }
    }
    gguf_free(ctx);
    return result;
}

// Locates and dlopen-loads this extension's own bundled ggml backend
// MODULEs (see the top-level CMakeLists.txt's GGML_BACKEND_DL comment for
// why they're MODULEs rather than hard-linked). Without GGML_BACKEND_PATH
// set, ggml's own default search (ggml_backend_load_all(), see
// ggml-backend-reg.cpp's ggml_backend_load_best()) only checks the
// executable directory and cwd -- for a Python extension loaded by the
// `python` interpreter, that's the interpreter's own directory, never ours.
// So we resolve our own module path via dladdr() (same technique
// ggml.c:130 already uses on Linux/macOS for backtraces) and pass an
// explicit dir_path to ggml_backend_load_all_from_path() instead.
//
// Two on-disk layouts, both real (see top-level CMakeLists.txt's comment on
// CMAKE_INSTALL_BINDIR): a packaged wheel puts the extension at
// mineru_llama_cpp/_mineru_llama_cpp*.so with backends alongside it at
// mineru_llama_cpp/bin/ (one level down); scikit-build-core's editable
// "inplace" mode instead builds the extension straight into
// src/mineru_llama_cpp/ while backend MODULEs land in the *build tree's*
// CMAKE_RUNTIME_OUTPUT_DIRECTORY, which for inplace editable builds is the
// repo root's bin/ (CMAKE_BINARY_DIR == CMAKE_SOURCE_DIR for that mode --
// see the top-level CMakeLists.txt comment) -- two directories up from
// src/mineru_llama_cpp/. Both are tried, nearest first; a middle
// one-level-up candidate is kept too in case either layout's nesting
// depth ever shifts by one.
//
// This matters beyond "GPU acceleration missing": with GGML_BACKEND_DL=ON,
// ggml_add_backend(CPU) (ggml/src/CMakeLists.txt) makes the CPU backend a
// MODULE too, same as every accelerator -- there is no compiled-in static
// CPU registration to fall back on (that only exists when
// GGML_BACKEND_DL=OFF, via the GGML_USE_CPU compile define). If no backend
// at all gets registered, llama_numa_init() immediately hits
// `GGML_ASSERT(dev && "CPU backend is not loaded")` (src/llama.cpp) --
// a hard abort, not a catchable C++ exception. Finding this directory is
// therefore load-bearing for every backend, not just optional accelerators.

// 说明已打包却未注册的后端，区分缺少运行库与没有可用设备的回退原因。
void log_packaged_backend_status(const std::filesystem::path & directory) {
    for (const char * name : {"vulkan", "metal"}) {
#if defined(_WIN32)
        const auto module = directory / (std::string("ggml-") + name + ".dll");
#else
        const auto module = directory / (std::string("libggml-") + name + ".so");
#endif
        std::error_code error;
        if (!std::filesystem::exists(module, error)) continue;
        auto * reg = ggml_backend_reg_by_name(std::string(name) == "metal" ? "MTL" : name);
        if (!reg) {
            LOG_DBG("mineru-llama-cpp: packaged %s backend did not register; check missing/incompatible runtime libraries\n", name);
        } else if (!ggml_backend_reg_dev_count(reg)) {
            LOG_DBG("mineru-llama-cpp: packaged %s backend has no usable GPU; check drivers/devices\n", name);
        }
    }
}

// CPU 请求只加载计算后端，其余请求沿用完整后端发现流程。
void load_packaged_backends_from(const std::filesystem::path & directory, bool cpu_only) {
    if (cpu_only) {
        ggml_backend_load_cpu_from_path(directory.string().c_str());
    } else {
        ggml_backend_load_all_from_path(directory.string().c_str());
        log_packaged_backend_status(directory);
    }
}
#if defined(__linux__) || defined(__APPLE__)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
// 提供稳定地址，定位当前扩展的安装目录。
void engine_core_module_anchor() {}

void load_packaged_backends(bool cpu_only) {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&engine_core_module_anchor), &info) == 0 || !info.dli_fname) {
        return;
    }

    const std::filesystem::path module_dir = std::filesystem::path(info.dli_fname).parent_path();

    const std::filesystem::path candidates[] = {
        module_dir / "bin",                                    // wheel: mineru_llama_cpp/bin
        module_dir.parent_path() / "bin",                       // defensive middle case
        module_dir.parent_path().parent_path() / "bin",         // editable inplace: <repo-root>/bin
    };

    std::error_code ec;
    for (const auto & candidate : candidates) {
        if (std::filesystem::is_directory(candidate, ec)) {
            load_packaged_backends_from(candidate, cpu_only);
            return;
        }
        ec.clear();
    }
}
#elif defined(_WIN32)
// Anchor function whose address we pass to GetModuleHandleExW -- the address
// must live inside this DLL (.pyd), so resolving "what module contains this
// instruction pointer" gives us a handle to _mineru_llama_cpp.pyd itself.
// No inline attribute needed on Windows, but the function must be a real,
// non-static symbol (not optimized away) for its address to be meaningful.
void engine_core_module_anchor() {}

void load_packaged_backends(bool cpu_only) {
    HMODULE hmod = nullptr;
    // GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: look up the module that contains
    // the given address (here, our anchor function). FLAG_UNCHANGED_REFCOUNT
    // avoids incrementing the module's refcount (we don't want a FreeLibrary
    // obligation on ourselves).
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&engine_core_module_anchor),
            &hmod)) {
        return;
    }

    wchar_t path_buf[MAX_PATH];
    DWORD len = GetModuleFileNameW(hmod, path_buf, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) {
        return;  // failed, or path too long (would need dynamic sizing -- rare for our layout)
    }

    const std::filesystem::path module_dir = std::filesystem::path(std::wstring(path_buf, len)).parent_path();

    // Same three candidate layouts as Linux/macOS -- see the comment there.
    // ggml_backend_load_all_from_path takes a const char *, so we need the
    // directory's narrow-string form. std::filesystem::path::string() yields
    // the system's default codepage (CP_ACP on Windows); for ASCII-only
    // install paths this is fine, and our directory name "bin" is ASCII.
    const std::filesystem::path candidates[] = {
        module_dir / "bin",
        module_dir.parent_path() / "bin",
        module_dir.parent_path().parent_path() / "bin",
    };

    std::error_code ec;
    for (const auto & candidate : candidates) {
        if (std::filesystem::is_directory(candidate, ec)) {
            load_packaged_backends_from(candidate, cpu_only);
            return;
        }
        ec.clear();
    }
}
#else
void load_packaged_backends(bool) {}
#endif

EngineCore::Timings extract_timings(const json & j) {
    EngineCore::Timings t;
    if (!j.contains("timings")) return t;
    const auto & tj = j.at("timings");
    t.prompt_n             = tj.value("prompt_n", 0);
    t.prompt_ms            = tj.value("prompt_ms", 0.0);
    t.prompt_per_second    = tj.value("prompt_per_second", 0.0);
    t.predicted_n          = tj.value("predicted_n", 0);
    t.predicted_ms         = tj.value("predicted_ms", 0.0);
    t.predicted_per_second = tj.value("predicted_per_second", 0.0);
    return t;
}

// 从已加载设备中选择一个后端，模型与视觉投影使用相同设备集合。
void select_engine_backend(common_params & params, int requested_layers, const std::string & requested) {
    std::vector<mineru_llama_cpp::BackendDevice> candidates;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        auto * device = ggml_backend_dev_get(index);
        const auto type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        const auto backend = mineru_llama_cpp::canonical_backend_name(
            ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)));
        candidates.push_back({backend, type == GGML_BACKEND_DEVICE_TYPE_IGPU, index});
    }
    const auto selected = mineru_llama_cpp::select_backend_devices(candidates, requested, requested_layers == 0);
    params.devices.clear();
    for (size_t index : selected) {
        auto * device = ggml_backend_dev_get(index);
        params.devices.push_back(device);
        LOG_INF("mineru-llama-cpp: selected %s (%s), backend=%s\n", ggml_backend_dev_name(device),
                ggml_backend_dev_description(device), ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)));
    }
    // llama_model_params.devices 是以 nullptr 终止的数组，GPU 列表也必须显式终止。
    params.devices.push_back(nullptr);
    if (selected.empty()) {
        // 仅含终止项的非空列表阻止 llama.cpp 再次自动挑选其他 GPU。
        params.n_gpu_layers = 0;
        params.mmproj_use_gpu = false;
        LOG_INF("mineru-llama-cpp: selected CPU (requested=%s); no eligible GPU backend or CPU forced\n", requested.c_str());
    } else {
        params.mmproj_use_gpu = true;
        params.mmproj_device = params.devices.front();
        for (const auto & candidate : candidates)
            if (std::find(selected.begin(), selected.end(), candidate.index) == selected.end())
                LOG_DBG("mineru-llama-cpp: skipped backend %s for this engine\n", candidate.backend.c_str());
    }
}

} // namespace

EngineCore::EngineCore(const std::string & model_path, const std::string & mmproj_path,
                        int n_ctx_seq, int n_gpu_layers, int n_parallel,
                        int32_t verbosity, int32_t n_threads) {
    const char * environment = std::getenv("MINERU_LLAMA_CPP_BACKEND");
    const std::string requested = environment ? environment : "auto";
    const bool cpu_only = mineru_llama_cpp::requests_cpu_backend(requested, n_gpu_layers == 0);
    params_.model.path          = model_path;
    params_.mmproj.path         = mmproj_path;
    params_.n_gpu_layers        = n_gpu_layers;
    params_.mmproj_use_gpu      = (n_gpu_layers > 0);
    params_.n_parallel          = n_parallel;

    // We expose per-slot context (n_ctx_seq), not llama.cpp's total n_ctx.
    // n_ctx_seq == 0 means "the model's training context length"; read it
    // straight from GGUF metadata since llama_model_n_ctx_train() isn't
    // available until after the model loads (which is too late -- the KV
    // cache is sized during load). If the read fails, fall back to letting
    // llama.cpp default n_ctx itself (params_.n_ctx = 0).
    if (n_ctx_seq == 0) {
        n_ctx_seq = read_n_ctx_train_from_gguf(model_path);
    }
    params_.n_ctx = (n_ctx_seq > 0) ? n_ctx_seq * n_parallel : 0;

    // NOT unified: with kv_unified = false the KV cache is hard-partitioned
    // into n_parallel private streams of n_ctx / n_parallel cells each. Since
    // we size n_ctx = n_ctx_seq * n_parallel above, each slot gets exactly
    // n_ctx_seq cells that no other slot can touch. This makes the
    // "failed to find a memory slot" failure (a slot starved because its
    // peers hold the shared pool) structurally impossible: a request either
    // fits in its own n_ctx_seq (runs, or cleanly stops with finish_reason=
    // "length" at the cap) or is rejected up front if its prompt alone
    // exceeds n_ctx_seq -- never a load-dependent mid-batch failure. Total KV
    // memory is identical to the unified case (n_ctx cells either way); the
    // only thing that changes is partitioned vs shared. For an offline batch
    // engine that values predictable stability over letting one request
    // borrow idle peers' capacity, the hard partition is the right trade.
    params_.kv_unified          = false;
    params_.cont_batching       = true;
    
    // Engine requests disable prompt reuse, so retaining idle slots only
    // adds RAM-cache eviction work during offline multi-modal batches.
    params_.cache_idle_slots    = false;
    params_.cache_ram_mib       = 0;

    params_.special             = true;
    params_.sleep_idle_seconds  = -1;  // never sleep (else model gets unloaded mid-use)
    params_.chat_template       = "";  // use the model's own built-in template
    params_.use_jinja           = true;
    params_.verbosity           = verbosity;
    params_.cpuparams.n_threads = n_threads;

    // common_init() wires llama's own log callback (llama_log_set) through
    // to common_log -- without it, LLAMA_LOG_DEBUG lines from llama.cpp
    // internals (e.g. llama_context::set_embeddings) bypass the verbosity
    // threshold entirely and print unconditionally, same as every llama.cpp
    // CLI tool calls this before touching the backend (see e.g.
    // tools/server/server.cpp).
    common_init();
    common_log_set_verbosity_thold(params_.verbosity);

    // CLI entry points funnel cpuparams.n_threads through this before it
    // ever reaches llama_context -- skipping it means -1 ("auto") is never
    // resolved and silently falls back to GGML_DEFAULT_N_THREADS (4) deep
    // inside ggml-cpu.c, regardless of how many cores are actually
    // available.
    postprocess_cpu_params(params_.cpuparams, nullptr);
    // 新上游在加载模型时创建批处理线程池，必须先解析其 -1 默认值。
    postprocess_cpu_params(params_.cpuparams_batch, &params_.cpuparams);

    // Must run before llama_backend_init(): that function only falls back to
    // ggml's default search path (ggml_backend_load_all(), executable-dir +
    // cwd) if nothing has been registered yet (see its
    // `if (!ggml_backend_reg_count())` guard in src/llama.cpp) -- calling
    // load_packaged_backends() first means that guard sees backends already
    // registered and skips its own (Python-interpreter-directory) search,
    // which would never find our bundled MODULEs anyway.
    load_packaged_backends(cpu_only);
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        throw std::runtime_error("Packaged CPU backend is unavailable; check the installation's bin directory");
    }
    llama_backend_init();
    llama_numa_init(params_.numa);

    try {
        select_engine_backend(params_, n_gpu_layers, requested);
        if (!ctx_.load_model(params_)) throw std::runtime_error("load_model failed");
    } catch (...) {
        llama_backend_free();
        throw;
    }
    meta_ = std::make_unique<server_context_meta>(ctx_.get_meta());
    loop_thread_ = std::thread([this] { lifecycle_.run([this] { ctx_.start_loop(); }); });
    started_ = true;
}

EngineCore::~EngineCore() {
    if (started_) {
        lifecycle_.begin_close();
        ctx_.terminate();
        if (loop_thread_.joinable()) loop_thread_.join();
    }
    llama_backend_free();
}

EngineCore::GenerateResult EngineCore::generate(const std::string & body_json) {
    check_health();
    // 无效 JSON 的 common_json_error 继续在绑定层映射为请求异常。
    json body = json::parse(body_json);

    server_response_reader rd = ctx_.get_response_reader();
    server_task task(SERVER_TASK_TYPE_COMPLETION);

    {
        std::lock_guard<std::mutex> lk(parse_mu_);
        task.id = rd.get_new_id();

        std::vector<raw_buffer> files;
        // Reuses llama-server's own OpenAI-compatible chat parsing ("Path A"):
        // renders the jinja chat template, extracts image_url data URIs into
        // files, and normalizes sampling params into `parsed`. Throws
        // std::runtime_error/std::invalid_argument on bad image_url etc.
        json parsed = oaicompat_chat_params_parse(body, meta_->chat_params, files);
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_.get_llama_context()));
        task.params = server_schema::eval_llama_cmpl_schema(
            vocab, params_, meta_->logit_bias_eog, parsed);
        task.params.stream       = false;
        task.params.res_type     = TASK_RESPONSE_TYPE_NONE;
        task.params.cache_prompt = false;  // don't reuse KV across requests (fixes Tier2 #3's "empty prompt" bug)
        task.cli        = true;
        task.cli_prompt = parsed.at("prompt").get<std::string>();
        task.cli_files  = std::move(files);
    }

    check_health();
    rd.post_task(std::move(task));

    GenerateResult result;
    auto r = rd.next([this]() { return lifecycle_.stopped(); });
    check_health();
    if (!r) {
        result.is_error   = true;
        result.error_json = R"json({"type":"server_error","message":"no result (stopped)"})json";
        return result;
    }
    if (r->is_error()) {
        result.is_error   = true;
        result.error_json = r->to_json().dump();
        return result;
    }

    json j = r->to_json();
    result.content         = j.value("content", std::string());
    result.finish_reason    = map_finish_reason(j.value("stop_type", std::string()));
    result.tokens_evaluated = j.value("tokens_evaluated", 0);
    result.tokens_predicted = j.value("tokens_predicted", 0);
    result.timings          = extract_timings(j);
    return result;
}

EngineCore::StreamHandle::StreamHandle(std::unique_ptr<server_response_reader> rd, const EngineLifecycle & lifecycle)
    : rd_(std::move(rd)), lifecycle_(lifecycle) {}

EngineCore::Chunk EngineCore::StreamHandle::next_chunk() {
    Chunk chunk;
    if (done_) {
        // Caller kept calling after the final chunk; report as an immediate
        // final (empty-delta) chunk rather than blocking on a reader that
        // has nothing left to give.
        chunk.is_final = true;
        chunk.finish_reason = "stop";
        return chunk;
    }

    lifecycle_.check_health();
    auto r = rd_->next([this]() { return lifecycle_.stopped(); });
    lifecycle_.check_health();
    if (!r) {
        chunk.is_error   = true;
        chunk.error_json = R"json({"type":"server_error","message":"no result (stopped)"})json";
        done_ = true;
        return chunk;
    }
    if (r->is_error()) {
        chunk.is_error   = true;
        chunk.error_json = r->to_json().dump();
        done_ = true;
        return chunk;
    }

    // The first partial chunk (is_begin, an SSE-header signal with no
    // content) serializes to a null json, not an object — see server-task.cpp
    // server_task_result_cmpl_partial::to_json(). Treat it as an empty delta.
    json j = r->to_json();
    if (!j.is_null()) {
        chunk.delta = j.value("content", std::string());
    }

    if (r->is_stop()) {
        chunk.is_final          = true;
        chunk.finish_reason     = map_finish_reason(j.is_null() ? std::string() : j.value("stop_type", std::string()));
        chunk.tokens_evaluated  = j.is_null() ? 0 : j.value("tokens_evaluated", 0);
        chunk.tokens_predicted  = j.is_null() ? 0 : j.value("tokens_predicted", 0);
        chunk.timings           = j.is_null() ? Timings{} : extract_timings(j);
        done_ = true;
    }
    return chunk;
}

EngineCore::StreamHandle EngineCore::generate_stream(const std::string & body_json) {
    check_health();
    json body = json::parse(body_json);  // throws on malformed input; propagates to binding layer

    // IMPORTANT: construct the ONE server_response_reader that will live for
    // this stream's entire lifetime, then mutate it in place through the
    // pointer (get_new_id(), post_task()) — never copy or move the
    // underlying object again after this line.
    //
    // Note on the line below: std::make_unique<T>(get_response_reader())
    // does NOT elide a copy here — make_unique takes its argument through a
    // forwarding-reference template parameter, which is not a
    // guaranteed-copy-elision context, so the compiler genuinely invokes
    // server_response_reader's implicit copy constructor once (confirmed via
    // -Wdeprecated-copy-with-user-provided-dtor). This is harmless ONLY
    // because the reader returned by get_response_reader() is still empty
    // (no task posted, id_tasks empty) at this exact point, so the copy's
    // destructor-triggered stop() is a no-op. The invariant we must actually
    // preserve is: never touch a *populated* reader (one that has had
    // post_task() called on it) through anything other than the single
    // unique_ptr from here on — see the correctness note at the top of
    // Phase B.
    auto reader = std::make_unique<server_response_reader>(ctx_.get_response_reader());

    server_task task(SERVER_TASK_TYPE_COMPLETION);
    {
        std::lock_guard<std::mutex> lk(parse_mu_);
        task.id = reader->get_new_id();

        std::vector<raw_buffer> files;
        json parsed = oaicompat_chat_params_parse(body, meta_->chat_params, files);
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_.get_llama_context()));
        task.params = server_schema::eval_llama_cmpl_schema(
            vocab, params_, meta_->logit_bias_eog, parsed);
        task.params.stream       = true;
        task.params.res_type     = TASK_RESPONSE_TYPE_NONE;
        task.params.cache_prompt = false;
        task.cli        = true;
        task.cli_prompt = parsed.at("prompt").get<std::string>();
        task.cli_files  = std::move(files);
    }

    check_health();
    reader->post_task(std::move(task));
    return StreamHandle(std::move(reader), lifecycle_);  // reader 身份及生命周期引用随迭代器保留。
}
