// CPython Limited API 绑定：仅负责引用、GIL、异常及数据转换。
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "engine_core.h"
#include "output_utf8.h"
#include "json.h"
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
struct PythonError {};

// 管理新引用；析构必须发生在持有 GIL 的作用域中。
class OwnedRef {
public:
    // 接管可空的新引用。
    explicit OwnedRef(PyObject * value = nullptr) : value_(value) {}
    // 释放引用，避免异常路径泄漏。
    ~OwnedRef() { Py_XDECREF(value_); }
    OwnedRef(const OwnedRef &) = delete;
    OwnedRef & operator=(const OwnedRef &) = delete;
    // 返回借用引用。
    PyObject * get() const { return value_; }
    // 将新引用交给 Python 调用者或容器。
    PyObject * release() { return std::exchange(value_, nullptr); }
private:
    PyObject * value_;
};

// 阻塞调用释放 GIL；C++ 异常离开作用域时也恢复线程状态。
class ReleasedGIL {
public:
    // 保存当前线程状态并释放 GIL。
    ReleasedGIL() : state_(PyEval_SaveThread()) {}
    // 在继续使用 Python API 前恢复 GIL。
    ~ReleasedGIL() { PyEval_RestoreThread(state_); }
    ReleasedGIL(const ReleasedGIL &) = delete;
    ReleasedGIL & operator=(const ReleasedGIL &) = delete;
private:
    PyThreadState * state_;
};

struct CoreObject { PyObject_HEAD EngineCore * core; };
struct StreamObject {
    PyObject_HEAD
    EngineCore::StreamHandle * handle;
    PyObject * owner;
    bool finished;
};
struct ModuleState { PyObject * core_type; PyObject * stream_type; };

// 检查 Python 操作失败，保留当前异常供 C 入口返回。
PyObject * checked(PyObject * object) {
    if (!object) throw PythonError{};
    return object;
}

// 从 str、bytes 或 bytearray 复制完整 UTF-8 字节，保留内嵌零字节。
std::string string_argument(PyObject * object) {
    Py_ssize_t size = 0;
    const char * data = nullptr;
    if (PyUnicode_Check(object)) {
        data = PyUnicode_AsUTF8AndSize(object, &size);
        if (!data) {
            // 旧 pybind11 将无法编码的 str 作为参数类型错误，保留异常类别。
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError, "string cannot be encoded as UTF-8");
            throw PythonError{};
        }
    } else if (PyBytes_Check(object)) {
        char * bytes = nullptr;
        if (PyBytes_AsStringAndSize(object, &bytes, &size) < 0) throw PythonError{};
        data = bytes;
    } else if (PyByteArray_Check(object)) {
        data = PyByteArray_AsString(object);
        if (!data) throw PythonError{};
        size = PyByteArray_Size(object);
    } else {
        PyErr_SetString(PyExc_TypeError, "expected str or bytes");
        throw PythonError{};
    }
    return std::string(data, static_cast<size_t>(size));
}

// 保留 pybind11 的 SupportsInt/SupportsIndex 行为，拒绝浮点数和超出 C int 的值。
int integer_argument(PyObject * object) {
    if (PyFloat_Check(object) || !PyNumber_Check(object)) {
        PyErr_SetString(PyExc_TypeError, "expected an integer-compatible value");
        throw PythonError{};
    }
    OwnedRef number(PyNumber_Index(object));
    if (!number.get()) {
        PyErr_Clear();
        OwnedRef converted(checked(PyNumber_Long(object)));
        const long value = PyLong_AsLong(converted.get());
        if (PyErr_Occurred() || value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
            PyErr_Clear(); PyErr_SetString(PyExc_TypeError, "integer value is out of range"); throw PythonError{};
        }
        return static_cast<int>(value);
    }
    const long value = PyLong_AsLong(number.get());
    if (PyErr_Occurred() || value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        PyErr_Clear(); PyErr_SetString(PyExc_TypeError, "integer value is out of range"); throw PythonError{};
    }
    return static_cast<int>(value);
}

// 保留生产异常分类，将服务端错误转换为 Python 异常。
[[noreturn]] void raise_mapped_error(const std::string & type, const std::string & message,
                                    const std::string & details = "{}") {
    OwnedRef module(checked(PyImport_ImportModule("mineru_llama_cpp.exceptions")));
    const char * name = type == "exceed_context_size_error" ? "ContextExceededError" :
                        type == "invalid_request_error" ? "InvalidRequestError" :
                        type == "engine_dead_error" ? "EngineUnavailableError" : "EngineError";
    OwnedRef exception(checked(PyObject_GetAttrString(module.get(), name)));
    OwnedRef json_module(checked(PyImport_ImportModule("json")));
    OwnedRef diagnostic(checked(PyObject_CallMethod(json_module.get(), "loads", "s", details.c_str())));
    OwnedRef args(checked(Py_BuildValue("(s)", message.c_str())));
    OwnedRef kwargs(checked(Py_BuildValue("{s:s,s:O}", "error_type", type.c_str(), "details", diagnostic.get())));
    PyObject * code = PyDict_GetItemString(diagnostic.get(), "code");
    if (code && PyDict_SetItemString(kwargs.get(), "code", code) < 0) throw PythonError{};
    OwnedRef value(checked(PyObject_Call(exception.get(), args.get(), kwargs.get())));
    PyErr_SetObject(exception.get(), value.get());
    throw PythonError{};
}

// 将原生首次致命异常映射为专用类型，保留原始 C++ 类型和消息。
[[noreturn]] void raise_engine_dead(const EngineTerminated & error) {
    common_json details = {{"type", "engine_dead_error"}, {"code", "engine_dead"},
                           {"message", error.what()}, {"native_type", error.native_type}};
    raise_mapped_error("engine_dead_error", error.what(), details.dump());
}

// 解析错误 JSON 并保留完整响应；格式损坏时仍保留原始诊断文本。
[[noreturn]] void raise_from_error_json(const std::string & text) {
    std::string type = "server_error", message = text;
    common_json details = {{"raw_response", text}};
    try {
        auto error = common_json::parse(text);
        if (error.is_object()) {
            type = error.value("type", type);
            message = error.value("message", text);
            details = error;
        }
    } catch (...) {}
    raise_mapped_error(type, message, details.dump());
}

// 接管字典值的新引用，写入失败时也释放引用。
void put(PyObject * dictionary, const char * key, PyObject * value) {
    OwnedRef owned(checked(value));
    if (PyDict_SetItemString(dictionary, key, owned.get()) < 0) throw PythonError{};
}

// 返回 None 的新引用，供流式未结束字段使用。
PyObject * none() { Py_INCREF(Py_None); return Py_None; }

// 将统计按原有字段及 Python 数值类型序列化。
PyObject * timings_to_dict(const EngineCore::Timings & t) {
    OwnedRef result(checked(PyDict_New()));
    put(result.get(), "prompt_n", PyLong_FromLong(t.prompt_n));
    put(result.get(), "prompt_ms", PyFloat_FromDouble(t.prompt_ms));
    put(result.get(), "prompt_per_second", PyFloat_FromDouble(t.prompt_per_second));
    put(result.get(), "predicted_n", PyLong_FromLong(t.predicted_n));
    put(result.get(), "predicted_ms", PyFloat_FromDouble(t.predicted_ms));
    put(result.get(), "predicted_per_second", PyFloat_FromDouble(t.predicted_per_second));
    return result.release();
}

// 保留非法生成字节映射为 InvalidRequestError 的契约。
PyObject * decode_generated_content(const std::string & content, const std::string & reason) {
    PyObject * decoded = mineru_llama_cpp::decode_output_content(content, reason);
    if (!decoded && PyErr_ExceptionMatches(PyExc_UnicodeDecodeError)) {
        PyObject * type = nullptr, * value = nullptr, * traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);
        PyErr_NormalizeException(&type, &value, &traceback);
        OwnedRef error_type(type), error_value(value), error_traceback(traceback);
        OwnedRef text(checked(PyObject_Str(value ? value : Py_None)));
        raise_mapped_error("invalid_request_error", string_argument(text.get()));
    }
    return checked(decoded);
}

// 将边界上的 C++ 异常转换为与原绑定一致的内置异常。
void translate_exception() {
    try { throw; }
    catch (const PythonError &) {}
    catch (const EngineTerminated & error) {
        try { raise_engine_dead(error); } catch (const PythonError &) {}
    }
    catch (const std::bad_alloc &) { PyErr_NoMemory(); }
    catch (const std::invalid_argument & error) { PyErr_SetString(PyExc_ValueError, error.what()); }
    catch (const std::out_of_range & error) { PyErr_SetString(PyExc_IndexError, error.what()); }
    catch (const std::exception & error) { PyErr_SetString(PyExc_RuntimeError, error.what()); }
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "unknown native exception"); }
}

// 分配零初始化包装；构造失败后仍能安全销毁。
PyObject * core_new(PyTypeObject * type, PyObject *, PyObject *) {
    return PyType_GenericAlloc(type, 0);
}

// 校验参数并创建引擎，拒绝重复初始化覆盖原有资源。
int core_init(CoreObject * self, PyObject * args, PyObject * kwargs) {
    static const char * names[] = {"model_path", "mmproj_path", "n_ctx_seq", "n_gpu_layers",
                                  "n_parallel", "verbosity", "n_threads", nullptr};
    PyObject * model = nullptr, * projector = nullptr;
    PyObject * context, * layers, * parallel, * verbosity, * threads;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOO:_EngineCore",
            const_cast<char **>(names), &model, &projector, &context, &layers,
            &parallel, &verbosity, &threads)) return -1;
    if (self->core) {
        PyErr_SetString(PyExc_RuntimeError, "_EngineCore is already initialized");
        return -1;
    }
    try {
        self->core = new EngineCore(string_argument(model), string_argument(projector),
                                   integer_argument(context), integer_argument(layers), integer_argument(parallel),
                                   integer_argument(verbosity), integer_argument(threads));
        return 0;
    } catch (...) { translate_exception(); return -1; }
}

// 释放原生对象，再按堆类型规则释放包装及类型引用。
void core_dealloc(CoreObject * self) {
    delete self->core;
    auto * type = Py_TYPE(reinterpret_cast<PyObject *>(self));
    auto free_object = reinterpret_cast<freefunc>(PyType_GetSlot(type, Py_tp_free));
    free_object(reinterpret_cast<PyObject *>(self));
    Py_DECREF(type);
}

// 防止未初始化包装进入原生代码导致空指针崩溃。
void require_core(CoreObject * self) {
    if (!self->core) {
        PyErr_SetString(PyExc_RuntimeError, "_EngineCore is not initialized");
        throw PythonError{};
    }
}

// 接收位置或关键字 body 参数，维持旧扩展调用方式。
std::string parse_body(PyObject * args, PyObject * kwargs) {
    static const char * names[] = {"body", nullptr};
    PyObject * body = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", const_cast<char **>(names), &body))
        throw PythonError{};
    return string_argument(body);
}

// 释放 GIL 等待同步结果，恢复后构造原有返回字典。
PyObject * core_generate(CoreObject * self, PyObject * args, PyObject * kwargs) {
    try {
        require_core(self);
        auto body = parse_body(args, kwargs);
        EngineCore::GenerateResult result;
        try { ReleasedGIL release; result = self->core->generate(body); }
        catch (const EngineTerminated & error) { raise_engine_dead(error); }
        catch (const std::exception & error) { raise_mapped_error("invalid_request_error", error.what()); }
        if (result.is_error) raise_from_error_json(result.error_json);
        OwnedRef output(checked(PyDict_New()));
        put(output.get(), "content", decode_generated_content(result.content, result.finish_reason));
        put(output.get(), "finish_reason", PyUnicode_FromString(result.finish_reason.c_str()));
        put(output.get(), "tokens_evaluated", PyLong_FromLong(result.tokens_evaluated));
        put(output.get(), "tokens_predicted", PyLong_FromLong(result.tokens_predicted));
        put(output.get(), "timings", timings_to_dict(result.timings));
        return output.release();
    } catch (...) { translate_exception(); return nullptr; }
}

// 提供无模型调用、无新增任务的真实循环状态探测。
PyObject * core_check_health(CoreObject * self, PyObject *) {
    try {
        require_core(self);
        self->core->check_health();
        return none();
    } catch (...) { translate_exception(); return nullptr; }
}

// 只允许 generate_stream 创建迭代器，禁止无 reader 的空实例。
PyObject * stream_new(PyTypeObject *, PyObject *, PyObject *) {
    PyErr_SetString(PyExc_TypeError, "_StreamIterator cannot be constructed directly");
    return nullptr;
}

// 创建单一 reader，并保持引擎存活直至 reader 销毁。
PyObject * core_stream(CoreObject * self, PyObject * args, PyObject * kwargs) {
    try {
        require_core(self);
        auto body = parse_body(args, kwargs);
        std::unique_ptr<EngineCore::StreamHandle> handle;
        try { handle = std::make_unique<EngineCore::StreamHandle>(self->core->generate_stream(body)); }
        catch (const EngineTerminated & error) { raise_engine_dead(error); }
        catch (const std::exception & error) { raise_mapped_error("invalid_request_error", error.what()); }
        auto * state = static_cast<ModuleState *>(PyType_GetModuleState(Py_TYPE(reinterpret_cast<PyObject *>(self))));
        if (!state) throw PythonError{};
        auto * output = reinterpret_cast<StreamObject *>(checked(
            PyType_GenericAlloc(reinterpret_cast<PyTypeObject *>(state->stream_type), 0)));
        output->handle = handle.release();
        output->owner = reinterpret_cast<PyObject *>(self);
        Py_INCREF(output->owner);
        return reinterpret_cast<PyObject *>(output);
    } catch (...) { translate_exception(); return nullptr; }
}

// 返回只读 EOS 文本属性。
PyObject * core_eos(CoreObject * self, void *) {
    try {
        require_core(self);
        const auto & text = self->core->eos_token_str();
        return mineru_llama_cpp::decode_output_content(text, "stop");
    } catch (...) { translate_exception(); return nullptr; }
}

// 先销毁 reader 再释放引擎引用，避免访问已释放上下文。
void stream_dealloc(StreamObject * self) {
    delete self->handle;
    Py_XDECREF(self->owner);
    auto * type = Py_TYPE(reinterpret_cast<PyObject *>(self));
    auto free_object = reinterpret_cast<freefunc>(PyType_GetSlot(type, Py_tp_free));
    free_object(reinterpret_cast<PyObject *>(self));
    Py_DECREF(type);
}

// __iter__ 返回自身的新引用。
PyObject * stream_iter(PyObject * self) { Py_INCREF(self); return self; }

// 阻塞读取时释放 GIL，结束块返回一次后始终停止迭代。
PyObject * stream_next(StreamObject * self) {
    if (self->finished) return nullptr;
    try {
        EngineCore::Chunk chunk;
        { ReleasedGIL release; chunk = self->handle->next_chunk(); }
        if (chunk.is_error) raise_from_error_json(chunk.error_json);
        OwnedRef output(checked(PyDict_New()));
        put(output.get(), "delta", decode_generated_content(chunk.delta, chunk.is_final ? chunk.finish_reason : "stop"));
        self->finished = chunk.is_final;
        put(output.get(), "finish_reason", chunk.is_final ? PyUnicode_FromString(chunk.finish_reason.c_str()) : none());
        put(output.get(), "tokens_evaluated", chunk.is_final ? PyLong_FromLong(chunk.tokens_evaluated) : none());
        put(output.get(), "tokens_predicted", chunk.is_final ? PyLong_FromLong(chunk.tokens_predicted) : none());
        put(output.get(), "timings", chunk.is_final ? timings_to_dict(chunk.timings) : none());
        return output.release();
    } catch (...) { translate_exception(); return nullptr; }
}

PyMethodDef core_methods[] = {
    {"check_health", reinterpret_cast<PyCFunction>(core_check_health), METH_NOARGS, "检查真实后台循环状态。"},
    {"generate", reinterpret_cast<PyCFunction>(core_generate), METH_VARARGS | METH_KEYWORDS, "同步生成并返回统计。"},
    {"generate_stream", reinterpret_cast<PyCFunction>(core_stream), METH_VARARGS | METH_KEYWORDS, "创建流式 reader。"},
    {nullptr, nullptr, 0, nullptr}
};
PyGetSetDef core_properties[] = {
    {const_cast<char *>("eos_token_str"), reinterpret_cast<getter>(core_eos), nullptr,
     const_cast<char *>("模型 EOS 文本。"), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};
PyType_Slot core_slots[] = {
    {Py_tp_new, reinterpret_cast<void *>(core_new)}, {Py_tp_init, reinterpret_cast<void *>(core_init)},
    {Py_tp_dealloc, reinterpret_cast<void *>(core_dealloc)}, {Py_tp_methods, core_methods},
    {Py_tp_getset, core_properties}, {0, nullptr}
};
PyType_Slot stream_slots[] = {
    {Py_tp_new, reinterpret_cast<void *>(stream_new)}, {Py_tp_dealloc, reinterpret_cast<void *>(stream_dealloc)},
    {Py_tp_iter, reinterpret_cast<void *>(stream_iter)}, {Py_tp_iternext, reinterpret_cast<void *>(stream_next)},
    {0, nullptr}
};
PyType_Spec core_spec = {"mineru_llama_cpp._mineru_llama_cpp._EngineCore", sizeof(CoreObject), 0, Py_TPFLAGS_DEFAULT, core_slots};
PyType_Spec stream_spec = {"mineru_llama_cpp._mineru_llama_cpp._StreamIterator", sizeof(StreamObject), 0, Py_TPFLAGS_DEFAULT, stream_slots};

// 用模块私有状态保存堆类型，避免解释器之间共享 Python 引用。
int module_exec(PyObject * module) {
    try {
        auto * state = static_cast<ModuleState *>(PyModule_GetState(module));
        state->core_type = checked(PyType_FromModuleAndSpec(module, &core_spec, nullptr));
        state->stream_type = checked(PyType_FromModuleAndSpec(module, &stream_spec, nullptr));
        if (PyModule_AddObjectRef(module, "_EngineCore", state->core_type) < 0 ||
            PyModule_AddObjectRef(module, "_StreamIterator", state->stream_type) < 0 ||
            PyModule_AddStringConstant(module, "_llama_cpp_commit", MINERU_LLAMA_CPP_LLAMA_COMMIT) < 0)
            throw PythonError{};
        return 0;
    } catch (...) { translate_exception(); return -1; }
}

// 向 GC 报告模块状态对堆类型的引用。
int module_traverse(PyObject * module, visitproc visit, void * arg) {
    auto * state = static_cast<ModuleState *>(PyModule_GetState(module));
    Py_VISIT(state->core_type); Py_VISIT(state->stream_type);
    return 0;
}

// 清理模块与类型的引用环。
int module_clear(PyObject * module) {
    auto * state = static_cast<ModuleState *>(PyModule_GetState(module));
    Py_CLEAR(state->core_type); Py_CLEAR(state->stream_type);
    return 0;
}
PyModuleDef_Slot module_slots[] = {{Py_mod_exec, reinterpret_cast<void *>(module_exec)}, {0, nullptr}};
PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_mineru_llama_cpp", "MinerU 原生引擎。",
                          sizeof(ModuleState), nullptr, module_slots, module_traverse, module_clear, nullptr};
} // namespace

// 使用多阶段初始化创建 abi3 模块。
PyMODINIT_FUNC PyInit__mineru_llama_cpp() { return PyModuleDef_Init(&module_def); }
