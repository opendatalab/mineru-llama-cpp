#pragma once
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <typeinfo>

// 仅用于已确认退出的引擎，不能由普通服务端请求错误构造。
struct EngineTerminated : std::runtime_error {
    std::string native_type;
    // 保留首次原生异常的类型名与消息；C++ 不提供可移植的栈回溯。
    EngineTerminated(const std::string & message, const std::string & type)
        : std::runtime_error(message), native_type(type) {}
};

class EngineLifecycle {
public:
    // 在后台线程边界捕获可恢复到宿主的异常；abort/访问违规仍交由进程管理器处理。
    template<class Function> void run(Function function) noexcept {
        std::exception_ptr failure;
        try { function(); } catch (...) { failure = std::current_exception(); }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!stopped_) failure_ = failure;
        stopped_ = true;
    }

    // 关闭只停止接单，不覆盖已经记录的首次致命原因。
    void begin_close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
    }

    // 响应 reader 用此状态退出等待，不终止上游结果队列或宿主进程。
    bool stopped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopped_ || closing_;
    }

    // 快速状态快照只确认循环是否已退出，不将推理繁忙或无心跳视为死亡。
    void check_health() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_) {
            try { std::rethrow_exception(failure_); }
            catch (const std::exception & error) { throw EngineTerminated(error.what(), typeid(error).name()); }
            catch (...) { throw EngineTerminated("unknown native loop exception", "unknown"); }
        }
        if (closing_) throw EngineTerminated("engine is closing", "");
        if (stopped_) throw EngineTerminated("engine loop exited unexpectedly", "");
    }
private:
    mutable std::mutex mutex_;
    std::exception_ptr failure_;
    bool stopped_ = false;
    bool closing_ = false;
};
