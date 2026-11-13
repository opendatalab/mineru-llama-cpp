#include "engine_lifecycle.h"
#include <cassert>
#include <thread>
#include <vector>

// 验证后台退出锁存、原始异常诊断、并发读取及正常关闭。
int main() {
    EngineLifecycle state;
    state.check_health();
    std::thread loop([&] { state.run([] { throw std::logic_error("first failure"); }); });
    loop.join();
    assert(state.stopped());
    state.run([] { throw std::runtime_error("second failure"); });
    std::vector<std::thread> readers;
    for (int i = 0; i < 8; ++i) readers.emplace_back([&] {
        try { state.check_health(); assert(false); }
        catch (const EngineTerminated & error) {
            assert(std::string(error.what()) == "first failure");
            assert(error.native_type == typeid(std::logic_error).name());
        }
    });
    for (auto & reader : readers) reader.join();
    state.begin_close();
    try { state.check_health(); assert(false); }
    catch (const EngineTerminated & error) { assert(std::string(error.what()) == "first failure"); }
    EngineLifecycle exited;
    exited.run([] {});
    assert(exited.stopped());
    try { exited.check_health(); assert(false); }
    catch (const EngineTerminated & error) { assert(std::string(error.what()).find("exited") != std::string::npos); }
    EngineLifecycle closing;
    closing.begin_close();
    closing.run([] {});
    try { closing.check_health(); assert(false); }
    catch (const EngineTerminated & error) { assert(std::string(error.what()).find("closing") != std::string::npos); }
    EngineLifecycle unknown;
    unknown.run([] { throw 42; });
    try { unknown.check_health(); assert(false); }
    catch (const EngineTerminated & error) { assert(error.native_type == "unknown"); }
}
