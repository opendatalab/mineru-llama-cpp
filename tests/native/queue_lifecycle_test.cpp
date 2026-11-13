#include "server-queue.h"
#include "engine_lifecycle.h"
#include <cassert>
#include <future>
#include <string>
#include <thread>

// 覆盖提前停止、运行中停止及并发启动停止，验证停止状态不可逆。
int main(int argc, char ** argv) {
    assert(argc == 2);
    const std::string scenario = argv[1];
    if (scenario == "fatal-reader") {
        // 使用真实上游响应队列验证：后台循环抛错后，所有等待者通过状态轮询退出。
        server_queue queue;
        server_response responses;
        EngineLifecycle lifecycle;
        queue.on_new_task([](server_task &&, bool) { return true; });
        queue.on_update_slots([] { throw std::logic_error("decode loop failure"); });
        std::vector<std::unique_ptr<server_response_reader>> readers;
        for (int i = 0; i < 4; ++i) {
            auto reader = std::make_unique<server_response_reader>(queue, responses, 1);
            server_task task(SERVER_TASK_TYPE_COMPLETION);
            task.id = reader->get_new_id();
            reader->post_task(std::move(task));
            readers.emplace_back(std::move(reader));
        }
        std::vector<std::future<void>> waiting;
        for (auto & reader : readers) waiting.emplace_back(std::async(std::launch::async, [&lifecycle, ptr = reader.get()] {
            auto result = ptr->next([&] { return lifecycle.stopped(); });
            assert(!result);
            try { lifecycle.check_health(); assert(false); }
            catch (const EngineTerminated & error) { assert(std::string(error.what()) == "decode loop failure"); }
        }));
        std::thread loop([&] { lifecycle.run([&] { queue.start_loop(); }); });
        loop.join();
        for (auto & future : waiting) {
            assert(future.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
            future.get();
        }
        return 0;
    }
    const int repeats = scenario == "concurrent" ? 100 : 1;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        server_queue queue;
        int updates = 0;
        queue.on_new_task([](server_task &&, bool) { return true; });
        queue.on_update_slots([&] {
            ++updates;
            // Also lets the old implementation finish after losing an early stop.
            queue.terminate();
        });
        if (scenario == "early-stop") {
            queue.terminate();
            queue.terminate();
            queue.start_loop();
            assert(updates == 0);
        } else if (scenario == "running-stop") {
            queue.start_loop();
            assert(updates == 1);
        } else if (scenario == "concurrent") {
            std::thread loop([&] { queue.start_loop(); });
            queue.terminate();
            loop.join();
            assert(updates <= 1);
        } else {
            return 2;
        }
        // A stopped queue instance must not be revived by a delayed loop entry.
        const int finished_updates = updates;
        queue.start_loop();
        assert(updates == finished_updates);
    }
}
