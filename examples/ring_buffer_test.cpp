#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <numeric>
#include "audio/audio_ring_buffer.h"

using namespace digital_human::audio;

int main() {
    std::cout << "========== RingBuffer Test ==========" << std::endl;

    // ==========================================
    // Test 1: Basic write → read
    // ==========================================
    std::cout << "\n[Test 1] Basic write/read..." << std::endl;
    {
        RingBuffer<int> rb(16);

        int src[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        size_t written = rb.write(src, 10);
        bool writeOk = (written == 10);

        int dst[10] = {0};
        size_t read = rb.read(dst, 10);
        bool readOk = (read == 10);

        bool dataOk = true;
        for (int i = 0; i < 10; i++) {
            if (dst[i] != src[i]) dataOk = false;
        }

        std::cout << (writeOk ? "  [PASS]" : "  [FAIL]") << " Write 10 elements" << std::endl;
        std::cout << (readOk ? "  [PASS]" : "  [FAIL]") << " Read 10 elements" << std::endl;
        std::cout << (dataOk ? "  [PASS]" : "  [FAIL]") << " Data intact" << std::endl;
    }

    // ==========================================
    // Test 2: Wrap-around (write past capacity)
    // ==========================================
    std::cout << "\n[Test 2] Wrap-around..." << std::endl;
    {
        RingBuffer<int> rb(8);

        // Write 5, read 3 → leaves 2 at positions 3,4
        int src1[5] = {10, 20, 30, 40, 50};
        rb.write(src1, 5);
        int tmp[3];
        rb.read(tmp, 3);

        // Now write 6 more → wraps around
        int src2[6] = {60, 70, 80, 90, 100, 110};
        size_t written = rb.write(src2, 6);
        bool writeOk = (written == 6);

        // Read all 8 elements
        int dst[8];
        size_t read = rb.read(dst, 8);
        bool readOk = (read == 8);

        // Expected: [40, 50, 60, 70, 80, 90, 100, 110]
        int expected[8] = {40, 50, 60, 70, 80, 90, 100, 110};
        bool dataOk = true;
        for (int i = 0; i < 8; i++) {
            if (dst[i] != expected[i]) dataOk = false;
        }

        std::cout << (writeOk ? "  [PASS]" : "  [FAIL]") << " Write 6 after partial read" << std::endl;
        std::cout << (readOk ? "  [PASS]" : "  [FAIL]") << " Read 8 (wrapped data)" << std::endl;
        std::cout << (dataOk ? "  [PASS]" : "  [FAIL]") << " Wrapped data intact" << std::endl;
    }

    // ==========================================
    // Test 3: Empty / Full states
    // ==========================================
    std::cout << "\n[Test 3] Empty / Full states..." << std::endl;
    {
        RingBuffer<int> rb(4);

        bool initEmpty = rb.empty();
        bool initNotFull = !rb.full();
        bool initAvailRead = (rb.availableRead() == 0);
        bool initAvailWrite = (rb.availableWrite() == 4);

        std::cout << (initEmpty ? "  [PASS]" : "  [FAIL]") << " Initially empty" << std::endl;
        std::cout << (initNotFull ? "  [PASS]" : "  [FAIL]") << " Initially not full" << std::endl;
        std::cout << (initAvailRead ? "  [PASS]" : "  [FAIL]") << " availableRead() = 0" << std::endl;
        std::cout << (initAvailWrite ? "  [PASS]" : "  [FAIL]") << " availableWrite() = 4" << std::endl;

        int src[4] = {1, 2, 3, 4};
        rb.write(src, 4);

        bool nowFull = rb.full();
        bool nowNotEmpty = !rb.empty();
        bool nowAvailWrite = (rb.availableWrite() == 0);
        bool nowAvailRead = (rb.availableRead() == 4);

        std::cout << (nowFull ? "  [PASS]" : "  [FAIL]") << " Full after filling" << std::endl;
        std::cout << (nowNotEmpty ? "  [PASS]" : "  [FAIL]") << " Not empty after filling" << std::endl;
        std::cout << (nowAvailWrite ? "  [PASS]" : "  [FAIL]") << " availableWrite() = 0" << std::endl;
        std::cout << (nowAvailRead ? "  [PASS]" : "  [FAIL]") << " availableRead() = 4" << std::endl;
    }

    // ==========================================
    // Test 4: Write to full → returns 0
    // ==========================================
    std::cout << "\n[Test 4] Write to full buffer..." << std::endl;
    {
        RingBuffer<int> rb(4);
        int src[4] = {1, 2, 3, 4};
        rb.write(src, 4);

        int extra[2] = {5, 6};
        size_t written = rb.write(extra, 2);

        bool writeBlocked = (written == 0);
        std::cout << (writeBlocked ? "  [PASS]" : "  [FAIL]")
                  << " Write to full returns 0 (got " << written << ")" << std::endl;
    }

    // ==========================================
    // Test 5: Read from empty → returns 0
    // ==========================================
    std::cout << "\n[Test 5] Read from empty..." << std::endl;
    {
        RingBuffer<int> rb(4);
        int dst[4];
        size_t read = rb.read(dst, 4);

        bool readBlocked = (read == 0);
        std::cout << (readBlocked ? "  [PASS]" : "  [FAIL]")
                  << " Read from empty returns 0" << std::endl;
    }

    // ==========================================
    // Test 6: skip()
    // ==========================================
    std::cout << "\n[Test 6] skip()..." << std::endl;
    {
        RingBuffer<int> rb(8);
        int src[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        rb.write(src, 8);

        size_t skipped = rb.skip(3);
        bool skipOk = (skipped == 3);
        bool availOk = (rb.availableRead() == 5);

        // Remaining should be [3, 4, 5, 6, 7]
        int dst[5];
        rb.read(dst, 5);
        bool dataOk = (dst[0] == 3 && dst[4] == 7);

        std::cout << (skipOk ? "  [PASS]" : "  [FAIL]") << " Skipped 3" << std::endl;
        std::cout << (availOk ? "  [PASS]" : "  [FAIL]") << " 5 remaining" << std::endl;
        std::cout << (dataOk ? "  [PASS]" : "  [FAIL]") << " Correct remaining data" << std::endl;
    }

    // ==========================================
    // Test 7: reset()
    // ==========================================
    std::cout << "\n[Test 7] reset()..." << std::endl;
    {
        RingBuffer<int> rb(4);
        int src[3] = {1, 2, 3};
        rb.write(src, 3);

        rb.reset();

        bool emptyOk = rb.empty();
        bool availOk = (rb.availableRead() == 0);
        bool writeAvailOk = (rb.availableWrite() == 4);

        std::cout << (emptyOk ? "  [PASS]" : "  [FAIL]") << " Empty after reset" << std::endl;
        std::cout << (availOk ? "  [PASS]" : "  [FAIL]") << " availableRead() = 0" << std::endl;
        std::cout << (writeAvailOk ? "  [PASS]" : "  [FAIL]") << " availableWrite() = capacity" << std::endl;
    }

    // ==========================================
    // Test 8: capacity()
    // ==========================================
    std::cout << "\n[Test 8] capacity()..." << std::endl;
    {
        RingBuffer<int> rb(1024);
        bool capOk = (rb.capacity() == 1024);
        std::cout << (capOk ? "  [PASS]" : "  [FAIL]") << " capacity = 1024" << std::endl;
    }

    // ==========================================
    // Test 9: Partial write (not enough space)
    // ==========================================
    std::cout << "\n[Test 9] Partial write..." << std::endl;
    {
        RingBuffer<int> rb(4);
        int src1[3] = {1, 2, 3};
        rb.write(src1, 3);

        // Only 1 space left, try to write 5
        int src2[5] = {4, 5, 6, 7, 8};
        size_t written = rb.write(src2, 5);

        bool partialOk = (written == 1);
        bool fullOk = rb.full();

        std::cout << (partialOk ? "  [PASS]" : "  [FAIL]")
                  << " Partial write returns 1 (got " << written << ")" << std::endl;
        std::cout << (fullOk ? "  [PASS]" : "  [FAIL]") << " Buffer now full" << std::endl;
    }

    // ==========================================
    // Test 10: Multi-threaded SPSC
    // ==========================================
    std::cout << "\n[Test 10] Multi-threaded SPSC..." << std::endl;
    {
        const size_t capacity = 1024;
        const int totalItems = 100000;
        RingBuffer<int> rb(capacity);

        std::atomic<bool> producerDone{false};
        std::atomic<int> producedCount{0};
        std::atomic<int> consumedCount{0};
        std::vector<int> consumed(totalItems, -1);

        // Producer: write sequential numbers
        std::thread producer([&]() {
            int val = 0;
            while (val < totalItems) {
                size_t avail = rb.availableWrite();
                if (avail > 0) {
                    size_t n = std::min(avail, size_t(50));
                    n = std::min(n, size_t(totalItems - val));
                    // Prepare batch
                    std::vector<int> batch(n);
                    for (size_t i = 0; i < n; i++) batch[i] = val++;
                    size_t written = rb.write(batch.data(), n);
                    producedCount.fetch_add(static_cast<int>(written));
                } else {
                    std::this_thread::yield();
                }
            }
            producerDone.store(true);
        });

        // Consumer: read and verify sequential
        std::thread consumer([&]() {
            int expectedVal = 0;
            while (!producerDone.load() || rb.availableRead() > 0) {
                size_t avail = rb.availableRead();
                if (avail > 0) {
                    size_t n = std::min(avail, size_t(50));
                    std::vector<int> batch(n);
                    size_t read = rb.read(batch.data(), n);
                    for (size_t i = 0; i < read; i++) {
                        int idx = consumedCount.fetch_add(1);
                        if (idx < totalItems) consumed[idx] = batch[i];
                    }
                } else {
                    std::this_thread::yield();
                }
            }
        });

        producer.join();
        consumer.join();

        // Verify: all items received, no gaps, no duplicates
        bool countOk = (consumedCount.load() == totalItems);
        bool seqOk = true;
        for (int i = 0; i < totalItems && seqOk; i++) {
            if (consumed[i] != i) seqOk = false;
        }

        std::cout << (countOk ? "  [PASS]" : "  [FAIL]")
                  << " All " << totalItems << " items transferred (got "
                  << consumedCount.load() << ")" << std::endl;
        std::cout << (seqOk ? "  [PASS]" : "  [FAIL]")
                  << " Sequential order preserved" << std::endl;
    }

    std::cout << "\n========== Test Complete ==========" << std::endl;
    return 0;
}
