#!/bin/bash
# 运行全部单元测试，输出通过/失败统计
cd "$(dirname "$0")/build/bin" || exit 1

# 测试列表（按类型分组）
LOGIC_TESTS=(
    frame_scheduler_test
    render_thread_test
    av_sync_test
    audio_sync_test
    ring_buffer_test
)

AUDIO_TESTS=(
    preemphasis_test
    rmsnorm_test
    noise_reduction_test
    vad_test
    audio_framer_test
    cmvn_test
    mel_feature_test
    audio_processor_test
)

PIPELINE_TESTS=(
    pipeline_test
    bugfix_v2_test
    bugfix_verification_test
    v1_review_fixes_test
)

# 需要外部资源（模型/音频/图像）的测试，单独运行
INTEGRATION_TESTS=(
    full_pipeline_test
    inference_worker_test
    output_processor_test
    face_detector_test
    face_aligner_test
    face_mask_generator_test
    image_load_test
)

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
FAILED_TESTS=()

run_test() {
    local t=$1
    local timeout_sec=${2:-60}
    if [ ! -x "./$t" ]; then
        echo "[SKIP] $t (binary not found)"
        ((SKIP_COUNT++))
        return
    fi
    local output
    output=$(timeout "$timeout_sec" ./"$t" 2>&1)
    local rc=$?
    # 从输出中提取通过/失败总数
    local pass_line
    pass_line=$(echo "$output" | grep -E "通过:|Passed:|PASS.*[0-9]+" | tail -3)
    local summary
    summary=$(echo "$output" | tail -8 | tr -d '\r')
    if [ $rc -eq 0 ]; then
        echo "[PASS] $t (rc=0)"
        echo "$summary" | sed 's/^/        /'
        ((PASS_COUNT++))
    elif [ $rc -eq 124 ]; then
        echo "[TIMEOUT] $t (>${timeout_sec}s)"
        ((FAIL_COUNT++))
        FAILED_TESTS+=("$t")
    else
        echo "[FAIL] $t (rc=$rc)"
        echo "$summary" | sed 's/^/        /'
        ((FAIL_COUNT++))
        FAILED_TESTS+=("$t")
    fi
}

echo "=========================================="
echo "  Logic Tests (调度器/同步/队列)"
echo "=========================================="
for t in "${LOGIC_TESTS[@]}"; do
    run_test "$t" 30
done

echo
echo "=========================================="
echo "  Audio Tests (音频预处理/特征)"
echo "=========================================="
for t in "${AUDIO_TESTS[@]}"; do
    run_test "$t" 30
done

echo
echo "=========================================="
echo "  Pipeline Tests (Pipeline 集成)"
echo "=========================================="
for t in "${PIPELINE_TESTS[@]}"; do
    run_test "$t" 60
done

echo
echo "=========================================="
echo "  Integration Tests (视觉/推理)"
echo "=========================================="
for t in "${INTEGRATION_TESTS[@]}"; do
    run_test "$t" 90
done

echo
echo "=========================================="
echo "  测试汇总"
echo "=========================================="
echo "  通过: $PASS_COUNT"
echo "  失败: $FAIL_COUNT"
echo "  跳过: $SKIP_COUNT"
if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "  失败测试: ${FAILED_TESTS[*]}"
fi
echo "=========================================="

# CI 门禁契约：任何失败或超时都必须以非零状态退出，否则流水线会误判成功。
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "[FATAL] $FAIL_COUNT 个测试失败，脚本以非零状态退出。" >&2
    exit 1
fi

exit 0
