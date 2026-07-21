#!/usr/bin/env bash
# Run all Lua tests in tests/luajit3_test through luajit-rs
# Usage: ./run_tests.sh [--release]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="$SCRIPT_DIR/tests/luajit3_test"
RELEASE=""

if [ "$1" = "--release" ]; then
    RELEASE="--release"
    BIN="target/release/luajit-rs"
else
    BIN="target/debug/luajit-rs"
fi

if [ ! -f "$BIN" ]; then
    echo "Building luajit-rs..."
    cargo build $RELEASE
fi

# Tests expected to fail (unsupported features / NYI)
declare -A EXPECTED_FAIL=(
)

passed=0
failed=0
skipped=()

echo "Running tests from luajit3_test/"
printf '=%.0s' {1..60}
echo

for f in "$TEST_DIR"/*.lua; do
    name=$(basename "$f")

    if [ -n "${EXPECTED_FAIL[$name]}" ]; then
        echo "  SKIP  $name  -- ${EXPECTED_FAIL[$name]}"
        skipped+=("$name")
        continue
    fi

    output=$("$BIN" "$f" 2>&1) && ec=$? || ec=$?

    if [ $ec -eq 0 ]; then
        echo "  PASS  $name"
        (( ++passed ))
    else
        msg="${output:0:120}"
        [ ${#output} -gt 120 ] && msg="${msg}..."
        echo "  FAIL  $name  -- $msg"
        (( ++failed ))
    fi
done

printf '=%.0s' {1..60}
echo
echo "Passed: $passed | Failed: $failed | Skipped: ${#skipped[@]}"

if [ ${#skipped[@]} -gt 0 ]; then
    echo "Skipped:"
    for s in "${skipped[@]}"; do
        echo "  $s -- ${EXPECTED_FAIL[$s]}"
    done
fi

exit $failed
