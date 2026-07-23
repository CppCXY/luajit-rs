-- Benchmark: Table operations
local iterations = 10000000

local function check(name, cond, msg)
    if not cond then
        print("FAIL: " .. name .. (msg and ": " .. msg or ""))
        os.exit(1)
    end
end

print("=== Table Operations Benchmark ===")
print("Iterations:", iterations)

-- Array creation and access (reduced iterations due to GC overhead)
local array_iters = iterations / 100
local start = os.clock()
for i = 1, array_iters do
    local t = {1, 2, 3, 4, 5}
    local x = t[1] + t[5]
end
local elapsed = os.clock() - start
print(string.format("Array creation & access: %.3f seconds (%.2f M ops/sec)", elapsed, array_iters / elapsed / 1000000))

-- Table insertion
start = os.clock()
local t = {}
for i = 1, iterations do
    t[i] = i
end
elapsed = os.clock() - start
print(string.format("Table insertion: %.3f seconds (%.2f M inserts/sec)", elapsed, iterations / elapsed / 1000000))
-- Verify
check("insert[1]", t[1] == 1)
check("insert[mid]", t[iterations/2] == iterations/2)
check("insert[last]", t[iterations] == iterations)

-- Table iteration
start = os.clock()
local sum = 0
for i = 1, iterations do
    sum = sum + t[i]
end
elapsed = os.clock() - start
print(string.format("Table access: %.3f seconds (%.2f M accesses/sec)", elapsed, iterations / elapsed / 1000000))
-- Verify: sum of 1..N = N*(N+1)/2
local expected_sum = (iterations * (iterations + 1)) / 2
check("table sum", sum == expected_sum, string.format("got %d want %d", sum, expected_sum))

-- Hash table operations
start = os.clock()
local ht = {}
for i = 1, 100000 do
    ht["key" .. i] = i
end
elapsed = os.clock() - start
print(string.format("Hash table insertion (100k): %.3f seconds", elapsed))
-- Verify
check("hash[1]", ht["key1"] == 1)
check("hash[mid]", ht["key50000"] == 50000)
check("hash[last]", ht["key100000"] == 100000)

-- ipairs iteration (reduced)
local ipairs_iters = 10
start = os.clock()
sum = 0
for i = 1, ipairs_iters do
    for idx, val in ipairs(t) do
        sum = sum + val
    end
end
elapsed = os.clock() - start
print(string.format("ipairs iteration (%dx%d): %.3f seconds", ipairs_iters, iterations, elapsed))
-- Verify: sum of 1..N done ipairs_iters times
check("ipairs sum", sum == expected_sum * ipairs_iters, string.format("got %d want %d", sum, expected_sum * ipairs_iters))

print("ALL OK")
