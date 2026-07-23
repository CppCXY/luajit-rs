-- Benchmark: Arithmetic operations
local iterations = 100000000

local function check(name, cond, msg)
    if not cond then print("FAIL: " .. name .. (msg and ": " .. msg or "")); os.exit(1) end
end

print("=== Arithmetic Benchmark ===")
print("Iterations:", iterations)

-- Integer addition
local start = os.clock()
local sum = 0
for i = 1, iterations do
    sum = sum + i
end
local elapsed = os.clock() - start
local expected = (iterations * (iterations + 1)) / 2
print(string.format("Integer addition: sum: %d %.3f seconds (%.2f M ops/sec)", sum, elapsed, iterations / elapsed / 1000000))
check("int sum", sum == expected, string.format("got %d want %d", sum, expected))

-- Floating point
start = os.clock()
local result = 1.0
for i = 1, iterations do
    result = result * 1.0000001
end
elapsed = os.clock() - start
print(string.format("Float multiplication: result: %.6f %.3f seconds (%.2f M ops/sec)", result, elapsed, iterations / elapsed / 1000000))
check("float ok", result > 1.0)

-- Mixed operations
start = os.clock()
local x, y, z = 0, 0, 0
for i = 1, iterations do
    x = i + 5
    y = x * 2
    z = y - 3
end
elapsed = os.clock() - start
print(string.format("Mixed operations: z: %d %.3f seconds (%.2f M ops/sec)", z, elapsed, iterations / elapsed / 1000000))
-- Last iteration: i=N, x=N+5, y=(N+5)*2, z=(N+5)*2-3
local exp_z = (iterations + 5) * 2 - 3
check("mixed z", z == exp_z, string.format("got %d want %d", z, exp_z))

print("ALL OK")
