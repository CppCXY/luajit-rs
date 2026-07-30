-- Benchmark: Allocation sinking effectiveness
-- Run: luajit-rs bench_jit_sink.lua
--      luajit-rs -j off bench_jit_sink.lua

local function bench(name, fn, iters, check)
  collectgarbage("collect")
  collectgarbage("collect")
  local start = os.clock()
  local r = fn(iters)
  local elapsed = os.clock() - start
  local rate = iters / elapsed / 1e6
  print(string.format("  %-40s %8.3f s  (%7.1f M ops/s)",
    name, elapsed, rate))
  if check then check(r) end
  return rate
end

local iters = 5000000

print("=== Allocation Sinking Benchmarks ===")
print(string.format("Iterations: %d", iters))
print()

-- T1: Table literal in hot loop
local t1_rate = bench("t = {a,b,c,d,e} in loop", function(n)
  local s = 0
  for i = 1, n do
    local t = {1, 2, 3, 4, 5}
    s = s + t[1] + t[5]
  end
  return s
end, iters, function(r) assert(r == 6 * iters, "t1: " .. r) end)

-- T2: String concat chain
local t2_rate = bench('"a".."b".."c".."d" in loop', function(n)
  local r = ""
  for i = 1, n do
    r = "a" .. "b" .. "c" .. "d"
  end
  return r
end, iters, function(r) assert(r == "abcd", "t2: " .. tostring(r)) end)

-- T3: ipairs over fixed-size array
local t_arr = {}
for i = 1, 50 do t_arr[i] = i end
local t3_rate = bench("ipairs() over 50-elt array", function(n)
  local s = 0
  for _ = 1, n do
    for _, v in ipairs(t_arr) do
      s = s + v
    end
  end
  return s
end, iters, function(r) assert(r == 1275 * iters, "t3: " .. r) end)

-- T4: pairs over fixed-size hash
local t_hash = {}
for i = 1, 50 do t_hash["k" .. i] = i end
local t4_rate = bench("pairs() over 50-key hash", function(n)
  local s = 0
  for _ = 1, n do
    for k, v in pairs(t_hash) do
      s = s + v
    end
  end
  return s
end, iters, function(r) assert(r == 1275 * iters, "t4: " .. r) end)

-- T5: table.concat
local t_cat = {}
for i = 1, 200 do t_cat[i] = tostring(i % 10) end
local t5_rate = bench("table.concat(200 items)", function(n)
  local r = ""
  for _ = 1, n do
    r = table.concat(t_cat)
  end
  return r
end, 50000)

-- T6: string.sub
local big_str = string.rep("x", 5000)
local t6_rate = bench("string.sub(1,5) on 5KB str", function(n)
  local r = ""
  for _ = 1, n do
    r = string.sub(big_str, 1, 5)
  end
  return r
end, iters, function(r) assert(r == "xxxxx") end)

-- T7: tostring on numbers (boxing)
local t7_rate = bench("tostring(number) in loop", function(n)
  local r = ""
  for i = 1, n do
    r = tostring(i)
  end
  return r
end, 500000)

-- T8: new table with initial values
local t8_rate = bench("{1,2,3,4,5} simple read", function(n)
  local s = 0
  for i = 1, n do
    local t = {10, 20, 30}
    s = s + t[1]
  end
  return s
end, iters, function(r) assert(r == 10 * iters, "t8: " .. r) end)

-- T9: #table in loop
local t_len = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
local t9_rate = bench("#table in loop", function(n)
  local s = 0
  for _ = 1, n do
    s = s + #t_len
  end
  return s
end, iters, function(r) assert(r == 10 * iters) end)

-- Summary
print()
print("== Summary (higher = better) ==")
local rates = {
  {"{1,2,3,4,5} literal", t1_rate},
  {"str concat chain", t2_rate},
  {"ipairs()", t3_rate},
  {"pairs()", t4_rate},
  {"table.concat", t5_rate},
  {"string.sub", t6_rate},
  {"tostring(num)", t7_rate},
  {"{10,20,30} read", t8_rate},
  {"#table", t9_rate},
}
for _, e in ipairs(rates) do
  print(string.format("  %-30s %7.1f M ops/s", e[1], e[2]))
end
print()
print("ALL OK")
