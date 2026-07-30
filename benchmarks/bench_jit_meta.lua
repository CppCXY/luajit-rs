-- Benchmark: Metamethod JIT coverage
-- Tests __index, __newindex, __call, __add, __len performance

local function bench(name, fn, iters, check)
  collectgarbage("collect")
  collectgarbage("collect")
  local start = os.clock()
  local r = fn(iters)
  local elapsed = os.clock() - start
  local rate = iters / elapsed / 1e6
  print(string.format("  %-42s %8.3f s  (%7.1f M ops/s)",
    name, elapsed, rate))
  if check then check(r) end
  return rate
end

local iters = 5000000
local small = 1000000
print("=== Metamethod JIT Coverage Benchmarks ===")
print()

-- ─── __index (function) ───────────────────────────────────────────

local idx_func_mt = {
  __index = function(t, k)
    if k == "x" then return 42
    elseif k == "y" then return 99
    else return nil end
  end
}
local proxy_func = setmetatable({}, idx_func_mt)
local m1 = bench("__index (func): proxy.x", function(n)
  local s = 0
  for _ = 1, n do s = s + proxy_func.x end
  return s
end, iters, function(r) assert(r == 42 * iters) end)

-- ─── __index (table) ──────────────────────────────────────────────

local delegate = { x = 99, y = 100 }
local proxy_tab = setmetatable({}, { __index = delegate })
local m2 = bench("__index (table): proxy->delegate.x", function(n)
  local s = 0
  for _ = 1, n do s = s + proxy_tab.x end
  return s
end, iters, function(r) assert(r == 99 * iters) end)

-- ─── __newindex ───────────────────────────────────────────────────

local nx_data = { x = 0 }
local nx_mt = {
  __index = function(t, k) return nx_data[k] end,
  __newindex = function(t, k, v) nx_data[k] = v end,
}
local nx_proxy = setmetatable({}, nx_mt)
local m3_nx_last = 0
bench("__newindex: proxy.x = i", function(n)
  for i = 1, n do
    nx_proxy.x = i
  end
  m3_nx_last = nx_proxy.x
end, small, function() assert(m3_nx_last == small, "nx: " .. m3_nx_last) end)

-- ─── __add ────────────────────────────────────────────────────────

local vec_mt = {
  __add = function(a, b) return { x = a.x + b.x, y = a.y + b.y } end,
}
local function newvec(x, y) return setmetatable({ x = x, y = y }, vec_mt) end
local va, vb = newvec(1, 2), newvec(3, 4)
local m4 = bench("__add: vector + vector", function(n)
  local s = 0
  for _ = 1, n do
    local v = va + vb
    s = s + v.x
  end
  return s
end, small, function(r) assert(r == 4 * small) end)

-- ─── __len ────────────────────────────────────────────────────────

local len_proxy = setmetatable({}, {
  __len = function() return 123 end,
})
local m5 = bench("__len: #proxy", function(n)
  local s = 0
  for _ = 1, n do s = s + #len_proxy end
  return s
end, iters, function(r) assert(r == 123 * iters, "len: " .. r) end)

-- ─── OOP pattern (get + set) ──────────────────────────────────────

local Point = { x = 0, y = 0 }
Point.__index = Point
function Point:move(dx, dy) self.x = self.x + dx; self.y = self.y + dy end
local pobj = setmetatable({ x = 0, y = 0 }, Point)
local m6 = bench("OOP: obj:move(1,2) in loop", function(n)
  for _ = 1, n do
    pobj:move(1, 2)
  end
  return pobj.x
end, small, function(r) assert(r == small) end)

-- ─── Baseline: raw table (no metatable) ───────────────────────────

local raw_tab = { x = 42 }
local m7 = bench("BASELINE: raw_tab.x (no meta)", function(n)
  local s = 0
  for _ = 1, n do s = s + raw_tab.x end
  return s
end, iters, function(r) assert(r == 42 * iters) end)

-- ─── Summary ──────────────────────────────────────────────────────

print()
print("== Summary (higher = better) ==")
print(string.format("  %-35s %7.1f M ops/s  (baseline)", "raw_tab.x", m7))
print(string.format("  %-35s %7.1f M ops/s  (%.1f%%)", "__index (func)", m1, m1/m7*100))
print(string.format("  %-35s %7.1f M ops/s  (%.1f%%)", "__index (table)", m2, m2/m7*100))
print(string.format("  %-35s %7.1f M ops/s", "__add (vec)", m4))
print(string.format("  %-35s %7.1f M ops/s", "__len", m5))
print(string.format("  %-35s %7.1f M ops/s", "OOP (move)", m6))
print()
print("ALL OK")
