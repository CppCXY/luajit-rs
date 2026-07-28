-- Test debug: setupvalue, getlocal, setlocal

-- === debug.getlocal ===
local function local_test()
  local x = 42
  return debug.getlocal(1, 1)
end
local _, val = local_test()
assert(val == 42, "getlocal: " .. tostring(val))
assert(debug.getlocal(function() end, 99) == nil, "getlocal OOB")

-- === debug.setlocal ===
local function set_test()
  local x = 1
  debug.setlocal(1, 1, 100)
  return x
end
assert(set_test() == 100, "setlocal")

-- === debug.setupvalue / getupvalue (closed-over upvalues) ===
local function make_closure()
  local x = 10
  return function() return x end, function(v) x = v end
end
local getter, setter = make_closure()
local val = select(2, debug.getupvalue(getter, 1))
assert(type(val) == "number", "getupvalue returns number")
if val == 10 then
  debug.setupvalue(getter, 1, 99)
  assert(getter() == 99, "setupvalue works")
end

print("debug test OK")
