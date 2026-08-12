local ffi = require("ffi")

do --- basic int callback
  local cb = ffi.cast("int (*)(int, int, int)", function(a, b, c)
    return a + b + c
  end)
  assert(cb(10, 99, 13) == 122)
  assert(cb(-42, 17, 12345) == -42 + 17 + 12345)
end

do --- double and float callbacks
  local d = ffi.cast("double (*)(double, float, double)", function(a, b, c)
    return a + b + c
  end)
  assert(d(7.125, -123.25, 9999.33) == 7.125 - 123.25 + 9999.33)

  local f = ffi.cast("float (*)(double, float, double)", function(a, b, c)
    return a + b + c
  end)
  assert(f(7.125, -123.25, 9999.33) == 9883.205078125)
end

do --- int64_t callback
  local cb = ffi.cast("int64_t (*)(int64_t, int64_t)", function(a, b)
    return a + b
  end)
  assert(cb(12345678901234567LL, 70000000000000001LL) == 12345678901234567LL + 70000000000000001LL)
end

do --- qsort through a callback
  ffi.cdef[[
  void qsort(void *base, int nmemb, int size, void *cmp);
  ]]
  local arr = ffi.new("int[8]", 3, 1, 4, 1, 5, 9, 2, 6)
  local function cmp(pa, pb)
    local a, b = ffi.cast("int *", pa)[0], ffi.cast("int *", pb)[0]
    if a < b then return -1 elseif a > b then return 1 else return 0 end
  end
  ffi.C.qsort(arr, 8, ffi.sizeof("int"), ffi.cast("int (*)(const void *, const void *)", cmp))
  for i = 0, 6 do assert(arr[i] <= arr[i + 1]) end
  assert(arr[0] == 1 and arr[7] == 9)
end

do --- callback errors surface from the enclosing C call
  local ok, err = pcall(function()
    local cb = ffi.cast("int (*)(void)", function() error("boom") return 1 end)
    return cb()
  end)
  assert(ok == false)
  assert(string.find(err, "boom"))
end

do --- invalid callbacks fail when called (result conversion)
  assert(pcall(function()
    local cb = ffi.cast("int (*)(void)", function() end)
    return cb()
  end) == false)
  assert(pcall(function()
    local cb = ffi.cast("int (*)(void)", function(a) return a + 1 end)
    return cb()
  end) == false)
  assert(ffi.cast("void (*)(void)", function() end) ~= nil)
end
