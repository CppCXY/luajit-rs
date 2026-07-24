-- luajit3 test suite runner
local tests = {
  "stmt_continue",
  "stmt_const",
  "stmt_compound",
  "number_underscore",
  "expr_bit",
  "expr_bit_bitop",
  "expr_coal",
  "expr_cond",
  "expr_customary",
  -- "expr_nav",       -- NYI: navigation operator
  "expr_shortfunc",
}

local npassed, nfailed, nskipped = 0, 0, 0

local function readfile(path)
  local f, err = io.open(path)
  if f == nil then return nil, err end
  local content = f:read("*a")
  f:close()
  return content
end

local function dotest(name)
  local path = name .. ".lua"
  local src, err = readfile(path)
  if src == nil then
    print(string.format("SKIP %s: %s", name, err))
    nskipped = nskipped + 1
    return
  end
  local f, err = loadstring(src, "=" .. name)
  if f == nil then
    print(string.format("FAIL %s (compile): %s", name, err))
    nfailed = nfailed + 1
    return
  end
  local ok, msg = pcall(f)
  if ok then
    npassed = npassed + 1
  else
    nfailed = nfailed + 1
    print(string.format("FAIL %s: %s", name, tostring(msg)))
  end
end

for _, t in ipairs(tests) do
  dotest(t)
end

print(string.format("\n=== %d passed, %d failed, %d skipped ===", npassed, nfailed, nskipped))
if nfailed > 0 then
  os.exit(1)
end
