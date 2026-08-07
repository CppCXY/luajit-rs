local function f(s, p)
  local i,e = string.find(s, p)
  if i then return string.sub(s, i, e) end
end
print("r1:", f("]]]�b", "[^]]"))
print("r2:", f("abc]]]", "[^]]"))
