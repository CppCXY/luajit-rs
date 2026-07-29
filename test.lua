local ffi = require("ffi")
print(1ull + 1)
ffi.cdef[[
typedef struct {
    int x;
    int y;
} Point;
]]

-- 创建结构体实例
local p1 = ffi.new("Point", {10, 20})
print(p1.x, p1.y)  --> 10  20

-- 创建结构体数组
local points = ffi.new("Point[3]", {{1,2}, {3,4}, {5,6}})
for i = 0, 2 do
    print(points[i].x, points[i].y)
end