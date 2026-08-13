/* luaconf.h — configuration for luajit-rs-cpi (LuaJIT 2.1 / Lua 5.1 API). */

#ifndef LUA_LUACONF_H
#define LUA_LUACONF_H

#define LUA_VERSION_MAJOR "5"
#define LUA_VERSION_MINOR "1"
#define LUA_VERSION_NUM 501
#define LUA_VERSION_RELEASE "5"

#define LUAJIT_VERSION "LuaJIT 2.1.0-rs"
#define LUAJIT_VERSION_NUM 20100
#define LUAJIT_VERSION_SYM luaJIT_version_2_1_0_rs

#define LUA_NUMBER double
#define LUA_INTEGER ptrdiff_t

#define LUA_PATH_DEFAULT ""
#define LUA_CPATH_DEFAULT ""

#define LUA_MAXCAPTURES 32
#define LUA_BUFFERSIZE 8192

#define LUAL_BUFFERSIZE 8192

#define LUA_IDSIZE 60

#define LUA_DIRSEP "/"

#define LUA_EXTRASPACE 0

#if defined(_WIN32)
#  define LUAI_IS32INT 0
#  define LUA_EXPORT __declspec(dllexport)
#else
#  define LUA_EXPORT __attribute__((visibility("default")))
#endif

#endif
