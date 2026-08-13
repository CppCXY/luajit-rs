/* lualib.h — standard library openers for luajit-rs-cpi. */

#ifndef LUA_LUALIB_H
#define LUA_LUALIB_H

#include "lua.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LUA_LIBNAME "lua"

#define LUA_BASELIBNAME "_G"
#define LUA_TABLIBNAME "table"
#define LUA_STRLIBNAME "string"
#define LUA_MATHLIBNAME "math"
#define LUA_IOLIBNAME "io"
#define LUA_OSLIBNAME "os"
#define LUA_COLIBNAME "coroutine"
#define LUA_DBLIBNAME "debug"
#define LUA_BITLIBNAME "bit"
#define LUA_JITLIBNAME "jit"
#define LUA_FFILIBNAME "ffi"
#define LUA_LOADLIBNAME "package"

LUA_EXPORT void luaL_openlibs(lua_State *L);

#ifdef __cplusplus
}
#endif

#endif
