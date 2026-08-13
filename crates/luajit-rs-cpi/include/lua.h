/* lua.h — core C API for luajit-rs-cpi (LuaJIT 2.1 / Lua 5.1 surface). */

#ifndef LUA_H
#define LUA_H

#include <stddef.h>
#include <stdarg.h>

#include "luaconf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lua_State lua_State;

typedef double lua_Number;
typedef ptrdiff_t lua_Integer;

typedef int (*lua_CFunction)(lua_State *L);

/* status codes for lua_pcall and lua_load */
#define LUA_YIELD 1
#define LUA_ERRRUN 2
#define LUA_ERRSYNTAX 3
#define LUA_ERRMEM 4
#define LUA_ERRERR 5

/* basic types */
#define LUA_TNONE (-1)
#define LUA_TNIL 0
#define LUA_TBOOLEAN 1
#define LUA_TLIGHTUSERDATA 2
#define LUA_TNUMBER 3
#define LUA_TSTRING 4
#define LUA_TTABLE 5
#define LUA_TFUNCTION 6
#define LUA_TUSERDATA 7
#define LUA_TTHREAD 8

/* pseudo-indices */
#define LUA_REGISTRYINDEX (-10000)
#define LUA_ENVIRONINDEX (-10001)
#define LUA_GLOBALSINDEX (-10002)

#define lua_upvalueindex(i) (LUA_GLOBALSINDEX - (i))

/* pre-defined references */
#define LUA_NOREF (-2)
#define LUA_REFNIL (-1)

#define LUA_MULTRET (-1)
#define LUA_OK 0

LUA_EXPORT lua_State *luaL_newstate(void);
LUA_EXPORT void lua_close(lua_State *L);
LUA_EXPORT void luaL_openlibs(lua_State *L);

LUA_EXPORT int lua_gettop(lua_State *L);
LUA_EXPORT void lua_settop(lua_State *L, int idx);
LUA_EXPORT void lua_pop(lua_State *L, int n);
LUA_EXPORT void lua_pushvalue(lua_State *L, int idx);
LUA_EXPORT void lua_remove(lua_State *L, int idx);
LUA_EXPORT void lua_replace(lua_State *L, int idx);
LUA_EXPORT int lua_absindex(lua_State *L, int idx);

LUA_EXPORT void lua_pushnil(lua_State *L);
LUA_EXPORT void lua_pushnumber(lua_State *L, lua_Number n);
LUA_EXPORT void lua_pushinteger(lua_State *L, lua_Integer n);
LUA_EXPORT void lua_pushboolean(lua_State *L, int b);
LUA_EXPORT void lua_pushstring(lua_State *L, const char *s);
LUA_EXPORT void lua_pushlstring(lua_State *L, const char *s, size_t len);
LUA_EXPORT void lua_pushcfunction(lua_State *L, lua_CFunction f);
LUA_EXPORT void lua_pushcclosure(lua_State *L, lua_CFunction f, int n);

LUA_EXPORT int lua_type(lua_State *L, int idx);
LUA_EXPORT const char *lua_typename(lua_State *L, int tp);
LUA_EXPORT int lua_isnil(lua_State *L, int idx);
LUA_EXPORT int lua_isboolean(lua_State *L, int idx);
LUA_EXPORT int lua_isnumber(lua_State *L, int idx);
LUA_EXPORT int lua_isstring(lua_State *L, int idx);
LUA_EXPORT int lua_istable(lua_State *L, int idx);
LUA_EXPORT int lua_isfunction(lua_State *L, int idx);
LUA_EXPORT int lua_isuserdata(lua_State *L, int idx);

LUA_EXPORT lua_Number lua_tonumber(lua_State *L, int idx);
LUA_EXPORT lua_Integer lua_tointeger(lua_State *L, int idx);
LUA_EXPORT int lua_toboolean(lua_State *L, int idx);
LUA_EXPORT const char *lua_tolstring(lua_State *L, int idx, size_t *len);
LUA_EXPORT size_t lua_objlen(lua_State *L, int idx);
LUA_EXPORT void *lua_touserdata(lua_State *L, int idx);
#define lua_tostring(L, i) lua_tolstring(L, (i), NULL)

LUA_EXPORT void lua_newtable(lua_State *L);
LUA_EXPORT void lua_createtable(lua_State *L, int narr, int nrec);
LUA_EXPORT void lua_gettable(lua_State *L, int idx);
LUA_EXPORT void lua_settable(lua_State *L, int idx);
LUA_EXPORT void lua_rawget(lua_State *L, int idx);
LUA_EXPORT void lua_rawset(lua_State *L, int idx);
LUA_EXPORT void lua_rawgeti(lua_State *L, int idx, int n);
LUA_EXPORT void lua_rawseti(lua_State *L, int idx, int n);
LUA_EXPORT void lua_getfield(lua_State *L, int idx, const char *k);
LUA_EXPORT void lua_setfield(lua_State *L, int idx, const char *k);
LUA_EXPORT void lua_getglobal(lua_State *L, const char *name);
LUA_EXPORT void lua_setglobal(lua_State *L, const char *name);
LUA_EXPORT void lua_register(lua_State *L, const char *name, lua_CFunction f);
LUA_EXPORT int lua_next(lua_State *L, int idx);

LUA_EXPORT void *lua_newuserdata(lua_State *L, size_t size);

LUA_EXPORT int lua_getmetatable(lua_State *L, int objindex);
LUA_EXPORT void lua_setmetatable(lua_State *L, int objindex);

LUA_EXPORT void lua_call(lua_State *L, int nargs, int nresults);
LUA_EXPORT int lua_pcall(lua_State *L, int nargs, int nresults, int errfunc);

LUA_EXPORT int lua_error(lua_State *L);
LUA_EXPORT const char *lua_pushfstring(lua_State *L, const char *fmt, ...);

LUA_EXPORT int luaL_loadstring(lua_State *L, const char *s);
LUA_EXPORT int luaL_dostring(lua_State *L, const char *s);

/* registry references (LuaJIT-compatible) */
LUA_EXPORT int luaL_ref(lua_State *L, int t);
LUA_EXPORT void luaL_unref(lua_State *L, int t, int ref);
LUA_EXPORT int lua_rawget_ref(lua_State *L, int ref);

#ifdef __cplusplus
}
#endif

#endif
