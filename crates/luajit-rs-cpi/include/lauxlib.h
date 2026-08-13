/* lauxlib.h — auxiliary library for luajit-rs-cpi. */

#ifndef LUA_LAUXLIB_H
#define LUA_LAUXLIB_H

#include <stddef.h>
#include <stdio.h>

#include "lua.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct luaL_Reg {
    const char *name;
    lua_CFunction func;
} luaL_Reg;

LUA_EXPORT int luaL_newmetatable(lua_State *L, const char *tname);
LUA_EXPORT void luaL_getmetatable(lua_State *L, const char *tname);
LUA_EXPORT void luaL_setmetatable(lua_State *L, const char *tname);
LUA_EXPORT void *luaL_checkudata(lua_State *L, int ud, const char *tname);

LUA_EXPORT lua_Number luaL_checknumber(lua_State *L, int narg);
LUA_EXPORT lua_Integer luaL_checkinteger(lua_State *L, int narg);
LUA_EXPORT const char *luaL_checklstring(lua_State *L, int narg, size_t *len);
LUA_EXPORT const char *luaL_checkstring(lua_State *L, int narg);
#define luaL_checkint(L, n) ((int)luaL_checkinteger((L), (n)))
#define luaL_optstring(L, n, d) \
    (lua_isnil((L), (n)) || lua_isstring((L), (n)) ? luaL_checkstring((L), (n)) : (d))
#define luaL_optnumber(L, n, d) \
    (lua_isnil((L), (n)) || lua_isnumber((L), (n)) ? luaL_checknumber((L), (n)) : (d))
#define luaL_optinteger(L, n, d) \
    (lua_isnil((L), (n)) || lua_isnumber((L), (n)) ? luaL_checkinteger((L), (n)) : (d))

LUA_EXPORT int luaL_error(lua_State *L, const char *fmt, ...);
LUA_EXPORT int luaL_argerror(lua_State *L, int narg, const char *fmt, ...);
LUA_EXPORT int luaL_typerror(lua_State *L, int narg, const char *tname);

LUA_EXPORT void luaL_setfuncs(lua_State *L, const luaL_Reg *l);
LUA_EXPORT int luaL_newlib(lua_State *L, const luaL_Reg *l);
LUA_EXPORT int luaL_register(lua_State *L, const char *libname, const luaL_Reg *l);

#define luaL_argcheck(L, cond, numarg, extramsg) \
    ((void)((cond) || luaL_argerror(L, (numarg), (extramsg))))

#ifdef __cplusplus
}
#endif

#endif
