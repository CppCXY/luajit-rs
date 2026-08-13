/* testmod.c — a real C extension module for the require() end-to-end test.
 * Compiled at test time into a shared library and loaded via
 * package.loadlib + require. */

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#include "lua.h"
#include "lauxlib.h"

static int t_add(lua_State *L) {
    double a = luaL_checknumber(L, 1);
    double b = luaL_checknumber(L, 2);
    lua_pushnumber(L, a + b);
    return 1;
}

static int t_sayhi(lua_State *L) {
    const char *s = luaL_checkstring(L, 1);
    lua_pushfstring(L, "hi %s", s);
    return 1;
}

static const luaL_Reg t_reg[] = {
    {"add", t_add},
    {"sayhi", t_sayhi},
    {NULL, NULL},
};

EXPORT int luaopen_testmod(lua_State *L) {
    luaL_register(L, "testmod", t_reg);
    return 1;
}
