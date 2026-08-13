/*
 * ljrs_shim.c — the C half of the luajit-rs-cpi error machinery.
 *
 * LuaJIT's C API uses longjmp for `lua_error`/`luaL_error`: an error
 * raised inside a C function unwinds to the nearest protected call. The
 * engine below is Rust, so a longjmp must NEVER cross a live Rust frame
 * (skipped destructors = UB). The discipline:
 *
 *  - Every C function invoked from Lua runs inside `ljrs_cfunc_invoke`,
 *    which sets a jmp_buf (layer 1). `ljrs_raise` longjmps there, so the
 *    skipped frames are pure C frames (the module function + this shim).
 *  - Error-raising entry points are C functions that (a) call a Rust
 *    helper which fully returns, (b) then `ljrs_raise`. No Rust frame is
 *    ever left live above the longjmp.
 *  - `ljrs_raise` with an empty jmp chain aborts (LuaJIT panics on an
 *    unprotected error).
 */

#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#define LJRS_THREAD __declspec(thread)
#define LJRS_EXPORT __declspec(dllexport)
#else
#define LJRS_THREAD __thread
#define LJRS_EXPORT __attribute__((visibility("default")))
#endif

/* ---- Rust helpers (defined in lib.rs) ---------------------------------- */
extern void ljrs_error_set(void *L, const char *msg, size_t len);
extern void ljrs_error_take(void *L);
extern int ljrs_call_impl(void *L, int nargs, int nresults);
extern void *ljrs_checkudata(void *L, int idx, const char *tname);

/* ---- exported Rust API functions used here ------------------------------ */
extern int lua_type(void *L, int idx);
extern const char *lua_typename(void *L, int tp);
extern int lua_isnumber(void *L, int idx);
extern double lua_tonumber(void *L, int idx);
extern long long lua_tointeger(void *L, int idx);
extern int lua_isstring(void *L, int idx);
extern const char *lua_tolstring(void *L, int idx, size_t *len);
extern void lua_pushnumber(void *L, double n);
extern void lua_pushstring(void *L, const char *s);
extern void lua_pushnil(void *L);
extern void *lua_newuserdata(void *L, size_t sz);
extern void lua_pushcfunction(void *L, void *f);

/* ---- jump chain ---------------------------------------------------------- */

struct ljrs_jmp {
    jmp_buf env;
    struct ljrs_jmp *prev;
};

LJRS_THREAD struct ljrs_jmp *ljrs_jmp_top = 0;

/* Raise a pending error: longjmp to the nearest C-function invoke frame. */
static void ljrs_raise(void) {
    if (ljrs_jmp_top) {
        longjmp(ljrs_jmp_top->env, 1);
    }
    fputs("ljrs: unprotected error in call to Lua API\n", stderr);
    abort();
}

/* Invoke a C function (registered from C) with a protection frame. The
 * machine-code trampoline jumps here; returns the function's result count,
 * or -1 when the function raised (the error object is in the state). */
LJRS_EXPORT int ljrs_cfunc_invoke(void *L, void *fn) {
    struct ljrs_jmp j;
    j.prev = ljrs_jmp_top;
    ljrs_jmp_top = &j;
    int r;
    if (setjmp(j.env)) {
        r = -1;
    } else {
        r = ((int (*)(void *))fn)(L);
    }
    ljrs_jmp_top = j.prev;
    return r;
}

/* ---- error raising entry points ------------------------------------------ */

/* "bad argument #%d (%s)" — LuaJIT includes the function name via debug
 * info; simplified for now. */
static void ljrs_arerror(void *L, int idx, const char *fmt, ...) {
    char msg[256];
    int n = snprintf(msg, sizeof(msg), "bad argument #%d (", idx);
    if (n < 0) n = 0;
    if (n < (int)sizeof(msg)) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg + n, sizeof(msg) - (size_t)n, fmt, ap);
        va_end(ap);
    }
    msg[sizeof(msg) - 1] = ')';
    msg[sizeof(msg) - 2] = '\0';
    ljrs_error_set(L, msg, strlen(msg));
    ljrs_raise();
}

LJRS_EXPORT int lua_error(void *L) {
    ljrs_error_take(L); /* pop the error object off the stack */
    ljrs_raise();
    return 0;
}

LJRS_EXPORT int luaL_error(void *L, const char *fmt, ...) {
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    ljrs_error_set(L, msg, strlen(msg));
    ljrs_raise();
    return 0;
}

LJRS_EXPORT int luaL_argerror(void *L, int idx, const char *fmt, ...) {
    char msg[256];
    int n = snprintf(msg, sizeof(msg), "bad argument #%d (", idx);
    if (n < 0) n = 0;
    if (n < (int)sizeof(msg)) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg + n, sizeof(msg) - (size_t)n, fmt, ap);
        va_end(ap);
    }
    msg[sizeof(msg) - 1] = ')';
    msg[sizeof(msg) - 2] = '\0';
    ljrs_error_set(L, msg, strlen(msg));
    ljrs_raise();
    return 0;
}

LJRS_EXPORT int luaL_typerror(void *L, int idx, const char *tname) {
    const char *tn = lua_typename(L, lua_type(L, idx));
    ljrs_arerror(L, idx, "%s expected, got %s", tname, tn);
    return 0;
}

/* ---- unprotected call ---------------------------------------------------- */

LJRS_EXPORT void lua_call(void *L, int nargs, int nresults) {
    if (ljrs_call_impl(L, nargs, nresults) != 0) {
        ljrs_raise();
    }
}

/* ---- argument checking --------------------------------------------------- */

LJRS_EXPORT double luaL_checknumber(void *L, int idx) {
    if (!lua_isnumber(L, idx)) {
        const char *tn = lua_typename(L, lua_type(L, idx));
        ljrs_arerror(L, idx, "number expected, got %s", tn);
    }
    return lua_tonumber(L, idx);
}

LJRS_EXPORT long long luaL_checkinteger(void *L, int idx) {
    if (!lua_isnumber(L, idx)) {
        const char *tn = lua_typename(L, lua_type(L, idx));
        ljrs_arerror(L, idx, "number expected, got %s", tn);
    }
    return lua_tointeger(L, idx);
}

LJRS_EXPORT const char *luaL_checklstring(void *L, int idx, size_t *len) {
    if (!lua_isstring(L, idx)) {
        const char *tn = lua_typename(L, lua_type(L, idx));
        ljrs_arerror(L, idx, "string expected, got %s", tn);
    }
    return lua_tolstring(L, idx, len);
}

LJRS_EXPORT const char *luaL_checkstring(void *L, int idx) {
    return luaL_checklstring(L, idx, NULL);
}

LJRS_EXPORT void *luaL_checkudata(void *L, int idx, const char *tname) {
    void *p = ljrs_checkudata(L, idx, tname);
    if (!p) {
        const char *tn = lua_typename(L, lua_type(L, idx));
        ljrs_arerror(L, idx, "%s expected, got %s", tname, tn);
    }
    return p;
}

/* ---- test C functions (exercise the longjmp machinery from real C) ------- */

LJRS_EXPORT int ljrs_test_simple(void *L) {
    lua_pushnumber(L, 6.0);
    lua_pushnumber(L, 7.0);
    return 2;
}

LJRS_EXPORT int ljrs_test_error(void *L) {
    return luaL_error(L, "boom from C %d", 42);
}

LJRS_EXPORT int ljrs_test_check(void *L) {
    double a = luaL_checknumber(L, 1);
    double b = luaL_checknumber(L, 2);
    lua_pushnumber(L, a * b);
    return 1;
}

LJRS_EXPORT int ljrs_test_ud(void *L) {
    void *p = lua_newuserdata(L, 16);
    ((int *)p)[0] = 1234;
    return 1;
}

/* ---- misc string helpers -------------------------------------------------- */

extern void lua_pushlstring(void *L, const char *s, size_t len);

/* Minimal printf-style formatter (%s %d %f %p %c %%) pushing the result. */
LJRS_EXPORT const char *lua_pushfstring(void *L, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    lua_pushlstring(L, buf, strlen(buf));
    return (const char *)lua_tolstring(L, -1, NULL);
}

/* ---- a luaopen_-style module written in real C -------------------------- */

extern int luaL_register(void *L, const char *libname, const void *reg);
extern void lua_pushstring(void *L, const char *s);

static int ljrs_mod_add(void *L) {
    double a = luaL_checknumber(L, 1);
    double b = luaL_checknumber(L, 2);
    lua_pushnumber(L, a + b);
    return 1;
}

static int ljrs_mod_greet(void *L) {
    const char *s = luaL_checkstring(L, 1);
    char buf[128];
    snprintf(buf, sizeof(buf), "hello, %s", s);
    lua_pushstring(L, buf);
    return 1;
}

typedef struct {
    const char *name;
    void *func;
} ljrs_reg;

static const ljrs_reg ljrs_testmod_reg[] = {
    {"add", (void *)ljrs_mod_add},
    {"greet", (void *)ljrs_mod_greet},
    {NULL, NULL},
};

LJRS_EXPORT int ljrs_test_luaopen(void *L) {
    return luaL_register(L, "testmod", ljrs_testmod_reg);
}
