//! Introspective sort for tables with custom comparators.
//!
//! When no comparator is supplied, the stdlib delegates to Rust's
//! `slice::sort_unstable_by`, which is an ipnsort-derived algorithm
//! (O(*n* log *n*), no allocation).  When a Lua comparator function is
//! passed, we use an introspective (introsort) implementation: quicksort
//! with median-of-three pivot that falls back to heapsort after
//! `2*floor(log2(n))` recursion levels.

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::value::LuaValue;

/// Sort `items` in-place using the Lua comparator function `comp`.
/// `comp` is a `GcPtr<GcFunc>` already validated as a function.
pub fn introsort(
    l: &mut LuaState,
    items: &mut [(i32, LuaValue)],
    comp: crate::gc::GcPtr<crate::func::GcFunc>,
) -> LuaResult<()> {
    let n = items.len();
    if n <= 1 {
        return Ok(());
    }
    let max_depth = (n.next_power_of_two().trailing_zeros() * 2) as usize;
    introsort_impl(l, items, comp, 0, n - 1, max_depth)
}

fn introsort_impl(
    l: &mut LuaState,
    items: &mut [(i32, LuaValue)],
    comp: crate::gc::GcPtr<crate::func::GcFunc>,
    lo: usize,
    hi: usize,
    depth: usize,
) -> LuaResult<()> {
    if hi - lo < 16 {
        insertion_sort(l, items, comp, lo, hi)?;
        return Ok(());
    }
    if depth == 0 {
        heapsort(l, items, comp, lo, hi)?;
        return Ok(());
    }
    let p = partition(l, items, comp, lo, hi)?;
    if p > lo {
        introsort_impl(l, items, comp, lo, p - 1, depth - 1)?;
    }
    if p < hi {
        introsort_impl(l, items, comp, p + 1, hi, depth - 1)?;
    }
    Ok(())
}

fn partition(
    l: &mut LuaState,
    items: &mut [(i32, LuaValue)],
    comp: crate::gc::GcPtr<crate::func::GcFunc>,
    lo: usize,
    hi: usize,
) -> LuaResult<usize> {
    let mid = lo + (hi - lo) / 2;
    if compare_lua(l, comp, items[mid].1, items[lo].1)? {
        items.swap(lo, mid);
    }
    if compare_lua(l, comp, items[hi].1, items[lo].1)? {
        items.swap(lo, hi);
    }
    if compare_lua(l, comp, items[hi].1, items[mid].1)? {
        items.swap(mid, hi);
    }
    let pivot = items[mid].1;
    items.swap(mid, hi - 1);
    let mut i = lo;
    let mut j = hi - 1;
    loop {
        i += 1;
        while i < j && compare_lua(l, comp, items[i].1, pivot)? {
            i += 1;
        }
        j -= 1;
        while j > i && compare_lua(l, comp, pivot, items[j].1)? {
            j -= 1;
        }
        if i >= j {
            break;
        }
        items.swap(i, j);
    }
    items.swap(i, hi - 1);
    Ok(i)
}

fn insertion_sort(
    l: &mut LuaState,
    items: &mut [(i32, LuaValue)],
    comp: crate::gc::GcPtr<crate::func::GcFunc>,
    lo: usize,
    hi: usize,
) -> LuaResult<()> {
    for i in lo + 1..=hi {
        let mut j = i;
        while j > lo {
            if compare_lua(l, comp, items[j].1, items[j - 1].1)? {
                items.swap(j, j - 1);
                j -= 1;
            } else {
                break;
            }
        }
    }
    Ok(())
}

fn heapsort(
    l: &mut LuaState,
    items: &mut [(i32, LuaValue)],
    comp: crate::gc::GcPtr<crate::func::GcFunc>,
    lo: usize,
    hi: usize,
) -> LuaResult<()> {
    let n = hi - lo + 1;
    for i in (0..n / 2).rev() {
        sift_down(l, items, comp, lo, n, i)?;
    }
    for i in (1..n).rev() {
        items.swap(lo, lo + i);
        sift_down(l, items, comp, lo, i, 0)?;
    }
    Ok(())
}

fn sift_down(
    l: &mut LuaState,
    items: &mut [(i32, LuaValue)],
    comp: crate::gc::GcPtr<crate::func::GcFunc>,
    lo: usize,
    n: usize,
    root: usize,
) -> LuaResult<()> {
    let mut r = root;
    loop {
        let left = 2 * r + 1;
        if left >= n {
            break;
        }
        let right = left + 1;
        let mut largest = r;
        if compare_lua(l, comp, items[lo + largest].1, items[lo + left].1)? {
            largest = left;
        }
        if right < n && compare_lua(l, comp, items[lo + largest].1, items[lo + right].1)? {
            largest = right;
        }
        if largest == r {
            break;
        }
        items.swap(lo + r, lo + largest);
        r = largest;
    }
    Ok(())
}

/// Call the Lua comparator `comp(a, b)` and return `true` iff `a < b`.
fn compare_lua(
    l: &mut LuaState,
    comp: crate::gc::GcPtr<crate::func::GcFunc>,
    a: LuaValue,
    b: LuaValue,
) -> LuaResult<bool> {
    let func_slot = l.top;
    l.stack[func_slot] = LuaValue::func(comp);
    l.stack[func_slot + 1] = LuaValue::NIL;
    l.stack[func_slot + 2] = a;
    l.stack[func_slot + 3] = b;
    l.top = func_slot + 4;
    let saved_base = l.base;
    l.base = func_slot;
    let nret = crate::vm::execute(l, l.base, 2, 1)?;
    let r = if nret > 0 { l.stack[func_slot].is_truthy() } else { false };
    l.top = l.base + nret;
    l.base = saved_base;
    Ok(r)
}
