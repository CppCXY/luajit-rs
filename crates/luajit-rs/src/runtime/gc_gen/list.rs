//! GcList — O(1) add/remove collection of GC object header pointers.

use super::header::GcHeader;
use std::ptr::NonNull;

pub struct GcList {
    objects: Vec<NonNull<GcHeader>>,
}

impl Default for GcList {
    fn default() -> Self {
        Self::new()
    }
}

impl GcList {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn add(&mut self, header: NonNull<GcHeader>) {
        let idx = self.objects.len() as u32;
        unsafe { header.as_ref() }.index.set(idx);
        self.objects.push(header);
    }

    pub fn remove(&mut self, header: NonNull<GcHeader>) {
        let idx = unsafe { header.as_ref() }.index.get() as usize;
        let last = self.objects.len() - 1;
        if idx < last {
            self.objects.swap(idx, last);
            unsafe { self.objects[idx].as_ref() }.index.set(idx as u32);
        }
        self.objects.pop();
    }

    pub fn take_all(&mut self) -> Vec<NonNull<GcHeader>> {
        std::mem::take(&mut self.objects)
    }

    pub fn add_all(&mut self, objects: Vec<NonNull<GcHeader>>) {
        for h in objects {
            self.add(h);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = NonNull<GcHeader>> + '_ {
        self.objects.iter().copied()
    }
}
