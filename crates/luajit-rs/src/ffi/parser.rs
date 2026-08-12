//! C declaration parser for FFI `cdef`.
//!
//! Simplified Rust port of LuaJIT's `lj_cparse.c`. Handles:
//! * Basic types: `int`, `float`, `double`, `char`, `void`, etc.
//! * Typedefs: `typedef int foo_t;`
//! * Structs/unions with fields
//! * Pointers, arrays, function types (limited)

use crate::ffi::{CT, CTState, CType, CTypeID, ct_info, ctinfo, ctype_align, ctype_isfunc};

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    Eof,
    Ident,
    Integer,
    String,
    // Operators & punctuation
    Star,
    Amp,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Colon,
    Ellipsis,
    Eql,
    Question,
    Minus,
    Plus,
    LAngle,
    RAngle,
    Slash,
    // Keywords
    KwVoid,
    KwChar,
    KwShort,
    KwInt,
    KwLong,
    KwFloat,
    KwDouble,
    KwSigned,
    KwUnsigned,
    KwBool,
    KwComplex,
    KwStruct,
    KwUnion,
    KwEnum,
    KwTypedef,
    KwExtern,
    KwStatic,
    KwConst,
    KwVolatile,
    // Calling-convention / attribute keywords (parsed, ignored on x64).
    KwCdecl,
    KwStdcall,
    KwFastcall,
    KwRestrict,
    KwInline,
    KwAttribute,
    KwExtension,
}

#[derive(Clone)]
struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    pub buf: Vec<u8>,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Lexer {
            src: src.as_bytes(),
            pos: 0,
            buf: Vec::new(),
        }
    }

    fn peek(&self) -> u8 {
        if self.pos < self.src.len() {
            self.src[self.pos]
        } else {
            0
        }
    }

    fn advance(&mut self) -> u8 {
        let c = self.peek();
        if c != 0 {
            self.pos += 1;
        }
        if c == b'\\' {
            let n = self.peek();
            if n == b'\n' || n == b'\r' {
                self.pos += 1;
                if n == b'\r' && self.peek() == b'\n' {
                    self.pos += 1;
                }
                return self.advance();
            }
        }
        c
    }

    fn skip_ws(&mut self) {
        loop {
            match self.peek() {
                b' ' | b'\t' | b'\n' | b'\r' => {
                    self.advance();
                }
                b'/' => {
                    self.advance();
                    if self.peek() == b'*' {
                        self.advance();
                        loop {
                            let c = self.advance();
                            if c == 0 {
                                return;
                            }
                            if c == b'*' && self.peek() == b'/' {
                                self.advance();
                                break;
                            }
                        }
                    } else if self.peek() == b'/' {
                        self.advance();
                        while self.peek() != 0 && self.peek() != b'\n' {
                            self.advance();
                        }
                    } else {
                        return;
                    }
                }
                _ => return,
            }
        }
    }

    fn ident_tail(&mut self) -> Token {
        loop {
            let c = self.peek();
            if c.is_ascii_alphanumeric() || c == b'_' {
                let c2 = self.advance();
                self.buf.push(c2);
            } else {
                break;
            }
        }
        match std::str::from_utf8(&self.buf).unwrap() {
            "void" => Token::KwVoid,
            "char" => Token::KwChar,
            "short" => Token::KwShort,
            "int" => Token::KwInt,
            "long" => Token::KwLong,
            "float" => Token::KwFloat,
            "double" => Token::KwDouble,
            "signed" => Token::KwSigned,
            "unsigned" => Token::KwUnsigned,
            "bool" | "_Bool" => Token::KwBool,
            "_Complex" | "complex" => Token::KwComplex,
            "struct" => Token::KwStruct,
            "union" => Token::KwUnion,
            "enum" => Token::KwEnum,
            "typedef" => Token::KwTypedef,
            "extern" => Token::KwExtern,
            "static" => Token::KwStatic,
            "const" => Token::KwConst,
            "volatile" => Token::KwVolatile,
            "__cdecl" | "_cdecl" => Token::KwCdecl,
            "__stdcall" | "_stdcall" => Token::KwStdcall,
            "__fastcall" | "_fastcall" => Token::KwFastcall,
            "__restrict" | "restrict" | "__restrict__" => Token::KwRestrict,
            "inline" | "__inline" | "__inline__" => Token::KwInline,
            "__attribute__" | "__attribute" => Token::KwAttribute,
            "__extension__" => Token::KwExtension,
            _ => Token::Ident,
        }
    }

    fn number_tail(&mut self) -> Token {
        loop {
            let ch = self.peek();
            if ch.is_ascii_hexdigit()
                || matches!(ch, b'x' | b'X' | b'u' | b'U' | b'l' | b'L' | b'.')
            {
                let c2 = self.advance();
                self.buf.push(c2);
            } else {
                break;
            }
        }
        Token::Integer
    }

    fn next_token(&mut self) -> Token {
        self.skip_ws();
        let c = self.advance();
        match c {
            0 => Token::Eof,
            b'*' => Token::Star,
            b'&' => Token::Amp,
            b'(' => Token::LParen,
            b')' => Token::RParen,
            b'{' => Token::LBrace,
            b'}' => Token::RBrace,
            b'[' => Token::LBracket,
            b']' => Token::RBracket,
            b',' => Token::Comma,
            b';' => Token::Semicolon,
            b':' => Token::Colon,
            b'=' => Token::Eql,
            b'?' => Token::Question,
            b'.' => {
                if self.peek() == b'.' {
                    self.advance();
                    if self.peek() == b'.' {
                        self.advance();
                        Token::Ellipsis
                    } else {
                        Token::Ellipsis
                    }
                } else {
                    Token::Eof
                }
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                self.buf.clear();
                self.buf.push(c);
                self.ident_tail()
            }
            b'0'..=b'9' => {
                self.buf.clear();
                self.buf.push(c);
                self.number_tail()
            }
            b'"' | b'\'' => {
                self.buf.clear();
                let quote = c;
                loop {
                    let c2 = self.advance();
                    if c2 == 0 || c2 == quote {
                        break;
                    }
                    self.buf.push(c2);
                }
                Token::String
            }
            b'-' => Token::Minus,
            b'+' => Token::Plus,
            b'<' => Token::LAngle,
            b'>' => Token::RAngle,
            b'/' => Token::Slash,
            _ => Token::Eof,
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct DeclSpec {
    flags: u32,
    type_id: u32,
}

/// A declarator element, in declaration order (index 0 is the base/return
/// type; each following element wraps the one before it).
enum DeclElem {
    Base(u32),
    Ptr(u32),        // pointer (qualifier flags)
    Func(Vec<u32>),  // function (parameter ctype ids)
    Array(u32),      // array (element count; u32::MAX = `?`)
}

struct Parser<'a> {
    lex: Lexer<'a>,
    tok: Token,
    cts: &'a mut CTState,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str, cts: &'a mut crate::ffi::CTState) -> Self {
        let mut p = Parser {
            lex: Lexer::new(src),
            tok: Token::Eof,
            cts,
        };
        p.next();
        p
    }

    fn next(&mut self) {
        self.tok = self.lex.next_token();
    }

    fn expect(&mut self, t: Token) -> Result<(), String> {
        if self.tok == t {
            self.next();
            Ok(())
        } else {
            Err(format!("expected {:?}, got {:?}", t, self.tok))
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        if self.tok == Token::Ident {
            let s = String::from_utf8(self.lex.buf.clone()).unwrap();
            self.next();
            Ok(s)
        } else {
            Err(format!("expected identifier, got {:?}", self.tok))
        }
    }

    // -- Declaration specifiers --

    fn parse_decl_spec(&mut self) -> Result<DeclSpec, String> {
        let mut decl = DeclSpec {
            flags: 0,
            type_id: CTypeID::Int32 as u32,
        };
        let mut seen_type = false;
        loop {
            match self.tok {
                Token::KwConst => {
                    decl.flags |= ctinfo::CONST;
                    self.next();
                }
                Token::KwVolatile => {
                    decl.flags |= ctinfo::VOLATILE;
                    self.next();
                }
                Token::KwCdecl | Token::KwStdcall | Token::KwFastcall
                | Token::KwRestrict | Token::KwInline | Token::KwExtension => {
                    // Calling conventions are a no-op on x64; ignore them.
                    self.next();
                }
                Token::KwAttribute => {
                    // Skip GCC `__attribute__((...))`.
                    self.skip_attribute();
                }
                Token::KwUnsigned => {
                    decl.flags |= ctinfo::UNSIGNED;
                    self.next();
                }
                Token::KwSigned => {
                    self.next();
                }
                Token::KwLong => {
                    self.next();
                    if self.tok == Token::KwLong {
                        self.next();
                        decl.flags |= ctinfo::LONG; /* 'long long' same as 'long' for now */
                    } else {
                        decl.flags |= ctinfo::LONG;
                    }
                }
                Token::KwBool => {
                    decl.flags |= ctinfo::BOOL | ctinfo::UNSIGNED;
                    decl.type_id = CTypeID::Int8 as u32;
                    seen_type = true;
                    self.next();
                }
                Token::KwVoid => {
                    if seen_type {
                        break;
                    }
                    decl.type_id = CTypeID::Void as u32;
                    seen_type = true;
                    self.next();
                }
                Token::KwChar => {
                    if seen_type {
                        break;
                    }
                    decl.type_id = CTypeID::CChar as u32;
                    seen_type = true;
                    self.next();
                }
                Token::KwInt => {
                    if seen_type {
                        break;
                    }
                    decl.type_id = CTypeID::Int32 as u32;
                    seen_type = true;
                    self.next();
                }
                Token::KwFloat => {
                    if seen_type {
                        break;
                    }
                    decl.type_id = CTypeID::Float as u32;
                    decl.flags |= ctinfo::FP;
                    seen_type = true;
                    self.next();
                }
                Token::KwDouble => {
                    if seen_type {
                        break;
                    }
                    decl.type_id = CTypeID::Double as u32;
                    decl.flags |= ctinfo::FP;
                    seen_type = true;
                    self.next();
                }
                Token::KwStruct | Token::KwUnion => {
                    if seen_type {
                        break;
                    }
                    decl.type_id = self.parse_struct_or_union()?;
                    seen_type = true;
                }
                Token::KwEnum => {
                    if seen_type {
                        break;
                    }
                    decl.type_id = self.parse_enum()?;
                    seen_type = true;
                }
                Token::KwComplex => {
                    if seen_type {
                        break;
                    }
                    self.next(); // eat complex
                    // Check for "complex float"
                    if self.tok == Token::KwFloat {
                        self.next();
                        decl.type_id = CTypeID::ComplexFloat as u32;
                    } else {
                        decl.type_id = CTypeID::ComplexDouble as u32;
                    }
                    seen_type = true;
                }
                // Handle typedef'd type names (int8_t, uint32_t, etc.)
                Token::Ident => {
                    if seen_type {
                        break;
                    }
                    let name = String::from_utf8_lossy(&self.lex.buf).to_string();
                    if let Some(id) = crate::ffi::lib::quick_type_id(&name) {
                        decl.type_id = id;
                        self.next();
                        seen_type = true;
                    } else if let Some(&id) = self.cts.names.get(&name) {
                        decl.type_id = id;
                        self.next();
                        seen_type = true;
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        Ok(decl)
    }

    /// Skip a GCC `__attribute__((...))` (self.tok == KwAttribute on entry).
    fn skip_attribute(&mut self) {
        self.next(); // eat __attribute__
        if self.tok == Token::LParen {
            self.next();
            if self.tok == Token::LParen {
                self.next();
                while self.tok != Token::RParen && self.tok != Token::Eof {
                    self.next();
                }
                if self.tok == Token::RParen {
                    self.next();
                }
            }
            if self.tok == Token::RParen {
                self.next();
            }
        }
    }

    /// Peek the token after the current one (without consuming).
    fn peek_next(&mut self) -> Token {
        let mut lx = self.lex.clone();
        lx.next_token()
    }

    /// Is the current `(` the start of an inner declarator group (a
    /// function-pointer declarator), rather than a parameter list?
    fn decl_is_group(&mut self) -> bool {
        matches!(
            self.peek_next(),
            Token::Star
                | Token::LParen
                | Token::KwCdecl
                | Token::KwStdcall
                | Token::KwFastcall
        )
    }

    /// Parse a full C declarator after the declaration specifiers, returning
    /// `(name, type_id)`. Handles pointers (`*`), parenthesized groups
    /// (`(*name)`), array suffixes (`[N]` / `[?]`) and function suffixes
    /// (`(params)`), using LuaJIT's declaration-stack model.
    fn parse_declarator(&mut self, base: u32) -> Result<(Option<String>, u32), String> {
        let mut stack = vec![DeclElem::Base(base)];
        let mut pos = 0usize;
        let name = self.decl_declarator(&mut stack, &mut pos)?;
        let mut id = match stack[0] {
            DeclElem::Base(b) => b,
            _ => unreachable!(),
        };
        for e in &stack[1..] {
            id = match e {
                DeclElem::Ptr(q) => {
                    let p = crate::ffi::lib::make_ptr_type(self.cts, id);
                    if *q != 0 {
                        self.cts.tab[p as usize].info |= *q;
                    }
                    p
                }
                DeclElem::Func(params) => {
                    crate::ffi::lib::make_func_type(self.cts, id, params.clone())
                }
                DeclElem::Array(n) => crate::ffi::lib::make_array_type(self.cts, id, *n),
                DeclElem::Base(_) => unreachable!(),
            };
        }
        Ok((name, id))
    }

    fn decl_declarator(
        &mut self,
        stack: &mut Vec<DeclElem>,
        pos: &mut usize,
    ) -> Result<Option<String>, String> {
        let mut name = None;
        // Head of declarator: pointer operators (calling conventions and
        // qualifiers between them are ignored).
        while self.tok == Token::Star
            || self.tok == Token::KwCdecl
            || self.tok == Token::KwStdcall
            || self.tok == Token::KwFastcall
        {
            let mut star = false;
            let mut q = 0u32;
            while self.tok == Token::Star
                || self.tok == Token::KwCdecl
                || self.tok == Token::KwStdcall
                || self.tok == Token::KwFastcall
                || self.tok == Token::KwConst
                || self.tok == Token::KwVolatile
            {
                match self.tok {
                    Token::Star => star = true,
                    Token::KwConst => q |= ctinfo::CONST,
                    Token::KwVolatile => q |= ctinfo::VOLATILE,
                    _ => {}
                }
                self.next();
            }
            if star {
                stack.insert(*pos + 1, DeclElem::Ptr(q));
                *pos += 1;
            } else {
                break;
            }
        }

        // Inner declarator group or direct name.
        if self.tok == Token::LParen && self.decl_is_group() {
            self.next(); // consume '('
            let saved = *pos;
            name = self.decl_declarator(stack, pos)?;
            self.expect(Token::RParen)?;
            *pos = saved;
        } else if self.tok == Token::Ident {
            name = Some(self.ident()?);
        }

        // Tail of declarator: array and function suffixes.
        loop {
            if self.tok == Token::LBracket {
                self.next();
                let mut n = u32::MAX;
                if self.tok == Token::Integer
                    && let Ok(v) = String::from_utf8_lossy(&self.lex.buf).parse::<u32>()
                {
                    n = v.max(1);
                    self.next();
                }
                self.expect(Token::RBracket)?;
                stack.insert(*pos + 1, DeclElem::Array(n));
            } else if self.tok == Token::LParen {
                self.next(); // consume '('
                let params = self.parse_func_params()?;
                stack.insert(*pos + 1, DeclElem::Func(params));
            } else {
                break;
            }
        }
        Ok(name)
    }

    /// Parse a function parameter list `(params)`, terminated by `)`.
    /// Each parameter is a full abstract/named declarator.
    fn parse_func_params(&mut self) -> Result<Vec<u32>, String> {
        let mut params = Vec::new();
        while self.tok != Token::RParen && self.tok != Token::Eof {
            // `(void)` — an empty parameter list.
            if self.tok == Token::KwVoid {
                let save_pos = self.lex.pos;
                let save_buf = self.lex.buf.clone();
                self.next();
                if self.tok == Token::RParen {
                    self.next();
                    return Ok(params);
                }
                self.lex.pos = save_pos;
                self.lex.buf = save_buf;
                self.tok = Token::KwVoid;
            }
            let pdecl = self.parse_decl_spec()?;
            let (_name, ptype) = self.parse_declarator(pdecl.type_id)?;
            params.push(ptype);
            if self.tok == Token::Comma {
                self.next();
            }
        }
        self.expect(Token::RParen)?;
        Ok(params)
    }

    // -- Struct / Union --

    fn parse_struct_or_union(&mut self) -> Result<u32, String> {
        let is_union = self.tok == Token::KwUnion;
        self.next(); // eat struct/union
        let tag: Option<String> = if self.tok == Token::Ident {
            let t = String::from_utf8_lossy(&self.lex.buf).to_string();
            self.next();
            Some(t)
        } else {
            None
        };

        if self.tok != Token::LBrace {
            // Forward declaration: reuse an existing tag, else create.
            if let Some(tag) = &tag
                && let Some(&id) = self.cts.tags.get(tag)
            {
                return Ok(id);
            }
            let id = self.cts.top;
            let sinfo = ct_info(CT::Struct, if is_union { ctinfo::UNION } else { 0 });
            self.cts.tab.push(CType {
                info: sinfo,
                size: 0,
                sib: 0,
                next: 0,
                name: 0,
            });
            self.cts.top = self
                .cts
                .top
                .checked_add(1)
                .ok_or_else(|| "too many C types (overflow)".to_string())?;
            if let Some(tag) = &tag {
                self.cts.tags.insert(tag.clone(), id);
            }
            return Ok(id);
        }
        self.next(); // {

        let _first_field_id = self.cts.top;
        let mut total_size: u64 = 0;
        let mut max_align: u32 = 1;
        let mut field_infos: Vec<(String, u32, u32)> = Vec::new(); // (name, type_id, offset)
        let mut field_ids: Vec<u32> = Vec::new();
        let mut prev_fdecl_type: Option<u32> = None;
        let mut guard: usize = 0;

        while self.tok != Token::RBrace && self.tok != Token::Eof {
            guard += 1;
            if guard > 10000 {
                return Err(format!("infinite loop in struct body, tok={:?}", self.tok));
            }
            // `type a, b, c;` — a comma-separated field name continues
            // with the previous declaration specifier.
            let mut fdecl = if self.tok == Token::Ident {
                if let Some(prev_type) = prev_fdecl_type {
                    DeclSpec {
                        flags: 0,
                        type_id: prev_type,
                    }
                } else {
                    self.parse_decl_spec()?
                }
            } else {
                self.parse_decl_spec()?
            };

            // Pointer declarators (`type *name`).
            while self.tok == Token::Star {
                fdecl.type_id = crate::ffi::lib::make_ptr_type(self.cts, fdecl.type_id);
                self.next();
            }

            // Skip pointer/declarator tokens (*, **, etc.) before the field name.
            while self.tok == Token::Star || self.tok == Token::Slash {
                self.next();
            }

            // Read field name(s) — comes before array brackets in C.
            let field_name = if self.tok == Token::Ident {
                let name = String::from_utf8_lossy(&self.lex.buf).to_string();
                self.next();
                name
            } else {
                String::new()
            };

            // Parse array declarator brackets (e.g. f[2][3]). Even a
            // one-element `t[1]` is an array type.
            let mut array_multiplier: u32 = 1;
            let mut is_array = false;
            while self.tok == Token::LBracket {
                is_array = true;
                self.next(); // eat [
                let mut array_num: u32 = 1;
                if self.tok == Token::Question {
                    // Variable-length array: size u32::MAX, resolved at
                    // allocation time.
                    array_num = u32::MAX;
                    self.next();
                } else if self.tok == Token::Integer {
                    if let Ok(n) = String::from_utf8_lossy(&self.lex.buf).parse::<u32>() {
                        array_num = n.max(1);
                    }
                    self.next();
                }
                while self.tok != Token::RBracket && self.tok != Token::Eof {
                    self.next();
                }
                if self.tok == Token::RBracket {
                    self.next(); // eat ]
                }
                array_multiplier = array_multiplier.saturating_mul(array_num);
            }

            let field_type_id = if array_multiplier > 1 || is_array {
                let elem_ct = self.cts.get(fdecl.type_id);
                let total_sz = elem_ct.size.saturating_mul(array_multiplier);
                let info = ct_info(CT::Array, 0) | fdecl.type_id;
                // Search for existing array type.
                let existing = (0..self.cts.top as usize)
                    .find(|&i| self.cts.tab[i].info == info && self.cts.tab[i].size == total_sz);
                if let Some(id) = existing {
                    id as u32
                } else {
                    let id = self.cts.top;
                    self.cts.tab.push(CType {
                        info,
                        size: total_sz,
                        sib: 0,
                        next: 0,
                        name: 0,
                    });
                    self.cts.top = id + 1;
                    id
                }
            } else {
                fdecl.type_id
            };
            // Comma-separated fields (`type a, b;`) share the *computed*
            // field type (an array declarator changes it).
            prev_fdecl_type = Some(field_type_id);

            // Bitfield
            if self.tok == Token::Colon {
                self.next(); // eat :
                while self.tok != Token::Comma
                    && self.tok != Token::Semicolon
                    && self.tok != Token::RBrace
                    && self.tok != Token::Eof
                {
                    self.next();
                }
            }
            if self.tok == Token::Comma {
                self.next();
            }
            if self.tok == Token::Semicolon {
                self.next();
                prev_fdecl_type = None; // New declaration specifier starts.
            }

            // Extract field info before any mutable ops on cts
            let field_size = {
                let ct = self.cts.get(field_type_id);
                (ct.size as u64, 1u32 << ctype_align(ct.info))
            };
            max_align = max_align.max(field_size.1);
            let align = field_size.1 as u64;
            let field_offset = if is_union {
                0u64
            } else {
                (total_size + align - 1) & !(align - 1)
            };
            total_size = if is_union {
                total_size.max(field_offset + field_size.0)
            } else {
                (field_offset + align - 1) & !(align - 1)
            };

            let finfo = ct_info(CT::Field, 0) | field_type_id;
            let field_id = self.cts.top;
            self.cts.tab.push(CType {
                info: finfo,
                size: field_offset as u32,
                sib: 0,
                next: 0,
                name: 0,
            });
            field_ids.push(field_id);
            if !field_name.is_empty() {
                field_infos.push((field_name, field_type_id, field_offset as u32));
            }
            self.cts.top = self
                .cts
                .top
                .checked_add(1)
                .ok_or_else(|| "too many C types (overflow)".to_string())?;
            total_size = if is_union {
                total_size.max(field_offset + field_size.0)
            } else {
                (field_offset + field_size.0 + align - 1) & !(align - 1)
            };
        }
        self.expect(Token::RBrace)?;

        total_size = (total_size + max_align as u64 - 1) & !(max_align as u64 - 1);

        // Link field siblings (the actual field ids — pointer types may
        // have been created between fields).
        for (i, &fid) in field_ids.iter().enumerate() {
            self.cts.tab[fid as usize].sib = field_ids.get(i + 1).copied().unwrap_or(0) as u16;
        }

        // The struct type itself (insert at end, after fields). A
        // previously forward-declared tag gets its entry completed.
        let first_field = field_ids.first().copied().unwrap_or(0);
        let sinfo = ct_info(CT::Struct, if is_union { ctinfo::UNION } else { 0 })
            | first_field
            | (max_align.trailing_zeros() << ctinfo::SHIFT_ALIGN);
        let struct_id = if let Some(tag) = &tag
            && let Some(&fwd_id) = self.cts.tags.get(tag)
        {
            self.cts.tab[fwd_id as usize] = CType {
                info: sinfo,
                size: total_size as u32,
                sib: 0,
                next: 0,
                name: 0,
            };
            fwd_id
        } else {
            self.cts.tab.push(CType {
                info: sinfo,
                size: total_size as u32,
                sib: 0,
                next: 0,
                name: 0,
            });
            self.cts.top = self
                .cts
                .top
                .checked_add(1)
                .ok_or_else(|| "too many C types (overflow)".to_string())?;
            let id = self.cts.top - 1;
            if let Some(tag) = &tag {
                self.cts.tags.insert(tag.clone(), id);
            }
            id
        };
        // Register field names
        for (name, type_id, offset) in field_infos {
            self.cts
                .field_names
                .insert((struct_id, name), (type_id, offset));
        }
        Ok(struct_id)
    }

    // -- Enum --

    fn parse_enum(&mut self) -> Result<u32, String> {
        self.next(); // eat enum
        if self.tok == Token::Ident {
            self.next();
        } // optional tag

        let mut next_val: i32 = 0;
        if self.tok == Token::LBrace {
            self.next();
            while self.tok != Token::RBrace && self.tok != Token::Eof {
                if self.tok == Token::Ident {
                    let name = String::from_utf8_lossy(&self.lex.buf).to_string();
                    self.next();
                    let v = if self.tok == Token::Eql {
                        self.next();
                        self.parse_enum_value()
                    } else {
                        next_val
                    };
                    self.cts.constants.insert(name, v);
                    next_val = v.wrapping_add(1);
                }
                if self.tok == Token::Comma {
                    self.next();
                }
            }
            self.expect(Token::RBrace)?;
        }
        // Enum is always int32
        Ok(CTypeID::Int32 as u32)
    }

    /// Parse a simple enum constant expression (integer/hex literal, an
    /// optional unary minus, or a reference to an earlier constant). Any
    /// remaining operators are skipped.
    fn parse_enum_value(&mut self) -> i32 {
        let mut neg = false;
        if self.tok == Token::Minus {
            neg = true;
            self.next();
        }
        let v = match self.tok {
            Token::Integer => {
                let s = String::from_utf8_lossy(&self.lex.buf).to_string();
                let v = if s.len() > 2 && (&s[..2] == "0x" || &s[..2] == "0X") {
                    i32::from_str_radix(&s[2..], 16).unwrap_or(0)
                } else {
                    s.parse::<i32>().unwrap_or(0)
                };
                self.next();
                v
            }
            Token::Ident => {
                let name = String::from_utf8_lossy(&self.lex.buf).to_string();
                self.next();
                self.cts.constants.get(&name).copied().unwrap_or(0)
            }
            _ => 0,
        };
        while !matches!(self.tok, Token::Comma | Token::RBrace | Token::Eof) {
            self.next();
        }
        if neg {
            -v
        } else {
            v
        }
    }

    // -- Typedef --

    fn parse_typedef(&mut self) -> Result<(), String> {
        self.next(); // eat typedef
        let decl = self.parse_decl_spec()?;
        let (name, type_id) = self.parse_declarator(decl.type_id)?;
        let name = name.ok_or_else(|| "typedef requires a name".to_string())?;
        let info = ct_info(CT::Typedef, 0) | type_id;
        let sz = self.cts.get(type_id).size;
        let id = self.cts.top;
        self.cts.tab.push(CType {
            info,
            size: sz,
            sib: 0,
            next: 0,
            name: 0,
        });
        self.cts.top = self
            .cts
            .top
            .checked_add(1)
            .ok_or_else(|| "too many C types (overflow)".to_string())?;
        self.cts.names.insert(name, id);
        // Skip declarator suffix / initializer.
        self.skip_until_semicolon();
        Ok(())
    }

    fn skip_until_semicolon(&mut self) {
        let mut depth = 0u32;
        let mut guard: usize = 0;
        loop {
            guard += 1;
            if guard > 10000 {
                self.tok = Token::Eof;
                return;
            }
            match self.tok {
                Token::Semicolon | Token::Eof => {
                    if depth == 0 {
                        if self.tok == Token::Semicolon {
                            self.next();
                        }
                        return;
                    }
                }
                Token::LParen | Token::LBrace | Token::LBracket => {
                    depth += 1;
                    self.next();
                }
                Token::RParen | Token::RBrace | Token::RBracket => {
                    depth = depth.saturating_sub(1);
                    self.next();
                }
                _ => {
                    self.next();
                }
            }
        }
    }

    // -- Top-level dispatch --

    fn parse_declaration(&mut self) -> Result<(), String> {
        match self.tok {
            Token::KwTypedef => self.parse_typedef(),
            Token::KwStruct | Token::KwUnion => {
                self.parse_struct_or_union()?;
                self.skip_until_semicolon();
                Ok(())
            }
            Token::KwEnum => {
                self.parse_enum()?;
                self.skip_until_semicolon();
                Ok(())
            }
            Token::Eof => Ok(()),
            _ => {
                let decl = self.parse_decl_spec()?;
                let (name, type_id) = self.parse_declarator(decl.type_id)?;
                // A function declarator (`name(params)` or `(*name)(params)`)
                // registers the prototype so ffi.C lookups can validate call
                // arguments against the declared parameter types.
                if let Some(name) = name {
                    let raw = self.cts.get(type_id);
                    if ctype_isfunc(raw.info) {
                        self.register_func_decl(&name, type_id)?;
                        return Ok(());
                    }
                }
                self.skip_until_semicolon();
                Ok(())
            }
        }
    }

    /// Register a function prototype `name` with its `CT::Func` ctype id,
    /// parsing an optional `asm("symbol")` redirect.
    fn register_func_decl(&mut self, name: &str, func_id: u32) -> Result<(), String> {
        let mut asm_name: Option<String> = None;
        while self.tok != Token::Semicolon && self.tok != Token::Eof {
            if self.tok == Token::Ident
                && String::from_utf8_lossy(&self.lex.buf).eq_ignore_ascii_case("asm")
            {
                self.next();
                if self.tok == Token::LParen {
                    self.next();
                    if self.tok == Token::String {
                        asm_name = Some(String::from_utf8_lossy(&self.lex.buf).to_string());
                    }
                }
            }
            self.next();
        }
        if self.tok == Token::Semicolon {
            self.next();
        }
        self.cts.names.insert(name.to_string(), func_id);
        self.cts.symbols.insert(
            name.to_string(),
            asm_name.unwrap_or_else(|| name.to_string()),
        );
        Ok(())
    }
}

/// Parse C declarations and register types in `CTState`.
pub fn parse(cts: &mut CTState, src: &str) -> Result<(), String> {
    let mut p = Parser::new(src, cts);
    let mut guard: usize = 0;
    while p.tok != Token::Eof {
        guard += 1;
        if guard > 10000 {
            return Err(format!("infinite loop in parse, tok={:?}", p.tok));
        }
        p.parse_declaration()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_types() {
        let mut cts = CTState::new();
        let base = cts.top;
        parse(&mut cts, "typedef int foo_t;").unwrap();
        assert!(cts.top > base, "should have added a type");
    }

    #[test]
    fn parse_struct() {
        let mut cts = CTState::new();
        parse(&mut cts, "struct point { int x; int y; };").unwrap();
        // Should have created: struct + 2 fields = 3 new entries
        assert!(cts.top >= 28, "should have struct+fields");
    }

    #[test]
    fn parse_unsigned_long_long() {
        let mut cts = CTState::new();
        parse(&mut cts, "typedef unsigned long long ull_t;").unwrap();
        assert!(cts.top > 25, "should have added ull_t typedef");
    }
}
