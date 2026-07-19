//! C declaration parser for FFI `cdef`.
//!
//! Simplified Rust port of LuaJIT's `lj_cparse.c`. Handles:
//! * Basic types: `int`, `float`, `double`, `char`, `void`, etc.
//! * Typedefs: `typedef int foo_t;`
//! * Structs/unions with fields
//! * Pointers, arrays, function types (limited)

use crate::ffi::{CType, CT, ctinfo, ct_info, ctype_align};

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    Eof, Ident, Integer,
    // Operators & punctuation
    Star, Amp, LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Comma, Semicolon, Colon, Ellipsis, Eql,
    // Keywords
    KwVoid, KwChar, KwShort, KwInt, KwLong, KwFloat, KwDouble,
    KwSigned, KwUnsigned, KwBool, KwComplex,
    KwStruct, KwUnion, KwEnum,
    KwTypedef, KwExtern, KwStatic, KwConst, KwVolatile,
}

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    pub buf: Vec<u8>,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Lexer { src: src.as_bytes(), pos: 0, buf: Vec::new() }
    }

    fn peek(&self) -> u8 {
        if self.pos < self.src.len() { self.src[self.pos] } else { 0 }
    }

    fn advance(&mut self) -> u8 {
        let c = self.peek();
        if c != 0 { self.pos += 1; }
        if c == b'\\' {
            let n = self.peek();
            if n == b'\n' || n == b'\r' {
                self.pos += 1;
                if n == b'\r' && self.peek() == b'\n' { self.pos += 1; }
                return self.advance();
            }
        }
        c
    }

    fn skip_ws(&mut self) {
        loop {
            match self.peek() {
                b' ' | b'\t' | b'\n' | b'\r' => { self.advance(); }
                b'/' => {
                    self.advance();
                    if self.peek() == b'*' {
                        self.advance();
                        loop {
                            let c = self.advance();
                            if c == 0 { return; }
                            if c == b'*' && self.peek() == b'/' { self.advance(); break; }
                        }
                    } else if self.peek() == b'/' {
                        self.advance();
                        while self.peek() != 0 && self.peek() != b'\n' { self.advance(); }
                    } else { return; }
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
            } else { break; }
        }
        match std::str::from_utf8(&self.buf).unwrap() {
            "void" => Token::KwVoid, "char" => Token::KwChar,
            "short" => Token::KwShort, "int" => Token::KwInt,
            "long" => Token::KwLong, "float" => Token::KwFloat,
            "double" => Token::KwDouble, "signed" => Token::KwSigned,
            "unsigned" => Token::KwUnsigned, "bool" | "_Bool" => Token::KwBool,
            "_Complex" | "complex" => Token::KwComplex,
            "struct" => Token::KwStruct, "union" => Token::KwUnion,
            "enum" => Token::KwEnum, "typedef" => Token::KwTypedef,
            "extern" => Token::KwExtern, "static" => Token::KwStatic,
            "const" => Token::KwConst, "volatile" => Token::KwVolatile,
            _ => Token::Ident,
        }
    }

    fn number_tail(&mut self) -> Token {
        loop {
            let ch = self.peek();
            if ch.is_ascii_hexdigit() || matches!(ch, b'x'|b'X'|b'u'|b'U'|b'l'|b'L'|b'.') {
                let c2 = self.advance();
                self.buf.push(c2);
            } else { break; }
        }
        Token::Integer
    }

    fn next_token(&mut self) -> Token {
        self.skip_ws();
        let c = self.advance();
        match c {
            0 => Token::Eof,
            b'*' => Token::Star, b'&' => Token::Amp,
            b'(' => Token::LParen, b')' => Token::RParen,
            b'{' => Token::LBrace, b'}' => Token::RBrace,
            b'[' => Token::LBracket, b']' => Token::RBracket,
            b',' => Token::Comma, b';' => Token::Semicolon,
            b':' => Token::Colon,
            b'=' => Token::Eql,
            b'.' => {
                if self.peek() == b'.' { self.advance(); if self.peek()==b'.' { self.advance(); Token::Ellipsis } else { Token::Ellipsis } }
                else { Token::Eof }
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                self.buf.clear();
                self.buf.push(c);
                self.ident_tail()
            }
            b'0'..=b'9' => { self.buf.clear(); self.buf.push(c); self.number_tail()}
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

struct Parser<'a> {
    lex: Lexer<'a>,
    tok: Token,
    cts: &'a mut crate::ffi::CTState,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str, cts: &'a mut crate::ffi::CTState) -> Self {
        let mut p = Parser { lex: Lexer::new(src), tok: Token::Eof, cts };
        p.next();
        p
    }

    fn next(&mut self) { self.tok = self.lex.next_token(); }

    fn expect(&mut self, t: Token) -> Result<(), String> {
        if self.tok == t { self.next(); Ok(()) }
        else { Err(format!("expected {:?}, got {:?}", t, self.tok)) }
    }

    fn ident(&mut self) -> Result<String, String> {
        if self.tok == Token::Ident {
            let s = String::from_utf8(self.lex.buf.clone()).unwrap();
            self.next(); Ok(s)
        } else { Err(format!("expected identifier, got {:?}", self.tok)) }
    }

    // -- Declaration specifiers --

    fn parse_decl_spec(&mut self) -> Result<DeclSpec, String> {
        let mut decl = DeclSpec { flags: 0, type_id: crate::ffi::CTypeID::Int32 as u32 };
        let mut seen_type = false;
        loop {
            match self.tok {
                Token::KwConst  => { decl.flags |= ctinfo::CONST; self.next(); }
                Token::KwVolatile => { decl.flags |= ctinfo::VOLATILE; self.next(); }
                Token::KwUnsigned => { decl.flags |= ctinfo::UNSIGNED; self.next(); }
                Token::KwSigned => { self.next(); }
                Token::KwLong => {
                    self.next();
                    if self.tok == Token::KwLong { self.next(); decl.flags |= ctinfo::LONG; /* 'long long' same as 'long' for now */ }
                    else { decl.flags |= ctinfo::LONG; }
                }
                Token::KwBool => {
                    decl.flags |= ctinfo::BOOL | ctinfo::UNSIGNED;
                    decl.type_id = crate::ffi::CTypeID::Int8 as u32;
                    seen_type = true; self.next();
                }
                Token::KwVoid => {
                    if seen_type { break; }
                    decl.type_id = crate::ffi::CTypeID::Void as u32;
                    seen_type = true; self.next();
                }
                Token::KwChar => {
                    if seen_type { break; }
                    decl.type_id = crate::ffi::CTypeID::CChar as u32;
                    seen_type = true; self.next();
                }
                Token::KwInt => {
                    if seen_type { break; }
                    decl.type_id = crate::ffi::CTypeID::Int32 as u32;
                    seen_type = true; self.next();
                }
                Token::KwFloat => {
                    if seen_type { break; }
                    decl.type_id = crate::ffi::CTypeID::Float as u32;
                    decl.flags |= ctinfo::FP; seen_type = true; self.next();
                }
                Token::KwDouble => {
                    if seen_type { break; }
                    decl.type_id = crate::ffi::CTypeID::Double as u32;
                    decl.flags |= ctinfo::FP; seen_type = true; self.next();
                }
                Token::KwStruct | Token::KwUnion => {
                    if seen_type { break; }
                    decl.type_id = self.parse_struct_or_union()?;
                    seen_type = true;
                }
                Token::KwEnum => {
                    if seen_type { break; }
                    decl.type_id = self.parse_enum()?;
                    seen_type = true;
                }
                _ => break,
            }
        }
        Ok(decl)
    }

    // -- Struct / Union --

    fn parse_struct_or_union(&mut self) -> Result<u32, String> {
        let is_union = self.tok == Token::KwUnion;
        self.next(); // eat struct/union
        if self.tok == Token::Ident { self.next(); } // optional tag

        if self.tok != Token::LBrace {
            // Forward declaration
            let id = self.cts.top;
            let sinfo = ct_info(CT::Struct, if is_union { ctinfo::UNION } else { 0 });
            self.cts.tab.push(CType { info: sinfo, size: 0, sib: 0, next: 0, name: 0 });
            self.cts.top += 1;
            return Ok(id);
        }
        self.next(); // {

        let first_field_id = self.cts.top;
        let mut total_size: u32 = 0;
        let mut max_align: u32 = 1;

        while self.tok != Token::RBrace && self.tok != Token::Eof {
            let fdecl = self.parse_decl_spec()?;

            // Read field name(s), bitfields
            loop {
                if self.tok == Token::Ident { self.next(); }
                if self.tok == Token::Colon {
                    self.next(); // eat :
                    while self.tok != Token::Comma && self.tok != Token::Semicolon
                        && self.tok != Token::RBrace && self.tok != Token::Eof {
                        self.next();
                    }
                }
                if self.tok == Token::Comma { self.next(); continue; }
                break;
            }
            if self.tok == Token::Semicolon { self.next(); }

            // Extract field info before any mutable ops on cts
            let field_size = {
                let ct = self.cts.get(fdecl.type_id);
                (ct.size, 1u32 << ctype_align(ct.info))
            };
            max_align = max_align.max(field_size.1);
            total_size = (total_size + field_size.1 - 1) & !(field_size.1 - 1);

            let finfo = ct_info(CT::Field, 0) | fdecl.type_id;
            self.cts.tab.push(CType {
                info: finfo, size: total_size, sib: 0, next: 0, name: 0,
            });
            self.cts.top += 1;
            total_size += field_size.0;
        }
        self.expect(Token::RBrace)?;

        total_size = (total_size + max_align - 1) & !(max_align - 1);

        // Link field siblings
        let num_fields = self.cts.top - first_field_id;
        for i in 0..num_fields {
            let idx = (first_field_id + i) as usize;
            let sib = if i + 1 < num_fields { (first_field_id + i + 1) as u16 } else { 0 };
            self.cts.tab[idx].sib = sib;
        }

        // The struct type itself (insert at end, after fields)
        let sinfo = ct_info(CT::Struct, if is_union { ctinfo::UNION } else { 0 })
            | first_field_id
            | (max_align.trailing_zeros() << ctinfo::SHIFT_ALIGN);
        self.cts.tab.push(CType { info: sinfo, size: total_size, sib: 0, next: 0, name: 0 });
        self.cts.top += 1;
        Ok(self.cts.top - 1)
    }

    // -- Enum --

    fn parse_enum(&mut self) -> Result<u32, String> {
        self.next(); // eat enum
        if self.tok == Token::Ident { self.next(); } // optional tag

        if self.tok == Token::LBrace {
            self.next();
            while self.tok != Token::RBrace && self.tok != Token::Eof {
                if self.tok == Token::Ident {
                    self.next();
                    if self.tok == Token::Eql {
                        self.next();
                        while self.tok != Token::Comma && self.tok != Token::RBrace && self.tok != Token::Eof {
                            self.next();
                        }
                    }
                }
                if self.tok == Token::Comma { self.next(); }
            }
            self.expect(Token::RBrace)?;
        }
        // Enum is always int32
        Ok(crate::ffi::CTypeID::Int32 as u32)
    }

    // -- Typedef --

    fn parse_typedef(&mut self) -> Result<(), String> {
        self.next(); // eat typedef
        let decl = self.parse_decl_spec()?;
        let name = self.ident()?;
        let info = ct_info(CT::Typedef, 0) | decl.type_id;
        let sz = self.cts.get(decl.type_id).size;
        let id = self.cts.top;
        self.cts.tab.push(CType { info, size: sz, sib: 0, next: 0, name: 0 });
        self.cts.top += 1;
        self.cts.names.insert(name, id);
        // Skip declarator suffix
        self.skip_until_semicolon();
        Ok(())
    }

    fn skip_until_semicolon(&mut self) {
        let mut depth = 0u32;
        loop {
            match self.tok {
                Token::Semicolon | Token::Eof => { if depth == 0 { if self.tok == Token::Semicolon { self.next(); } return; } }
                Token::LParen | Token::LBrace | Token::LBracket => { depth += 1; self.next(); }
                Token::RParen | Token::RBrace | Token::RBracket => { depth = depth.saturating_sub(1); self.next(); }
                _ => { self.next(); }
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
                let _decl = self.parse_decl_spec()?;
                self.skip_until_semicolon();
                Ok(())
            }
        }
    }
}

/// Parse C declarations and register types in `CTState`.
pub fn parse(cts: &mut crate::ffi::CTState, src: &str) -> Result<(), String> {
    let mut p = Parser::new(src, cts);
    while p.tok != Token::Eof {
        p.parse_declaration()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::CTState;

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
