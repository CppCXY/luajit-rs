use std::panic::panic_any;

pub use crate::string::{Interner, StrId};

#[derive(Debug)]
pub struct CompileError(pub String);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tok {
    Char(u8),
    And,
    Break,
    Const,
    Continue,
    Do,
    Else,
    Elseif,
    End,
    False,
    For,
    Function,
    Goto,
    If,
    In,
    Local,
    Nil,
    Not,
    Or,
    Repeat,
    Return,
    Then,
    True,
    Until,
    While,
    Concat,
    Dots,
    Eq,
    Ge,
    Le,
    Ne,
    Nav,
    Coal,
    Shl,
    Shr,
    Sar,
    AndAnd,
    OrOr,
    NeBang,
    Arrow,
    Label,
    Number,
    Name,
    Str,
    Eof,
}

const KEYWORDS: &[(&[u8], Tok)] = &[
    (b"and", Tok::And),
    (b"break", Tok::Break),
    (b"const", Tok::Const),
    (b"continue", Tok::Continue),
    (b"do", Tok::Do),
    (b"else", Tok::Else),
    (b"elseif", Tok::Elseif),
    (b"end", Tok::End),
    (b"false", Tok::False),
    (b"for", Tok::For),
    (b"function", Tok::Function),
    (b"goto", Tok::Goto),
    (b"if", Tok::If),
    (b"in", Tok::In),
    (b"local", Tok::Local),
    (b"nil", Tok::Nil),
    (b"not", Tok::Not),
    (b"or", Tok::Or),
    (b"repeat", Tok::Repeat),
    (b"return", Tok::Return),
    (b"then", Tok::Then),
    (b"true", Tok::True),
    (b"until", Tok::Until),
    (b"while", Tok::While),
];

pub fn tok2str(tok: Tok) -> String {
    match tok {
        Tok::Char(c) => {
            if c.is_ascii_control() {
                format!("char({})", c)
            } else {
                (c as char).to_string()
            }
        }
        Tok::And => "and".into(),
        Tok::Break => "break".into(),
        Tok::Const => "const".into(),
        Tok::Continue => "continue".into(),
        Tok::Do => "do".into(),
        Tok::Else => "else".into(),
        Tok::Elseif => "elseif".into(),
        Tok::End => "end".into(),
        Tok::False => "false".into(),
        Tok::For => "for".into(),
        Tok::Function => "function".into(),
        Tok::Goto => "goto".into(),
        Tok::If => "if".into(),
        Tok::In => "in".into(),
        Tok::Local => "local".into(),
        Tok::Nil => "nil".into(),
        Tok::Not => "not".into(),
        Tok::Or => "or".into(),
        Tok::Repeat => "repeat".into(),
        Tok::Return => "return".into(),
        Tok::Then => "then".into(),
        Tok::True => "true".into(),
        Tok::Until => "until".into(),
        Tok::While => "while".into(),
        Tok::Concat => "..".into(),
        Tok::Dots => "...".into(),
        Tok::Eq => "==".into(),
        Tok::Ge => ">=".into(),
        Tok::Le => "<=".into(),
        Tok::Ne => "~=".into(),
        Tok::Nav => "?.".into(),
        Tok::Coal => "??".into(),
        Tok::Shl => "<<".into(),
        Tok::Shr => ">>".into(),
        Tok::Sar => "~>>".into(),
        Tok::AndAnd => "&&".into(),
        Tok::OrOr => "||".into(),
        Tok::NeBang => "!=".into(),
        Tok::Arrow => "->".into(),
        Tok::Label => "::".into(),
        Tok::Number => "<number>".into(),
        Tok::Name => "<name>".into(),
        Tok::Str => "<string>".into(),
        Tok::Eof => "<eof>".into(),
    }
}

const LEX_EOF: i32 = -1;

#[derive(Clone, Copy, Default)]
pub struct TokVal {
    pub num: f64,
    pub str: StrId,
}

pub struct LexState {
    src: Vec<u8>,
    pos: usize,
    pub c: i32,
    pub tok: Tok,
    pub tokval: TokVal,
    pub lookahead: Tok,
    lookaheadval: TokVal,
    sb: Vec<u8>,
    pub linenumber: u32,
    pub lastline: u32,
    pub chunkname: String,
    pub strs: Interner,
}

#[inline]
fn is_ident(c: i32) -> bool {
    (c >= b'0' as i32 && c <= b'9' as i32)
        || (c >= b'A' as i32 && c <= b'Z' as i32)
        || (c >= b'a' as i32 && c <= b'z' as i32)
        || c == b'_' as i32
}

#[inline]
fn is_digit(c: i32) -> bool {
    c >= b'0' as i32 && c <= b'9' as i32
}

#[inline]
fn is_xdigit(c: i32) -> bool {
    is_digit(c)
        || (c >= b'A' as i32 && c <= b'F' as i32)
        || (c >= b'a' as i32 && c <= b'f' as i32)
}

#[inline]
fn is_space(c: i32) -> bool {
    (c >= 9 && c <= 13) || c == 32
}

impl LexState {
    pub fn new(src: Vec<u8>, chunkname: String) -> LexState {
        let mut ls = LexState {
            src,
            pos: 0,
            c: 0,
            tok: Tok::Eof,
            tokval: TokVal::default(),
            lookahead: Tok::Eof,
            lookaheadval: TokVal::default(),
            sb: Vec::new(),
            linenumber: 1,
            lastline: 1,
            chunkname,
            strs: Interner::default(),
        };
        ls.next_char();
        if ls.c == 0xef && ls.pos + 1 < ls.src.len() && ls.src[ls.pos] == 0xbb && ls.src[ls.pos + 1] == 0xbf {
            ls.pos += 2;
            ls.next_char();
        }
        if ls.c == b'#' as i32 {
            loop {
                ls.next_char();
                if ls.c == LEX_EOF {
                    return ls;
                }
                if ls.is_eol() {
                    break;
                }
            }
            ls.newline();
        }
        ls
    }

    pub fn error(&self, msg: &str) -> ! {
        panic_any(CompileError(format!(
            "{}:{}: {}",
            self.chunkname, self.linenumber, msg
        )))
    }

    pub fn err_token_str(&self, near: &str, msg: &str) -> ! {
        self.error(&format!("{} near '{}'", msg, near));
    }

    pub fn err_near(&self, msg: &str) -> ! {
        let near = match self.tok {
            Tok::Name | Tok::Str => {
                String::from_utf8_lossy(self.strs.get(self.tokval.str)).into_owned()
            }
            Tok::Number => String::from_utf8_lossy(&self.sb).into_owned(),
            t => tok2str(t),
        };
        self.err_token_str(&near, msg);
    }

    #[inline]
    fn next_char(&mut self) -> i32 {
        self.c = if self.pos < self.src.len() {
            let c = self.src[self.pos] as i32;
            self.pos += 1;
            c
        } else {
            LEX_EOF
        };
        self.c
    }

    #[inline]
    fn save(&mut self, c: i32) {
        self.sb.push(c as u8);
    }

    #[inline]
    fn save_next(&mut self) -> i32 {
        self.save(self.c);
        self.next_char()
    }

    #[inline]
    fn is_eol(&self) -> bool {
        self.c == b'\n' as i32 || self.c == b'\r' as i32
    }

    fn newline(&mut self) {
        let old = self.c;
        debug_assert!(self.is_eol());
        self.next_char();
        if self.is_eol() && self.c != old {
            self.next_char();
        }
        self.linenumber += 1;
        if self.linenumber >= 0x7fffff00 {
            self.error("chunk has too many lines");
        }
    }

    fn lex_number(&mut self) -> f64 {
        let mut xp = b'e' as i32;
        let mut c = self.c;
        debug_assert!(is_digit(c));
        if c == b'0' as i32 {
            self.save(c);
            loop {
                c = self.next_char();
                if c != b'_' as i32 {
                    break;
                }
            }
            if (c | 0x20) == b'x' as i32 {
                xp = b'p' as i32;
            }
        }
        while is_ident(self.c)
            || self.c == b'.' as i32
            || ((self.c == b'-' as i32 || self.c == b'+' as i32) && (c | 0x20) == xp)
        {
            if self.c != b'_' as i32 {
                c = self.c;
                self.save(self.c);
            }
            self.next_char();
        }
        match crate::strscan::scan_number(&self.sb) {
            Some(n) => n,
            None => self.error("malformed number"),
        }
    }

    fn skip_eq(&mut self) -> i32 {
        let mut count = 0;
        let s = self.c;
        debug_assert!(s == b'[' as i32 || s == b']' as i32);
        while self.save_next() == b'=' as i32 && count < 0x20000000 {
            count += 1;
        }
        if self.c == s {
            count
        } else {
            -count - 1
        }
    }

    fn long_string(&mut self, is_str: bool, sep: i32) -> Option<StrId> {
        self.save_next();
        if self.is_eol() {
            self.newline();
        }
        loop {
            if self.c == LEX_EOF {
                self.error(if is_str {
                    "unfinished long string"
                } else {
                    "unfinished long comment"
                });
            } else if self.c == b']' as i32 {
                if self.skip_eq() == sep {
                    self.save_next();
                    break;
                }
            } else if self.c == b'\n' as i32 || self.c == b'\r' as i32 {
                self.save(b'\n' as i32);
                self.newline();
                if !is_str {
                    self.sb.clear();
                }
            } else {
                self.save_next();
            }
        }
        if is_str {
            let start = 2 + sep as usize;
            let end = self.sb.len() - start;
            let id = self.strs.intern(&self.sb[start..end].to_vec());
            Some(id)
        } else {
            None
        }
    }

    fn string(&mut self) -> StrId {
        let delim = self.c;
        self.save_next();
        while self.c != delim {
            match self.c {
                LEX_EOF => self.error("unfinished string"),
                c if c == b'\n' as i32 || c == b'\r' as i32 => self.error("unfinished string"),
                c if c == b'\\' as i32 => {
                    let mut c = self.next_char();
                    match c as u8 {
                        b'a' => c = 7,
                        b'b' => c = 8,
                        b'f' => c = 12,
                        b'n' => c = b'\n' as i32,
                        b'r' => c = b'\r' as i32,
                        b't' => c = b'\t' as i32,
                        b'v' => c = 11,
                        b'x' => {
                            c = (self.next_char() & 15) << 4;
                            if !is_digit(self.c) {
                                if !is_xdigit(self.c) {
                                    self.error("invalid escape sequence");
                                }
                                c += 9 << 4;
                            }
                            c += self.next_char() & 15;
                            if !is_digit(self.c) {
                                if !is_xdigit(self.c) {
                                    self.error("invalid escape sequence");
                                }
                                c += 9;
                            }
                        }
                        b'u' => {
                            if self.next_char() != b'{' as i32 {
                                self.error("invalid escape sequence");
                            }
                            self.next_char();
                            c = 0;
                            loop {
                                c = (c << 4) | (self.c & 15);
                                if !is_digit(self.c) {
                                    if !is_xdigit(self.c) {
                                        self.error("invalid escape sequence");
                                    }
                                    c += 9;
                                }
                                if c >= 0x110000 {
                                    self.error("invalid escape sequence");
                                }
                                if self.next_char() == b'}' as i32 {
                                    break;
                                }
                            }
                            self.next_char();
                            if c < 0x800 {
                                if c >= 0x80 {
                                    self.save(0xc0 | (c >> 6));
                                    self.save(0x80 | (c & 0x3f));
                                } else {
                                    self.save(c);
                                }
                            } else {
                                if c >= 0x10000 {
                                    self.save(0xf0 | (c >> 18));
                                    self.save(0x80 | ((c >> 12) & 0x3f));
                                } else {
                                    if c >= 0xd800 && c < 0xe000 {
                                        self.error("invalid escape sequence");
                                    }
                                    self.save(0xe0 | (c >> 12));
                                }
                                self.save(0x80 | ((c >> 6) & 0x3f));
                                self.save(0x80 | (c & 0x3f));
                            }
                            continue;
                        }
                        b'z' => {
                            self.next_char();
                            while is_space(self.c) {
                                if self.is_eol() {
                                    self.newline();
                                } else {
                                    self.next_char();
                                }
                            }
                            continue;
                        }
                        b'\n' | b'\r' => {
                            self.save(b'\n' as i32);
                            self.newline();
                            continue;
                        }
                        b'\\' | b'"' | b'\'' => {}
                        _ => {
                            if c == LEX_EOF {
                                continue;
                            }
                            if !is_digit(c) {
                                self.error("invalid escape sequence");
                            }
                            c -= b'0' as i32;
                            if is_digit(self.next_char()) {
                                c = c * 10 + (self.c - b'0' as i32);
                                if is_digit(self.next_char()) {
                                    c = c * 10 + (self.c - b'0' as i32);
                                    if c > 255 {
                                        self.error("invalid escape sequence");
                                    }
                                    self.next_char();
                                }
                            }
                            self.save(c);
                            continue;
                        }
                    }
                    self.save(c);
                    self.next_char();
                }
                _ => {
                    self.save_next();
                }
            }
        }
        self.save_next();
        self.strs.intern(&self.sb[1..self.sb.len() - 1].to_vec())
    }

    fn scan(&mut self) -> (Tok, TokVal) {
        self.sb.clear();
        loop {
            if is_ident(self.c) {
                if is_digit(self.c) {
                    let n = self.lex_number();
                    return (Tok::Number, TokVal { num: n, str: 0 });
                }
                loop {
                    self.save_next();
                    if !is_ident(self.c) {
                        break;
                    }
                }
                let bytes = self.sb.clone();
                let id = self.strs.intern(&bytes);
                for (kw, t) in KEYWORDS {
                    if *kw == &bytes[..] {
                        return (*t, TokVal { num: 0.0, str: id });
                    }
                }
                return (Tok::Name, TokVal { num: 0.0, str: id });
            }
            match self.c {
                c if c == b'\n' as i32 || c == b'\r' as i32 => {
                    self.newline();
                }
                c if c == b' ' as i32 || c == 9 || c == 11 || c == 12 => {
                    self.next_char();
                }
                c if c == b'-' as i32 => {
                    self.next_char();
                    if self.c != b'-' as i32 {
                        if self.c != b'>' as i32 {
                            return (Tok::Char(b'-'), TokVal::default());
                        }
                        self.next_char();
                        return (Tok::Arrow, TokVal::default());
                    }
                    self.next_char();
                    if self.c == b'[' as i32 {
                        let sep = self.skip_eq();
                        self.sb.clear();
                        if sep >= 0 {
                            self.long_string(false, sep);
                            self.sb.clear();
                            continue;
                        }
                    }
                    while !self.is_eol() && self.c != LEX_EOF {
                        self.next_char();
                    }
                }
                c if c == b'[' as i32 => {
                    let sep = self.skip_eq();
                    if sep >= 0 {
                        let id = self.long_string(true, sep).unwrap();
                        return (Tok::Str, TokVal { num: 0.0, str: id });
                    } else if sep == -1 {
                        return (Tok::Char(b'['), TokVal::default());
                    } else {
                        self.error("invalid long string delimiter");
                    }
                }
                c if c == b'=' as i32 => {
                    self.next_char();
                    if self.c != b'=' as i32 {
                        return (Tok::Char(b'='), TokVal::default());
                    }
                    self.next_char();
                    return (Tok::Eq, TokVal::default());
                }
                c if c == b'<' as i32 => {
                    self.next_char();
                    if self.c == b'=' as i32 {
                        self.next_char();
                        return (Tok::Le, TokVal::default());
                    }
                    if self.c == b'<' as i32 {
                        self.next_char();
                        return (Tok::Shl, TokVal::default());
                    }
                    return (Tok::Char(b'<'), TokVal::default());
                }
                c if c == b'>' as i32 => {
                    self.next_char();
                    if self.c == b'=' as i32 {
                        self.next_char();
                        return (Tok::Ge, TokVal::default());
                    }
                    if self.c == b'>' as i32 {
                        self.next_char();
                        return (Tok::Shr, TokVal::default());
                    }
                    return (Tok::Char(b'>'), TokVal::default());
                }
                c if c == b'~' as i32 => {
                    self.next_char();
                    if self.c == b'=' as i32 {
                        self.next_char();
                        return (Tok::Ne, TokVal::default());
                    }
                    if self.c == b'>' as i32 {
                        self.next_char();
                        if self.c != b'>' as i32 {
                            self.error("unexpected symbol");
                        }
                        self.next_char();
                        return (Tok::Sar, TokVal::default());
                    }
                    return (Tok::Char(b'~'), TokVal::default());
                }
                c if c == b'!' as i32 => {
                    self.next_char();
                    if self.c != b'=' as i32 {
                        return (Tok::Char(b'!'), TokVal::default());
                    }
                    self.next_char();
                    return (Tok::NeBang, TokVal::default());
                }
                c if c == b':' as i32 => {
                    self.next_char();
                    if self.c != b':' as i32 {
                        return (Tok::Char(b':'), TokVal::default());
                    }
                    self.next_char();
                    return (Tok::Label, TokVal::default());
                }
                c if c == b'?' as i32 => {
                    self.next_char();
                    if self.c == b'.' as i32 {
                        self.next_char();
                        return (Tok::Nav, TokVal::default());
                    }
                    if self.c == b'?' as i32 {
                        self.next_char();
                        return (Tok::Coal, TokVal::default());
                    }
                    return (Tok::Char(b'?'), TokVal::default());
                }
                c if c == b'&' as i32 => {
                    self.next_char();
                    if self.c != b'&' as i32 {
                        return (Tok::Char(b'&'), TokVal::default());
                    }
                    self.next_char();
                    return (Tok::AndAnd, TokVal::default());
                }
                c if c == b'|' as i32 => {
                    self.next_char();
                    if self.c != b'|' as i32 {
                        return (Tok::Char(b'|'), TokVal::default());
                    }
                    self.next_char();
                    return (Tok::OrOr, TokVal::default());
                }
                c if c == b'"' as i32 || c == b'\'' as i32 => {
                    let id = self.string();
                    return (Tok::Str, TokVal { num: 0.0, str: id });
                }
                c if c == b'.' as i32 => {
                    if self.save_next() == b'.' as i32 {
                        self.next_char();
                        if self.c == b'.' as i32 {
                            self.next_char();
                            return (Tok::Dots, TokVal::default());
                        }
                        return (Tok::Concat, TokVal::default());
                    } else if !is_digit(self.c) {
                        return (Tok::Char(b'.'), TokVal::default());
                    } else {
                        let n = self.lex_number();
                        return (Tok::Number, TokVal { num: n, str: 0 });
                    }
                }
                LEX_EOF => return (Tok::Eof, TokVal::default()),
                c => {
                    self.next_char();
                    return (Tok::Char(c as u8), TokVal::default());
                }
            }
        }
    }

    pub fn next(&mut self) {
        self.lastline = self.linenumber;
        if self.lookahead == Tok::Eof {
            let (tok, val) = self.scan();
            self.tok = tok;
            self.tokval = val;
        } else {
            self.tok = self.lookahead;
            self.tokval = self.lookaheadval;
            self.lookahead = Tok::Eof;
        }
    }

    pub fn peek(&mut self) -> Tok {
        debug_assert!(self.lookahead == Tok::Eof);
        let (tok, val) = self.scan();
        self.lookahead = tok;
        self.lookaheadval = val;
        tok
    }
}
