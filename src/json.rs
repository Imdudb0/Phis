//! Minimal self-contained JSON serializer and parser.

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn to_pretty_string(&self) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, 0);
        out
    }

    fn write_pretty(&self, out: &mut String, indent: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(n) => out.push_str(&n.to_string()),
            Json::Float(v) => {
                if !v.is_finite() {
                    out.push_str("null");
                } else if *v == 0.0 {
                    out.push_str("0.0");
                } else {
                    let s = v.to_string();
                    out.push_str(&s);
                    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                        out.push_str(".0");
                    }
                }
            }
            Json::Str(s) => write_escaped_str(out, s),
            Json::Arr(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                for (idx, item) in items.iter().enumerate() {
                    for _ in 0..(indent + 2) {
                        out.push(' ');
                    }
                    item.write_pretty(out, indent + 2);
                    if idx + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                for _ in 0..indent {
                    out.push(' ');
                }
                out.push(']');
            }
            Json::Obj(pairs) => {
                if pairs.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                for (idx, (k, v)) in pairs.iter().enumerate() {
                    for _ in 0..(indent + 2) {
                        out.push(' ');
                    }
                    write_escaped_str(out, k);
                    out.push_str(": ");
                    v.write_pretty(out, indent + 2);
                    if idx + 1 < pairs.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                for _ in 0..indent {
                    out.push(' ');
                }
                out.push('}');
            }
        }
    }

    pub fn parse(input: &str) -> Result<Json, String> {
        let mut p = Parser {
            bytes: input.as_bytes(),
            pos: 0,
        };
        let v = p.parse_value()?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            return Err(format!("trailing bytes at offset {}", p.pos));
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Float(v) => Some(*v),
            Json::Int(n) => Some(*n as f64),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Int(n) if *n >= 0 => Some(*n as usize),
            _ => None,
        }
    }
}

fn write_escaped_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, expected: u8) -> Result<(), String> {
        self.skip_ws();
        match self.bytes.get(self.pos) {
            Some(&b) if b == expected => {
                self.pos += 1;
                Ok(())
            }
            Some(&b) => Err(format!(
                "expected '{}' at {}, found '{}'",
                expected as char, self.pos, b as char
            )),
            None => Err(format!("expected '{}' at EOF", expected as char)),
        }
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        match self.peek() {
            Some(b'n') => self.parse_null(),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'"') => self.parse_string().map(Json::Str),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(b) => Err(format!("unexpected '{}' at {}", b as char, self.pos)),
            None => Err("unexpected EOF".to_string()),
        }
    }

    fn parse_null(&mut self) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Json::Null)
        } else {
            Err(format!("invalid literal at {}", self.pos))
        }
    }

    fn parse_bool(&mut self) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Json::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Json::Bool(false))
        } else {
            Err(format!("invalid boolean at {}", self.pos))
        }
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        let mut is_float = false;
        match self.bytes.get(self.pos) {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(format!("invalid number at {}", start)),
        }
        if self.bytes.get(self.pos) == Some(&b'.') {
            is_float = true;
            self.pos += 1;
            if !matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                return Err(format!("invalid fractional part at {}", self.pos));
            }
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                return Err(format!("invalid exponent at {}", self.pos));
            }
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|e| e.to_string())?;
        if is_float {
            let v: f64 = s.parse().map_err(|e| format!("bad float {s}: {e}"))?;
            Ok(Json::Float(v))
        } else if let Ok(n) = s.parse::<i64>() {
            Ok(Json::Int(n))
        } else {
            let v: f64 = s.parse().map_err(|e| format!("bad number {s}: {e}"))?;
            Ok(Json::Float(v))
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        if self.pos + 4 > self.bytes.len() {
            return Err("truncated \\u escape".to_string());
        }
        let s = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
            .map_err(|e| e.to_string())?;
        self.pos += 4;
        u16::from_str_radix(s, 16).map_err(|e| format!("bad hex \\u{s}: {e}"))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        let mut chunk_start = self.pos;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                let seg = std::str::from_utf8(&self.bytes[chunk_start..self.pos])
                    .map_err(|e| e.to_string())?;
                out.push_str(seg);
                self.pos += 1;
                return Ok(out);
            } else if b == b'\\' {
                let seg = std::str::from_utf8(&self.bytes[chunk_start..self.pos])
                    .map_err(|e| e.to_string())?;
                out.push_str(seg);
                self.pos += 1;
                let esc = *self
                    .bytes
                    .get(self.pos)
                    .ok_or_else(|| "truncated escape".to_string())?;
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0c'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let code = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&code) {
                            if !self.bytes[self.pos..].starts_with(b"\\u") {
                                return Err("unpaired high surrogate".to_string());
                            }
                            self.pos += 2;
                            let low = self.parse_hex4()?;
                            if !(0xDC00..=0xDFFF).contains(&low) {
                                return Err("invalid low surrogate".to_string());
                            }
                            let cp = 0x10000
                                + (((code as u32 - 0xD800) << 10) | (low as u32 - 0xDC00));
                            let ch = char::from_u32(cp)
                                .ok_or_else(|| "invalid unicode scalar".to_string())?;
                            out.push(ch);
                        } else {
                            let ch = char::from_u32(code as u32)
                                .ok_or_else(|| "invalid unicode scalar".to_string())?;
                            out.push(ch);
                        }
                    }
                    other => return Err(format!("invalid escape \\{}", other as char)),
                }
                chunk_start = self.pos;
            } else if b < 0x20 {
                return Err(format!("unescaped control char at {}", self.pos));
            } else {
                self.pos += 1;
            }
        }
        Err("unterminated string".to_string())
    }

    fn parse_array(&mut self) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.pos)),
            }
        }
    }

    fn parse_object(&mut self) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.skip_ws();
            let k = self.parse_string()?;
            self.expect(b':')?;
            let v = self.parse_value()?;
            pairs.push((k, v));
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(pairs));
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.pos)),
            }
        }
    }
}