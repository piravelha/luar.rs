use std::fmt::{Display, Error, Formatter};
use std::iter::Peekable;
use std::str::Chars;

#[derive(Debug, Clone)]
struct Location {
    file: String,
    line: i32,
    column: i32,
}

impl Location {
    pub fn new(file: String, line: i32, column: i32) -> Location {
        Location { file, line, column }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    EOF,
    Int,
    Local,
    Equals,
    Plus,
    Minus,
    Times,
    Div,
    Caret,
    Hash,
    Concat,
    Ellipsis,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Comma,
    SemiColon,
    Colon,
    Dot,
    Function,
    If,
    Then,
    Else,
    Elseif,
    While,
    Do,
    For,
    In,
    End,
    Repeat,
    Until,
    Return,
    Break,
    Name,
    True,
    False,
    Nil,

    IntType,
    BooleanType,
    NilType,
}

#[derive(Debug, Clone)]
struct Token {
    kind: TokenKind,
    value: String,
    location: Location,
}

impl Token {
    pub fn new(kind: TokenKind, value: String, location: Location) -> Token {
        Token {
            kind,
            value: value.to_string(),
            location,
        }
    }
}

impl Display for Token {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        write!(fmt, "[{:?}:'{}']", self.kind, self.value)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum TokenErrorKind {
    InvalidCharacter(char),
}

#[derive(Debug, Clone)]
struct TokenError {
    location: Location,
    kind: TokenErrorKind,
}

impl TokenError {
    pub fn new(location: Location, kind: TokenErrorKind) -> TokenError {
        TokenError { location, kind }
    }
}

#[derive(Debug, Clone)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Concat,
}

#[derive(Debug, Clone)]
enum Expr {
    IntLit(i32),
    Var(String),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    FuncCall(Box<Expr>, Vec<Expr>),
}

#[derive(Debug, Clone)]
enum Stmt {
    VarDecl(String, Type, Expr),
    Return(Expr),
}

#[derive(Debug, Clone)]
enum Type {
    Int,
    Boolean,
    Nil,
}

#[derive(Debug, Clone)]
struct Chunk(Vec<Stmt>);

#[derive(Debug, Clone)]
struct ParseError {
    location: Location,
    expected: Vec<TokenKind>,
    got: TokenKind,
}

impl ParseError {
    pub fn new(
        location: Location,
        expected: Vec<TokenKind>,
        got: TokenKind,
    ) -> ParseError {
        ParseError { location, expected, got }
    }
}

fn update_line_and_column(char: &char, line: &mut i32, column: &mut i32) {
    if *char == '\n' {
        *line += 1;
        *column = 1;
    } else {
        *column += 1;
    }
}

fn trim_left(chars: &mut Peekable<Chars>, line: &mut i32, column: &mut i32) {
    while let Some(char) = chars.clone().peek() {
        if char.is_whitespace() {
            update_line_and_column(&char, line, column);
            chars.next();
        } else {
            break;
        }
    }
}

fn starts_with(chars: &mut Peekable<Chars>, prefix: &str, column: &mut i32) -> bool {
    let is_name = prefix.chars().nth(0).unwrap().is_alphabetic();
    let result = chars
        .clone()
        .take(prefix.len() + 1)
        .collect::<String>();
    
    let rl = result.len();
    let l = if rl == prefix.len() + 1 {
        rl - 1
    } else if rl > 1 {
        rl
    } else {
        0
    };

    if result.len() > 1
        && result.chars()
        .nth(result.len() - 1)
        .unwrap()
        .is_alphabetic()
        && is_name
        && result.len() == prefix.len() + 1
    {
        false
    } else if
        result.len() > 1
        && &result[..l]
        == prefix
    {
        for _ in 0..prefix.len() {
            *column += 1;
            chars.next();
        }
        true
    } else {
        false
    }
}

fn get_next_token(
    file: &str,
    line: &mut i32,
    column: &mut i32,
    chars: &mut Peekable<Chars>,
) -> Result<Token, TokenError> {
    trim_left(chars, line, column);
    let loc = Location::new(file.to_string(), *line, *column);

    let lits = vec![
        ("local", TokenKind::Local),
        ("=", TokenKind::Equals),
        ("+", TokenKind::Plus),
        ("-", TokenKind::Minus),
        ("*", TokenKind::Times),
        ("/", TokenKind::Div),
        ("^", TokenKind::Caret),
        ("#", TokenKind::Hash),
        ("...", TokenKind::Ellipsis),
        ("..", TokenKind::Concat),
        ("(", TokenKind::LeftParen),
        (")", TokenKind::RightParen),
        ("[", TokenKind::LeftBracket),
        ("]", TokenKind::RightBracket),
        ("{", TokenKind::LeftBrace),
        ("}", TokenKind::RightBrace),
        (",", TokenKind::Comma),
        (";", TokenKind::SemiColon),
        (":", TokenKind::Colon),
        (".", TokenKind::Dot),
        ("function", TokenKind::Function),
        ("if", TokenKind::If),
        ("then", TokenKind::Then),
        ("else", TokenKind::Else),
        ("elseif", TokenKind::Elseif),
        ("while", TokenKind::While),
        ("do", TokenKind::Do),
        ("for", TokenKind::For),
        ("in", TokenKind::In),
        ("end", TokenKind::End),
        ("repeat", TokenKind::Repeat),
        ("until", TokenKind::Until),
        ("return", TokenKind::Return),
        ("break", TokenKind::Break),
    ];

    for (lit, kind) in lits.into_iter() {
        if starts_with(chars, lit, column) {
            return Ok(Token::new(kind, lit.to_string(), loc));
        }
    }

    let true_lit = "true".to_string();
    if starts_with(chars, &true_lit, column) {
        return Ok(Token::new(TokenKind::True, true_lit, loc));
    }

    let false_lit = "false".to_string();
    if starts_with(chars, &false_lit, column) {
        return Ok(Token::new(TokenKind::False, false_lit, loc));
    }

    let nil_lit = "nil".to_string();
    if starts_with(chars, &nil_lit, column) {
        return Ok(Token::new(TokenKind::Nil, nil_lit, loc));
    }

    let first = match chars.peek() {
        None => {
            let val = "".to_string();
            let tok = Token::new(TokenKind::EOF, val, loc);
            return Ok(tok);
        }
        Some(char) => char,
    };

    if first.is_ascii_digit() {
        let mut num = vec![];

        while let Some(char) = chars.clone().peek() {
            if char.is_ascii_digit() {
                num.push(char.clone());
                update_line_and_column(&char, line, column);
                chars.next();
            } else {
                break;
            }
        }

        let num = num.into_iter().collect();
        Ok(Token::new(TokenKind::Int, num, loc))
    } else if first.is_alphabetic() || *first == '_' {
        let mut name = vec![first.clone()];
        chars.next();

        while let Some(char) = chars.clone().peek() {
            if char.is_alphanumeric() || *char == '_' {
                name.push(char.clone());
                update_line_and_column(&char, line, column);
                chars.next();
            } else {
                break;
            }
        }

        let name = name.into_iter().collect();
        Ok(Token::new(TokenKind::Name, name, loc))
    } else {
        let char = chars.next().unwrap();
        Err(TokenError::new(loc, TokenErrorKind::InvalidCharacter(char)))
    }
}

fn get_tokens(file: &str, code: &str) -> Result<Vec<Token>, Vec<TokenError>> {
    let mut tokens = vec![];
    let line = &mut 1;
    let column = &mut 1;

    let mut chars = code.chars().peekable();

    loop {
        let token = get_next_token(file, line, column, &mut chars);
        tokens.push(token.clone());
        match token {
            Ok(Token {
                kind: TokenKind::EOF,
                ..
            }) => break,
            _ => continue,
        }
    }

    let mut errs = vec![];
    let mut toks = vec![];

    for token in tokens.into_iter() {
        match token {
            Ok(t) => toks.push(t),
            Err(e) => errs.push(e),
        }
    }

    if errs.len() > 0 {
        Err(errs)
    } else {
        Ok(toks)
    }
}

fn parse_int_lit(
    tokens: Vec<Token>,
) -> Result<(Expr, Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    match first.kind {
        TokenKind::Int => {
            let rest = tokens[1..].to_vec();
            let n = str::parse::<i32>(&first.value).unwrap();
            Ok((Expr::IntLit(n), rest))
        },
        _ => Err(ParseError::new(first.location, vec![TokenKind::Int], first.kind)),
    }
}

fn parse_var(
    tokens: Vec<Token>,
) -> Result<(Expr, Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    match first.kind {
        TokenKind::Name => {
            let rest = tokens[1..].to_vec();
            Ok((Expr::Var(first.value), rest))
        },
        _ => Err(ParseError::new(first.location, vec![TokenKind::Name], first.kind)),
    }
}

fn parse_atom(
    tokens: Vec<Token>,
) -> Result<(Expr, Vec<Token>), ParseError> {
    let err = match parse_int_lit(tokens.clone()) {
        ok @ Ok(_) => return ok,
        Err(e) => e,
    };
    match parse_var(tokens) {
        ok @ Ok(_) => ok,
        Err(e) => if err.got == e.got {
            Err(ParseError::new(err.location, [
                err.expected,
                e.expected,
            ].concat(), err.got))
        } else {
            Err(e)
        },
    }
}

fn parse_pow_op(
    tokens: Vec<Token>,
) -> Result<(BinaryOp, Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    match first.kind {
        TokenKind::Caret => {
            let rest = tokens[1..].to_vec();
            Ok((BinaryOp::Pow, rest))
        },
        _ => Err(ParseError::new(first.location, vec![TokenKind::Caret], first.kind)),
    }
}

fn parse_pow_expr(
    tokens: Vec<Token>,
) -> Result<(Expr, Vec<Token>), ParseError> {
    let (mut result, mut tokens) = parse_atom(tokens)?;

    loop {
        let (op, new_tokens) = match parse_pow_op(tokens.clone()) {
            Ok(ok) => ok,
            Err(_) => break,
        };
        let (next, ref mut new_tokens) = parse_atom(new_tokens)?;
        tokens = new_tokens.to_vec();
        result = Expr::Binary(Box::new(result), op, Box::new(next));
    }
    Ok((result, tokens))
}

fn parse_mul_op(
    tokens: Vec<Token>,
) -> Result<(BinaryOp, Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    let rest = tokens[1..].to_vec();

    match first.kind {
        TokenKind::Times => Ok((BinaryOp::Mul, rest)),
        TokenKind::Div => Ok((BinaryOp::Div, rest)),
        _ => Err(ParseError::new(first.location, vec![
            TokenKind::Times,
            TokenKind::Div,
        ], first.kind)),
    }
}

fn parse_mul_expr(
    tokens: Vec<Token>,
) -> Result<(Expr, Vec<Token>), ParseError> {
    let (mut result, mut tokens) = parse_pow_expr(tokens)?;
    
    loop {
        let (op, new_tokens) = match parse_mul_op(tokens.clone()) {
            Ok(ok) => ok,
            Err(_) => break,
        };
        let (next, ref mut new_tokens) = parse_pow_expr(new_tokens)?;
        tokens = new_tokens.to_vec();
        result = Expr::Binary(Box::new(result), op, Box::new(next));
    }
    Ok((result, tokens))
}

fn parse_add_op(
    tokens: Vec<Token>,
) -> Result<(BinaryOp, Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    let rest = tokens[1..].to_vec();

    match first.kind {
        TokenKind::Plus => Ok((BinaryOp::Add, rest)),
        TokenKind::Minus => Ok((BinaryOp::Sub, rest)),
        TokenKind::Concat => Ok((BinaryOp::Concat, rest)),
        _ => Err(ParseError::new(first.location, vec![
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Concat,
        ], first.kind)),
    }
}

fn parse_add_expr(
    tokens: Vec<Token>,
) -> Result<(Expr, Vec<Token>), ParseError> {
    let (mut result, mut tokens) = parse_mul_expr(tokens)?;
    
    loop {
        let (op, new_tokens) = match parse_add_op(tokens.clone()) {
            Ok(ok) => ok,
            Err(_) => break,
        };
        let (next, ref mut new_tokens) = parse_mul_expr(new_tokens)?;
        tokens = new_tokens.to_vec();
        result = Expr::Binary(Box::new(result), op, Box::new(next));
    }
    Ok((result, tokens))
}

fn parse_expr(
    tokens: Vec<Token>,
) -> Result<(Expr, Vec<Token>), ParseError> {
    parse_add_expr(tokens)
}

fn parse_local_keyword(
    tokens: Vec<Token>,
) -> Result<((), Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    let rest = tokens[1..].to_vec();

    match first.kind {
        TokenKind::Local => Ok(((), rest)),
        _ => Err(ParseError::new(first.location, vec![
            TokenKind::Local,
        ], first.kind)),
    }
}

fn parse_equals(
    tokens: Vec<Token>,
) -> Result<((), Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    let rest = tokens[1..].to_vec();

    match first.kind {
        TokenKind::Equals => Ok(((), rest)),
        _ => Err(ParseError::new(first.location, vec![
            TokenKind::Equals,
        ], first.kind)),
    }
}

fn parse_colon(
    tokens: Vec<Token>,
) -> Result<((), Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    let rest = tokens[1..].to_vec();

    match first.kind {
        TokenKind::Colon => Ok(((), rest)),
        _ => Err(ParseError::new(first.location, vec![
            TokenKind::Colon,
        ], first.kind)),
    }
}

fn parse_var_decl(
    tokens: Vec<Token>,
) -> Result<(Stmt, Vec<Token>), ParseError> {
    let (_, tokens) = parse_local_keyword(tokens)?;
    let (var, tokens) = parse_var(tokens)?;
    let var = match var {
        Expr::Var(name) => name,
        _ => todo!(),
    };
    let (_, tokens) = parse_colon(tokens)?;
    let (typ, tokens) = parse_type(tokens)?;
    let (_, tokens) = parse_equals(tokens)?;
    let (expr, tokens) = parse_expr(tokens)?;
    Ok((Stmt::VarDecl(var, typ, expr), tokens))
}

fn parse_return_keyword(
    tokens: Vec<Token>,
) -> Result<((), Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    let rest = tokens[1..].to_vec();

    match first.kind {
        TokenKind::Return => Ok(((), rest)),
        _ => Err(ParseError::new(first.location, vec![
            TokenKind::Return,
        ], first.kind)),
    }
}

fn parse_return(
    tokens: Vec<Token>,
) -> Result<(Stmt, Vec<Token>), ParseError> {
    let (_, tokens) = parse_return_keyword(tokens)?;
    let (expr, tokens) = parse_expr(tokens)?;
    Ok((Stmt::Return(expr), tokens))
}

fn parse_stmt(
    tokens: Vec<Token>,
) -> Result<(Stmt, Vec<Token>), ParseError> {
    let err = match parse_var_decl(tokens.clone()) {
        ok @ Ok(_) => return ok,
        Err(e) => e,
    };

    match parse_return(tokens) {
        ok @ Ok(_) => ok,
        Err(e) => if err.got == e.got {
            Err(ParseError::new(err.location, [
                err.expected,
                e.expected,
            ].concat(), err.got))
        } else {
            Err(err)
        }
    }
}

fn parse_int_type(
    tokens: Vec<Token>,
) -> Result<(Type, Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    let rest = tokens[1..].to_vec();

    match first.value.as_str() {
        "int" => Ok((Type::Int, rest)),
        _ => Err(ParseError::new(first.location, vec![
            TokenKind::IntType,
        ], first.kind)),
    }
}

fn parse_boolean_type(
    tokens: Vec<Token>,
) -> Result<(Type, Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    let rest = tokens[1..].to_vec();

    match first.value.as_str() {
        "boolean" => Ok((Type::Int, rest)),
        _ => Err(ParseError::new(first.location, vec![
            TokenKind::BooleanType,
        ], first.kind)),
    }
}

fn parse_nil_type(
    tokens: Vec<Token>,
) -> Result<(Type, Vec<Token>), ParseError> {
    let first = tokens[0].clone();
    let rest = tokens[1..].to_vec();

    match first.value.as_str() {
        "nil" => Ok((Type::Int, rest)),
        _ => Err(ParseError::new(first.location, vec![
            TokenKind::NilType,
        ], first.kind)),
    }
}

fn parse_type(
    tokens: Vec<Token>,
) -> Result<(Type, Vec<Token>), ParseError> {
    let mut expected = vec![];

    match parse_int_type(tokens.clone()) {
        ok @ Ok(_) => return ok,
        Err(e) => expected.extend(e.expected),
    };

    match parse_boolean_type(tokens.clone()) {
        ok @ Ok(_) => return ok,
        Err(e) => expected.extend(e.expected),
    };

    match parse_nil_type(tokens) {
        ok @ Ok(_) => ok,
        Err(e) => Err(ParseError::new(e.location, [
            expected,
            e.expected
        ].concat(), e.got)) 
    }
}

fn parse_chunk(
    mut tokens: Vec<Token>,
) -> (Chunk, Vec<Token>, ParseError) {
    let mut stmts = vec![];
    let err;

    loop {
        let (stmt, ref mut new_tokens) = match parse_stmt(tokens.clone()) {
            Ok(ok) => ok,
            Err(e) => {
                err = e;
                break;
            },
        };
        stmts.push(stmt);
        tokens = new_tokens.to_vec();
    }

    (Chunk(stmts), tokens, err)
}

fn parse_program(
    tokens: Vec<Token>,
) -> Result<Chunk, ParseError> {
    let (chunk, tokens, err) = parse_chunk(tokens);
    let first = tokens[0].clone();

    match first.kind {
        TokenKind::EOF => Ok(chunk),
        _ => Err(err),
    }
}

fn main() {
    let input = "
        local four : int = 2 * 2
        return four ^ 2
    ";
    let tokens = get_tokens("", &input);
    match tokens {
        Err(es) => es.into_iter().for_each(|e| println!("{:?}", e)),
        Ok(ts) => {
            let res = parse_program(ts);
            println!("{:?}", res);
        }
    }
}
