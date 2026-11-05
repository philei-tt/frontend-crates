// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reproductions of MiniMax-M3.1 tool-call failures seen on OpenRouter's `toolcall_formats`
//! suite and MiniMax-Provider-Verifier against a Dynamo frontend. Each test states the output a
//! schema validator accepts; the raw model text is what M3.1 emits for that case.

use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParser, UnifiedParserExt, assemble, create_unified_parser_for_family,
};
use serde_json::{Value, json};

const NS: &str = "]<]minimax[>[";

fn parser(schema: Value) -> Box<dyn UnifiedParser> {
    let tools = vec![Tool {
        name: "record".into(),
        description: None,
        parameters: schema,
        strict: None,
    }];
    create_unified_parser_for_family("minimax_m3", &tools).unwrap()
}

fn param(key: &str, value: &str) -> String {
    format!("{NS}<{key}>{value}{NS}</{key}>")
}

fn call(body: &str) -> String {
    format!("{NS}<tool_call>\n{NS}<invoke name=\"record\">{body}{NS}</invoke>\n{NS}</tool_call>")
}

/// Parses `input` whole, then split at every char boundary, and returns the arguments of the
/// single tool call; panics if the splits disagree.
fn args(schema: Value, input: &str) -> Value {
    let run = |chunks: &[&str]| {
        let mut p = parser(schema.clone());
        let mut out = Vec::new();
        for c in chunks {
            out.extend(p.push(c).unwrap());
        }
        out.extend(p.finish().unwrap().events);
        let calls: Vec<Value> = assemble(&out)
            .into_iter()
            .filter_map(|e| match e {
                UnifiedEvent::ToolCall { arguments, .. } => Some(arguments),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1, "expected one call, got {calls:?} for {input:?}");
        calls.into_iter().next().unwrap()
    };
    let whole = run(&[input]);
    for (at, _) in input.char_indices().skip(1) {
        assert_eq!(run(&[&input[..at], &input[at..]]), whole, "split at {at}");
    }
    let chars: Vec<String> = input.chars().map(String::from).collect();
    let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
    assert_eq!(run(&chars), whole, "one character per chunk");
    whole
}

// --- 1. required + nullable key omitted (34 of 135 OR toolcall_formats cases) ---------------
// MiniMax's chat template drops None arguments when rendering history, so the model learned to
// omit null fields. A validator then reports a missing required key.

#[test]
fn required_nullable_anyof_key_omitted_becomes_null() {
    let schema = json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}
        },
        "required": ["city", "note"]
    });
    let got = args(schema, &call(&param("city", "Paris")));
    assert_eq!(got, json!({"city": "Paris", "note": null}));
}

#[test]
fn required_nullable_type_array_key_omitted_becomes_null() {
    let schema = json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "limit": {"type": ["integer", "null"]}
        },
        "required": ["city", "limit"]
    });
    let got = args(schema, &call(&param("city", "Paris")));
    assert_eq!(got, json!({"city": "Paris", "limit": null}));
}

#[test]
fn required_non_nullable_key_omitted_stays_omitted() {
    // Never invent a value the schema does not allow.
    let schema = json!({
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city", "days"]
    });
    let got = args(schema, &call(&param("city", "Paris")));
    assert_eq!(got, json!({"city": "Paris"}));
}

#[test]
fn optional_nullable_key_omitted_stays_omitted() {
    let schema = json!({
        "type": "object",
        "properties": {"city": {"type": "string"}, "note": {"type": ["string", "null"]}},
        "required": ["city"]
    });
    let got = args(schema, &call(&param("city", "Paris")));
    assert_eq!(got, json!({"city": "Paris"}));
}

// --- 2. trailing newline kept in string arguments (5 of 135) --------------------------------

#[test]
fn string_argument_trailing_newline_before_close_is_dropped() {
    let schema = json!({
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"]
    });
    let got = args(schema, &call(&param("text", "caf\u{e9} \u{2713}\n")));
    assert_eq!(got, json!({"text": "caf\u{e9} \u{2713}"}));
}

#[test]
fn string_argument_leading_newline_after_open_is_dropped() {
    let schema = json!({
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"]
    });
    let got = args(schema, &call(&param("text", "\nhello")));
    assert_eq!(got, json!({"text": "hello"}));
}

#[test]
fn string_argument_inner_newlines_are_kept() {
    let schema = json!({
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"]
    });
    let got = args(schema, &call(&param("text", "line one\nline two")));
    assert_eq!(got, json!({"text": "line one\nline two"}));
}

fn text_schema() -> Value {
    json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
}

#[test]
fn string_argument_crlf_edges_are_dropped() {
    let got = args(text_schema(), &call(&param("text", "\r\nhi\r\n")));
    assert_eq!(got, json!({"text": "hi"}));
}

#[test]
fn string_argument_only_one_edge_newline_is_dropped() {
    let got = args(text_schema(), &call(&param("text", "\n\nbody\n\n")));
    assert_eq!(got, json!({"text": "\nbody\n"}));
}

#[test]
fn string_argument_of_only_a_newline_becomes_empty() {
    let got = args(text_schema(), &call(&param("text", "\n")));
    assert_eq!(got, json!({"text": ""}));
}

#[test]
fn string_argument_edge_spaces_are_kept() {
    let got = args(text_schema(), &call(&param("text", " padded \n ")));
    assert_eq!(got, json!({"text": " padded \n "}));
}

#[test]
fn nullable_string_union_trailing_newline_is_dropped() {
    let schema = json!({
        "type": "object",
        "properties": {"text": {"type": ["string", "null"]}},
        "required": ["text"]
    });
    let got = args(schema, &call(&param("text", "hello\n")));
    assert_eq!(got, json!({"text": "hello"}));
}

#[test]
fn required_nullable_filled_when_call_has_no_parameters() {
    let schema = json!({
        "type": "object",
        "properties": {"note": {"type": ["string", "null"]}},
        "required": ["note"]
    });
    let got = args(schema, &call(""));
    assert_eq!(got, json!({"note": null}));
}

#[test]
fn required_nullable_present_value_is_not_overwritten() {
    let schema = json!({
        "type": "object",
        "properties": {"city": {"type": "string"}, "note": {"type": ["string", "null"]}},
        "required": ["city", "note"]
    });
    let body = format!("{}{}", param("note", "hi"), param("city", "Paris"));
    let got = args(schema, &call(&body));
    assert_eq!(got, json!({"note": "hi", "city": "Paris"}));
}

// --- 3. top-level oneOf / anyOf parameters (verifier 14_07) ---------------------------------

#[test]
fn top_level_oneof_types_integer_and_list() {
    let schema = json!({
        "oneOf": [
            {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}},
                "required": ["id"]
            },
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        ]
    });
    let body = format!(
        "{}{}",
        param("id", "42"),
        param("tags", &format!("{NS}<item>a{NS}</item>{NS}<item>b{NS}</item>"))
    );
    let got = args(schema, &call(&body));
    assert_eq!(got, json!({"id": 42, "tags": ["a", "b"]}));
}

#[test]
fn top_level_anyof_types_number() {
    let schema = json!({
        "anyOf": [
            {"type": "object", "properties": {"price": {"type": "number"}}, "required": ["price"]},
            {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}
        ]
    });
    let got = args(schema, &call(&param("price", "19.5")));
    assert_eq!(got, json!({"price": 19.5}));
}

// --- 4. other typing shapes from the GLM fuzz classes ---------------------------------------

#[test]
fn mixed_type_enum_keeps_integer_member() {
    let schema = json!({
        "type": "object",
        "properties": {"level": {"enum": ["auto", 1, 2]}},
        "required": ["level"]
    });
    let got = args(schema, &call(&param("level", "2")));
    assert_eq!(got, json!({"level": 2}));
}

#[test]
fn ref_to_defs_types_boolean() {
    let schema = json!({
        "type": "object",
        "properties": {"flag": {"$ref": "#/$defs/Flag"}},
        "required": ["flag"],
        "$defs": {"Flag": {"type": "boolean"}}
    });
    let got = args(schema, &call(&param("flag", "true")));
    assert_eq!(got, json!({"flag": true}));
}
