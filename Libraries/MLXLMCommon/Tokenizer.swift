// Copyright © 2024 Apple Inc.

import Foundation

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol Tokenizer: Sendable {
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?

    var bosToken: String? { get }
    var eosToken: String? { get }
    var unknownToken: String? { get }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]
}

extension Tokenizer {
    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokenIds: [Int]) -> String {
        decode(tokenIds: tokenIds, skipSpecialTokens: false)
    }

    public var eosTokenId: Int? {
        guard let eosToken else { return nil }
        return convertTokenToId(eosToken)
    }

    public var unknownTokenId: Int? {
        guard let unknownToken else { return nil }
        return convertTokenToId(unknownToken)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]]
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: tools, additionalContext: nil)
    }
}

public enum TokenizerError: LocalizedError {
    case missingChatTemplate

    public var errorDescription: String? {
        switch self {
        case .missingChatTemplate:
            "This tokenizer does not have a chat template."
        }
    }
}

public protocol StreamingDetokenizer: IteratorProtocol<String> {
    mutating func append(token: Int)
}

public struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    let tokenizer: any Tokenizer

    var segmentTokens = [Int]()
    var segment = ""

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public mutating func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokenIds: segmentTokens)
        } else {
            segment = ""
        }
    }

    public mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokenIds: segmentTokens)

        // A trailing REPLACEMENT CHARACTER means the newest token ends inside a multi-byte UTF-8
        // sequence (byte-level BPE splits e.g. " \u{1F44B}" into " \xF0\x9F\x91" + "\x8B"). Hold
        // everything until a later token completes it. Checked on the last SCALAR of the decoded window
        // (a Character check can miss an FFFD fused into a cluster, e.g. after a Prepend scalar).
        if newSegment.unicodeScalars.last == "\u{FFFD}" {
            return nil
        }

        // Diff in UNICODE SCALARS, not Characters. A token that adds a combining mark, ZWJ, skin-tone
        // modifier, variation selector or a flag's second regional indicator extends the previous
        // grapheme cluster without changing `String.count`, so grapheme arithmetic silently dropped
        // those scalars ("\u{1F468}\u{200D}\u{1F469}" streamed as "\u{1F468}"). Concatenating the
        // scalar pieces reproduces the full decode exactly.
        let all = newSegment.unicodeScalars
        let committed = segment.unicodeScalars.count
        let new =
            all.count > committed
            ? String(all[all.index(all.startIndex, offsetBy: committed)...]) : ""

        // Reset the decode window at safe boundaries so `decode(segmentTokens)` stays O(1) amortized
        // instead of re-decoding an unbounded segment EVERY token (an O(n^2) host cost on long,
        // newline-sparse generations). Only reset once this step produced VISIBLE text that does not
        // end in a partial sequence: a byte-fallback decoder (`<0xNN>` tokens) renders a trailing byte
        // run as NOTHING until a normal token follows, so resetting on an empty `new` would orphan the
        // run's leading bytes and later surface them as U+FFFD.
        if !new.isEmpty && (new.hasSuffix("\n") || segmentTokens.count >= 16) {
            startNewSegment()
        } else {
            self.segment = newSegment
        }

        return new
    }
}

/// Rewrites Python Jinja `tojson(sort_keys=..., separators=...)` filter calls into a
/// pure-Jinja compact JSON macro compatible with `swift-jinja` (`tojson` in `swift-jinja`
/// only accepts `indent` and `ensure_ascii`).
public enum ChatTemplateSanitizer {
    // Emit `{`, `}`, `[`, `]` as string expressions `{{- "{" -}}` so `swift-jinja`'s
    // `Lexer.preprocess` (`\{\s+\{%-` -> `"{ {%"`) never injects leading spaces inside objects.
    private static let compactJSONMacro = #"""
{%- macro _mlx_compact_json(v, sort_keys=false) -%}
{%- if v is mapping -%}{{- "{" -}}
{%- if sort_keys -%}{%- for k, val in v | dictsort(case_sensitive=true) -%}{{ (k | tojson) ~ ":" ~ _mlx_compact_json(val, true) }}{%- if not loop.last -%},{%- endif -%}{%- endfor -%}
{%- else -%}{%- for k in v -%}{{ (k | tojson) ~ ":" ~ _mlx_compact_json(v[k], false) }}{%- if not loop.last -%},{%- endif -%}{%- endfor -%}{%- endif -%}
{{- "}" -}}{%- elif v is sequence and v is not string -%}{{- "[" -}}{%- for item in v -%}{{ _mlx_compact_json(item, sort_keys) }}{%- if not loop.last -%},{%- endif -%}{%- endfor -%}{{- "]" -}}
{%- else -%}{{ v | tojson }}{%- endif -%}
{%- endmacro -%}

"""#

    private static let tojsonCallPattern: NSRegularExpression = {
        // Safe static pattern matching `<identifier_or_dotted_path> | tojson(`
        try! NSRegularExpression(pattern: #"([A-Za-z_][A-Za-z0-9_.]*)\s*\|\s*tojson\("#)
    }()

    /// Returns a sanitized chat template if `template` uses Python-only `tojson` kwargs
    /// (`sort_keys` or `separators`), or `nil` when no rewrite is needed.
    public static func sanitize(_ template: String) -> String? {
        guard template.contains("tojson("),
            template.contains("sort_keys") || template.contains("separators")
        else {
            return nil
        }
        if template.contains("_mlx_compact_json") {
            return template
        }

        var cursor = template.startIndex
        var output = ""
        var didRewrite = false

        while cursor < template.endIndex {
            let searchRange = NSRange(cursor ..< template.endIndex, in: template)
            guard let match = tojsonCallPattern.firstMatch(in: template, range: searchRange),
                let matchRange = Range(match.range, in: template),
                let identRange = Range(match.range(at: 1), in: template)
            else {
                output.append(contentsOf: template[cursor...])
                break
            }

            output.append(contentsOf: template[cursor ..< matchRange.lowerBound])

            // Scan forward from the opening `(` of `tojson(` for its balanced `)`.
            var idx = matchRange.upperBound
            var depth = 1
            var activeQuote: Character? = nil
            var isEscaped = false

            while idx < template.endIndex && depth > 0 {
                let ch = template[idx]
                if let q = activeQuote {
                    if isEscaped {
                        isEscaped = false
                    } else if ch == "\\" {
                        isEscaped = true
                    } else if ch == q {
                        activeQuote = nil
                    }
                } else {
                    if ch == "\"" || ch == "'" {
                        activeQuote = ch
                    } else if ch == "(" {
                        depth += 1
                    } else if ch == ")" {
                        depth -= 1
                    }
                }
                idx = template.index(after: idx)
            }

            guard depth == 0 else {
                output.append(contentsOf: template[matchRange.lowerBound...])
                break
            }

            let closeParenIndex = template.index(before: idx)
            let argsText = String(template[matchRange.upperBound ..< closeParenIndex])
            if argsText.contains("sort_keys") || argsText.contains("separators") {
                let sortKeys =
                    argsText.range(
                        of: #"sort_keys\s*=\s*(true|True)"#,
                        options: .regularExpression
                    ) != nil
                let ident = String(template[identRange])
                output.append("_mlx_compact_json(\(ident), \(sortKeys ? "true" : "false"))")
                didRewrite = true
            } else {
                output.append(contentsOf: template[matchRange.lowerBound ..< idx])
            }
            cursor = idx
        }

        guard didRewrite else { return nil }
        return compactJSONMacro + output
    }

    /// Inspects `chat_template.jinja` or `tokenizer_config.json` in `directory` and returns
    /// a sanitized chat template override only when `tojson(sort_keys=..., separators=...)` is present.
    public static func loadOverrideIfNeeded(from directory: URL) -> String? {
        let jinjaURL = directory.appendingPathComponent("chat_template.jinja")
        if let raw = try? String(contentsOf: jinjaURL, encoding: .utf8) {
            return sanitize(raw)
        }

        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: configURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        if let raw = json["chat_template"] as? String {
            return sanitize(raw)
        }
        if let entries = json["chat_template"] as? [[String: Any]] {
            if let defaultEntry = entries.first(where: { ($0["name"] as? String) == "default" }),
                let raw = defaultEntry["template"] as? String
            {
                return sanitize(raw)
            }
            if let firstRaw = entries.first?["template"] as? String {
                return sanitize(firstRaw)
            }
        }
        return nil
    }
}
