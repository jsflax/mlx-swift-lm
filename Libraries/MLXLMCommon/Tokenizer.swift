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
