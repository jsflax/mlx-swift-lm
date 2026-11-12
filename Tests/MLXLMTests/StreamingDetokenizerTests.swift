// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

/// A byte-level stub tokenizer: each token id stands for a run of raw UTF-8 bytes, and `decode`
/// concatenates the bytes of all tokens BEFORE decoding them, exactly like swift-transformers'
/// `ByteLevelDecoder` (`String(decoding:as: UTF8.self)`, one U+FFFD per invalid/truncated subpart).
///
/// With `byteFallback`, ids at or above `byteFallbackBase` are `<0xNN>` byte tokens decoded like
/// swift-transformers' `ByteFallbackDecoder`: a run of byte tokens is decoded together, and a
/// trailing run that no normal token has closed yet renders as nothing.
private struct ByteStubTokenizer: MLXLMCommon.Tokenizer {
    static let byteFallbackBase = 1_000

    var vocabulary: [Int: [UInt8]] = [:]
    var byteFallback = false

    mutating func add(_ bytes: [UInt8]) -> Int {
        let id = vocabulary.count
        vocabulary[id] = bytes
        return id
    }

    mutating func add(_ text: String) -> Int { add(Array(text.utf8)) }

    mutating func addByteTokens(for text: String) -> [Int] {
        text.utf8.map { byte in
            let id = Self.byteFallbackBase + Int(byte)
            vocabulary[id] = [byte]
            return id
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        guard byteFallback else {
            return String(decoding: tokenIds.flatMap { vocabulary[$0] ?? [] }, as: UTF8.self)
        }
        var text = ""
        var run: [UInt8] = []
        for id in tokenIds {
            let bytes = vocabulary[id] ?? []
            if id >= Self.byteFallbackBase {
                run += bytes
            } else {
                if !run.isEmpty {
                    text += String(decoding: run, as: UTF8.self)
                    run.removeAll()
                }
                text += String(decoding: bytes, as: UTF8.self)
            }
        }
        return text  // a trailing byte run is not rendered until a normal token closes it
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// Stream `tokens` one at a time, the way `TextToolTokenLoopHandler` and the owned-render decode
/// loops do, and return every chunk the detokenizer released.
private func stream(_ tokens: [Int], through tokenizer: ByteStubTokenizer) -> [String] {
    var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
    var chunks: [String] = []
    for token in tokens {
        detokenizer.append(token: token)
        if let chunk = detokenizer.next(), !chunk.isEmpty { chunks.append(chunk) }
    }
    return chunks
}

private func expectFaithful(
    _ tokens: [Int], _ tokenizer: ByteStubTokenizer, equals expected: String,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let chunks = stream(tokens, through: tokenizer)
    #expect(chunks.joined() == expected, sourceLocation: sourceLocation)
    #expect(
        !chunks.contains { $0.unicodeScalars.contains("\u{FFFD}") },
        "a chunk carried U+FFFD: \(chunks.map(\.debugDescription))", sourceLocation: sourceLocation)
}

struct StreamingDetokenizerTests {

    /// The observed Orb-room reply: Qwen3.5 tokenizes " 👋" as " \xF0\x9F\x91" + "\x8B"
    /// (ids 59720, 233). The wave must come out whole, never as "Hey! \u{FFFD} How".
    @Test func emojiSplitAcrossTwoByteTokens() {
        var tokenizer = ByteStubTokenizer()
        let hey = tokenizer.add("Hey")
        let bang = tokenizer.add("!")
        let spaceAndLead = tokenizer.add([0x20, 0xF0, 0x9F, 0x91])
        let tail = tokenizer.add([0x8B])
        let how = tokenizer.add(" How's it going?")

        let chunks = stream([hey, bang, spaceAndLead, tail, how], through: tokenizer)

        #expect(chunks.joined() == "Hey! \u{1F44B} How's it going?")
        #expect(chunks.contains(" \u{1F44B}"))
        #expect(!chunks.contains { $0.unicodeScalars.contains("\u{FFFD}") })
    }

    /// The 16-token window reset fires on the very token that completes the emoji, and on the
    /// token after it; the reset must neither duplicate, drop nor corrupt any scalar.
    @Test(arguments: [13, 14, 15, 16])
    func emojiCompletingAtWindowReset(prefixLength: Int) {
        var tokenizer = ByteStubTokenizer()
        let filler = tokenizer.add("a")
        let spaceAndLead = tokenizer.add([0x20, 0xF0, 0x9F, 0x91])
        let tail = tokenizer.add([0x8B])
        let how = tokenizer.add(" How")
        let tokens = Array(repeating: filler, count: prefixLength) + [spaceAndLead, tail, how, filler]
        // prefixLength 14: the completing "\x8B" is exactly the 16th token of the window.
        expectFaithful(
            tokens, tokenizer,
            equals: String(repeating: "a", count: prefixLength) + " \u{1F44B} Howa")
    }

    /// Every way of cutting a 4-byte emoji into two byte tokens, at every offset around the
    /// 16-token reset (including a cut that completes exactly on the reset token).
    @Test func everyTwoTokenSplitAtEveryWindowOffset() {
        let emoji = Array("\u{1F44B}".utf8)
        for cut in 1 ..< emoji.count {
            for prefixLength in 0 ... 20 {
                var tokenizer = ByteStubTokenizer()
                let filler = tokenizer.add("x")
                let head = tokenizer.add(Array(emoji[..<cut]))
                let rest = tokenizer.add(Array(emoji[cut...]))
                let after = tokenizer.add("!")
                let tokens = Array(repeating: filler, count: prefixLength) + [head, rest, after]
                expectFaithful(
                    tokens, tokenizer,
                    equals: String(repeating: "x", count: prefixLength) + "\u{1F44B}!")
            }
        }
    }

    /// Tokens that EXTEND the previous grapheme cluster (ZWJ sequences, flags, skin tones,
    /// variation selectors, combining marks) leave `String.count` unchanged; they must still stream.
    @Test(arguments: [
        ["\u{1F468}", "\u{200D}", "\u{1F469}", "\u{200D}", "\u{1F467}"],  // family (ZWJ)
        ["\u{1F1FA}", "\u{1F1F8}"],  // flag: two regional indicators
        ["\u{1F44D}", "\u{1F3FB}"],  // thumbs up + skin tone
        ["\u{2764}", "\u{FE0F}"],  // heart + VS16
        ["e", "\u{0301}"],  // combining acute
    ])
    func clusterExtendingTokens(pieces: [String]) {
        for prefixLength in [0, 13, 14, 15, 16] {
            var tokenizer = ByteStubTokenizer()
            let filler = tokenizer.add("a")
            let tail = tokenizer.add(" ok")
            let tokens =
                Array(repeating: filler, count: prefixLength) + pieces.map { tokenizer.add($0) }
                + [tail]
            expectFaithful(
                tokens, tokenizer,
                equals: String(repeating: "a", count: prefixLength) + pieces.joined() + " ok")
        }
    }

    /// `<0xNN>` byte-fallback tokens (Llama/Gemma-style vocabularies): a byte run straddling the
    /// window reset must not orphan its leading bytes.
    @Test(arguments: Array(10 ... 17))
    func byteFallbackRunAcrossWindowReset(prefixLength: Int) {
        var tokenizer = ByteStubTokenizer(byteFallback: true)
        let filler = tokenizer.add("a")
        let how = tokenizer.add(" How")
        let tokens =
            Array(repeating: filler, count: prefixLength) + tokenizer.addByteTokens(for: "\u{1F44B}")
            + [how]
        expectFaithful(
            tokens, tokenizer,
            equals: String(repeating: "a", count: prefixLength) + "\u{1F44B} How")
    }

    /// A partial sequence is held, not emitted, while it is still incomplete.
    @Test func partialSequenceIsHeldUntilComplete() {
        var tokenizer = ByteStubTokenizer()
        let lead = tokenizer.add([0xF0, 0x9F])
        let middle = tokenizer.add([0x91])
        let last = tokenizer.add([0x8B])
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        detokenizer.append(token: lead)
        #expect(detokenizer.next() == nil)
        detokenizer.append(token: middle)
        #expect(detokenizer.next() == nil)
        detokenizer.append(token: last)
        #expect(detokenizer.next() == "\u{1F44B}")
    }
}
