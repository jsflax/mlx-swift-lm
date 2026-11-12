import Foundation

/// Counts at the actual prepared-input boundary, before model/cache work.
public struct ChatSessionPreparedContext: Sendable {
    public let cachedTokenCount: Int?
    public let inputTokenCount: Int
    public let requestedOutputTokens: Int?

    public init(cachedTokenCount: Int?, inputTokenCount: Int, requestedOutputTokens: Int?) {
        self.cachedTokenCount = cachedTokenCount
        self.inputTokenCount = inputTokenCount
        self.requestedOutputTokens = requestedOutputTokens
    }

    /// Exact logical position after the joined, non-speculative iterator.
    func position(afterGeneratedTokens count: Int) -> Int? {
        guard count >= 0, let prompt = try? promptTokenCount() else { return nil }
        let (total, overflow) = prompt.addingReportingOverflow(count)
        return overflow ? nil : total
    }

    public func promptTokenCount() throws -> Int {
        guard let cachedTokenCount, cachedTokenCount >= 0, inputTokenCount > 0 else {
            throw ChatSessionContextError.unknownCachePosition
        }
        let (total, overflow) = cachedTokenCount.addingReportingOverflow(inputTokenCount)
        guard !overflow else { throw ChatSessionContextError.unknownCachePosition }
        return total
    }
}

public enum ChatSessionContextError: Error, Sendable, Equatable {
    case unknownCachePosition
    case invalidOutputAllowance
    case noRoom(promptTokens: Int, contextWindow: Int)
    case unsupportedSpeculativeAdmission
}

/// Logical consumption, not resident KV length. Hybrid/recurrent caches need not
/// expose a meaningful offset. Read only after the exact generator task joins.
final class ChatSessionTokenCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var value: Int { lock.withLock { count } }
    func record(_ count: Int) { lock.withLock { self.count = max(self.count, count) } }
}

/// The raw iterator count survives tool parsing, detokenization and EOS filtering.
struct ChatSessionCountingIterator<Base: TokenIteratorProtocol>: TokenIteratorProtocol {
    private var base: Base
    let counter: ChatSessionTokenCounter
    init(_ base: Base, counter: ChatSessionTokenCounter) { self.base = base; self.counter = counter }
    var maxTokens: Int? { base.maxTokens }
    var tokenCount: Int { base.tokenCount }
    var promptPrefillTime: TimeInterval { base.promptPrefillTime }
    mutating func next() -> Int? {
        let result = base.next()
        counter.record(base.tokenCount)
        return result
    }
}
