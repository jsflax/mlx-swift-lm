// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXOptimizers
import XCTest

@testable import MLXLMCommon

/// See also ChatSessionIntegrationTests
public class ChatSessionTests: XCTestCase {

    private struct RecordedMessage: Equatable, Sendable {
        var role: Chat.Message.Role
        var content: String
    }

    private struct RecordingMessageGenerator: MessageGenerator {
        let continuation: AsyncStream<[RecordedMessage]>.Continuation

        func generate(messages: [Chat.Message]) -> [Message] {
            continuation.yield(messages.map { .init(role: $0.role, content: $0.content) })

            return DefaultMessageGenerator().generate(messages: messages)
        }
    }

    private func model(processor: TestInputProcessor = TestInputProcessor()) -> ModelContext {
        let config = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64, attentionHeads: 4,
            headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        let model = Gemma3TextModel(config)
        quantize(model: model, groupSize: 64, bits: 4)

        // Force evaluation of all model weights before concurrent usage
        // This ensures all weight promises are realized and avoids race conditions
        eval(model)

        return .init(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer)
    }

    private let generationParameters = GenerateParameters(maxTokens: 50)

    private let targetLength = 1

    func testChatSessionSync() async throws {
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        let result1 = try await session.respond(to: "hello")
        XCTAssertGreaterThan(result1.count, targetLength, result1)
        let result2 = try await session.respond(to: "hello again")
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionAsync() async throws {
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        var result1 = ""
        for try await part in session.streamResponse(to: "hello") {
            result1 += part
        }
        XCTAssertGreaterThan(result1.count, targetLength, result1)

        var result2 = ""
        for try await part in session.streamResponse(to: "hello again") {
            result2 += part
        }
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionRespondToMessages() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)

        let result = try await session.respond(to: [
            .user("hello"),
            .assistant("hi"),
            .user("hello again"),
        ])
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testChatSessionStreamResponseToMessages() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)

        var result = ""
        for try await part in session.streamResponse(to: [
            .user("hello"),
            .assistant("hi"),
            .user("hello again"),
        ]) {
            result += part
        }
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testStructuredContinuationAvoidsReplayingHistoryAcrossToolTurns() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: TestTokenizer(),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let history: [Chat.Message] = (0 ..< 8).flatMap { index in
            [
                .user("question \(index)"),
                .assistant("answer \(index)"),
            ]
        }
        let continuations: [[Chat.Message]] = [
            [.tool("first tool result")],
            [.tool("second tool result")],
            [.user("final answer")],
        ]
        let session = ChatSession(
            model(processor: processor),
            history: history,
            generateParameters: GenerateParameters(maxTokens: 1))

        for messages in continuations {
            _ = try await session.respond(to: messages)
        }
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages {
            calls.append(call)
        }

        XCTAssertEqual(calls.map(\.count), [history.count + 1, 1, 1])
        XCTAssertEqual(calls[0].map(\.role), history.map(\.role) + [.tool])
        XCTAssertEqual(calls[1].map(\.role), [.tool])
        XCTAssertEqual(calls[2].map(\.role), [.user])

        let actualPreparedMessageCount = calls.reduce(0) { $0 + $1.count }
        let replayedHistoryPreparedMessageCount = continuations.indices.reduce(0) {
            $0 + history.count + $1 + 1
        }
        XCTAssertLessThan(actualPreparedMessageCount, replayedHistoryPreparedMessageCount)
    }

    func testChatSessionAsyncInterrupt() async throws {
        // interrupt the streamResponse and continue with another request
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        for _ in 0 ..< 10 {
            var result1 = ""
            for try await part in session.streamResponse(to: "hello") {
                result1 += part
                break
            }

            // at this point the performStreaming/generate code may still be running.
            // the next call can corrupt the state if not thread safe

            var result2 = ""
            for try await part in session.streamResponse(to: "hello again") {
                result2 += part
                if result2.count > 100 {
                    break
                }
            }
        }

        // since we are interrupting we need to wait for everything to finish
        // (avoids shutdown issues if this is the last/only test). because the
        // streaming task is not a synchronous shutdown
        await session.synchronize()
    }

    func testCancelledPreparationWithoutAdmissionDrainsBeforeCompletionAndCacheReadmission()
        async throws
    {
        let drainOrCompletion = expectation(description: "drain entered or producer completed")
        drainOrCompletion.assertForOverFulfill = false
        let readerRequested = expectation(description: "cache reader requested access")
        let probe = PreparationDrainProbe(
            drainOrCompletion: drainOrCompletion, readerRequested: readerRequested)
        let session = preparationDrainSession(probe: probe, cancelDuringPreparation: true)
        XCTAssertNil(session.generationAdmission)
        session.synchronizePreparedWork = { probe.drain(hold: true) }
        defer {
            probe.releaseDrain()
            // A deliberately broken old-guard control skips the observed drain.
            // Clean up its real queued work after the owned tasks have joined,
            // outside the event oracle even when a later XCTUnwrap throws.
            Stream().synchronize()
        }

        let producer = session.streamDetailsWithTask(to: [.user("cancel during preparation")])
        let completion = Task {
            await producer.task.value
            // The old nil-hook guard never enters the drain. Wake that control
            // through producer completion instead of hanging on a missing event.
            probe.producerJoined()
        }
        await fulfillment(of: [drainOrCompletion], timeout: 5)

        let readerSession = CacheReaderSession(session)
        let reader = Task {
            probe.requestCacheRead()
            await readerSession.session.withCache { _ in
                probe.record(.cacheReadmitted)
            }
        }
        await fulfillment(of: [readerRequested], timeout: 5)
        probe.releaseDrain()
        await completion.value
        await reader.value

        var eventCount = 0
        do {
            for try await _ in producer.stream { eventCount += 1 }
            XCTFail("Cancellation after preparation must finish with CancellationError")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
        XCTAssertEqual(eventCount, 0, "No generation producer should start after cancellation")
        let events = probe.events
        XCTAssertEqual(events.filter { $0 == .prepared }.count, 1)
        XCTAssertFalse(events.contains(.generated))
        XCTAssertFalse(events.contains(.drainTimedOut), "The test drain gate must be released")
        XCTAssertEqual(events.filter { $0 == .drainEntered }.count, 1)
        XCTAssertEqual(events.filter { $0 == .drained }.count, 1)
        let drained = try XCTUnwrap(events.firstIndex(of: .drained))
        XCTAssertLessThan(drained, try XCTUnwrap(events.firstIndex(of: .producerJoined)))
        XCTAssertLessThan(drained, try XCTUnwrap(events.firstIndex(of: .cacheReadmitted)))
        XCTAssertLessThan(try XCTUnwrap(events.firstIndex(of: .prepared)), drained)
        // The request overlaps the held drain; this witness does not claim to
        // inspect the private mutex's waiter queue.
        XCTAssertLessThan(try XCTUnwrap(events.firstIndex(of: .cacheReadRequested)), drained)
    }

    func testSuccessfulPreparationPreservesOptionalAdmissionAndGeneratorJoining() async throws {
        for installAdmission in [false, true] {
            let probe = PreparationDrainProbe()
            let session = preparationDrainSession(probe: probe, cancelDuringPreparation: false)
            session.synchronizePreparedWork = { probe.drain(hold: false) }
            if installAdmission {
                session.generationAdmission = .init(
                    validate: { probe.record(.validated) },
                    start: { create in
                        probe.record(.started)
                        try create()
                    })
            }

            let producer = session.streamDetailsWithTask(to: [.user("one token")])
            var receivedCompletion = false
            for try await event in producer.stream {
                if case .info = event { receivedCompletion = true }
            }
            await producer.task.value
            probe.record(.producerJoined)
            await session.withCache { cache in
                XCTAssertNotNil(cache)
                probe.record(.cacheReadmitted)
            }

            let events = probe.events
            XCTAssertTrue(receivedCompletion)
            XCTAssertEqual(events.filter { $0 == .prepared }.count, 1)
            XCTAssertEqual(events.filter { $0 == .generated }.count, 1)
            XCTAssertEqual(events.filter { $0 == .validated }.count, installAdmission ? 1 : 0)
            XCTAssertEqual(events.filter { $0 == .started }.count, installAdmission ? 1 : 0)
            XCTAssertEqual(events.filter { $0 == .drained }.count, installAdmission ? 1 : 0,
                "A joined no-hook generator already drained its work")
            let generated = try XCTUnwrap(events.firstIndex(of: .generated))
            let joined = try XCTUnwrap(events.firstIndex(of: .producerJoined))
            XCTAssertLessThan(generated, joined)
            XCTAssertLessThan(joined, try XCTUnwrap(events.firstIndex(of: .cacheReadmitted)))
            if installAdmission {
                XCTAssertLessThan(try XCTUnwrap(events.firstIndex(of: .validated)),
                    try XCTUnwrap(events.firstIndex(of: .started)))
                XCTAssertLessThan(try XCTUnwrap(events.firstIndex(of: .started)), generated)
                let drained = try XCTUnwrap(events.firstIndex(of: .drained))
                XCTAssertLessThan(generated, drained)
                XCTAssertLessThan(drained, joined)
            }
        }
    }

    private func preparationDrainSession(
        probe: PreparationDrainProbe, cancelDuringPreparation: Bool
    ) -> ChatSession {
        let processor = TestInputProcessor()
        let context = ModelContext(
            configuration: processor.configuration,
            model: PreparationCancellationModel(
                probe: probe, cancelDuringPreparation: cancelDuringPreparation),
            processor: processor, tokenizer: processor.tokenizer)
        return ChatSession(context, generateParameters: .init(maxTokens: 1, temperature: 0))
    }

    private final class PreparationCancellationModel: Module, LanguageModel, KVCacheDimensionProvider {
        let probe: PreparationDrainProbe
        let cancelDuringPreparation: Bool
        var kvHeads: [Int] { [1] }

        init(probe: PreparationDrainProbe, cancelDuringPreparation: Bool) {
            self.probe = probe
            self.cancelDuringPreparation = cancelDuringPreparation
            super.init()
        }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            probe.record(.prepared)
            if cancelDuringPreparation {
                withUnsafeCurrentTask { $0?.cancel() }
            }
            // Preparation returns successfully. The real TokenIterator converts
            // these logits and queues asyncEval before it checks cancellation.
            return .logits(.init(logits: MLXArray.zeros([1, 1, 100])))
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            probe.record(.generated)
            return MLXArray.zeros([1, 1, 100])
        }
    }

    private final class PreparationDrainProbe: @unchecked Sendable {
        enum Event: Equatable, Sendable {
            case prepared, validated, started, generated
            case drainEntered, drainTimedOut, drained, producerJoined
            case cacheReadRequested, cacheReadmitted
        }
        private let lock = NSLock()
        private var storedEvents: [Event] = []
        private let drainPermit = DispatchSemaphore(value: 0)
        private let drainOrCompletion: XCTestExpectation?
        private let readerRequested: XCTestExpectation?

        init(
            drainOrCompletion: XCTestExpectation? = nil,
            readerRequested: XCTestExpectation? = nil
        ) {
            self.drainOrCompletion = drainOrCompletion
            self.readerRequested = readerRequested
        }

        func producerJoined() {
            record(.producerJoined)
            drainOrCompletion?.fulfill()
        }

        func requestCacheRead() {
            record(.cacheReadRequested)
            readerRequested?.fulfill()
        }

        var events: [Event] { lock.withLock { storedEvents } }
        func record(_ event: Event) { lock.withLock { storedEvents.append(event) } }
        func releaseDrain() { drainPermit.signal() }

        func drain(hold: Bool) {
            record(.drainEntered)
            drainOrCompletion?.fulfill()
            // Two five-second handshakes plus cleanup grace; timeout is a test
            // failure, never an unbounded blocked Swift worker.
            if hold, drainPermit.wait(timeout: .now() + 15) == .timedOut {
                record(.drainTimedOut)
            }
            // This replaces only the call boundary, not the actual MLX fence.
            Stream().synchronize()
            record(.drained)
        }
    }

    /// Configuration is frozen before producer start. This holder permits only
    /// the existing internally serialized withCache read from a second task.
    private final class CacheReaderSession: @unchecked Sendable {
        let session: ChatSession
        init(_ session: ChatSession) { self.session = session }
    }

    func testChatSessionWithTools() async throws {
        let model = model()
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": [
                                "type": "string",
                                "description": "City name",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["location"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model, generateParameters: generationParameters, tools: tools
        )

        let result = try await session.respond(to: "What is the weather in SF?")
        XCTAssertGreaterThan(result.count, targetLength, result)

        // second turn to verify tools persist through cache
        let result2 = try await session.respond(to: "How about NYC?")
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionWithToolsStreaming() async throws {
        let model = model()
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model, generateParameters: generationParameters, tools: tools
        )

        var result = ""
        for try await part in session.streamResponse(to: "hello") {
            result += part
        }
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    // MARK: - KV Cache

    func testCurrentCacheNilBeforeGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    func testCurrentCacheAfterGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
    }

    func testInitWithKVCache() async throws {
        // build a cache from an initial session
        let container = ModelContainer(context: model())
        let initial = ChatSession(container, generateParameters: generationParameters)
        _ = try await initial.respond(to: "hello")

        try await initial.withCache { [targetLength, generationParameters] cache in
            XCTAssertNotNil(cache)

            if let cache {
                // restore the cache into a new session and verify generation continues
                let restored = ChatSession(
                    container,
                    cache: cache.map { $0.copy() },
                    generateParameters: generationParameters)
                let result = try await restored.respond(to: "hello again")
                XCTAssertGreaterThan(result.count, targetLength, result)
            }
        }
    }

    func testSaveCacheThrowsBeforeGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        do {
            try await session.saveCache(to: url)
            XCTFail("expected ChatSessionError.noCacheAvailable")
        } catch ChatSessionError.noCacheAvailable {
            // expected
        }
    }

    func testSaveAndRestoreCache() async throws {
        let ctx = model()
        let initial = ChatSession(ctx, generateParameters: generationParameters)
        _ = try await initial.respond(to: "hello")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        try await initial.saveCache(to: url)

        let (loadedCache, _) = try loadPromptCache(url: url)
        let restored = ChatSession(
            ctx, cache: loadedCache, generateParameters: generationParameters)
        let result = try await restored.respond(to: "hello again")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testCurrentCacheNilForHistorySessionBeforeGeneration() async throws {
        // .history state should behave like .empty: no cache until first generation
        let history: [Chat.Message] = [.user("hello"), .assistant("hi")]
        let session = ChatSession(
            model(), history: history, generateParameters: generationParameters)
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    func testCurrentCacheNonNilForHistorySessionAfterGeneration() async throws {
        // after generation from .history state, cache transitions to .kvcache
        let history: [Chat.Message] = [.user("hello"), .assistant("hi")]
        let session = ChatSession(
            model(),
            history: history,
            generateParameters: generationParameters)
        _ = try await session.respond(to: "hello again")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
    }

    func testCurrentCacheNilAfterClear() async throws {
        // clear() resets to .empty; currentCache() should return nil again
        let session = ChatSession(model(), generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
        await session.clear()
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    /// Uses a tiny synthetic model only to establish an occupied cache. The
    /// hypothetical admission and rejected real preparation perform no forward.
    func testDeferredSchemaPreparationCountsActualCacheAndMatchesRealAdmission() async throws {
        let processor = TestInputProcessor(tokenizer: CatalogCountingTokenizer(),
            configuration: ModelConfiguration(id: "test"), messageGenerator: DefaultMessageGenerator())
        let oldTools: [ToolSpec] = [["type": "function", "function": ["name": "discover"] as ToolSpec]]
        let newTools = oldTools + [["type": "function", "function": ["name": "read_record"] as ToolSpec]]
        let session = ChatSession(model(processor: processor), instructions: "Current instructions",
                                  generateParameters: GenerateParameters(maxTokens: 1), tools: oldTools)
        _ = try await session.respond(to: "Read a record")
        let replies: [Chat.Message] = [.tool("Actual preceding reply"), .tool("Selected read_record"), .tool("Pending current tool result")]
        let old = try await session.prepareContinuationContext(to: replies, tools: oldTools)
        let proposed = try await session.prepareContinuationContext(to: replies, tools: newTools)
        XCTAssertGreaterThan(try XCTUnwrap(proposed.cachedTokenCount), 0)
        XCTAssertEqual(old.cachedTokenCount, proposed.cachedTokenCount)
        XCTAssertEqual(proposed.inputTokenCount - old.inputTokenCount, 5)
        XCTAssertEqual(try proposed.promptTokenCount(), try XCTUnwrap(proposed.cachedTokenCount) + proposed.inputTokenCount)
        XCTAssertEqual(session.tools?.count, 1, "A proposal cannot publish tool changes")
        let actual = PreparedContextRecorder()
        session.tools = newTools
        session.contextAdmission = { value in actual.record(value); throw ProvisionalAdmissionRefused() }
        do {
            _ = try await session.respond(to: replies)
            XCTFail("Admission must fail before generation")
        } catch is ProvisionalAdmissionRefused { }
        let measured = try XCTUnwrap(actual.value)
        XCTAssertEqual(measured.cachedTokenCount, proposed.cachedTokenCount)
        XCTAssertEqual(measured.inputTokenCount, proposed.inputTokenCount)
        let retry = try await session.prepareContinuationContext(to: replies, tools: newTools)
        XCTAssertEqual(try retry.promptTokenCount(), try proposed.promptTokenCount(), "Failed preparation must preserve cache position")
    }

    func testDeferredSchemaPreparationKeepsRestoredHistoryAndPendingReplyOrder() async throws {
        let (recorded, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(tokenizer: CatalogCountingTokenizer(), configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let session = ChatSession(model(processor: processor), instructions: "Keep instructions",
            history: [.user("Protected user"), .assistant("Discovery call")], generateParameters: GenerateParameters(maxTokens: 1))
        let prepared = try await session.prepareContinuationContext(
            to: [.tool("Denied earlier call"), .tool("Discovery reply"), .tool("Pending current tool result")], tools: [])
        continuation.finish()
        var calls: [[RecordedMessage]] = []
        for await call in recorded { calls.append(call) }
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].map(\.content), ["Keep instructions", "Protected user", "Discovery call",
            "Denied earlier call", "Discovery reply", "Pending current tool result"])
        XCTAssertEqual(prepared.cachedTokenCount, 0)
        await session.withCache { XCTAssertNil($0, "Read-only preparation cannot realize a cache") }
    }

    private struct ProvisionalAdmissionRefused: Error {}
    private final class PreparedContextRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: ChatSessionPreparedContext?
        var value: ChatSessionPreparedContext? { lock.withLock { stored } }
        func record(_ value: ChatSessionPreparedContext) { lock.withLock { stored = value } }
    }
    private struct CatalogCountingTokenizer: MLXLMCommon.Tokenizer {
        let base = TestTokenizer()
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { base.encode(text: text, addSpecialTokens: addSpecialTokens) }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens) }
        func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }
        func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var eosTokenId: Int? { base.eosTokenId }
        var unknownToken: String? { nil }
        var unknownTokenId: Int? { base.unknownTokenId }
        func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                               additionalContext: [String: any Sendable]?) throws -> [Int] {
            Array(repeating: 1, count: 8 + messages.count * 2 + (tools?.count ?? 0) * 5)
        }
    }

    /// something that looks like a view model
    @MainActor class ChatModel {
        let session: ChatSession

        public var messages = [Chat.Message]()

        private var task: Task<Void, Error>?
        public var isBusy: Bool {
            task != nil
        }

        init(model: ModelContext) {
            self.session = ChatSession(model)
        }

        public func cancel() {
            task?.cancel()
        }

        public func respond(_ message: String) {
            guard task == nil else { return }

            self.messages.append(.init(role: .user, content: message))
            self.messages.append(.init(role: .assistant, content: "..."))
            let lastIndex = self.messages.count - 1

            self.task = Task {
                var first = true
                for try await item in session.streamResponse(to: message) {
                    if first {
                        self.messages[lastIndex].content = item
                        first = false
                    } else {
                        self.messages[lastIndex].content += item
                    }
                }
                self.task = nil
            }
        }
    }

    @MainActor
    func testViewModel() async throws {
        let model = ChatModel(model: model())

        // start producing a response but interrupt it
        // triggers https://github.com/ml-explore/mlx-swift/pull/323
        model.respond("message1")
        try await Task.sleep(for: .milliseconds(50))
        model.cancel()

        // wait for it to finish
        while model.isBusy {
            try await Task.sleep(for: .milliseconds(10))
        }

        // try another message, wait for full completion (but cap the length)
        model.session.generateParameters = self.generationParameters
        model.respond("message2")
        while model.isBusy {
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}
