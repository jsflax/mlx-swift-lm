// Copyright © 2025 Apple Inc.

import CoreGraphics
import Foundation
import MLX

/// Configuration for speculative decoding in a `ChatSession`.
///
/// Speculative decoding uses a small draft model to propose candidate tokens
/// that the main model then verifies in a single forward pass, providing a
/// ~2–3× generation speedup with no quality degradation.
///
/// Both models must share the same tokenizer vocabulary.
///
/// Example usage:
/// ```swift
/// let main  = try await LLMModelFactory.shared.loadContainer(configuration: mainConfig)
/// let draft = try await LLMModelFactory.shared.loadContainer(configuration: draftConfig)
///
/// let session = ChatSession(
///     main,
///     speculativeDecoding: SpeculativeDecodingConfig(draftModel: draft, numDraftTokens: 5)
/// )
/// ```
public struct SpeculativeDecodingConfig: Sendable {

    /// The lightweight model used to propose candidate tokens.
    public let draftModel: ModelContainer

    /// Number of tokens proposed by the draft model per verification cycle.
    /// The default value of 5 offers a good balance between speed and accuracy.
    public let numDraftTokens: Int

    public init(draftModel: ModelContainer, numDraftTokens: Int = 5) {
        self.draftModel = draftModel
        self.numDraftTokens = numDraftTokens
    }
}

/// Simplified API for multi-turn conversations with LLMs and VLMs.
///
/// For example:
///
/// ```swift
/// let modelContainer = try await loadModelContainer(id: "mlx-community/Qwen3-4B-4bit")
/// let session = ChatSession(modelContainer)
/// print(try await session.respond(to: "What are two things to see in San Francisco?"))
/// print(try await session.respond(to: "How about a great place to eat?"))
/// ```
///
/// To enable speculative decoding for faster generation, pass a `SpeculativeDecodingConfig`:
///
/// ```swift
/// let draft = try await LLMModelFactory.shared.loadContainer(configuration: draftConfig)
/// let session = ChatSession(
///     modelContainer,
///     speculativeDecoding: SpeculativeDecodingConfig(draftModel: draft)
/// )
/// ```
///
/// - Note: `ChatSession` is not thread-safe. Each session should be used from a single
///   task/thread at a time. The underlying `ModelContainer` handles thread safety for
///   model operations.
public final class ChatSession {

    enum Cache {
        case empty
        case kvcache([KVCache], draftKVCache: [KVCache]?, logicalTokenCount: Int? = nil, exclusivelyOwned: Bool = false)
        case history([Chat.Message])
    }

    private let model: ModelContainer
    public var instructions: String?
    private let cache: SerialAccessContainer<Cache>
    public var processing: UserInput.Processing
    public var generateParameters: GenerateParameters
    /// Opt-in host admission over the exact prepared input and tracked logical
    /// cache position. Return this round's output allowance; throwing performs no
    /// model forward and does not replace an empty/restored cache.
    public var contextAdmission: (@Sendable (ChatSessionPreparedContext) throws -> Int)?
    /// Optional asynchronous freshness check and synchronous producer fence.
    /// Called after tokenizer/iterator preparation, immediately around generateTask.
    public var generationAdmission: ChatSessionGenerationAdmission?
    public var additionalContext: [String: any Sendable]?
    public var tools: [ToolSpec]?
    public var toolDispatch: (@Sendable (ToolCall) async throws -> String)?

    /// Speculative decoding configuration, nil if disabled.
    public let speculativeDecoding: SpeculativeDecodingConfig?

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.empty)
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.empty)
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Initialize the `ChatSession` with an existing message history.
    ///
    /// This enables "Prompt Re-hydration" for persistent chat applications.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - history: The full array of messages to restore (including system prompt)
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        history: consuming [Chat.Message],
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.history(history))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Initialize the `ChatSession` with an existing message history.
    ///
    /// This enables "Prompt Re-hydration" for persistent chat applications.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session
    ///   - history: The full array of messages to restore (including system prompt)
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        history: [Chat.Message],
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.history(history))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Initialize the `ChatSession` with a pre-built KV cache.
    ///
    /// This enables prefix caching: build a KV cache from a long shared context (e.g. a
    /// system prompt and document) once, save it via ``saveCache(to:)``, and restore it
    /// across multiple sessions to avoid re-prefilling the same tokens each time.
    ///
    /// > Important: If the cache was built from a session that already included system
    /// > instructions, do not pass the same `instructions` here — they would be
    /// > re-tokenized on each call to ``respond(to:role:images:videos:audios:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]` previously loaded with ``loadPromptCache(url:)``,
    ///     matching the given model
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        cache: consuming [KVCache],
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.kvcache(cache, draftKVCache: nil))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Initialize the `ChatSession` with a pre-built KV cache.
    ///
    /// This enables prefix caching: build a KV cache from a long shared context (e.g. a
    /// system prompt and document) once, save it via ``saveCache(to:)``, and restore it
    /// across multiple sessions to avoid re-prefilling the same tokens each time.
    ///
    /// > Important: If the cache was built from a session that already included system
    /// > instructions, do not pass the same `instructions` here — they would be
    /// > re-tokenized on each call to ``respond(to:role:images:videos:audios:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]` previously loaded with ``loadPromptCache(url:)``,
    ///     matching the given model
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        cache: consuming [KVCache],
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.kvcache(cache, draftKVCache: nil))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video],
        audios: consuming [UserInput.Audio]
    ) async throws -> String {
        try await respond(to: [
            .init(role: role, content: prompt, images: images, videos: videos, audios: audios)
        ])
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    ///   - audio: optional audio (for use with VLMs)
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        role: Chat.Message.Role = .user,
        image: consuming UserInput.Image? = nil,
        video: consuming UserInput.Video? = nil,
        audio: consuming UserInput.Audio? = nil
    ) async throws -> String {
        try await respond(
            to: prompt,
            role: role,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? [],
            audios: audio.map { [$0] } ?? []
        )
    }

    /// Produces a response after appending a batch of structured chat messages.
    ///
    /// Use this to continue an existing session with non-user roles, such as one
    /// or more tool results, while preserving the session's KV cache.
    ///
    /// - Important: Initializing a new session from history must prefill that
    ///   history once. Reuse the same session with this method for subsequent
    ///   tool or agent turns to avoid repeatedly pre-filling the accumulated
    ///   transcript.
    ///
    /// - Parameter messages: chat messages to append before generation
    /// - Returns: the model's response
    public func respond(
        to messages: consuming [Chat.Message]
    ) async throws -> String {
        try Task.checkCancellation()
        let producer = streamMapWithTask(messages: messages) { $0.chunk }
        var output = ""
        do {
            for try await chunk in producer.stream {
                try Task.checkCancellation()
                output += chunk
            }
        } catch {
            producer.task.cancel()
            await producer.task.value
            throw error
        }
        if Task.isCancelled { producer.task.cancel() }
        await producer.task.value
        try Task.checkCancellation()
        return output
    }

    /// Produces a streaming response to a prompt as Strings.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image] = [],
        videos: consuming [UserInput.Video] = [],
        audios: consuming [UserInput.Audio] = []
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(to: prompt, role: role, images: images, videos: videos, audios: audios) {
            $0.chunk
        }
    }

    /// Produces a streaming response after appending a batch of structured chat messages.
    ///
    /// Use this to continue an existing session with non-user roles, such as one
    /// or more tool results, while preserving the session's KV cache.
    ///
    /// - Parameter messages: chat messages to append before generation
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to messages: consuming [Chat.Message]
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(messages: messages) {
            $0.chunk
        }
    }

    /// Produces a streaming response to a prompt as `Generation`.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: a stream of `Generation` from the model
    public func streamDetails(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image] = [],
        videos: consuming [UserInput.Video] = [],
        audios: consuming [UserInput.Audio] = [],
    ) -> AsyncThrowingStream<Generation, Error> {
        streamMap(to: prompt, role: role, images: images, videos: videos, audios: audios) {
            $0
        }
    }

    /// Produces a streaming response after appending a batch of structured chat messages as `Generation`.
    ///
    /// Use this to continue an existing session with non-user roles, such as one
    /// or more tool results, while preserving the session's KV cache.
    ///
    /// - Parameter messages: chat messages to append before generation
    /// - Returns: a stream of `Generation` from the model
    public func streamDetails(
        to messages: consuming [Chat.Message]
    ) -> AsyncThrowingStream<Generation, Error> {
        streamMap(messages: messages) {
            $0
        }
    }

    /// Streaming generation with the exact session producer task. Cancel and await this
    /// task when abandoning the stream; it joins generation and releases the cache owner.
    public func streamDetailsWithTask(
        to messages: consuming [Chat.Message]
    ) -> (stream: AsyncThrowingStream<Generation, Error>, task: Task<Void, Never>) {
        streamMapWithTask(messages: messages) { $0 }
    }

    /// Produces a streaming response to a prompt by transforming the
    /// raw `Generation` values.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: a stream of transformed values from the model
    private func streamMap<R: Sendable>(
        to prompt: String,
        role: Chat.Message.Role,
        images: consuming [UserInput.Image] = [],
        videos: consuming [UserInput.Video] = [],
        audios: consuming [UserInput.Audio] = [],
        transform: @Sendable @escaping (Generation) -> R?
    ) -> AsyncThrowingStream<R, Error> {
        streamMap(
            messages: [
                .init(role: role, content: prompt, images: images, videos: videos, audios: audios)
            ],
            transform: transform
        )
    }

    private func streamMap<R: Sendable>(
        messages: consuming [Chat.Message],
        transform: @Sendable @escaping (Generation) -> R?
    ) -> AsyncThrowingStream<R, Error> {
        streamMapWithTask(messages: messages, transform: transform).stream
    }

    private func streamMapWithTask<R: Sendable>(
        messages: consuming [Chat.Message],
        transform: @Sendable @escaping (Generation) -> R?
    ) -> (stream: AsyncThrowingStream<R, Error>, task: Task<Void, Never>) {
        let (stream, continuation) = AsyncThrowingStream<R, Error>.makeStream()

        // images and videos are not Sendable (MLXArray) but they are consumed
        // and are only being sent to the inner async
        let inputMessages = SendableBox<[Chat.Message]>(messages)

        let task = Task {
            [
                model,
                instructions, processing, tools, toolDispatch,
                additionalContext, cache, generateParameters, speculativeDecoding, contextAdmission, generationAdmission
            ] in
            do {
                try await cache.update { cache in
                    // Iterator preparation may enqueue work before the host's
                    // final admission refuses/cancels. Keep cache custody until
                    // that work joins, including when no genTask was created.
                    defer { if generationAdmission != nil { Stream().synchronize() } }

                    // these are all Sendable
                    let processor = await model.processor
                    let tokenizer = await model.tokenizer
                    let modelConfiguration = await model.configuration

                    var messages: [Chat.Message] = []
                    if let instructions {
                        messages.append(.system(instructions))
                    }

                    // prepare the cache, if needed.  note:
                    // this is using the LanguageModel (not Sendable) outside
                    // the protective lock.  Assuming the weights are not
                    // being mutated behind the scenes, this will obey the MLXArray
                    // contract that they be evaluated if used across threads.
                    // This is internal to the implementation and this technique
                    // should not be used in calling code.
                    //
                    // The benefit is that callers can be running multiple
                    // ChatSessions in parallel, as long as the instances
                    // are distinct.  In particular the KVCache cannot
                    // be shared and that is the lock that is held here.

                    let model = await model.perform { context in
                        SendableBox(context.model)
                    }.consume()

                    var kvCache: [KVCache]
                    var draftKVCache: [KVCache]?
                    var logicalTokenCount: Int?
                    let exclusivelyOwned: Bool
                    switch cache {
                    case .empty:
                        kvCache = model.newCache(parameters: generateParameters)
                        draftKVCache = nil
                        logicalTokenCount = 0
                        exclusivelyOwned = true

                    case .kvcache(let array, let storedDraftCache, let storedTokenCount, let owned):
                        kvCache = array
                        draftKVCache = storedDraftCache
                        logicalTokenCount = storedTokenCount
                        exclusivelyOwned = owned

                    case .history(let history):
                        // the KVCache is represented by a chat history
                        kvCache = model.newCache(parameters: generateParameters)
                        draftKVCache = nil
                        logicalTokenCount = 0
                        exclusivelyOwned = true
                        messages.append(contentsOf: history)
                    }

                    // prepare the input
                    messages.append(contentsOf: inputMessages.consume())

                    // loop can restart on tool calls
                    restart: while !messages.isEmpty {
                        let userInput = UserInput(
                            chat: messages,
                            processing: processing,
                            tools: tools, additionalContext: additionalContext)
                        let input = try await processor.prepare(input: userInput)
                        try Task.checkCancellation()
                        var roundParameters = generateParameters
                        let preparedContext = ChatSessionPreparedContext(cachedTokenCount: logicalTokenCount,
                            inputTokenCount: input.text.tokens.size, requestedOutputTokens: generateParameters.maxTokens,
                            residentCacheBytes: contextAdmission == nil || !exclusivelyOwned ? nil : KVCacheResidency.residentBytes(in: kvCache))
                        if let contextAdmission {
                            guard speculativeDecoding == nil else {
                                throw ChatSessionContextError.unsupportedSpeculativeAdmission
                            }
                            let allowance = try contextAdmission(preparedContext)
                            guard allowance > 0 else { throw ChatSessionContextError.invalidOutputAllowance }
                            roundParameters.maxTokens = allowance
                            try Task.checkCancellation()
                        }
                        // Admission uses this exact LMInput. Keep .empty/.history
                        // staged until it succeeds; update() does not roll back.
                        messages.removeAll()
                        let counter = ChatSessionTokenCounter()

                        // A failing model.prepare may have partially touched an
                        // already-published cache. Never reuse its old logical
                        // position; successful joined decoding publishes a new one.
                        if case .kvcache = cache {
                            cache = .kvcache(kvCache, draftKVCache: draftKVCache, logicalTokenCount: nil, exclusivelyOwned: exclusivelyOwned)
                        }
                        // Select the token iterator based on speculative decoding configuration.
                        let (genStream, genTask): (AsyncStream<Generation>, Task<Void, Never>)

                        if let speculativeDecoding {
                            // Extract the draft model from its container (same pattern as the main model).
                            let draftModel = await speculativeDecoding.draftModel.perform {
                                context in
                                SendableBox(context.model)
                            }.consume()

                            // Allocate the draft KV cache once and reuse it across turns,
                            // exactly like the main model's KV cache.
                            if draftKVCache == nil {
                                draftKVCache = draftModel.newCache(parameters: generateParameters)
                            }
                            let draftCache = draftKVCache!

                            let iterator = try SpeculativeTokenIterator(
                                input: input,
                                mainModel: model,
                                draftModel: draftModel,
                                mainCache: kvCache,
                                draftCache: draftCache,
                                parameters: roundParameters,
                                numDraftTokens: speculativeDecoding.numDraftTokens
                            )

                            try await generationAdmission?.validate()
                            try Task.checkCancellation()
                            func start() -> (AsyncStream<Generation>, Task<Void, Never>) {
                                MLXLMCommon.generateTask(promptTokenCount: input.text.tokens.size,
                                    modelConfiguration: modelConfiguration, tokenizer: tokenizer,
                                    iterator: iterator, tools: tools)
                            }
                            if let generationAdmission { (genStream, genTask) = try generationAdmission.start(start) }
                            else { (genStream, genTask) = start() }
                        } else {
                            // Standard path with no speculative decoding.
                            let iterator = ChatSessionCountingIterator(try TokenIterator(
                                input: input, model: model, cache: kvCache,
                                parameters: roundParameters), counter: counter)

                            try await generationAdmission?.validate()
                            try Task.checkCancellation()
                            func start() -> (AsyncStream<Generation>, Task<Void, Never>) {
                                MLXLMCommon.generateTask(promptTokenCount: input.text.tokens.size,
                                    modelConfiguration: modelConfiguration, tokenizer: tokenizer,
                                    iterator: iterator, tools: tools)
                            }
                            if let generationAdmission { (genStream, genTask) = try generationAdmission.start(start) }
                            else { (genStream, genTask) = start() }
                        }

                        cache = .kvcache(kvCache, draftKVCache: draftKVCache, logicalTokenCount: logicalTokenCount, exclusivelyOwned: exclusivelyOwned)
                        var pendingToolCalls: [ToolCall] = []

                        for await item in genStream {
                            // collect tool calls for dispatch; if no
                            // toolDispatch the caller handles them via
                            // the transform (streamDetails path)
                            if let toolCall = item.toolCall, toolDispatch != nil {
                                pendingToolCalls.append(toolCall)
                            } else if let value = transform(item) {
                                if case .terminated = continuation.yield(value) {
                                    break
                                }
                            }
                        }

                        // wait for the task to complete -- this is important in
                        // the case where we broke the loop early as the generation
                        // work may continue (briefly) and use the KVCache
                        if Task.isCancelled { genTask.cancel() }
                        await genTask.value
                        // TokenIterator consumes the prepared prompt and one
                        // cached position for every raw iteration, including EOS.
                        // This does not inspect recurrent-cache offset defaults.
                        if speculativeDecoding == nil {
                            logicalTokenCount = preparedContext.position(afterGeneratedTokens: counter.value)
                        } else { logicalTokenCount = nil }
                        cache = .kvcache(kvCache, draftKVCache: draftKVCache, logicalTokenCount: logicalTokenCount, exclusivelyOwned: exclusivelyOwned)
                        try Task.checkCancellation()

                        // dispatch all tool calls from this generation pass
                        if let toolDispatch, !pendingToolCalls.isEmpty,
                            !Task.isCancelled
                        {
                            for toolCall in pendingToolCalls {
                                let toolResult = try await toolDispatch(toolCall)
                                messages.append(.tool(toolResult))
                            }
                            continue restart
                        }
                    }

                    continuation.finish()
                }
            } catch {
                continuation.finish(throwing: error)
            }
        }

        continuation.onTermination = { _ in
            task.cancel()
        }

        return (stream, task)
    }

    /// Produces a streaming response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    ///   - audio: optional audio (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        image: consuming UserInput.Image? = nil,
        video: consuming UserInput.Video? = nil,
        audio: consuming UserInput.Audio? = nil
    ) -> AsyncThrowingStream<String, Error> {
        streamResponse(
            to: prompt,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? [],
            audios: audio.map { [$0] } ?? [],
        )
    }

    /// Prepare a hypothetical continuation without generating, publishing cache
    /// state or changing tools. The caller must serialize this with generation.
    /// Counts include the actual retained KV position, not a lossy history replay.
    public func prepareContinuationContext(to messages: consuming [Chat.Message], tools proposedTools: [ToolSpec]?)
        async throws -> ChatSessionPreparedContext {
        let inputMessages = SendableBox(messages)
        return try await cache.read { [model, instructions, processing, additionalContext,
                                      generateParameters, speculativeDecoding] cache in
            try Task.checkCancellation()
            guard speculativeDecoding == nil else { throw ChatSessionContextError.unsupportedSpeculativeAdmission }
            var body = instructions.map { [Chat.Message.system($0)] } ?? []
            let cachedTokens: Int?
            var residentBytes: UInt64? = 0
            switch cache {
            case .empty: cachedTokens = 0
            case .history(let history): cachedTokens = 0; body.append(contentsOf: history)
            case .kvcache(let current, _, let count, let owned):
                cachedTokens = count; residentBytes = owned ? KVCacheResidency.residentBytes(in: current) : nil
            }
            body.append(contentsOf: inputMessages.consume())
            let processor = await model.processor
            let input = try await processor.prepare(input: UserInput(chat: body, processing: processing,
                tools: proposedTools, additionalContext: additionalContext))
            try Task.checkCancellation()
            return ChatSessionPreparedContext(cachedTokenCount: cachedTokens,
                inputTokenCount: input.text.tokens.size, requestedOutputTokens: generateParameters.maxTokens,
                residentCacheBytes: residentBytes)
        }
    }

    /// Replace only transient KV with an authoritative replay supplied by its
    /// owner after the exact producer joined. A failed later summary can replay
    /// the same messages; clearing compute state is not clearing conversation.
    public func restoreHistory(_ history: consuming [Chat.Message]) async {
        let transfer = SendableBox(history)
        await cache.update { cache in cache = .history(transfer.consume()) }
    }

    /// Clear the session history and cache, preserving system instructions.
    public func clear() async {
        await cache.update { cache in
            cache = .empty
        }
    }

    /// Wait for exclusive access to the KVCache.
    ///
    /// This is useful for cases where a program is terminating and wants to ensure that any
    /// async operations are complete.
    public func synchronize() async {
        await cache.read { _ in }
    }

    /// Visit the current cache value, if realized as a `[KVCache]`.
    ///
    /// This method is meant for test support.
    func withCache<R: Sendable>(_ body: @Sendable ([KVCache]?) async throws -> R) async rethrows
        -> R?
    {
        try await cache.read { cache in
            switch cache {
            case .kvcache(let cache, _, _, _):
                return try await body(cache)
            default:
                return try await body(nil)
            }
        }
    }

    /// Saves the current KV cache to disk.
    ///
    /// Use one of the initializers that accept a `cache` parameter together with
    /// ``loadPromptCache(url:)`` to restore the saved cache in a future session.
    ///
    /// - Parameter url: the file URL to write the cache to
    /// - Throws: ``ChatSessionError/noCacheAvailable`` if no generation has occurred yet,
    ///   or any error thrown by the underlying file write
    public func saveCache(to url: URL) async throws {
        try await cache.read { cache in
            switch cache {
            case .kvcache(let cache, _, _, _):
                try savePromptCache(url: url, cache: cache)
            default:
                throw ChatSessionError.noCacheAvailable
            }
        }
    }
}

/// Errors thrown by ``ChatSession``.
public enum ChatSessionError: LocalizedError {
    /// ``ChatSession/saveCache(to:)`` was called before any generation occurred.
    case noCacheAvailable

    public var errorDescription: String? {
        "No KV cache is available. Call respond() or streamResponse() before saveCache(to:)."
    }
}


/// Keeps host-specific mailbox types out of the model library. The synchronous
/// start closure must execute exactly once or throw before producer creation.
public struct ChatSessionGenerationAdmission: Sendable {
    private let validateInput: @Sendable () async throws -> Void
    private let startProducer: @Sendable (() throws -> Void) throws -> Void
    public init(validate: @escaping @Sendable () async throws -> Void,
                start: @escaping @Sendable (() throws -> Void) throws -> Void) {
        validateInput = validate; startProducer = start
    }
    public func validate() async throws { try await validateInput() }
    public func start<Value>(_ create: () throws -> Value) throws -> Value {
        var value: Value?
        var started = false
        try startProducer {
            guard !started else { throw CancellationError() }
            started = true
            value = try create()
        }
        guard let value else { throw CancellationError() }
        return value
    }
}
