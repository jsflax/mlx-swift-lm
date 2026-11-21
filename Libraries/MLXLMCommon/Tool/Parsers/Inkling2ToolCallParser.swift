// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for Thinking Machines Inkling 2 tool calls:
/// `<|message_model|><fn_name><|content_invoke_tool_json|>{"name":"<fn_name>","args":{...}}<|end_message|>`
public struct Inkling2ToolCallParser: ToolCallParser, Sendable {
    public static let messageModelTag = "<|message_model|>"
    public static let invokeTag = "<|content_invoke_tool_json|>"
    public static let endMessageTag = "<|end_message|>"
    public static let endSamplingTag = "<|content_model_end_sampling|>"

    public let startTag: String? = Self.invokeTag
    public let endTag: String? = Self.endMessageTag

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseAll(content: content, tools: tools).first
    }

    public func parseAll(content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        guard content.contains(Self.invokeTag) else {
            if let call = parseSegment(header: nil, jsonSlice: content[...], tools: tools) {
                return [call]
            }
            return []
        }

        var results: [ToolCall] = []
        var searchStart = content.startIndex

        while searchStart < content.endIndex,
            let invokeRange = content.range(
                of: Self.invokeTag, range: searchStart ..< content.endIndex)
        {
            let prefixSlice = content[searchStart ..< invokeRange.lowerBound]
            let headerName = Self.extractHeaderFunctionName(from: prefixSlice)

            let nextInvokeStart =
                content.range(
                    of: Self.invokeTag,
                    range: invokeRange.upperBound ..< content.endIndex
                )?.lowerBound ?? content.endIndex

            let payloadEnd: String.Index
            let nextSearchStart: String.Index
            if let endRange = content.range(
                of: Self.endMessageTag,
                range: invokeRange.upperBound ..< nextInvokeStart
            ) {
                payloadEnd = endRange.lowerBound
                nextSearchStart = endRange.upperBound
            } else {
                payloadEnd = nextInvokeStart
                nextSearchStart = nextInvokeStart
            }

            let jsonSlice = content[invokeRange.upperBound ..< payloadEnd]
            if let call = parseSegment(header: headerName, jsonSlice: jsonSlice, tools: tools) {
                results.append(call)
            }

            searchStart = nextSearchStart
        }

        return results
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        parseAll(content: toolCallBuffer, tools: tools)
    }

    // MARK: - Helpers

    private static func extractHeaderFunctionName(from prefix: Substring) -> String? {
        let candidate: Substring
        if let lastMsgRange = prefix.range(of: Self.messageModelTag, options: .backwards) {
            candidate = prefix[lastMsgRange.upperBound...]
        } else {
            candidate = prefix
        }
        let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.contains("<"), !trimmed.contains(">") else {
            return nil
        }
        return trimmed
    }

    private func parseSegment(
        header: String?,
        jsonSlice: Substring,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        var cleaned = String(jsonSlice)
        if let endRange = cleaned.range(of: Self.endMessageTag) {
            cleaned = String(cleaned[..<endRange.lowerBound])
        }
        cleaned = cleaned.replacingOccurrences(of: Self.endSamplingTag, with: "")
        cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty,
            let data = cleaned.data(using: .utf8),
            let jsonObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        let jsonName = (jsonObject["name"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let funcName: String
        if let jsonName, !jsonName.isEmpty {
            funcName = jsonName
        } else if let header, !header.isEmpty {
            funcName = header
        } else {
            return nil
        }

        if let tools, !tools.isEmpty {
            guard Self.isDeclaredTool(name: funcName, in: tools) else {
                return nil
            }
        }

        let rawArgs = jsonObject["args"] ?? jsonObject["arguments"]
        let argumentsDict: [String: Any]
        if rawArgs == nil || rawArgs is NSNull {
            argumentsDict = [:]
        } else if let dict = rawArgs as? [String: Any] {
            argumentsDict = dict
        } else if let stringified = rawArgs as? String,
            let argsData = stringified.data(using: .utf8),
            let dict = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any]
        {
            argumentsDict = dict
        } else {
            return nil
        }

        let sendableArgs = argumentsDict.mapValues { asSendable($0) }
        return ToolCall(function: .init(name: funcName, arguments: sendableArgs))
    }

    private static func isDeclaredTool(
        name: String,
        in tools: [[String: any Sendable]]
    ) -> Bool {
        for tool in tools {
            if let functionSpec = tool["function"] as? [String: any Sendable],
                functionSpec["name"] as? String == name
            {
                return true
            }
            if tool["name"] as? String == name {
                return true
            }
        }
        return false
    }
}
