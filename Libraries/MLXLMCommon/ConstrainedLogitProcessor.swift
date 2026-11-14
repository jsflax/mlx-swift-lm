/// A reference processor whose completion state is shared with the decoder.
/// The host owns grammar/schema policy; the generation owner owns all mutation
/// from prompt preparation through the final iterator/stream drain.
public protocol ConstrainedLogitProcessor: AnyObject, LogitProcessor, Sendable {
    var isComplete: Bool { get }
}
