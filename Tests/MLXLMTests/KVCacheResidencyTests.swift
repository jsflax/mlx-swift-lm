import Testing
@testable import MLXLMCommon

/// Pure backing-layout/alias arithmetic; no tensors, weights or Metal commands.
struct KVCacheResidencyTests {
    @Test func onlyProvenContiguousBytesCanOffsetGrowth() {
        #expect(KVCacheResidency.contiguousBytes(shape: [1, 2, 256, 128], strides: [65536, 32768, 128, 1], itemBytes: 2) == 131072)
        // Broadcast and strided views can have logical nbytes larger than their
        // distinct resident backing. They must not create capacity credit.
        #expect(KVCacheResidency.contiguousBytes(shape: [2, 256, 128], strides: [0, 128, 1], itemBytes: 2) == nil)
        #expect(KVCacheResidency.contiguousBytes(shape: [2, 128], strides: [256, 2], itemBytes: 2) == nil)
        #expect(KVCacheResidency.contiguousBytes(shape: [2, 128], strides: [-128, 1], itemBytes: 2) == nil)
        #expect(KVCacheResidency.contiguousBytes(shape: [Int.max, 4], strides: [4, 1], itemBytes: 4) == nil)
    }

    @Test func aliasedBackingIsCountedOnlyOnce() {
        let ranges: [(UInt64, UInt64)] = [(100, 200), (140, 190), (180, 250), (300, 310), (100, 200)]
        #expect(KVCacheResidency.unionBytes(ranges) == 160)
        #expect(KVCacheResidency.unionBytes([(200, 100)]) == nil)
        #expect(KVCacheResidency.unionBytes([]) == 0)
    }
}
