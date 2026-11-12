import Foundation
import Cmlx
import MLX

/// Evidence of bytes already resident in a joined, exclusively owned cache.
/// This is a lower bound on distinct live backing bytes, not allocator capacity
/// or reclaimable memory. No tensor contents or addresses leave this operation.
public enum KVCacheResidency {
    public static func residentBytes(in cache: [KVCache]) -> UInt64? {
        // These layouts expose all relevant arrays without creating cache views.
        guard cache.count <= 4096,
              cache.allSatisfy({ $0 is KVCacheSimple || $0 is ArraysCache }) else { return nil }
        let arrays = cache.flatMap { $0.innerState() }
        guard arrays.count <= 8192 else { return nil }
        // A completed forward can leave lazily computed state. Materialize it
        // BEFORE the caller samples available/active memory, never count a
        // prospective allocation as if it were already charged.
        eval(arrays)
        var ranges: [(UInt64, UInt64)] = []
        for array in arrays {
            guard array.size > 0 else { continue }
            guard let strides = mlx_array_strides(array.ctx) else { continue }
            let shape = array.shape
            let strideValues = (0..<shape.count).map { Int(strides[$0]) }
            guard let bytes = contiguousBytes(shape: shape, strides: strideValues, itemBytes: array.itemSize),
                  let pointer = mlx_array_data_uint8(array.ctx) else { continue }
            let start = UInt64(UInt(bitPattern: pointer))
            let (end, overflow) = start.addingReportingOverflow(bytes)
            guard !overflow else { return nil }
            ranges.append((start, end))
        }
        // Views can alias across caches. Credit their union only once, retaining
        // every array through the measurement. Unproven/strided arrays get zero
        // credit and remain charged in the owner's memory snapshot.
        return withExtendedLifetime(arrays) { unionBytes(ranges) }
    }

    static func contiguousBytes(shape: [Int], strides: [Int], itemBytes: Int) -> UInt64? {
        guard shape.count == strides.count, shape.count <= 16, itemBytes > 0 else { return nil }
        var elements: UInt64 = 1
        for (dimension, stride) in zip(shape, strides).reversed() {
            guard dimension > 0, stride >= 0,
                  dimension == 1 || UInt64(stride) == elements else { return nil }
            let (next, overflow) = elements.multipliedReportingOverflow(by: UInt64(dimension))
            guard !overflow else { return nil }; elements = next
        }
        let (bytes, overflow) = elements.multipliedReportingOverflow(by: UInt64(itemBytes))
        return overflow ? nil : bytes
    }

    static func unionBytes(_ ranges: [(UInt64, UInt64)]) -> UInt64? {
        let sorted = ranges.sorted { $0.0 < $1.0 }
        var end: UInt64 = 0, total: UInt64 = 0
        for (start, upper) in sorted {
            guard upper >= start else { return nil }
            let increment = upper > max(start, end) ? upper - max(start, end) : 0
            let (next, overflow) = total.addingReportingOverflow(increment)
            guard !overflow else { return nil }
            total = next; end = max(end, upper)
        }
        return total
    }
}
