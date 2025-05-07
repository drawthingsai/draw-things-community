import Collections
import NNC

public final class WeightsCache {
  struct Item: Comparable {
    var file: String
    var weights: [(key: String, value: AnyTensor)]
    var size: Int
  }
  private var heap: Heap<Item>
  private var map: [String: Item]
  private let maxTotalCacheSize: Int  // Maximum number of items in the cache
  private var currentTotalSize: Int  // Optional: if you also have a total size limit

  public init(maxTotalCacheSize: Int) {
    self.maxTotalCacheSize = maxTotalCacheSize
    heap = Heap<Item>()
    map = [:]
    currentTotalSize = 0  // Initialize if tracking total size
  }

  public subscript(file: String) -> (size: Int, weights: [(key: String, value: AnyTensor)])? {
    get {
      guard let item = map[file] else { return nil }
      return (size: item.size, weights: item.weights)
    }
    set {
      guard let newValue = newValue else {
        if let index = map.index(forKey: file) {
          let item = map.remove(at: index)
          // If this exists, now loop through heap to remove.
          heap = Heap(map.values)
          currentTotalSize -= item.value.size
        }
        return
      }
      // If this item itself is larger than total cache size, we cannot fit.
      if newValue.size > maxTotalCacheSize {
        return
      }
      // If this item size is smaller than smallest in the cache, and we are at capacity, there is no need to insert.
      if let min = heap.min, newValue.size < min.size,
        newValue.size + currentTotalSize > maxTotalCacheSize
      {
        return
      }
      // If this item size plus the current size is larger, need to evict first.
      evict(for: newValue.size)
      let item = Item(file: file, weights: newValue.weights, size: newValue.size)
      map[file] = item
      heap.insert(item)
    }
  }

  public func removeAll() {
    map.removeAll()
    heap = Heap()
  }

  // Evict the smallest item if the cache is over capacity
  private func evict(for size: Int) {
    // Define `maxTotalCacheSize` for this logic
    while currentTotalSize + size > maxTotalCacheSize {
      if let smallestItem = heap.popMin() {
        map[smallestItem.file] = nil
        currentTotalSize -= smallestItem.size
      } else {
        break  // Heap is empty
      }
    }
  }

  // Get the current number of items in the cache
  public var count: Int {
    return heap.count
  }

  // Get the current total size of items in the cache
  public var totalSize: Int {
    return currentTotalSize
  }
}

extension WeightsCache.Item {
  // Conformance to Comparable for Min-Heap behavior based on 'size'
  static func < (lhs: WeightsCache.Item, rhs: WeightsCache.Item) -> Bool {
    // To make it a min-heap (smallest item at the top),
    // an item is "less than" another if its size is smaller.
    return lhs.size < rhs.size
  }

  // Conformance to Equatable (required by Comparable)
  static func == (lhs: WeightsCache.Item, rhs: WeightsCache.Item) -> Bool {
    return lhs.file == rhs.file && lhs.size == rhs.size
  }
}
