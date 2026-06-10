import Foundation
import XCTest

@testable import Tokenizer

final class TiktokenIncrementalDecoderTests: XCTestCase {
  private func makeTokenizer() -> TiktokenTokenizer {
    let vocabulary: [String: Int32] = [
      TiktokenTokenizer.bytesToUnicode(Data([0xf0, 0x9f])): 1,
      TiktokenTokenizer.bytesToUnicode(Data([0x98, 0x80])): 2,
      TiktokenTokenizer.bytesToUnicode(Data("A".utf8)): 3,
      TiktokenTokenizer.bytesToUnicode(Data("B".utf8)): 4,
    ]
    let vocabularyData = try! JSONEncoder().encode(vocabulary)
    let mergesData = Data("#version: 0.2\na b\n".utf8)
    return TiktokenTokenizer(
      vocabulary: vocabularyData, merges: mergesData,
      specialTokens: ["<|endoftext|>": 0, "<special>": 9],
      unknownToken: "<|endoftext|>", startToken: "<|endoftext|>",
      endToken: "<|endoftext|>")
  }

  func testSplitUTF8ScalarEmitsAfterCompletion() {
    let tokenizer = makeTokenizer()
    var decoder = tokenizer.makeTokenStreamer()
    XCTAssertEqual(decoder.append([1]), "")
    XCTAssertEqual(decoder.append([2]), "😀")
    XCTAssertEqual(decoder.finish(), "")
  }

  func testASCIIChunksStreamImmediately() {
    let tokenizer = makeTokenizer()
    var decoder = tokenizer.makeTokenStreamer()
    XCTAssertEqual(decoder.append([3]), "A")
    XCTAssertEqual(decoder.append([4]), "B")
    XCTAssertEqual(decoder.finish(), "")
  }

  func testSpecialTokenMappingsArePreserved() {
    let tokenizer = makeTokenizer()
    var decoder = tokenizer.makeTokenStreamer(specialTokens: [9: "<special>"])
    XCTAssertEqual(decoder.append([3, 9, 4]), "A<special>B")
    XCTAssertEqual(decoder.finish(), "")
  }

  func testFinishDropsIncompleteTrailingBytesLikeFullDecode() {
    let tokenizer = makeTokenizer()
    var decoder = tokenizer.makeTokenStreamer()
    XCTAssertEqual(decoder.append([1]), "")
    XCTAssertEqual(decoder.finish(), tokenizer.decode([1]))
  }
}
