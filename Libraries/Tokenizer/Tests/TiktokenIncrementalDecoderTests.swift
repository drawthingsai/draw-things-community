import BinaryResources
import Foundation
import XCTest

@testable import Tokenizer

final class TiktokenIncrementalDecoderTests: XCTestCase {
  func testSpecialTokenPrefilterPreservesRegexMatches() throws {
    guard #available(iOS 16.0, macOS 13.0, *) else { return }
    let tokenLists = [
      ["a", "ab", "abc"], ["abc", "ab", "a"], ["ab", "a", "abc"],
      ["<think>", "</think>", "<｜User｜>", "｜DSML｜"],
      ["[", "]", "-", "\\", "^", "$", ".", "*", "+", "?", "(", ")", "|"],
      ["é", "e\u{301}x", "👩🏽‍💻", "🇺🇸", "\r\n", "\n"],
      [], ["<|endoftext|>"], ["", "a"], ["a", ""],
      (0..<1300).map { "<｜place▁holder▁no▁\($0)｜>" },
    ]
    let texts = [
      "", "ordinary words 123 中文 日本語 العربية हिन्दी",
      "abc ab a abcd", "<think>x</think><｜User｜>｜DSML｜",
      "[]-\\^$.*+?()|", "e\u{301} é e\u{301}x 👩🏽‍💻 🇺🇸\r\n\n",
      "<think>\u{301}a\u{301}\u{0}\u{200b}",
    ]
    for tokens in tokenLists {
      let original = try Regex(
        "(\(tokens.map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "|")))")
      let updated = try Regex(TiktokenTokenizer.specialTokensPattern(tokens))
      for text in texts + [tokens.joined(separator: " text\n")] {
        XCTAssertEqual(
          text.matches(of: original).map(\.range), text.matches(of: updated).map(\.range),
          "Special-token match ranges changed for \(tokens.prefix(3))")
      }
    }
  }

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

  func testJoyAIPretokenizationMatchesDeepSeekReference() {
    let tokenizer = TiktokenTokenizer(
      vocabulary: BinaryResources.vocab_deepseek4_json,
      merges: BinaryResources.merges_deepseek4_txt,
      specialTokens: [
        "<｜begin▁of▁sentence｜>": 0,
        "<｜end▁of▁sentence｜>": 1,
        "<｜▁pad▁｜>": 2,
      ],
      unknownToken: "<｜▁pad▁｜>", startToken: "<｜begin▁of▁sentence｜>",
      endToken: "<｜end▁of▁sentence｜>", pretokenizer: .joyAI)

    let text = "    int x = 1234;\n中文かなABC\nfoo_bar?!\n"
    XCTAssertEqual(
      tokenizer.tokenize(text: text).0,
      [
        361, 688, 1_527, 438, 223, 6_895, 22, 510, 21_134, 49_534, 29_080, 201, 40_897,
        100_869, 33, 8_567,
      ])
    XCTAssertEqual(
      tokenizer.tokenize(
        text: "<｜begin▁of▁sentence｜>OK<｜end▁of▁sentence｜>"
      ).0,
      [0, 11_932, 1])
    XCTAssertEqual(tokenizer.tokenize(text: "\u{200b}\u{0000}").0, [35_020, 191])
    // Digits and CJK are split out before the main regex, so the spaces before them stay one
    // piece and "Swift" is not merged with the following CJK. Ids are from Hugging Face.
    XCTAssertEqual(
      tokenizer.tokenize(text: "        0,\n使用Swift编写  中文\r\ne\u{301}").0,
      [385, 18, 989, 2_541, 101_988, 32_000, 262, 21_134, 204, 201, 71, 17_793])
    for text in [
      "\u{200b}\u{0000}text", "text\u{200b}\u{0000}", "a\u{200b}\u{0000}b",
      "\u{200b}<｜begin▁of▁sentence｜>\u{0000}<｜end▁of▁sentence｜>\u{200b}",
    ] {
      XCTAssertEqual(tokenizer.decode(tokenizer.tokenize(text: text).0), text)
    }
  }

  // Expected ids come from Hugging Face tokenizers with Qwen/Qwen3.5-27B tokenizer.json.
  func testQwen3_5PretokenizationMatchesHuggingFace() {
    let tokenizer = TiktokenTokenizer(
      vocabulary: BinaryResources.vocab_qwen3_5_json,
      merges: BinaryResources.merges_qwen3_5_txt,
      specialTokens: ["<|endoftext|>": 248_044, "<|im_start|>": 248_045, "<|im_end|>": 248_046],
      unknownToken: "<|endoftext|>", startToken: "<|endoftext|>", endToken: "<|im_end|>",
      pretokenizer: .qwen3_5)

    // Indentation and repeated spaces are preserved; the rank-0 merge "Ġ Ġ" is loaded.
    let code = "    if x  == 1:\n        return  y \n"
    XCTAssertEqual(
      tokenizer.tokenize(text: code).0,
      [262, 413, 830, 220, 606, 220, 16, 25, 198, 285, 460, 220, 374, 695])
    XCTAssertEqual(tokenizer.decode(tokenizer.tokenize(text: code).0), code)
    // Single digits, NFC, "\r\n" and combining marks.
    XCTAssertEqual(
      tokenizer.tokenize(text: "v12345 cafe\u{301} caf\u{e9}\r\nक्षत्रिय").0,
      [85, 16, 17, 18, 19, 20, 50_203, 50_203, 317, 151_858, 150_422, 153_503])
    // A special token followed by a combining mark.
    XCTAssertEqual(
      tokenizer.tokenize(text: "<|im_start|>a  b<|im_end|>\u{338}").0,
      [248_045, 64, 220, 292, 248_046, 136, 116])
  }
}
