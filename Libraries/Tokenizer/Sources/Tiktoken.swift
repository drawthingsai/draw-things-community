import Foundation

public struct TiktokenTokenizer {
  private struct Pair: Hashable, Equatable {
    var first: Data
    var second: Data
    init(first: Data, second: Data) {
      self.first = first
      self.second = second
    }
  }
  private let decoder: [Int32: Data]
  private let bpeRanks: [Pair: Int]
  private let specialTokens: [Data]
  private let specialTokenStrings: [String]
  public let vocabulary: [Data: Int32]
  public let unknownToken: Int32
  public let startToken: Int32
  public let endToken: Int32

  public init(
    vocabulary: Data, merges: Data, specialTokens: [String: Int32] = [:],
    unknownToken: String = "<|endoftext|>", startToken: String = "<|endoftext|>",
    endToken: String = "<|endoftext|>"
  ) {
    let jsonVocabulary = try! JSONDecoder().decode([String: Int32].self, from: vocabulary)
    var decoder = [Int32: Data]()
    var vocabulary = [Data: Int32]()
    for (k, v) in jsonVocabulary {
      let bytes = Self.unicodeToBytes(k)
      decoder[v] = bytes
      vocabulary[bytes] = v
    }
    vocabulary.merge(
      specialTokens.compactMap { k, v in
        guard let data = k.data(using: .utf8) else { return nil }
        return (data, v)
      }
    ) { a, _ in a }
    self.vocabulary = vocabulary
    self.decoder = decoder
    let bpeMerges = (String(data: merges, encoding: .utf8) ?? "")
      .trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "\n")[
        1...]
    var bpeRanks = [Pair: Int]()
    for (i, merge) in bpeMerges.enumerated() {
      let splits = merge.split(separator: " ", maxSplits: 2)
      bpeRanks[
        Pair(
          first: Self.unicodeToBytes(String(splits[0])),
          second: Self.unicodeToBytes(String(splits[1])))] = i
    }
    self.bpeRanks = bpeRanks
    var specialTokenStrings = Set(specialTokens.keys)
    specialTokenStrings.insert(unknownToken)
    specialTokenStrings.insert(startToken)
    specialTokenStrings.insert(endToken)
    self.specialTokenStrings = Array(specialTokenStrings)
    let unknownToken = unknownToken.data(using: .utf8)!
    let startToken = startToken.data(using: .utf8)!
    let endToken = endToken.data(using: .utf8)!
    self.unknownToken = self.vocabulary[unknownToken]!
    self.startToken = self.vocabulary[startToken]!
    self.endToken = self.vocabulary[endToken]!
    var specialTokens = Set(
      specialTokens.keys.compactMap {
        $0.data(using: .utf8)
      })
    specialTokens.insert(unknownToken)
    specialTokens.insert(startToken)
    specialTokens.insert(endToken)
    self.specialTokens = Array(specialTokens)
  }

  public func decode(_ tokens: [Int32]) -> String {
    let data = tokens.flatMap {
      let token = decoder[$0, default: Data()]
      guard !token.isEmpty else { return [UInt8]() }
      return [UInt8](token)
    }
    return String(data: Data(data), encoding: .utf8) ?? ""
  }

  /// byte → Unicode scalar
  private static let byteEncoder: [UInt8: Character] = {
    // 1. printable bytes kept as-is
    var bs: [UInt8] = Array(33...126) + Array(161...172) + Array(174...255)
    var cs = bs.map { UInt32($0) }  // same code-points so far

    // 2. non-printables shifted to 256+
    var n: UInt32 = 0
    for b in UInt32(0)...255 where !bs.contains(UInt8(b)) {
      bs.append(UInt8(b))
      cs.append(256 + n)
      n += 1
    }

    // 3. build dictionary
    var table = [UInt8: Character]()
    for (b, c) in zip(bs, cs) {
      table[b] = Character(UnicodeScalar(c)!)
    }
    return table
  }()

  /// Unicode scalar (surrogate) → byte
  private static let byteDecoder: [Character: UInt8] = {
    var table = [Character: UInt8]()
    for (b, u) in byteEncoder { table[u] = b }
    return table
  }()

  // MARK: – Helpers

  /// Raw-byte **String** → surrogate **String** (reversible)
  static func bytesToUnicode(_ data: Data) -> String {
    var scalars = String.UnicodeScalarView()
    for byte in data {
      // mapping is total: `byte` is always found
      guard let unicodeScalars = byteEncoder[byte]?.unicodeScalars else { continue }
      scalars.append(contentsOf: unicodeScalars)
    }
    return String(scalars)
  }

  /// Surrogate **String** → original raw-byte **String**
  ///
  /// We re-assemble the bytes then materialise them with ISO-8859-1
  /// (“latin1”) because it maps 0x00…0xFF one-to-one into Unicode,
  /// guaranteeing loss-less round-trips.
  static func unicodeToBytes(_ text: String) -> Data {
    var buffer = [UInt8]()
    buffer.reserveCapacity(text.count)
    for ch in text {
      guard let b = byteDecoder[ch] else { continue }
      buffer.append(b)
    }
    return Data(buffer)
  }

  private func pretokenizeNoRegex(_ text: String) -> [Substring] {
    var tokens: [Substring] = []
    // Sliding window [last ..< i) holds the start of the current candidate token.
    var last = text.startIndex
    var i = text.startIndex
    var lastWasNewline = false

    // Local helper: flush whatever is between `last` and `end` into `tokens`
    @inline(__always)
    func flush(upTo end: String.Index) {
      if last < end { tokens.append(text[last..<end]) }
    }

    while i < text.endIndex {
      let ch = text[i]

      // ---- 1. Fixed meta-tokens -------------------------------------------------
      if let tok = specialTokenStrings.first(where: { text[i...].hasPrefix($0) }) {
        let tokEnd = text.index(i, offsetBy: tok.count)
        flush(upTo: i)
        tokens.append(text[i..<tokEnd])
        last = tokEnd
        i = tokEnd
        continue
      }

      // ---- 2. Single decimal digit ---------------------------------------------
      if ch.isNumber {
        flush(upTo: i)
        let next = text.index(after: i)
        tokens.append(text[i..<next])
        last = next
        i = next
        continue
      }

      // ---- 3. Apostrophe contractions ------------------------------------------
      if ch == "'" {
        let remain = text[i...]
        if remain.hasPrefix("'s") || remain.hasPrefix("'t") || remain.hasPrefix("'m")
          || remain.hasPrefix("'d")
        {

          flush(upTo: i)
          let next = text.index(i, offsetBy: 2)  // 'x
          tokens.append(text[i..<next])
          last = next
          i = next
          continue
        }
        if remain.hasPrefix("'re") || remain.hasPrefix("'ve") || remain.hasPrefix("'ll") {

          flush(upTo: i)
          let next = text.index(i, offsetBy: 3)  // 'xx
          tokens.append(text[i..<next])
          last = next
          i = next
          continue
        }
      }

      // ---- 4. Newline forces a split -------------------------------------------
      if ch.isNewline {
        lastWasNewline = true
        i = text.index(after: i)
        continue
      }
      if lastWasNewline {
        flush(upTo: i)
        last = i
        lastWasNewline = false
      }

      // ---- 5. Whitespace boundary ---------------------------------------------
      if ch.isWhitespace {
        flush(upTo: i)  // push token before the space
        let next = text.index(after: i)  // skip the space itself
        last = next
        i = next
        continue
      }

      // ---- 6. Punctuation / symbol (single char) ------------------------------
      if !ch.isLetter && !ch.isNumber && !ch.isWhitespace {

        flush(upTo: i)
        let next = text.index(after: i)
        tokens.append(text[i..<next])  // keep punct as its own token
        last = next
        i = next
        continue
      }

      // ---- 7. Otherwise just advance one character ----------------------------
      i = text.index(after: i)
    }

    // ---- 8. Flush tail -----------------------------------------------------------
    if last < text.endIndex {
      tokens.append(text[last..<text.endIndex])
    }

    return tokens
  }

  private static let tokenRegex: Any? = {
    if #available(iOS 16.0, macOS 13.0, *) {
      return try? Regex(
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
      )
    }
    return nil
  }()

  private func pretokenize(_ text: String) -> [Substring] {
    if #available(iOS 16.0, macOS 13.0, *) {
      if let tokenRegex = Self.tokenRegex as? Regex<AnyRegexOutput> {
        return text.matches(of: tokenRegex).map {
          text[$0.range]
        }
      }
    }
    return pretokenizeNoRegex(text)
  }

  public func tokenize(text: String, addSpecialTokens: Bool = false)
    -> ([Int32], [String])
  {
    var fixText = text.split(separator: " ").joined(separator: " ")
    if text.hasPrefix(" ") {
      fixText = " " + fixText
    }
    if text.hasSuffix(" ") {
      fixText = fixText + " "
    }
    // Logic for r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+\r?\n*|\s*\r?\n+|\s+(?!\S)|\s+"""
    // Implement this with for loop rather than regex so it is applicable with Swift 5.6.x
    let tokens = pretokenize(fixText)
    // token should match the token before sending to bpe mapping. Now do bpe merge.
    let bpeTokens = tokens.flatMap { token -> [Data] in
      guard let token = token.data(using: .utf8) else { return [] }
      guard !specialTokens.contains(token) else {
        return [token]
      }
      return bpe(token: token)
    }
    // With bpeTokens, we can query vocabulary and return index now.
    var strs: [String] = addSpecialTokens ? [""] : []
    var ids: [Int32] = addSpecialTokens ? [startToken] : []
    for bpeToken in bpeTokens {
      strs.append(String(data: bpeToken, encoding: .utf8) ?? "")
      ids.append(vocabulary[bpeToken, default: unknownToken])
    }
    return (ids, strs)
  }

  private func getPairs(word: [Data]) -> Set<Pair>? {
    guard word.count > 1 else {
      return nil
    }
    var pairs = Set<Pair>()
    var previousCharacter = word[0]
    for character in word.suffix(from: 1) {
      pairs.insert(Pair(first: previousCharacter, second: character))
      previousCharacter = character
    }
    return pairs
  }

  private func bpe(token: Data) -> [Data] {
    var word = [Data]()
    for character in token {
      word.append(Data([character]))
    }
    guard var pairs = getPairs(word: word) else {
      return word
    }
    while true {
      var bigram: Pair? = nil
      var minRank: Int? = nil
      for pair in pairs {
        if let rank = bpeRanks[pair] {
          guard let currentMinRank = minRank else {
            bigram = pair
            minRank = rank
            continue
          }
          if rank < currentMinRank {
            bigram = pair
            minRank = rank
          }
        }
      }
      guard let bigram = bigram else {
        break
      }
      var newWord = [Data]()
      var i = 0
      while i < word.count {
        guard let j = word[i...].firstIndex(of: bigram.first) else {
          newWord.append(contentsOf: word[i...])
          break
        }
        if i < j {
          newWord.append(contentsOf: word[i..<j])
        }
        i = j
        if word[i] == bigram.first && i < word.count - 1 && word[i + 1] == bigram.second {
          newWord.append(bigram.first + bigram.second)
          i += 2
        } else {
          newWord.append(word[i])
          i += 1
        }
      }
      word = newWord
      if word.count == 1 {
        break
      }
      pairs = getPairs(word: word)!  // word.count > 1, should be able to get pair.
    }
    return word
  }
}

extension TiktokenTokenizer: Tokenizer {
  public func tokenize(text: String, truncation: Bool, maxLength: Int, paddingToken: Int32?) -> (
    [String], [Int32], [Float], [String?], [Int]
  ) {
    var (ids, strs) = tokenize(text: text, addSpecialTokens: true)
    let lengthOfTokens = ids.count
    let paddingToken = paddingToken ?? unknownToken
    if ids.count < maxLength {
      for _ in ids.count..<maxLength {
        strs.append("")
        ids.append(paddingToken)
      }
    }
    return (strs, ids, [Float](repeating: 1, count: ids.count), strs, [lengthOfTokens])
  }
}

extension TiktokenTokenizer: TextualInversionPoweredTokenizer {
  public func textualInversion(for token: Int32) -> String? {
    return nil
  }
  public func isTextualInversion(_ token: Int32) -> Bool {
    return false
  }
  public var textualInversions: [String] { [] }
}
