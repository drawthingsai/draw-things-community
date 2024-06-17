import SentencePiece

public struct SentencePieceTokenizer: Tokenizer {
  public var startToken: Int32?
  public var endToken: Int32
  public var unknownToken: Int32 { 0 }
  private let sentencePiece: SentencePiece
  public init(file: String, startToken: Int32?, endToken: Int32) {
    vocabulary = [:]
    sentencePiece = SentencePiece(file: file)
    self.startToken = startToken
    self.endToken = endToken
  }

  public func tokenize(text: String, truncation: Bool, maxLength: Int, paddingToken: Int32?) -> (
    [String], [Int32], [Float], [String?], [Int]
  ) {
    let result = sentencePiece.encode(text)
    var strs = [String]()
    var ids = [Int32]()
    var canonicals = [String?]()
    if let startToken = startToken {
      strs.append("")
      ids.append(startToken)
      canonicals.append("<|startoftext|>")
    }
    if truncation {
      for (i, spt) in result.enumerated() {
        guard i < maxLength + 2 else { break }
        strs.append(spt.surface)
        ids.append(spt.id + 1)
        canonicals.append(spt.piece)
      }
    } else {
      for spt in result {
        strs.append(spt.surface)
        ids.append(spt.id + 1)
        canonicals.append(spt.piece)
      }
    }
    strs.append("")
    ids.append(endToken)
    canonicals.append("<|endoftext|>")
    let paddingToken = paddingToken ?? 1
    if ids.count < maxLength {
      for _ in ids.count..<maxLength {
        strs.append("")
        ids.append(paddingToken)
        canonicals.append("")
      }
    }
    return (strs, ids, [Float](repeating: 1, count: ids.count), canonicals, [ids.count - 2])
  }

  public var vocabulary: [String: Int32]
}

extension SentencePieceTokenizer: TextualInversionPoweredTokenizer {
  public func textualInversion(for token: Int32) -> String? {
    return nil
  }
  public func isTextualInversion(_ token: Int32) -> Bool {
    return false
  }
  public var textualInversions: [String] { [] }
}
