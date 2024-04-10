public struct TiledDiffusionConfiguration: Equatable {
  public struct Size: Equatable {
    public var width: Int
    public var height: Int
    public init(width: Int, height: Int) {
      self.width = width
      self.height = height
    }
  }
  public var isEnabled: Bool
  public var tileSize: Size
  public var tileOverlap: Int
  public init(isEnabled: Bool, tileSize: Size, tileOverlap: Int) {
    self.isEnabled = isEnabled
    self.tileSize = tileSize
    self.tileOverlap = tileOverlap
  }
}
