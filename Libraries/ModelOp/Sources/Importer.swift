import Dispatch

public enum Importer {
  public static let queue = DispatchQueue(
    label: "com.draw-things.model-import", qos: .userInitiated)
}
