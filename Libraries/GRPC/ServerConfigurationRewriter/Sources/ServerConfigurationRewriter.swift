import DataModels
import Foundation

public enum ServerConfigurationRewriteError: Error {
  case canNotLoadModel(String)
}

public protocol ServerConfigurationRewriter {
  func newConfiguration(
    configuration: GenerationConfiguration,
    progress: @escaping (_ bytesReceived: Int64, _ bytesExpected: Int64, _ index: Int, _ total: Int)
      -> Void,
    cancellation: @escaping (@escaping () -> Void) -> Void,
    completion: @escaping (Result<GenerationConfiguration, Error>) -> Void)

}
