import DataModels
import Foundation

public enum ServerConfigurationRewriteError: Error {
  case canNotLoadModel(String)
}

public protocol ServerConfigurationRewriter {
  func newConfiguration(
    configuration: GenerationConfiguration,
    cancellation: @escaping (@escaping () -> Void) -> Void,
    completion: @escaping (Result<GenerationConfiguration, Error>) -> Void)

}
