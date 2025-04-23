import DataModels
import Foundation

public enum ServerConfigurationRewriteError: Error {
  case canNotLoadModel(String)
}

public protocol ServerConfigurationRewriting {
  func newConfiguration(
    configuration: GenerationConfiguration,
    completion: @escaping (Result<GenerationConfiguration, Error>) -> Void)

}
