import DataModels
import Diffusion
import Foundation
import GRPC
import GRPCImageServiceModels
import GRPCServer
import ImageGenerator
import ModelZoo
import NNC
import RemoteImageGenerator

/// Configuration for remote image generation execution.
internal struct MediaGenerationRemoteConfiguration {
  public let serverURL: String
  public let port: Int
  public let useTLS: Bool
  public let authentication: MediaGenerationRemoteAuthenticationMode
  public let deviceName: String?

  public init(
    serverURL: String,
    port: Int = 7859,
    useTLS: Bool = false,
    authentication: MediaGenerationRemoteAuthenticationMode = .none,
    deviceName: String? = nil
  ) {
    self.serverURL = serverURL
    self.port = port
    self.useTLS = useTLS
    self.authentication = authentication
    self.deviceName = deviceName
  }
}

internal final class MediaGenerationRemoteExecutor: @unchecked Sendable {
  private let authenticationMode: MediaGenerationRemoteAuthenticationMode
  private let queue: DispatchQueue
  private var remoteGeneratorInstance: RemoteImageGenerator?

  init(configuration: MediaGenerationRemoteConfiguration) throws {
    self.authenticationMode = configuration.authentication
    self.queue = DispatchQueue(label: "com.drawthings.mediagenerationkit.remote", qos: .userInteractive)
    try configure(configuration)
  }

  func generate(
    prompt: String,
    negativePrompt: String = "",
    configuration: GenerationConfiguration,
    image: Data? = nil,
    mask: Data? = nil,
    hints: [MediaGenerationExecutionHint] = [],
    cancellationBridge: MediaGenerationCancellationBridge?,
    uploadProgress: ((Int, Int) -> Void)? = nil,
    downloadProgress: ((Int, Int) -> Void)? = nil,
    feedback: @escaping (ImageGeneratorSignpost, Tensor<FloatType>?) -> Bool,
    completion: @escaping (Result<[Tensor<FloatType>], MediaGenerationKitError>) -> Void
  ) {
    let generator: RemoteImageGenerator
    do {
      generator = try configuredGeneratorForExecution(
        requestedModel: configuration.model,
        uploadProgress: uploadProgress,
        downloadProgress: downloadProgress
      )
    } catch let error as MediaGenerationKitError {
      completion(.failure(error))
      return
    } catch {
      completion(.failure(.generationFailed(error.localizedDescription)))
      return
    }

    MediaGenerationExecutionUtilities.generate(
      on: queue,
      generator: generator,
      prompt: prompt,
      negativePrompt: negativePrompt,
      configuration: configuration,
      image: image,
      mask: mask,
      hints: hints,
      cancellationBridge: cancellationBridge,
      feedback: feedback,
      completion: completion
    )
  }

  private func configuredGeneratorForExecution(
    requestedModel: String?,
    uploadProgress: ((Int, Int) -> Void)?,
    downloadProgress: ((Int, Int) -> Void)?
  ) throws -> RemoteImageGenerator {
    guard let remoteGeneratorInstance else {
      throw MediaGenerationKitError.remoteNotConfigured
    }

    if case .cloudCompute = authenticationMode, let requestedModel {
      let remoteModels = try listRemoteModels()
      if !remoteModels.contains(requestedModel) {
        throw MediaGenerationKitError.modelNotFoundOnRemote(requestedModel)
      }
    }

    var generator = remoteGeneratorInstance
    generator.transferDataCallback = RemoteImageGenerator.TransferDataCallback(
      beginUpload: { totalBytes in
        uploadProgress?(0, totalBytes)
      },
      beginDownload: { totalBytes in
        downloadProgress?(0, totalBytes)
      },
      remoteDownloads: { bytesReceived, bytesExpected, item, itemsExpected in
        // Remote model download progress
      }
    )
    self.remoteGeneratorInstance = generator
    return generator
  }

  private func configure(_ config: MediaGenerationRemoteConfiguration) throws {
    try? disconnect()

    #if os(macOS)
      let deviceType: ImageGeneratorDeviceType = .laptop
    #else
      let deviceType: ImageGeneratorDeviceType = .phone
    #endif

    let sharedSecret: String?
    switch config.authentication {
    case .none:
      sharedSecret = nil
    case .sharedSecret(let secret):
      sharedSecret = secret
    case .cloudCompute:
      sharedSecret = nil
    }

    let deviceName = config.deviceName ?? "MediaGenerationKit"
    let client = ImageGenerationClientWrapper(deviceName: deviceName)

    do {
      try client.connect(
        host: config.serverURL,
        port: config.port,
        TLS: config.useTLS,
        hostnameVerification: config.useTLS,
        sharedSecret: sharedSecret
      )
    } catch {
      throw MediaGenerationKitError.generationFailed("connect failed: \(error)")
    }

    let echoGroup = DispatchGroup()
    echoGroup.enter()
    var echoSucceeded = false
    var serverIdentifier: UInt64 = 0

    client.echo { success, authenticated, resources, labHours, serverIdent in
      echoSucceeded = success
      serverIdentifier = serverIdent
      echoGroup.leave()
    }

    let echoResult = echoGroup.wait(timeout: .now() + 5)
    if echoResult == .timedOut || !echoSucceeded {
      try? client.disconnect()
      throw MediaGenerationKitError.generationFailed(
        echoResult == .timedOut ? "echo timed out (5s)" : "echo returned success=false")
    }

    let authHandler = createAuthenticationHandler(from: config.authentication)
    remoteGeneratorInstance = RemoteImageGenerator(
      name: deviceName,
      deviceType: deviceType,
      client: client,
      serverIdentifier: serverIdentifier,
      authenticationHandler: authHandler,
      requestExceedLimitHandler: nil
    )
  }

  private func disconnect() throws {
    try remoteGeneratorInstance?.client.disconnect()
    remoteGeneratorInstance = nil
  }

  private func listRemoteModels(timeout: TimeInterval = 10) throws -> [String] {
    guard let remoteGeneratorInstance else {
      throw MediaGenerationKitError.remoteNotConfigured
    }
    let client = remoteGeneratorInstance.client
    guard client.client != nil else {
      throw MediaGenerationKitError.remoteNotConfigured
    }

    let queryGroup = DispatchGroup()
    queryGroup.enter()

    var echoSucceeded = false
    var discovered = Set<String>()

    client.echo { success, authenticated, resources, labHours, serverIdentifier in
      defer { queryGroup.leave() }
      guard success else { return }
      echoSucceeded = true
      for model in resources.models {
        discovered.insert(model.file)
      }
      for file in resources.files {
        discovered.insert(file)
      }
    }

    let waitResult = queryGroup.wait(timeout: .now() + timeout)
    guard waitResult != .timedOut, echoSucceeded else {
      throw MediaGenerationKitError.generationFailed(
        waitResult == .timedOut
          ? "listRemoteModels echo timed out (\(timeout)s)" : "listRemoteModels echo success=false")
    }
    return discovered.sorted()
  }

  private func createAuthenticationHandler(
    from mode: MediaGenerationRemoteAuthenticationMode
  ) -> (
    (Bool, Data, GenerationConfiguration, Bool, Int, (@escaping () -> Void) -> Void) -> String?
  )? {
    switch mode {
    case .none, .sharedSecret:
      return nil
    case .cloudCompute(let apiKey, let baseURL, let appCheck):
      let authenticator = CloudAuthenticatorRegistry.shared.authenticator(
        apiKey: apiKey,
        baseURL: baseURL
      )
      return { _, encodedBlob, configuration, hasImage, shuffleCount, cancellation in
        let shortTermToken: String
        switch self.blockingShortTermToken(authenticator: authenticator, appCheck: appCheck) {
        case .success(let token):
          shortTermToken = token
        case .failure:
          return nil
        }

        let estimatedComputeUnits = ComputeUnits.from(
          configuration,
          hasImage: hasImage,
          shuffleCount: shuffleCount
        ).map(Double.init)

        return MediaGenerationCloudAuthentication.authenticate(
          shortTermToken: shortTermToken,
          encodedBlob: encodedBlob.base64EncodedString(),
          fromBridge: true,
          estimatedComputeUnits: estimatedComputeUnits,
          baseURL: baseURL,
          cancellation: cancellation
        )
      }
    }
  }

  private func blockingShortTermToken(
    authenticator: CloudAuthenticator,
    appCheck: AppCheckConfiguration,
    timeout: TimeInterval = 30
  ) -> Result<String, Error> {
    let semaphore = DispatchSemaphore(value: 0)
    var tokenResult: Result<String, Error> = .failure(MediaGenerationKitError.notConfigured)
    authenticator.getShortTermToken(appCheck: appCheck) {
      tokenResult = $0
      semaphore.signal()
    }
    let waitResult = semaphore.wait(timeout: .now() + timeout)
    guard waitResult != .timedOut else {
      return .failure(MediaGenerationKitError.generationFailed("authentication timed out"))
    }
    return tokenResult
  }
}
