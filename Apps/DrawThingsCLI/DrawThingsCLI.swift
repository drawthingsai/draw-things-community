import ArgumentParser
import BinaryResources
import ConfigurationZoo
import DataModels
import Dflat
import Diffusion
import Dispatch
import Downloader
import Foundation
import ImageGenerator
import LocalImageGenerator
import ModelOp
import ModelZoo
import NNC
import PNG
import SQLiteDflat
import ScriptDataModels
import Tokenizer
import Trainer

#if canImport(AVFoundation) && canImport(CoreMedia) && canImport(CoreVideo)
  import AVFoundation
  import CoreMedia
  import CoreVideo
#endif
#if canImport(CoreGraphics)
  import CoreGraphics
#endif
#if canImport(ImageIO)
  import ImageIO
#endif

private enum DrawThingsCLIError: LocalizedError {
  case invalidModelsDirectory(String)
  case invalidOutputPath(String)
  case invalidConfigurationJSON
  case invalidLoRAConfigurationJSON
  case missingModel
  case unsupportedModelInput(String)
  case missingModelFiles([String])
  case generationFailed
  case unsupportedTensorShape(String)
  case invalidImageDimensions(Int, Int)
  case pngEncodeFailed(String)
  case videoEncodeFailed(String)
  case unsupportedVideoOutput(String)
  case invalidInputImagePath(String)
  case invalidInputImage(String)

  var errorDescription: String? {
    switch self {
    case .invalidModelsDirectory(let path):
      return "Models directory path is not valid: \(path)"
    case .invalidOutputPath(let path):
      return "Output path extension must be .png, .mov, or .mp4: \(path)"
    case .invalidConfigurationJSON:
      return "Failed to parse configuration override JSON"
    case .invalidLoRAConfigurationJSON:
      return "Failed to parse JSON as LoRATrainingConfiguration"
    case .missingModel:
      return "--model is required"
    case .unsupportedModelInput(let model):
      return "Unable to resolve model from input: \(model)"
    case .missingModelFiles(let files):
      return
        "Missing model files:\n\(files.map { "  - \($0)" }.joined(separator: "\n"))\nUse --download-missing or run `\(CLIIdentity.command("models ensure --model ..."))`."
    case .generationFailed:
      return "Generation failed (no tensors returned)"
    case .unsupportedTensorShape(let shape):
      return "Unsupported output tensor shape: \(shape)"
    case .invalidImageDimensions(let width, let height):
      return "Image dimensions must be multiples of 64, got \(width)x\(height)"
    case .pngEncodeFailed(let outputPath):
      return "Failed to encode PNG at path: \(outputPath)"
    case .videoEncodeFailed(let outputPath):
      return "Failed to encode video at path: \(outputPath)"
    case .unsupportedVideoOutput(let outputPath):
      return
        "Video output is not supported on this platform for path: \(outputPath). Use .png output instead."
    case .invalidInputImagePath(let path):
      return "Input image path does not exist: \(path)"
    case .invalidInputImage(let path):
      return "Failed to decode input image: \(path)"
    }
  }
}

private struct ResolvedGenerationConfiguration {
  let configuration: GenerationConfiguration
  let recommendedNegativePrompt: String?
}

private enum NetworkAccessPolicy {
  static var offline = false
}

enum VideoExportFormat: String, ExpressibleByArgument {
  case prores4444
  case prores422hq
  case h264
  case hevc
}

private enum CLIIdentity {
  static let commandName = "draw-things-cli"

  static let version: String = {
    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
      !version.isEmpty
    {
      return version
    }
    let fallbackInfoPlist = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
      .appendingPathComponent("Apps/DrawThings/SupportingFiles/Info.plist")
    if let info = NSDictionary(contentsOf: fallbackInfoPlist),
      let version = info["CFBundleShortVersionString"] as? String,
      !version.isEmpty
    {
      return version
    }
    return "dev"
  }()

  static func command(_ arguments: String) -> String {
    return "\(commandName) \(arguments)"
  }
}

private enum CLIHelpText {
  static let root = """
    DESCRIPTION:
      Run local Draw Things inference and local LoRA training from the command line.

    COMMON COMMANDS:
      \(CLIIdentity.command("generate --model flux_2_klein_4b_q6p.ckpt --prompt \"a red cube on a table\""))
      \(CLIIdentity.command("models list --downloaded-only"))
      \(CLIIdentity.command("models ensure --model flux_2_dev_q8p.ckpt"))
      \(CLIIdentity.command("train lora --model flux_2_klein_4b_q6p.ckpt --dataset ./dataset"))
      \(CLIIdentity.command("completion zsh"))

    HELP:
      Use `\(CLIIdentity.command("<command> --help"))` for command-specific help.

    ENVIRONMENT:
      DRAWTHINGS_MODELS_DIR
        Default models directory when --models-dir is not provided.
    """

  static let generate = """
    DESCRIPTION:
      Resolve a local inference model, load recommended settings, apply JSON overrides,
      then apply explicit command-line overrides before generation.

    MODEL REFERENCES:
      --model accepts a model file id, a human-readable model name, an hf://owner/repo
      reference, an owner/repo reference, or a Hugging Face model URL.

    CONFIGURATION PRECEDENCE:
      1. Recommended settings for the resolved model.
      2. Overrides from --config-json or --config-file.
      3. Explicit command-line flags such as --steps, --cfg, --width, --height,
         --frames, --seed, and --strength.
      4. --negative-prompt overrides any recommended negative prompt.

    EXAMPLES:
      \(CLIIdentity.command("generate --model flux_2_klein_4b_q6p.ckpt --prompt \"a red cube on a table\""))
      \(CLIIdentity.command("generate --model flux_2_klein_4b_q6p.ckpt --prompt-file prompt.txt"))
      \(CLIIdentity.command("generate --model flux_2_klein_4b_q6p.ckpt --prompt \"studio portrait\" --image input.png --strength 0.35"))
      \(CLIIdentity.command("generate --model ltx_2.3_22b_distilled_q6p.ckpt --prompt \"ocean waves at sunset\" --frames 49 --output clip.mov"))
    """

  static let models = """
    DESCRIPTION:
      Inspect model mappings and ensure local model files exist before generation or training.
    """

  static let modelList = """
    DESCRIPTION:
      List official models in ModelZoo order, then append community models from cached or
      fetched catalog data.
    """

  static let modelEnsure = """
    DESCRIPTION:
      Resolve a model reference and download the model file, plus dependencies by default.

    EXAMPLE:
      \(CLIIdentity.command("models ensure --model flux_2_dev_q8p.ckpt"))
    """

  static let modelImport = """
    DESCRIPTION:
      Import a local checkpoint or safetensors artifact into Draw Things format, infer a
      custom model specification, and optionally download missing companion models.

    INPUTS:
      Local files only for now. Supported source formats depend on ModelImporter and
      commonly include .safetensors, .ckpt, .pth, .pt, .bin, and .zip.

    EXAMPLES:
      \(CLIIdentity.command("models import ./flux-2-klein-4b.safetensors"))
      \(CLIIdentity.command("models import ./model.safetensors --name \"My Model\" --trigger-word mytoken"))
      \(CLIIdentity.command("models import ./model.safetensors --dry-run"))
    """

  static let train = """
    DESCRIPTION:
      Training commands for local fine-tuning workflows.
    """

  static let completion = """
    DESCRIPTION:
      Generate a shell completion script for bash, zsh, or fish.

    EXAMPLES:
      \(CLIIdentity.command("completion zsh > ~/.zsh/completions/_draw-things-cli"))
      \(CLIIdentity.command("completion bash > /etc/bash_completion.d/draw-things-cli"))
      \(CLIIdentity.command("completion fish > ~/.config/fish/completions/draw-things-cli.fish"))
    """

  static let loraTrain = """
    DESCRIPTION:
      Train a LoRA adapter using a local dataset and a local base model.

    CONFIGURATION PRECEDENCE:
      1. LoRATrainingConfiguration.default.
      2. Partial overrides from --config-json.
      3. Explicit command-line flags such as --steps, --rank, --scale, --learning-rate,
         --width, --height, and execution flags.

    DATASET:
      --dataset should point to a directory of images. Matching .txt files are treated as captions.

    EXAMPLES:
      \(CLIIdentity.command("train lora --model flux_2_klein_4b_q6p.ckpt --dataset ./dataset --steps 200"))
      \(CLIIdentity.command("train lora --config-json '{\"base_model\":\"flux_2_klein_4b_q6p.ckpt\"}' --dataset ./dataset --dry-run"))
    """
}

private let modelsDirectoryHelp = ArgumentHelp(
  "Models directory.",
  discussion:
    "Resolution order: --models-dir, DRAWTHINGS_MODELS_DIR, <binary-dir>/Models (if it exists), then ~/Documents/Models."
)

private let modelReferenceHelp = ArgumentHelp(
  "Model file, model name, or Hugging Face repo/URL.",
  discussion:
    "Accepted forms: flux_2_klein_4b_q6p.ckpt, \"FLUX.2 [klein] 4B (6-bit)\", hf://owner/repo, owner/repo, or https://huggingface.co/owner/repo."
)

private let generateConfigJSONHelp = ArgumentHelp(
  "Inline JSON override in JSGenerationConfiguration format.",
  discussion:
    "This is merged onto the model's recommended settings. It is not treated as a complete configuration by itself."
)

private let generateConfigFileHelp = ArgumentHelp(
  "Path to a JSON override file in JSGenerationConfiguration format.",
  discussion:
    "The file is parsed as a partial override and merged onto the model's recommended settings.")

private let promptFileHelp = ArgumentHelp(
  "Read prompt text from a file, or `-` for stdin.",
  discussion:
    "Use this for long or multiline prompts. Mutually exclusive with the inline prompt flag.")

private let negativePromptFileHelp = ArgumentHelp(
  "Read negative prompt text from a file, or `-` for stdin.",
  discussion:
    "Use this for long or multiline negative prompts. Mutually exclusive with the inline negative prompt flag."
)

private let generateImageHelp = ArgumentHelp(
  "Input image path for img2img.",
  discussion:
    "The image is resized with aspect-preserving scale and center crop to match the requested output size."
)

private let videoFormatHelp = ArgumentHelp(
  "Video export format for .mov/.mp4 outputs.",
  discussion:
    "Accepted values: prores4444, prores422hq, h264, hevc. ProRes formats require .mov output."
)

private let modelImportArtifactHelp = ArgumentHelp(
  "Local model artifact to import.",
  discussion:
    "Typically a .safetensors, .ckpt, .pth, .pt, .bin, or .zip file. Remote URLs are not supported by this command yet."
)

private let modelImportScaleHelp = ArgumentHelp(
  "Default finetune scale in 64px units.",
  discussion:
    "For example, 16 means a default 1024px base resolution. If omitted, the CLI mirrors the app's import defaults for the detected model family."
)

private let trainConfigJSONHelp = ArgumentHelp(
  "Inline JSON override in LoRATrainingConfiguration format.",
  discussion:
    "This is merged onto LoRATrainingConfiguration.default. It is not treated as a complete training configuration by itself."
)

private let trainDatasetHelp = ArgumentHelp(
  "Dataset directory containing images and optional .txt captions.",
  discussion:
    "Images are loaded from the directory and matching .txt files are treated as captions when present."
)

private enum NetworkCacheResolver {
  static func primaryURL(fileName: String) throws -> URL {
    guard
      let cacheDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
        .first
    else {
      throw DrawThingsCLIError.invalidConfigurationJSON
    }
    return cacheDirectory.appendingPathComponent("net", isDirectory: true).appendingPathComponent(
      fileName)
  }

  static func candidateURLs(fileName: String, modelsDirectory: URL) -> [URL] {
    var urls: [URL] = []
    if let primary = try? primaryURL(fileName: fileName) {
      urls.append(primary)
    }
    let modelsDirectoryPath = modelsDirectory.standardizedFileURL.pathComponents
    if modelsDirectoryPath.suffix(3) == ["Data", "Documents", "Models"] {
      let dataDirectory =
        modelsDirectory
        .deletingLastPathComponent()
        .deletingLastPathComponent()
      urls.append(
        dataDirectory
          .appendingPathComponent("Library", isDirectory: true)
          .appendingPathComponent("Caches", isDirectory: true)
          .appendingPathComponent("net", isDirectory: true)
          .appendingPathComponent(fileName))
    }
    return urls
  }

  static func persist(_ data: Data, fileName: String) throws {
    let fileURL = try primaryURL(fileName: fileName)
    try FileManager.default.createDirectory(
      at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: fileURL, options: .atomic)
  }
}

struct ModelsDirectoryOptions: ParsableArguments {
  @Option(name: .long, help: modelsDirectoryHelp)
  var modelsDir: String?
}

struct GenerateModelResolutionOptions: ParsableArguments {
  @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

  @Option(name: .shortAndLong, help: modelReferenceHelp)
  var model: String?
}

struct GeneratePromptOptions: ParsableArguments {
  @Option(name: .shortAndLong, help: "Prompt text.")
  var prompt: String?

  @Option(name: .long, help: promptFileHelp)
  var promptFile: String?

  @Option(
    name: .long,
    help: ArgumentHelp(
      "Negative prompt text.",
      discussion: "If omitted, recommended settings may provide one."))
  var negativePrompt: String?

  @Option(name: .long, help: negativePromptFileHelp)
  var negativePromptFile: String?
}

struct GenerateSamplingOptions: ParsableArguments {
  @Option(name: .long, help: "Sampling steps.")
  var steps: Int?

  @Option(name: .long, help: "CFG guidance scale.")
  var cfg: Float?

  @Option(name: .long, help: "Output width in pixels (multiple of 64).")
  var width: Int?

  @Option(name: .long, help: "Output height in pixels (multiple of 64).")
  var height: Int?

  @Option(name: .long, help: "Number of frames for video-capable models.")
  var frames: Int?

  @Option(name: .long, help: "Denoising strength for img2img (0...1).")
  var strength: Float?

  @Option(name: .shortAndLong, help: "Random seed.")
  var seed: UInt32?
}

struct GenerateConfigurationOverrideOptions: ParsableArguments {
  @Option(name: .long, help: generateConfigJSONHelp)
  var configJSON: String?

  @Option(name: .long, help: generateConfigFileHelp)
  var configFile: String?
}

struct GenerateImageInputOptions: ParsableArguments {
  @Option(name: .long, help: generateImageHelp)
  var image: String?

  @Option(
    name: .customLong("init-image"),
    help: ArgumentHelp("Alias of --image.", visibility: .hidden))
  var initImage: String?

  @Option(
    name: .customLong("input-image"),
    help: ArgumentHelp("Alias of --image.", visibility: .hidden))
  var inputImage: String?
}

struct GenerateOutputOptions: ParsableArguments {
  @Option(
    name: .shortAndLong,
    help: ArgumentHelp(
      "Output path (.png, .mov, or .mp4).",
      discussion: "Use .png for image output and .mov/.mp4 for video-capable models."))
  var output: String = "output.png"

  @Option(name: .long, help: videoFormatHelp)
  var videoFormat: VideoExportFormat?
}

struct GenerateExecutionOptions: ParsableArguments {
  @Flag(name: .long, inversion: .prefixedNo, help: "Auto-download missing model files.")
  var downloadMissing: Bool = true

  @Flag(
    name: .long,
    help:
      "Disable network access. Uses cached community catalogs and recommended settings only, and never downloads models."
  )
  var offline: Bool = false
}

struct LoRATrainModelAndDatasetOptions: ParsableArguments {
  @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

  @Option(name: .shortAndLong, help: modelReferenceHelp)
  var model: String?

  @Option(name: .long, help: trainDatasetHelp)
  var dataset: String?
}

struct LoRATrainConfigurationOverrideOptions: ParsableArguments {
  @Option(name: .long, help: trainConfigJSONHelp)
  var configJSON: String?

  @Option(
    name: .long,
    help: ArgumentHelp(
      "Memory saver mode.",
      discussion: "Accepted values: minimal, balanced, speed, turbo."))
  var memorySaver: String?

  @Option(
    name: .long,
    help: ArgumentHelp(
      "Weights memory mode.",
      discussion: "Accepted values: cached or justInTime."))
  var weightsMemory: String?
}

struct LoRATrainOutputOptions: ParsableArguments {
  @Option(name: .shortAndLong, help: "Output LoRA filename prefix (without extension).")
  var output: String?

  @Option(name: .long, help: "Display name for LoRA metadata.")
  var name: String?

  @Option(name: .long, help: "Resume from a saved checkpoint filename/path.")
  var resume: String?

  @Option(
    name: .customLong("train-checkpoint"),
    help: ArgumentHelp("Alias of --resume.", visibility: .hidden))
  var trainCheckpoint: String?
}

struct LoRATrainTrainingOptions: ParsableArguments {
  @Option(name: .long, help: "Training steps.")
  var steps: Int?

  @Option(name: .long, help: "LoRA rank.")
  var rank: Int?

  @Option(name: .long, help: "LoRA scale.")
  var scale: Float?

  @Option(
    name: .long,
    help:
      "UNet learning rate as a float or range. Examples: 1e-4, [5e-5,1e-4], 5e-5:1e-4.")
  var learningRate: String?

  @Option(name: .long, help: "Gradient accumulation steps.")
  var gradientAccumulation: Int?

  @Option(name: .long, help: "Warmup steps.")
  var warmupSteps: Int?

  @Option(name: .long, help: "Save checkpoint every N steps (0 means only final).")
  var saveEvery: Int?

  @Option(name: .long, help: "Random seed.")
  var seed: UInt32?
}

struct LoRATrainDataAndModelOptions: ParsableArguments {
  @Option(name: .long, help: "CLIP skip layers.")
  var clipSkip: Int?

  @Option(name: .long, help: "Noise offset.")
  var noiseOffset: Float?

  @Option(name: .long, help: "Caption dropout rate.")
  var captionDropout: Float?

  @Option(name: .long, help: "Output resolution in pixels (square).")
  var resolution: Int?

  @Option(name: .long, help: "Output width in pixels (multiple of 64).")
  var width: Int?

  @Option(name: .long, help: "Output height in pixels (multiple of 64).")
  var height: Int?

  @Flag(
    name: .long,
    inversion: .prefixedNo,
    help: "Use image aspect ratio bucket training.")
  var useAspectRatio: Bool?

  @Flag(
    name: .long,
    inversion: .prefixedNo,
    help: "Use orthonormal LoRA down initialization.")
  var orthonormalDown: Bool?

  @Flag(
    name: .long,
    inversion: .prefixedNo,
    help: "Co-train the text model.")
  var cotrainTextModel: Bool?
}

struct LoRATrainExecutionOptions: ParsableArguments {
  @Flag(name: .long, help: "Validate and print resolved training config without training.")
  var dryRun: Bool = false

  @Flag(name: .long, inversion: .prefixedNo, help: "Auto-download missing model files.")
  var downloadMissing: Bool = true

  @Flag(
    name: .long,
    help:
      "Disable network access. Uses cached community catalogs only, and never downloads models.")
  var offline: Bool = false
}

struct LoRATrainCommandOptions: ParsableArguments {
  @OptionGroup(title: "Model And Dataset") var modelAndDataset: LoRATrainModelAndDatasetOptions
  @OptionGroup(title: "Configuration Overrides")
  var configurationOverrides: LoRATrainConfigurationOverrideOptions
  @OptionGroup(title: "Output And Resume") var output: LoRATrainOutputOptions
  @OptionGroup(title: "Training") var training: LoRATrainTrainingOptions
  @OptionGroup(title: "Data And Model") var dataAndModel: LoRATrainDataAndModelOptions
  @OptionGroup(title: "Execution") var execution: LoRATrainExecutionOptions
}

private enum ModelsDirectoryResolver {
  static func resolve(path: String?) throws -> URL {
    if let path, !path.isEmpty {
      return try normalizeAndEnsureDirectory(URL(fileURLWithPath: path, isDirectory: true))
    }
    if let envPath = ProcessInfo.processInfo.environment["DRAWTHINGS_MODELS_DIR"], !envPath.isEmpty
    {
      return try normalizeAndEnsureDirectory(URL(fileURLWithPath: envPath, isDirectory: true))
    }
    if let adjacent = executableAdjacentModelsDirectoryIfExists() {
      return adjacent
    }
    let fallback = try documentsModelsDirectory()
    return try normalizeAndEnsureDirectory(fallback)
  }

  private static func executableAdjacentModelsDirectoryIfExists() -> URL? {
    let executablePath = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    let modelsPath = executablePath.deletingLastPathComponent().appendingPathComponent(
      "Models", isDirectory: true)
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: modelsPath.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      return nil
    }
    return modelsPath
  }

  private static func documentsModelsDirectory() throws -> URL {
    guard
      let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    else {
      throw DrawThingsCLIError.invalidModelsDirectory("~/Documents/Models")
    }
    return documents.appendingPathComponent("Models", isDirectory: true)
  }

  private static func normalizeAndEnsureDirectory(_ url: URL) throws -> URL {
    let normalized = url.standardizedFileURL
    var isDirectory: ObjCBool = false
    let exists = FileManager.default.fileExists(atPath: normalized.path, isDirectory: &isDirectory)
    if exists && !isDirectory.boolValue {
      throw DrawThingsCLIError.invalidModelsDirectory(normalized.path)
    }
    if !exists {
      try FileManager.default.createDirectory(at: normalized, withIntermediateDirectories: true)
    }
    return normalized
  }
}

private enum ModelResolver {
  static func resolve(_ input: String, modelsDirectory: URL? = nil) -> ModelZoo.Specification? {
    if let specification = ModelZoo.resolveModelReference(input)?.specification {
      return specification
    }
    guard let modelsDirectory else { return nil }
    return CommunityModelResolver.resolve(
      input, modelsDirectory: modelsDirectory, allowNetwork: !NetworkAccessPolicy.offline)
  }

  static func suggestions(_ input: String, limit: Int = 5) -> [ModelZoo.Specification] {
    return ModelZoo.candidateSpecifications(forModelReference: input, limit: limit)
  }
}

private enum ConfigurationLoader {
  static func load(
    model: String, configJSON: String?, configFile: String?, modelsDirectory: URL
  ) throws -> ResolvedGenerationConfiguration {
    if configJSON != nil && configFile != nil {
      throw ValidationError("Use only one of --config-json or --config-file")
    }
    let overrideDictionary = try loadOverrideDictionary(
      configJSON: configJSON, configFile: configFile)
    let (recommendedConfiguration, recommendedNegativePrompt) =
      RecommendedSettingsResolver.resolve(
        model: model, overrideDictionary: overrideDictionary, modelsDirectory: modelsDirectory,
        allowNetwork: !NetworkAccessPolicy.offline)
    let configuration = try mergeOverrides(
      overrideDictionary, onto: recommendedConfiguration, forcingModel: model)
    return ResolvedGenerationConfiguration(
      configuration: configuration, recommendedNegativePrompt: recommendedNegativePrompt)
  }

  private static func loadOverrideDictionary(
    configJSON: String?, configFile: String?
  ) throws -> [String: Any]? {
    guard let rawString = try loadRawJSON(configJSON: configJSON, configFile: configFile) else {
      return nil
    }
    guard let data = rawString.data(using: .utf8),
      let dictionary = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
      throw DrawThingsCLIError.invalidConfigurationJSON
    }
    return dictionary
  }

  private static func mergeOverrides(
    _ overrideDictionary: [String: Any]?, onto baseConfiguration: GenerationConfiguration,
    forcingModel model: String
  ) throws -> GenerationConfiguration {
    guard
      let baseData = try? JSONEncoder().encode(
        JSGenerationConfiguration(configuration: baseConfiguration)),
      var mergedDictionary = try? JSONSerialization.jsonObject(with: baseData) as? [String: Any]
    else {
      throw DrawThingsCLIError.invalidConfigurationJSON
    }
    if let overrideDictionary {
      for (key, value) in overrideDictionary {
        mergedDictionary[key] = value
      }
    }
    mergedDictionary["model"] = model
    guard
      let mergedData = try? JSONSerialization.data(withJSONObject: mergedDictionary),
      let configuration = try? JSONDecoder().decode(
        JSGenerationConfiguration.self, from: mergedData
      )
      .createGenerationConfiguration()
    else {
      throw DrawThingsCLIError.invalidConfigurationJSON
    }
    return configuration
  }

  private static func loadRawJSON(configJSON: String?, configFile: String?) throws -> String? {
    if let configJSON {
      return configJSON
    }
    if let configFile {
      return try String(contentsOfFile: configFile, encoding: .utf8)
    }
    return nil
  }
}

private enum RecommendedSettingsResolver {
  static func resolve(
    model: String, overrideDictionary: [String: Any]?, modelsDirectory: URL, allowNetwork: Bool
  ) -> (GenerationConfiguration, String?) {
    let defaultConfiguration = GenerationConfiguration.default
    let loras = loras(from: overrideDictionary)
    guard
      let specification = findRecommendedSettings(
        model: model, loras: loras, modelsDirectory: modelsDirectory, allowNetwork: allowNetwork)
    else {
      var builder = GenerationConfigurationBuilder(from: defaultConfiguration)
      builder.model = model
      return (builder.build(), nil)
    }
    guard
      let defaultData = try? JSONEncoder().encode(
        JSGenerationConfiguration(configuration: defaultConfiguration)),
      var mergedDictionary = try? JSONSerialization.jsonObject(with: defaultData) as? [String: Any]
    else {
      var builder = GenerationConfigurationBuilder(from: defaultConfiguration)
      builder.model = model
      return (builder.build(), nil)
    }
    for (key, value) in specification.configuration {
      mergedDictionary[key] = value
    }
    mergedDictionary["model"] = model
    guard
      let mergedData = try? JSONSerialization.data(withJSONObject: mergedDictionary),
      let configuration = try? JSONDecoder().decode(
        JSGenerationConfiguration.self, from: mergedData
      )
      .createGenerationConfiguration()
    else {
      var builder = GenerationConfigurationBuilder(from: defaultConfiguration)
      builder.model = model
      return (builder.build(), nil)
    }
    return (configuration, specification.negative?.trimmingCharacters(in: .whitespacesAndNewlines))
  }

  private static func loras(from overrideDictionary: [String: Any]?) -> Set<String> {
    guard let loras = overrideDictionary?["loras"] as? [[String: Any]] else { return [] }
    return Set(loras.compactMap { $0["file"] as? String })
  }

  private static func findRecommendedSettings(
    model: String, loras: Set<String>, modelsDirectory: URL, allowNetwork: Bool,
    timeout: TimeInterval = 10
  ) -> ConfigurationZoo.Specification? {
    let configurations = refreshedCommunityConfigurations(
      timeout: timeout, modelsDirectory: modelsDirectory, allowNetwork: allowNetwork)
    guard !configurations.isEmpty else { return nil }
    let version = ModelZoo.versionForModel(model)
    let prefix = prefix(for: model)
    var bestMatch = matchWithLoRAs(
      configurations: configurations, loras,
      first: {
        ($0.configuration["model"] as? String) == model
      },
      second: {
        guard let configModel = $0.configuration["model"] as? String, !prefix.isEmpty else {
          return false
        }
        return configModel.hasPrefix(prefix)
      })
    if bestMatch == nil {
      bestMatch = matchWithLoRAs(configurations: configurations, loras) {
        $0.version == version
      }
    }
    return bestMatch
  }

  private static func refreshedCommunityConfigurations(
    timeout: TimeInterval, modelsDirectory: URL, allowNetwork: Bool
  ) -> [ConfigurationZoo.Specification] {
    if allowNetwork,
      let fetched = try? fetchCommunityConfigurations(timeout: timeout), !fetched.isEmpty
    {
      return fetched
    }
    let cached = cachedCommunityConfigurations(modelsDirectory: modelsDirectory)
    if !cached.isEmpty {
      return cached
    }
    return ConfigurationZoo.community
  }

  private static func prefix(for file: String) -> String {
    let stem = file.components(separatedBy: ".")[0]
    guard !stem.isEmpty else { return "" }
    var components = stem.components(separatedBy: "_")
    while let last = components.last, ["f16", "svd", "q5p", "q6p", "q8p"].contains(last) {
      components.removeLast()
    }
    return components.joined(separator: "_")
  }

  private static func matchWithLoRAs(
    configurations: [ConfigurationZoo.Specification], _ loras: Set<String>,
    first: (ConfigurationZoo.Specification) -> Bool,
    second: ((ConfigurationZoo.Specification) -> Bool)? = nil
  ) -> ConfigurationZoo.Specification? {
    guard !loras.isEmpty else {
      guard let second = second else {
        return configurations.first(where: first)
      }
      return configurations.first(where: first) ?? configurations.first(where: second)
    }
    guard let second = second else {
      return configurations.first {
        let isFirst = first($0)
        guard isFirst else { return false }
        guard let configLoras = $0.configuration["loras"] as? [[String: Any]] else { return false }
        return loras.isSubset(of: configLoras.compactMap { $0["file"] as? String })
      } ?? configurations.first(where: first)
    }
    return configurations.first {
      let isFirst = first($0)
      guard isFirst else { return false }
      guard let configLoras = $0.configuration["loras"] as? [[String: Any]] else { return false }
      return loras.isSubset(of: configLoras.compactMap { $0["file"] as? String })
    } ?? configurations.first {
      let isSecond = second($0)
      guard isSecond else { return false }
      guard let configLoras = $0.configuration["loras"] as? [[String: Any]] else { return false }
      return loras.isSubset(of: configLoras.compactMap { $0["file"] as? String })
    } ?? configurations.first(where: first) ?? configurations.first(where: second)
  }

  private static func fetchCommunityConfigurations(timeout: TimeInterval) throws
    -> [ConfigurationZoo.Specification]
  {
    guard let url = URL(string: "https://models.drawthings.ai/configs.json") else {
      return []
    }
    var request = URLRequest(url: url)
    request.timeoutInterval = timeout
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<Data, Error> = .failure(DrawThingsCLIError.invalidConfigurationJSON)
    URLSession.shared.dataTask(with: request) { data, response, error in
      defer { semaphore.signal() }
      if let error {
        result = .failure(error)
        return
      }
      guard let response = response as? HTTPURLResponse, 200...299 ~= response.statusCode else {
        result = .failure(DrawThingsCLIError.invalidConfigurationJSON)
        return
      }
      guard let data else {
        result = .failure(DrawThingsCLIError.invalidConfigurationJSON)
        return
      }
      result = .success(data)
    }.resume()
    guard semaphore.wait(timeout: .now() + timeout) != .timedOut else {
      return []
    }
    guard case .success(let data) = result else {
      return []
    }
    let specifications = parseCommunityConfigurations(data: data)
    guard !specifications.isEmpty else {
      return []
    }
    try? persistCommunityConfigurations(data)
    return specifications
  }

  private static func parseCommunityConfigurations(data: Data) -> [ConfigurationZoo.Specification] {
    guard let jsonSpecifications = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
    else {
      return []
    }
    return jsonSpecifications.compactMap { specification in
      guard let name = specification["name"] as? String,
        let configuration = specification["configuration"] as? [String: Any]
      else {
        return nil
      }
      return ConfigurationZoo.Specification(
        name: name,
        version: (specification["version"] as? String).flatMap { ModelVersion(rawValue: $0) },
        negative: specification["negative"] as? String,
        configuration: configuration)
    }
  }

  private static func cachedCommunityConfigurations(modelsDirectory: URL)
    -> [ConfigurationZoo.Specification]
  {
    for url in NetworkCacheResolver.candidateURLs(
      fileName: "configs.json", modelsDirectory: modelsDirectory)
    {
      guard let data = try? Data(contentsOf: url) else { continue }
      let configurations = parseCommunityConfigurations(data: data)
      if !configurations.isEmpty {
        return configurations
      }
    }
    return []
  }

  private static func persistCommunityConfigurations(_ data: Data) throws {
    try NetworkCacheResolver.persist(data, fileName: "configs.json")
  }
}

private enum CommunityModelResolver {
  static func resolve(
    _ input: String, modelsDirectory: URL, allowNetwork: Bool = true, timeout: TimeInterval = 10
  ) -> ModelZoo.Specification? {
    let local = localCommunitySpecifications(
      modelsDirectory: modelsDirectory, allowNetwork: false)
    if let specification = matchingSpecification(for: input, in: local) {
      primeOverrideMapping(with: specification)
      return specification
    }
    guard !isDownloadedFileReference(input) else {
      return nil
    }
    guard allowNetwork else {
      return nil
    }
    let fetched = (try? fetchCommunitySpecifications(timeout: timeout)) ?? []
    if let specification = matchingSpecification(for: input, in: fetched) {
      primeOverrideMapping(with: specification)
      return specification
    }
    return nil
  }

  static func allSpecifications(
    modelsDirectory: URL, allowNetwork: Bool = true, timeout: TimeInterval = 10
  )
    -> [ModelZoo.Specification]
  {
    let official = ModelZoo.availableSpecifications.filter { $0.remoteApiModelConfig == nil }
    let community = localCommunitySpecifications(
      modelsDirectory: modelsDirectory, allowNetwork: allowNetwork, timeout: timeout)
    var seen = Set<String>()
    var combined = [ModelZoo.Specification]()
    for specification in official + community where !seen.contains(specification.file) {
      seen.insert(specification.file)
      combined.append(specification)
    }
    return combined
  }

  private static func localCommunitySpecifications(
    modelsDirectory: URL, allowNetwork: Bool, timeout: TimeInterval = 10
  )
    -> [ModelZoo.Specification]
  {
    let cached = cachedCommunitySpecifications(modelsDirectory: modelsDirectory)
    if !cached.isEmpty {
      return cached.filter { $0.remoteApiModelConfig == nil }
    }
    guard allowNetwork else {
      return []
    }
    return ((try? fetchCommunitySpecifications(timeout: timeout)) ?? []).filter {
      $0.remoteApiModelConfig == nil
    }
  }

  private static func cachedCommunitySpecifications(modelsDirectory: URL)
    -> [ModelZoo.Specification]
  {
    for url in NetworkCacheResolver.candidateURLs(
      fileName: "models.json", modelsDirectory: modelsDirectory)
    {
      guard let data = try? Data(contentsOf: url) else { continue }
      let specifications = parseCommunitySpecifications(data: data)
      if !specifications.isEmpty {
        return specifications
      }
    }
    return []
  }

  private static func fetchCommunitySpecifications(timeout: TimeInterval) throws
    -> [ModelZoo.Specification]
  {
    guard let url = URL(string: "https://models.drawthings.ai/models.json") else {
      return []
    }
    var request = URLRequest(url: url)
    request.timeoutInterval = timeout
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<Data, Error> = .failure(DrawThingsCLIError.invalidConfigurationJSON)
    URLSession.shared.dataTask(with: request) { data, response, error in
      defer { semaphore.signal() }
      if let error {
        result = .failure(error)
        return
      }
      guard let response = response as? HTTPURLResponse, 200...299 ~= response.statusCode else {
        result = .failure(DrawThingsCLIError.invalidConfigurationJSON)
        return
      }
      guard let data else {
        result = .failure(DrawThingsCLIError.invalidConfigurationJSON)
        return
      }
      result = .success(data)
    }.resume()
    guard semaphore.wait(timeout: .now() + timeout) != .timedOut else {
      return []
    }
    guard case .success(let data) = result else {
      return []
    }
    let specifications = parseCommunitySpecifications(data: data)
    guard !specifications.isEmpty else {
      return []
    }
    try? NetworkCacheResolver.persist(data, fileName: "models.json")
    return specifications
  }

  private static func parseCommunitySpecifications(data: Data) -> [ModelZoo.Specification] {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let specifications = try? decoder.decode(
        [FailableDecodable<ModelZoo.Specification>].self, from: data
      ).compactMap({ $0.value })
    else {
      return []
    }
    return specifications
  }

  private static func primeOverrideMapping(with specification: ModelZoo.Specification) {
    ModelZoo.overrideMapping[specification.file] = specification
    if let huggingFaceLink = specification.huggingFaceLink,
      let repo = ModelZoo.normalizeHuggingFaceRepo(huggingFaceLink)
    {
      ModelZoo.huggingFaceRepoOverrideMapping[repo] = specification
    }
  }

  private static func matchingSpecification(
    for input: String, in specifications: [ModelZoo.Specification]
  ) -> ModelZoo.Specification? {
    if let exactFileMatch = specifications.first(where: { $0.file == input }) {
      return exactFileMatch
    }
    if let exactNameMatch = specifications.first(where: { $0.name == input }) {
      return exactNameMatch
    }
    guard let canonicalRepo = ModelZoo.normalizeHuggingFaceRepo(input) else {
      return nil
    }
    return specifications.first { specification in
      guard let huggingFaceLink = specification.huggingFaceLink,
        let specificationRepo = ModelZoo.normalizeHuggingFaceRepo(huggingFaceLink)
      else {
        return false
      }
      return specificationRepo == canonicalRepo
    }
  }

  private static func isDownloadedFileReference(_ input: String) -> Bool {
    let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return false }
    return ModelZoo.isModelDownloaded(trimmed)
  }
}

private enum SHARefresh {
  static func refresh(timeout: TimeInterval = 15) {
    guard !NetworkAccessPolicy.offline else { return }
    let endpoints = [
      "https://models.drawthings.ai/models_sha256.json",
      "https://models.drawthings.ai/uncurated_models_sha256.json",
    ]
    for endpoint in endpoints {
      guard let url = URL(string: endpoint) else { continue }
      if let map = try? fetchSHA(url: url, timeout: timeout) {
        ModelZoo.mergeFileSHA256(map)
      }
    }
  }

  private static func fetchSHA(url: URL, timeout: TimeInterval) throws -> [String: String] {
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<[String: String], Error> = .failure(
      DrawThingsCLIError.invalidConfigurationJSON)
    var request = URLRequest(url: url)
    request.timeoutInterval = timeout
    URLSession.shared.dataTask(with: request) { data, response, error in
      defer { semaphore.signal() }
      if let error {
        result = .failure(error)
        return
      }
      if let response = response as? HTTPURLResponse, !(200...299).contains(response.statusCode) {
        result = .failure(URLError(.badServerResponse))
        return
      }
      guard let data,
        let map = try? JSONDecoder().decode([String: String].self, from: data)
      else {
        result = .failure(DrawThingsCLIError.invalidConfigurationJSON)
        return
      }
      result = .success(map)
    }.resume()
    semaphore.wait()
    return try result.get()
  }
}

private enum ModelDownloader {
  private final class DownloadProgressPrinter {
    private let file: String
    private let index: Int
    private let total: Int
    private let byteFormatter: ByteCountFormatter
    private var lastLineLength: Int = 0
    private var hasRendered = false

    init(file: String, index: Int, total: Int) {
      self.file = file
      self.index = index
      self.total = total
      let byteFormatter = ByteCountFormatter()
      byteFormatter.countStyle = .file
      byteFormatter.allowedUnits = [.useBytes, .useKB, .useMB, .useGB, .useTB]
      self.byteFormatter = byteFormatter
    }

    func update(totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64, isComplete: Bool) {
      let line = renderLine(
        totalBytesWritten: totalBytesWritten,
        totalBytesExpectedToWrite: totalBytesExpectedToWrite,
        isComplete: isComplete)
      let padding = String(
        repeating: " ", count: max(0, lastLineLength - line.count))
      let output = "\r\(line)\(padding)\(isComplete ? "\n" : "")"
      FileHandle.standardOutput.write(Data(output.utf8))
      hasRendered = true
      lastLineLength = isComplete ? 0 : line.count
    }

    func finishLineIfNeeded() {
      guard hasRendered, lastLineLength > 0 else { return }
      FileHandle.standardOutput.write(Data("\n".utf8))
      lastLineLength = 0
    }

    private func renderLine(
      totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64, isComplete: Bool
    ) -> String {
      let progress =
        totalBytesExpectedToWrite > 0
        ? min(1, max(0, Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)))
        : (isComplete ? 1 : 0)
      let percent = Int((progress * 100).rounded())
      let barWidth = 24
      let filledWidth = min(barWidth, Int((Double(barWidth) * progress).rounded(.down)))
      let bar: String
      if filledWidth >= barWidth {
        bar = String(repeating: "=", count: barWidth)
      } else {
        bar =
          String(repeating: "=", count: filledWidth)
          + ">"
          + String(repeating: " ", count: max(0, barWidth - filledWidth - 1))
      }
      let written = byteFormatter.string(fromByteCount: totalBytesWritten)
      let totalBytes =
        totalBytesExpectedToWrite > 0
        ? byteFormatter.string(fromByteCount: totalBytesExpectedToWrite)
        : "?"
      return
        "[\(index)/\(total)] \(file) [\(bar)] \(String(format: "%3d", percent))% \(written)/\(totalBytes)"
    }
  }

  static func ensureFiles(
    _ files: [String], modelsDirectory: URL, downloadMissing: Bool
  ) throws {
    let orderedUniqueFiles = orderedSet(files)
    let missing = orderedUniqueFiles.filter { !ModelZoo.isModelDownloaded($0) }
    guard !missing.isEmpty else { return }

    if !downloadMissing {
      throw DrawThingsCLIError.missingModelFiles(missing)
    }
    if NetworkAccessPolicy.offline {
      throw ValidationError(
        "Offline mode is enabled and model files are missing:\n\(missing.map { "  - \($0)" }.joined(separator: "\n"))"
      )
    }

    SHARefresh.refresh()
    for (index, file) in missing.enumerated() {
      try downloadFile(
        file, index: index + 1, total: missing.count, modelsDirectory: modelsDirectory)
    }
  }

  private static func downloadFile(
    _ file: String, index: Int, total: Int, modelsDirectory _: URL
  ) throws {
    let encodedName = file.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? file
    guard let remoteURL = URL(string: "https://static.libnnc.org/\(encodedName)") else {
      throw ValidationError("Invalid remote URL for file \(file)")
    }
    let localURL = URL(fileURLWithPath: ModelZoo.filePathForModelDownloaded(file))
    try FileManager.default.createDirectory(
      at: localURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    let expectedSHA = ModelZoo.fileSHA256ForModelDownloaded(file)
    let semaphore = DispatchSemaphore(value: 0)
    var outputError: Error?
    let progressPrinter = DownloadProgressPrinter(file: file, index: index, total: total)
    let downloader = ResumableDownloader(
      remoteUrl: remoteURL, localUrl: localURL, sha256: expectedSHA)
    downloader.resume { totalBytesWritten, totalBytesExpectedToWrite, isComplete, error in
      if let error {
        progressPrinter.finishLineIfNeeded()
        outputError = error
        semaphore.signal()
        return
      }
      progressPrinter.update(
        totalBytesWritten: totalBytesWritten,
        totalBytesExpectedToWrite: totalBytesExpectedToWrite,
        isComplete: isComplete)
      if isComplete {
        semaphore.signal()
      }
    }
    semaphore.wait()
    if let outputError {
      throw outputError
    }
  }

  private static func orderedSet(_ files: [String]) -> [String] {
    var seen = Set<String>()
    var ordered = [String]()
    for file in files where !seen.contains(file) {
      seen.insert(file)
      ordered.append(file)
    }
    return ordered
  }
}

private func resolvedLocalFileURL(_ path: String) throws -> URL {
  let expanded = (path as NSString).expandingTildeInPath
  let url = URL(fileURLWithPath: expanded)
  guard FileManager.default.fileExists(atPath: url.path) else {
    throw ValidationError("File does not exist: \(path)")
  }
  return url.standardizedFileURL
}

private func resolvedOptionalLocalFileURL(_ path: String?) throws -> URL? {
  guard let path, !path.isEmpty else { return nil }
  return try resolvedLocalFileURL(path)
}

private func defaultImportedModelDisplayName(for artifactURL: URL) -> String {
  artifactURL.deletingPathExtension().lastPathComponent
}

private func defaultImportScale(for version: ModelVersion, artifactFileName: String) -> UInt16 {
  switch version {
  case .hunyuanVideo, .wan21_14b, .wan22_5b:
    return 12
  case .wan21_1_3b:
    return 8
  case .sdxlBase, .sdxlRefiner, .ssd1b, .hiDreamI1, .qwenImage, .zImage, .wurstchenStageC,
    .wurstchenStageB, .sd3, .sd3Large, .auraflow, .flux1, .flux2, .flux2_9b, .flux2_4b,
    .ltx2, .ltx2_3:
    return 16
  case .pixart:
    return artifactFileName.contains("512") ? 8 : 16
  case .v1, .v2, .kandinsky21, .svdI2v:
    return 8
  }
}

private func validateCustomTextEncoderSupport(
  version: ModelVersion, textEncoderURL: URL?, textEncoder2URL: URL?
) throws {
  if let textEncoder2URL, version != .sdxlBase {
    throw ValidationError(
      "--text-encoder-2 is only supported for Stable Diffusion XL Base imports, got \(textEncoder2URL.lastPathComponent)."
    )
  }
  guard textEncoderURL != nil || textEncoder2URL != nil else { return }
  switch version {
  case .v1, .v2, .sdxlBase, .ssd1b, .sdxlRefiner:
    return
  case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .pixart,
    .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .qwenImage,
    .wan22_5b, .zImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3:
    throw ValidationError(
      "Custom text encoder import is not supported for \(ModelZoo.humanReadableNameForVersion(version))."
    )
  }
}

private func projectedImportedOutputFiles(
  modelName: String, version: ModelVersion, includeAutoencoder: Bool, includeTextEncoder: Bool,
  includeTextEncoder2: Bool
) throws -> [String] {
  var files = ["\(modelName)_f16.ckpt"]
  if includeTextEncoder {
    switch version {
    case .v1:
      files.append("\(modelName)_clip_vit_l14_f16.ckpt")
    case .v2:
      files.append("\(modelName)_open_clip_vit_h14_f16.ckpt")
    case .sdxlBase, .ssd1b:
      files.append("\(modelName)_clip_vit_l14_f16.ckpt")
      files.append("\(modelName)_open_clip_vit_bigg14_f16.ckpt")
    case .sdxlRefiner:
      files.append("\(modelName)_open_clip_vit_bigg14_f16.ckpt")
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .pixart,
      .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .qwenImage,
      .wan22_5b, .zImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3:
      break
    }
  }
  if includeTextEncoder2 && version == .sdxlBase {
    let secondEncoder = "\(modelName)_open_clip_vit_bigg14_f16.ckpt"
    if !files.contains(secondEncoder) {
      files.append(secondEncoder)
    }
  }
  if includeAutoencoder {
    files.append("\(modelName)_vae_f16.ckpt")
  }
  return files
}

private func existingImportedOutputs(for files: [String]) -> [String] {
  files.filter { FileManager.default.fileExists(atPath: ModelZoo.filePathForModelDownloaded($0)) }
}

private func importPrefix(triggerWord: String?) -> String {
  let trimmed = triggerWord?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
  return trimmed.isEmpty ? "" : "\(trimmed) "
}

private func importedDependencyFiles(
  specification: ModelZoo.Specification,
  additionalModels: [(name: String, subtitle: String, file: String)],
  importedFiles: [String]
) -> [String] {
  let imported = Set(importedFiles)
  let files =
    additionalModels.map(\.file)
    + ModelZoo.filesToDownload(specification).map(\.file).filter { !imported.contains($0) }
  var seen = Set<String>()
  return files.filter { seen.insert($0).inserted }
}

private func printImportedModelSummary(
  specification: ModelZoo.Specification, version: ModelVersion, modifier: SamplerModifier,
  importedFiles: [String], dependencyFiles: [String]
) {
  let rows = [
    ["MODEL", specification.file],
    ["NAME", specification.name],
    [
      "VERSION",
      "\(ModelZoo.humanReadableNameForVersion(version)) (\(String(describing: version)))",
    ],
    ["MODIFIER", String(describing: modifier)],
    ["TEXT_ENCODER", specification.textEncoder ?? "-"],
    ["AUTOENCODER", specification.autoencoder ?? "-"],
    [
      "DEFAULT_SCALE",
      "\(specification.defaultScale) (\(Int(specification.defaultScale) * 64)x\(Int(specification.defaultScale) * 64))",
    ],
    ["PREFIX", specification.prefix.isEmpty ? "-" : specification.prefix],
  ]
  printTable(headers: ["FIELD", "VALUE"], rows: rows, maxWidths: [18, 88])
  if !importedFiles.isEmpty {
    print("")
    printTable(
      headers: ["IMPORTED_FILE"],
      rows: importedFiles.map { [$0] },
      maxWidths: [88])
  }
  if !dependencyFiles.isEmpty {
    print("")
    printTable(
      headers: ["COMPANION_FILE"],
      rows: dependencyFiles.map { [$0] },
      maxWidths: [88])
  }
}

private func createTemporaryDirectory() throws -> String {
  let path = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
  return path.path
}

private func createLocalImageGenerator(queue: DispatchQueue) throws -> (String, LocalImageGenerator)
{
  let tempDir = try createTemporaryDirectory()
  let workspace = SQLiteWorkspace(
    filePath: "\(tempDir)/config.sqlite3", fileProtectionLevel: .noProtection)
  let configurations = workspace.fetch(for: GenerationConfiguration.self).where(
    GenerationConfiguration.id == 0, limit: .limit(0))

  let tokenizerV1 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_json,
    merges: BinaryResources.merges_txt,
    textualInversions: [])
  let tokenizerV2 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_16e6_json,
    merges: BinaryResources.bpe_simple_vocab_16e6_txt,
    textualInversions: [])
  let tokenizerKandinsky = SentencePieceTokenizer(
    data: BinaryResources.xlmroberta_bpe_model, startToken: 0,
    endToken: 2, tokenShift: 1)
  let tokenizerXL = tokenizerV2
  let tokenizerT5 = SentencePieceTokenizer(
    data: BinaryResources.t5_spiece_model, startToken: nil,
    endToken: 1, tokenShift: 0)
  let tokenizerPileT5 = SentencePieceTokenizer(
    data: BinaryResources.pile_t5_spiece_model, startToken: nil,
    endToken: 2, tokenShift: 0)
  let tokenizerChatGLM3 = SentencePieceTokenizer(
    data: BinaryResources.chatglm3_spiece_model, startToken: nil,
    endToken: nil, tokenShift: 0)
  let tokenizerLlama3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_llama3_json, merges: BinaryResources.merges_llama3_txt,
    specialTokens: [
      "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009,
      "<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
    ], unknownToken: "<|end_of_text|>", startToken: "<|begin_of_text|>",
    endToken: "<|end_of_text|>")
  let tokenizerUMT5 = SentencePieceTokenizer(
    data: BinaryResources.umt5_spiece_model, startToken: nil,
    endToken: 1, tokenShift: 0)
  let tokenizerQwen25 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen2_5_json, merges: BinaryResources.merges_qwen2_5_txt,
    specialTokens: [
      "</tool_call>": 151658, "<tool_call>": 151657, "<|box_end|>": 151649, "<|box_start|>": 151648,
      "<|endoftext|>": 151643, "<|file_sep|>": 151664, "<|fim_middle|>": 151660,
      "<|fim_pad|>": 151662, "<|fim_prefix|>": 151659, "<|fim_suffix|>": 151661,
      "<|im_end|>": 151645, "<|im_start|>": 151644, "<|image_pad|>": 151655,
      "<|object_ref_end|>": 151647, "<|object_ref_start|>": 151646, "<|quad_end|>": 151651,
      "<|quad_start|>": 151650, "<|repo_name|>": 151663, "<|video_pad|>": 151656,
      "<|vision_end|>": 151653, "<|vision_pad|>": 151654, "<|vision_start|>": 151652,
    ], unknownToken: "<|endoftext|>", startToken: "<|endoftext|>", endToken: "<|endoftext|>")
  let tokenizerQwen3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen3_json, merges: BinaryResources.merges_qwen3_txt,
    specialTokens: [
      "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
      "<|object_ref_start|>": 151646, "<|object_ref_end|>": 151647, "<|box_start|>": 151648,
      "<|box_end|>": 151649, "<|quad_start|>": 151650, "<|quad_end|>": 151651,
      "<|vision_start|>": 151652, "<|vision_end|>": 151653, "<|vision_pad|>": 151654,
      "<|image_pad|>": 151655, "<|video_pad|>": 151656, "<tool_call>": 151657,
      "</tool_call>": 151658, "<|fim_prefix|>": 151659, "<|fim_middle|>": 151660,
      "<|fim_suffix|>": 151661, "<|fim_pad|>": 151662, "<|repo_name|>": 151663,
      "<|file_sep|>": 151664, "<tool_response>": 151665, "</tool_response>": 151666,
      "<think>": 151667, "</think>": 151668,
    ], unknownToken: "<|endoftext|>", startToken: "<|endoftext|>", endToken: "<|endoftext|>")
  let tokenizerMistral3 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_mistral3_json, merges: BinaryResources.merges_mistral3_txt,
    specialTokens: [
      "<unk>": 0, "<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4, "[AVAILABLE_TOOLS]": 5,
      "[/AVAILABLE_TOOLS]": 6, "[TOOL_RESULTS]": 7, "[/TOOL_RESULTS]": 8, "[TOOL_CALLS]": 9,
      "[IMG]": 10, "<pad>": 11, "[IMG_BREAK]": 12, "[IMG_END]": 13, "[PREFIX]": 14, "[MIDDLE]": 15,
      "[SUFFIX]": 16, "[SYSTEM_PROMPT]": 17, "[/SYSTEM_PROMPT]": 18, "[TOOL_CONTENT]": 19,
    ], unknownToken: "<unk>", startToken: "<s>", endToken: "</s>")
  let tokenizerGemma3 = SentencePieceTokenizer(
    data: BinaryResources.gemma3_spiece_model, startToken: 2, endToken: nil, tokenShift: 0)
  let generator = LocalImageGenerator(
    queue: queue, configurations: configurations, workspace: workspace, tokenizerV1: tokenizerV1,
    tokenizerV2: tokenizerV2, tokenizerXL: tokenizerXL, tokenizerKandinsky: tokenizerKandinsky,
    tokenizerT5: tokenizerT5, tokenizerPileT5: tokenizerPileT5,
    tokenizerChatGLM3: tokenizerChatGLM3, tokenizerLlama3: tokenizerLlama3,
    tokenizerUMT5: tokenizerUMT5, tokenizerQwen25: tokenizerQwen25,
    tokenizerQwen3: tokenizerQwen3, tokenizerMistral3: tokenizerMistral3,
    tokenizerGemma3: tokenizerGemma3
  )
  return (tempDir, generator)
}

private final class LocalGenerationRunner {
  private enum OutputDestination {
    case png(URL)
    case video(URL, containerExtension: String)
  }

  private struct ImageTensorShape {
    let width: Int
    let height: Int
    let channels: Int
  }

  private let queue = DispatchQueue(label: "com.drawthings.cli.generate", qos: .userInteractive)
  private let temporaryDirectory: String
  private let imageGenerator: LocalImageGenerator

  init() throws {
    let (temporaryDirectory, imageGenerator) = try createLocalImageGenerator(queue: queue)
    self.temporaryDirectory = temporaryDirectory
    self.imageGenerator = imageGenerator
    DeviceCapability.cacheUri = URL(fileURLWithPath: temporaryDirectory)
  }

  deinit {
    try? FileManager.default.removeItem(atPath: temporaryDirectory)
  }

  func generate(
    prompt: String, negativePrompt: String, configuration: GenerationConfiguration,
    outputPath: String,
    inputImage: Tensor<FloatType>?, videoFormat: VideoExportFormat?
  ) throws -> [String] {
    let trace = ImageGeneratorTrace(fromBridge: true)
    let progressPrinter = ProgressBarPrinter()
    let estimation = GenerationEstimation.default
    let hints = [(ControlHintType, [(AnyTensor, Float)])]()
    let (images, _, _) = queue.sync {
      imageGenerator.generate(
        trace: trace, image: inputImage, scaleFactor: 1, mask: nil, hints: hints, text: prompt,
        negativeText: negativePrompt, configuration: configuration, fileMapping: [:], keywords: [],
        cancellation: { _ in },
        feedback: { signpost, signposts, _ in
          let (elapsed, estimatedTotal) = GenerationEstimator.estimateUpToDateDuration(
            from: estimation, signpost: signpost, signposts: signposts)
          if estimatedTotal > 0 {
            let progress = Float(elapsed / estimatedTotal)
            progressPrinter.update(progress: progress)
          }
          return true
        })
    }
    guard let images, !images.isEmpty else {
      throw DrawThingsCLIError.generationFailed
    }
    return try saveOutputs(
      images, outputPath: outputPath, configuration: configuration, videoFormat: videoFormat)
  }

  private func saveOutputs(
    _ tensors: [Tensor<FloatType>], outputPath: String, configuration: GenerationConfiguration,
    videoFormat: VideoExportFormat?
  ) throws -> [String] {
    let destination = try normalizedOutputDestination(outputPath)
    switch destination {
    case .png(let outputURL):
      if videoFormat != nil {
        throw ValidationError("--video-format can only be used with .mov or .mp4 output")
      }
      let outputPaths = imageOutputPaths(baseOutputURL: outputURL, count: tensors.count)
      for (tensor, path) in zip(tensors, outputPaths) {
        try writePNG(tensor: tensor, to: path)
      }
      return outputPaths
    case .video(let outputURL, let containerExtension):
      let framesPerSecond = ModelZoo.framesPerSecondForModel(configuration.model ?? "")
      let path = try writeVideo(
        tensors: tensors, to: outputURL, containerExtension: containerExtension,
        framesPerSecond: framesPerSecond, videoFormat: videoFormat ?? .h264)
      return [path]
    }
  }

  private func normalizedOutputDestination(_ outputPath: String) throws -> OutputDestination {
    var url = URL(fileURLWithPath: outputPath)
    if url.pathExtension.isEmpty {
      url = url.appendingPathExtension("png")
    }
    let ext = url.pathExtension.lowercased()
    let destination: OutputDestination
    switch ext {
    case "png":
      destination = .png(url)
    case "mov", "mp4":
      destination = .video(url, containerExtension: ext)
    default:
      throw DrawThingsCLIError.invalidOutputPath(url.path)
    }
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    return destination
  }

  private func imageOutputPaths(baseOutputURL: URL, count: Int) -> [String] {
    if count == 1 {
      return [baseOutputURL.path]
    }
    let basename = baseOutputURL.deletingPathExtension().path
    let ext = baseOutputURL.pathExtension
    let digits = max(4, String(count - 1).count)
    return (0..<count).map { index in
      "\(basename)-\(String(format: "%0\(digits)d", index)).\(ext)"
    }
  }

  private func imageTensorShape(_ tensor: Tensor<FloatType>) throws -> ImageTensorShape {
    let shape = tensor.shape
    switch shape.count {
    case 4:
      return ImageTensorShape(width: shape[2], height: shape[1], channels: shape[3])
    case 3:
      return ImageTensorShape(width: shape[1], height: shape[0], channels: shape[2])
    default:
      throw DrawThingsCLIError.unsupportedTensorShape("\(shape)")
    }
  }

  private func pixelByte(_ value: FloatType) -> UInt8 {
    UInt8(min(max(Int((value + 1) * 127.5), 0), 255))
  }

  private func writePNG(tensor: Tensor<FloatType>, to outputPath: String) throws {
    let shape = try imageTensorShape(tensor)
    let height = shape.height
    let width = shape.width
    let channels = shape.channels
    guard channels >= 3 else {
      throw DrawThingsCLIError.unsupportedTensorShape("\(tensor.shape)")
    }
    let pixelCount = width * height
    var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: pixelCount)
    tensor.withUnsafeBytes {
      guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      for i in 0..<pixelCount {
        let base = i * channels
        rgba[i].r = pixelByte(fp16[base])
        rgba[i].g = pixelByte(fp16[base + 1])
        rgba[i].b = pixelByte(fp16[base + 2])
        rgba[i].a = 255
      }
    }
    let image = PNG.Data.Rectangular(
      packing: rgba, size: (x: width, y: height),
      layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
    do {
      try image.compress(path: outputPath, level: 4)
    } catch {
      throw DrawThingsCLIError.pngEncodeFailed(outputPath)
    }
  }

  private func writeVideo(
    tensors: [Tensor<FloatType>], to outputURL: URL, containerExtension: String,
    framesPerSecond: Double, videoFormat: VideoExportFormat
  ) throws -> String {
    #if canImport(AVFoundation) && canImport(CoreMedia) && canImport(CoreVideo)
      guard let first = tensors.first else {
        throw DrawThingsCLIError.generationFailed
      }
      let firstShape = try imageTensorShape(first)
      guard firstShape.channels >= 3 else {
        throw DrawThingsCLIError.unsupportedTensorShape("\(first.shape)")
      }
      for frame in tensors.dropFirst() {
        let frameShape = try imageTensorShape(frame)
        guard frameShape.channels >= 3 else {
          throw DrawThingsCLIError.unsupportedTensorShape("\(frame.shape)")
        }
        guard frameShape.width == firstShape.width, frameShape.height == firstShape.height else {
          throw DrawThingsCLIError.unsupportedTensorShape(
            "Inconsistent frame dimensions: expected \(firstShape.width)x\(firstShape.height), got \(frameShape.width)x\(frameShape.height)"
          )
        }
      }
      if FileManager.default.fileExists(atPath: outputURL.path) {
        try? FileManager.default.removeItem(at: outputURL)
      }
      if containerExtension == "mp4",
        videoFormat == .prores4444 || videoFormat == .prores422hq
      {
        throw ValidationError("ProRes video formats require .mov output")
      }
      let fileType: AVFileType = containerExtension == "mp4" ? .mp4 : .mov
      let videoSettings = videoSettings(
        for: videoFormat, width: firstShape.width, height: firstShape.height)
      let writer = try AVAssetWriter(outputURL: outputURL, fileType: fileType)
      let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
      writerInput.expectsMediaDataInRealTime = false
      let pixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
        kCVPixelBufferWidthKey as String: firstShape.width,
        kCVPixelBufferHeightKey as String: firstShape.height,
      ]
      let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: writerInput, sourcePixelBufferAttributes: pixelBufferAttributes)

      guard writer.canAdd(writerInput) else {
        throw DrawThingsCLIError.videoEncodeFailed(outputURL.path)
      }
      writer.add(writerInput)
      guard writer.startWriting() else {
        throw writer.error ?? DrawThingsCLIError.videoEncodeFailed(outputURL.path)
      }
      writer.startSession(atSourceTime: .zero)
      guard let pixelBufferPool = adaptor.pixelBufferPool else {
        throw DrawThingsCLIError.videoEncodeFailed(outputURL.path)
      }

      let frameDuration = frameDurationForVideo(frameRate: framesPerSecond)
      for (index, tensor) in tensors.enumerated() {
        while !writerInput.isReadyForMoreMediaData {
          Thread.sleep(forTimeInterval: 0.002)
        }
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &pixelBuffer)
        guard status == kCVReturnSuccess, let pixelBuffer else {
          throw DrawThingsCLIError.videoEncodeFailed(outputURL.path)
        }
        try populate(
          pixelBuffer: pixelBuffer, with: tensor, expected: firstShape, outputPath: outputURL.path)
        let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(index))
        guard adaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
          throw writer.error ?? DrawThingsCLIError.videoEncodeFailed(outputURL.path)
        }
      }

      writerInput.markAsFinished()
      let semaphore = DispatchSemaphore(value: 0)
      var finishError: Error?
      writer.finishWriting {
        if writer.status != .completed {
          finishError = writer.error ?? DrawThingsCLIError.videoEncodeFailed(outputURL.path)
        }
        semaphore.signal()
      }
      semaphore.wait()
      if let finishError {
        throw finishError
      }
      return outputURL.path
    #else
      throw DrawThingsCLIError.unsupportedVideoOutput(outputURL.path)
    #endif
  }

  #if canImport(AVFoundation) && canImport(CoreMedia) && canImport(CoreVideo)
    private func videoSettings(
      for format: VideoExportFormat, width: Int, height: Int
    ) -> [String: Any] {
      switch format {
      case .prores4444:
        return [
          AVVideoCodecKey: AVVideoCodecType.proRes4444.rawValue,
          AVVideoWidthKey: NSNumber(value: width),
          AVVideoHeightKey: NSNumber(value: height),
        ]
      case .prores422hq:
        return [
          AVVideoCodecKey: AVVideoCodecType.proRes422HQ.rawValue,
          AVVideoWidthKey: NSNumber(value: width),
          AVVideoHeightKey: NSNumber(value: height),
        ]
      case .h264:
        return [
          AVVideoCodecKey: AVVideoCodecType.h264.rawValue,
          AVVideoWidthKey: NSNumber(value: width),
          AVVideoHeightKey: NSNumber(value: height),
          AVVideoCompressionPropertiesKey: [
            AVVideoAverageBitRateKey: max(9_500_000, Int((Double(width * height) * 5).rounded())),
            AVVideoProfileLevelKey: AVVideoProfileLevelH264High41,
            AVVideoMaxKeyFrameIntervalKey: 30,
            AVVideoAllowFrameReorderingKey: true,
          ],
        ]
      case .hevc:
        return [
          AVVideoCodecKey: AVVideoCodecType.hevc.rawValue,
          AVVideoWidthKey: NSNumber(value: width),
          AVVideoHeightKey: NSNumber(value: height),
          AVVideoCompressionPropertiesKey: [
            AVVideoAverageBitRateKey: max(7_500_000, Int((Double(width * height) * 4).rounded())),
            AVVideoMaxKeyFrameIntervalKey: 30,
            AVVideoAllowFrameReorderingKey: true,
          ],
        ]
      }
    }
  #endif

  #if canImport(AVFoundation) && canImport(CoreMedia) && canImport(CoreVideo)
    private func frameDurationForVideo(frameRate: Double) -> CMTime {
      guard frameRate > 0 else {
        return CMTime(value: 1, timescale: 30)
      }
      if abs(frameRate - frameRate.rounded()) < 1e-12 {
        return CMTime(value: 1, timescale: Int32(frameRate.rounded()))
      }
      let timescale: Int32 = 60_000
      let value = max(1, Int64((Double(timescale) / frameRate).rounded()))
      return CMTime(value: value, timescale: timescale)
    }

    private func populate(
      pixelBuffer: CVPixelBuffer, with tensor: Tensor<FloatType>, expected: ImageTensorShape,
      outputPath: String
    ) throws {
      let shape = try imageTensorShape(tensor)
      guard shape.width == expected.width, shape.height == expected.height, shape.channels >= 3
      else {
        throw DrawThingsCLIError.unsupportedTensorShape("\(tensor.shape)")
      }
      CVPixelBufferLockBaseAddress(pixelBuffer, [])
      defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
      guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        throw DrawThingsCLIError.videoEncodeFailed(outputPath)
      }
      let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
      let destination = baseAddress.assumingMemoryBound(to: UInt8.self)
      tensor.withUnsafeBytes {
        guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
        for y in 0..<shape.height {
          let row = destination.advanced(by: y * bytesPerRow)
          for x in 0..<shape.width {
            let pixelIndex = y * shape.width + x
            let source = pixelIndex * shape.channels
            let destinationIndex = x * 4
            row[destinationIndex] = pixelByte(fp16[source + 2])  // B
            row[destinationIndex + 1] = pixelByte(fp16[source + 1])  // G
            row[destinationIndex + 2] = pixelByte(fp16[source])  // R
            row[destinationIndex + 3] = 255
          }
        }
      }
    }
  #endif
}

private func printTable(headers: [String], rows: [[String]], maxWidths: [Int]? = nil) {
  guard !headers.isEmpty else { return }
  let columnCount = headers.count
  let normalizedRows: [[String]] = rows.map { row in
    (0..<columnCount).map { index in index < row.count ? row[index] : "" }
  }
  var widths = headers.map(\.count)
  for row in normalizedRows {
    for (index, value) in row.enumerated() {
      widths[index] = max(widths[index], value.count)
    }
  }
  if let maxWidths {
    for index in 0..<min(columnCount, maxWidths.count) {
      widths[index] = min(widths[index], maxWidths[index])
    }
  }

  func truncated(_ value: String, width: Int) -> String {
    guard value.count > width else { return value }
    guard width > 3 else { return String(value.prefix(width)) }
    return String(value.prefix(width - 3)) + "..."
  }

  func formattedRow(_ row: [String]) -> String {
    row.enumerated().map { index, rawValue in
      let value = truncated(rawValue, width: widths[index])
      if value.count < widths[index] {
        return value + String(repeating: " ", count: widths[index] - value.count)
      }
      return value
    }.joined(separator: "  ")
  }

  print(formattedRow(headers))
  print(widths.map { String(repeating: "-", count: $0) }.joined(separator: "  "))
  for row in normalizedRows {
    print(formattedRow(row))
  }
}

private func printModelList(
  limit: Int? = nil, downloadedOnly: Bool = false, modelsDirectory: URL, allowNetwork: Bool = true
) {
  let officialFiles = Set(
    ModelZoo.availableSpecifications
      .filter { $0.remoteApiModelConfig == nil }
      .map(\.file))
  let specs = CommunityModelResolver.allSpecifications(
    modelsDirectory: modelsDirectory, allowNetwork: allowNetwork && !downloadedOnly)
  let filtered = downloadedOnly ? specs.filter { ModelZoo.isModelDownloaded($0) } : specs
  let output = limit.map { Array(filtered.prefix($0)) } ?? filtered
  if output.isEmpty {
    print("No models found.")
    return
  }
  let rows = output.map { spec in
    let downloaded = ModelZoo.isModelDownloaded(spec) ? "yes" : "no"
    let source = officialFiles.contains(spec.file) ? "official" : "community"
    let hf = (spec.huggingFaceLink?.isEmpty == false) ? (spec.huggingFaceLink ?? "") : "-"
    return [spec.file, spec.name, source, downloaded, hf]
  }
  printTable(
    headers: ["MODEL", "NAME", "SOURCE", "DOWNLOADED", "HUGGING_FACE"],
    rows: rows,
    maxWidths: [42, 42, 10, 10, 52])
}

private func printModelResolutionHelp(limit: Int = 20, modelsDirectory: URL) {
  printModelList(
    limit: limit, modelsDirectory: modelsDirectory, allowNetwork: !NetworkAccessPolicy.offline)
  print("Tip: run `\(CLIIdentity.command("models list"))` for the full list.")
}

private func unresolvedModelValidationError(_ input: String) -> ValidationError {
  let suggestions = ModelResolver.suggestions(input, limit: 5)
  guard !suggestions.isEmpty else {
    return ValidationError("Could not resolve --model '\(input)'.")
  }
  let lines = suggestions.map { "  - \($0.file) (\($0.name))" }.joined(separator: "\n")
  return ValidationError("Could not resolve --model '\(input)'.\nClosest matches:\n\(lines)")
}

private func requiredFiles(for configuration: GenerationConfiguration) -> [String] {
  ImageGeneratorUtils.filesToDownload(configuration, keywords: []).map(\.file)
}

private func createConfiguration(
  model: String, steps: Int?, cfg: Float?, width: Int?, height: Int?, frames: Int?, seed: UInt32?,
  strength: Float?, configJSON: String?, configFile: String?, modelsDirectory: URL
) throws -> ResolvedGenerationConfiguration {
  guard let modelSpec = ModelResolver.resolve(model, modelsDirectory: modelsDirectory) else {
    throw DrawThingsCLIError.unsupportedModelInput(model)
  }
  let resolvedConfiguration = try ConfigurationLoader.load(
    model: modelSpec.file, configJSON: configJSON, configFile: configFile,
    modelsDirectory: modelsDirectory)
  var builder = GenerationConfigurationBuilder(from: resolvedConfiguration.configuration)
  builder.model = modelSpec.file
  if let steps {
    guard steps >= 1 else { throw ValidationError("--steps must be >= 1") }
    builder.steps = UInt32(steps)
  }
  if let cfg {
    guard cfg >= 0 else { throw ValidationError("--cfg must be >= 0") }
    builder.guidanceScale = cfg
  }
  if let width {
    guard width % 64 == 0 else {
      throw DrawThingsCLIError.invalidImageDimensions(
        width, height ?? Int(builder.startHeight) * 64)
    }
    builder.startWidth = UInt16(width / 64)
  }
  if let height {
    guard height % 64 == 0 else {
      throw DrawThingsCLIError.invalidImageDimensions(width ?? Int(builder.startWidth) * 64, height)
    }
    builder.startHeight = UInt16(height / 64)
  }
  if let frames {
    guard frames >= 1 else { throw ValidationError("--frames must be >= 1") }
    builder.numFrames = UInt32(frames)
  }
  if let seed {
    builder.seed = seed
  }
  if let strength {
    guard (0...1).contains(strength) else { throw ValidationError("--strength must be in [0, 1]") }
    builder.strength = strength
  }
  return ResolvedGenerationConfiguration(
    configuration: builder.build(),
    recommendedNegativePrompt: resolvedConfiguration.recommendedNegativePrompt)
}

private func runLoRATraining(_ options: LoRATrainCommandOptions) throws {
  let configJSON = options.configurationOverrides.configJSON
  let (config, configJSONDictionary) = try LoRATrainConfigLoader.load(configJSON: configJSON)
  let resumeCheckpoint = try mergedAlias(
    primary: options.output.resume, alias: options.output.trainCheckpoint, primaryFlag: "--resume",
    aliasFlag: "--train-checkpoint")

  let modelsDirectory = try ModelsDirectoryResolver.resolve(
    path: options.modelAndDataset.modelsDirectoryOptions.modelsDir)
  ModelZoo.isExternalUrlsPreferred = true
  ModelZoo.externalUrls = [modelsDirectory]

  let modelInput = options.modelAndDataset.model ?? config.baseModel
  guard let modelInput else {
    printModelResolutionHelp(modelsDirectory: modelsDirectory)
    throw ValidationError("--model is required.")
  }
  guard
    let modelSpecification = ModelResolver.resolve(
      modelInput, modelsDirectory: modelsDirectory)
  else {
    printModelResolutionHelp(modelsDirectory: modelsDirectory)
    throw unresolvedModelValidationError(modelInput)
  }

  let datasetDirectory = options.modelAndDataset.dataset
  guard let datasetDirectory, !datasetDirectory.isEmpty else {
    throw ValidationError("--dataset is required.")
  }

  let output = options.output.output ?? config.name ?? "lora_output"
  let name = options.output.name ?? config.name ?? output
  let trainingSteps = options.training.steps ?? Int(config.trainingSteps)
  guard trainingSteps > 0 else { throw ValidationError("--steps must be > 0") }
  let rank = options.training.rank ?? Int(config.networkDim)
  guard rank > 0 else { throw ValidationError("--rank must be > 0") }
  let loraScale = options.training.scale ?? config.networkScale
  guard loraScale > 0 else { throw ValidationError("--scale must be > 0") }
  let unetLearningRate: ClosedRange<Float> =
    if let learningRate = options.training.learningRate {
      try parseLearningRateRange(learningRate)
    } else {
      min(
        config.unetLearningRate, config.unetLearningRateLowerBound)...max(
          config.unetLearningRate, config.unetLearningRateLowerBound)
    }
  let textModelLearningRate = config.textModelLearningRate
  let gradientAccumulation =
    options.training.gradientAccumulation ?? Int(config.gradientAccumulationSteps)
  guard gradientAccumulation > 0 else {
    throw ValidationError("--gradient-accumulation must be > 0")
  }
  let warmupSteps = options.training.warmupSteps ?? Int(config.warmupSteps)
  let saveEvery = options.training.saveEvery ?? Int(config.saveEveryNSteps)
  let seed = options.training.seed ?? config.seed
  let clipSkip = options.dataAndModel.clipSkip ?? Int(config.clipSkip)
  let noiseOffset = options.dataAndModel.noiseOffset ?? config.noiseOffset
  let captionDropout = options.dataAndModel.captionDropout ?? config.captionDropoutRate
  let maxTextLength = Int(config.maxTextLength)
  let useAspectRatio = options.dataAndModel.useAspectRatio ?? config.useImageAspectRatio
  let guidanceEmbedLowerBound = min(config.guidanceEmbedLowerBound, config.guidanceEmbedUpperBound)
  let guidanceEmbedUpperBound = max(config.guidanceEmbedLowerBound, config.guidanceEmbedUpperBound)
  let guidanceEmbed = guidanceEmbedLowerBound...guidanceEmbedUpperBound
  let denoisingLowerBound = min(max(config.denoisingStart, 0), 1)
  let denoisingUpperBound = min(max(config.denoisingEnd, 0), 1)
  let denoisingTimesteps =
    Int(
      (min(denoisingLowerBound, denoisingUpperBound) * 999).rounded())...Int(
      (max(denoisingLowerBound, denoisingUpperBound) * 999).rounded())
  let orthonormalDown = options.dataAndModel.orthonormalDown ?? config.orthonormalLoraDown
  let cotrainTextModel = options.dataAndModel.cotrainTextModel ?? config.cotrainTextModel
  let cotrainCustomEmbedding = config.cotrainCustomEmbedding
  let customEmbeddingLength = Int(config.customEmbeddingLength)
  let customEmbeddingLearningRate = config.customEmbeddingLearningRate
  let stopEmbeddingTrainingAtStep = Int(config.stopEmbeddingTrainingAtStep)
  let memorySaverFromConfig = try parseMemorySaverValue(
    LoRATrainConfigLoader.memorySaverValue(from: configJSONDictionary))
  let weightsMemoryFromConfig = try parseWeightsMemoryValue(
    LoRATrainConfigLoader.weightsMemoryValue(from: configJSONDictionary))
  let memorySaver =
    try options.configurationOverrides.memorySaver.map { try parseMemorySaver($0) }
    ?? memorySaverFromConfig ?? .balanced
  let weightsMemory =
    try options.configurationOverrides.weightsMemory.map { try parseWeightsMemory($0) }
    ?? weightsMemoryFromConfig ?? .cached
  let shouldDryRun = options.execution.dryRun

  let widthPx =
    options.dataAndModel.width ?? options.dataAndModel.resolution
    ?? Int(config.startWidth) * 64
  let heightPx =
    options.dataAndModel.height ?? options.dataAndModel.resolution
    ?? Int(config.startHeight) * 64
  guard widthPx > 0 && widthPx % 64 == 0 else {
    throw ValidationError("--width must be a positive multiple of 64")
  }
  guard heightPx > 0 && heightPx % 64 == 0 else {
    throw ValidationError("--height must be a positive multiple of 64")
  }
  let imageScale = DeviceCapability.Scale(
    widthScale: UInt16(widthPx / 64), heightScale: UInt16(heightPx / 64))
  let additionalScales: [DeviceCapability.Scale] =
    useAspectRatio
    ? config.additionalScales.map {
      DeviceCapability.Scale(widthScale: $0, heightScale: $0)
    } : []
  let resolvedShift: Float =
    config.resolutionDependentShift
    ? Float(ModelZoo.shiftFor((width: imageScale.widthScale, height: imageScale.heightScale)))
    : config.shift
  let powerEMA: ClosedRange<Float>? =
    config.powerEmaLowerBound > 0 && config.powerEmaUpperBound > 0
    ? min(
      config.powerEmaLowerBound, config.powerEmaUpperBound)...max(
        config.powerEmaLowerBound, config.powerEmaUpperBound) : nil
  let stepsBetweenRestarts = Int(config.stepsBetweenRestarts)

  let datasetInputs = try loadTrainingDataset(from: datasetDirectory)
  guard !datasetInputs.isEmpty else {
    throw ValidationError("No dataset images found in '\(datasetDirectory)'.")
  }

  print("Models directory: \(modelsDirectory.path)")
  print("Model: \(modelSpecification.file)")
  print("Dataset: \(datasetDirectory)")
  print("Name: \(name)")
  print("Training steps: \(trainingSteps)")
  print("Rank: \(rank), scale: \(loraScale)")
  print("Resolution: \(widthPx)x\(heightPx)")
  print("Dataset size: \(datasetInputs.count) sample(s)")
  if shouldDryRun {
    print("Dry run completed. Use the same command without --dry-run to start training.")
    return
  }

  let files = ModelZoo.filesToDownload(modelSpecification).map(\.file)
  try ModelDownloader.ensureFiles(
    files, modelsDirectory: modelsDirectory, downloadMissing: options.execution.downloadMissing)

  let tokenizers = createLoRATrainerTokenizers()
  let session = UUID().uuidString
  let trainer = LoRATrainer(
    tensorBoard: nil, comment: "\(output)-",
    model: modelSpecification.file, scale: imageScale, additionalScales: additionalScales,
    useImageAspectRatio: useAspectRatio, cotrainTextModel: cotrainTextModel,
    cotrainCustomEmbedding: cotrainCustomEmbedding, clipSkip: clipSkip,
    maxTextLength: maxTextLength, session: session, resumeIfPossible: false,
    imageInspector: inspectTrainingImage, imageLoader: loadTrainingTensor)

  let tokenizerStack: [TextualInversionPoweredTokenizer & Tokenizer]
  switch trainer.version {
  case .v1:
    tokenizerStack = [tokenizers.tokenizerV1]
  case .v2:
    tokenizerStack = [tokenizers.tokenizerV2]
  case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b:
    switch trainer.textEncoderVersion {
    case .chatglm3_6b:
      tokenizerStack = [tokenizers.tokenizerChatGLM3]
    case nil:
      tokenizerStack = [tokenizers.tokenizerV2]
    }
  case .flux1:
    tokenizerStack = [tokenizers.tokenizerV2, tokenizers.tokenizerT5]
  case .qwenImage:
    tokenizerStack = [tokenizers.tokenizerQwen25]
  default:
    throw ValidationError("Unsupported LoRA trainer model version: \(trainer.version)")
  }

  guard
    let (dataFrame, zeroCaption) = trainer.prepareDataset(
      inputs: datasetInputs, tokenizers: tokenizerStack,
      customEmbeddingLength: customEmbeddingLength,
      progressHandler: { state, index in
        switch state {
        case .imageEncoding:
          print("\rEncoding images: \(index)/\(datasetInputs.count)", terminator: "")
        case .conditionalEncoding:
          print("\rEncoding captions: \(index)/\(datasetInputs.count)", terminator: "")
        }
        fflush(stdout)
        return true
      })
  else {
    throw ValidationError("Failed to prepare training dataset.")
  }
  print("")

  let trainableKeys: [String] = trainer.version == .flux1 ? flux1TrainableKeys() : []
  let startTime = Date()
  var lastStep = 0

  let resumingLoRAFile: (String, Int)? = {
    guard let resumeCheckpoint, !resumeCheckpoint.isEmpty else { return nil }
    let fileName = URL(fileURLWithPath: resumeCheckpoint).lastPathComponent
    let nameWithoutSuffix = fileName.replacingOccurrences(of: "_lora_f32.ckpt", with: "")
    guard let stepString = nameWithoutSuffix.split(separator: "_").last,
      let step = Int(stepString)
    else {
      return nil
    }
    return (fileName, step)
  }()

  trainer.train(
    resumingLoRAFile: resumingLoRAFile,
    tokenizers: tokenizerStack, dataFrame: dataFrame, zeroCaption: zeroCaption,
    trainingSteps: trainingSteps, warmupSteps: warmupSteps,
    gradientAccumulationSteps: gradientAccumulation,
    rankOfLoRA: rank, scaleOfLoRA: loraScale,
    textModelLearningRate: textModelLearningRate, unetLearningRate: unetLearningRate,
    stepsBetweenRestarts: stepsBetweenRestarts, seed: seed, trainableKeys: trainableKeys,
    customEmbeddingLength: customEmbeddingLength,
    customEmbeddingLearningRate: customEmbeddingLearningRate,
    stopEmbeddingTrainingAtStep: stopEmbeddingTrainingAtStep,
    resolutionDependentShift: config.resolutionDependentShift, shift: resolvedShift,
    noiseOffset: noiseOffset, guidanceEmbed: guidanceEmbed,
    denoisingTimesteps: denoisingTimesteps,
    captionDropoutRate: captionDropout, orthonormalLoRADown: orthonormalDown,
    powerEMA: powerEMA,
    memorySaver: memorySaver, weightsMemory: weightsMemory
  ) { state, loss, checkpoint in
    switch state {
    case .compile:
      print("[LoRA] Compiling computation graph...")
    case .step(let step):
      lastStep = step
      let elapsed = Date().timeIntervalSince(startTime)
      let itPerSec = step > 0 ? Double(step) / elapsed : 0
      print(
        "[LoRA] Step \(step)/\(trainingSteps) | Loss: \(String(format: "%.4f", loss)) | \(String(format: "%.2f", itPerSec)) it/s"
      )
      let shouldSave =
        step == trainingSteps || (saveEvery > 0 && step > 0 && step % saveEvery == 0)
      if shouldSave {
        let filename = "\(output)_\(step)_lora_f32.ckpt"
        let outputPath = LoRAZoo.filePathForModelDownloaded(filename)
        checkpoint.makeLoRA(to: outputPath, scale: loraScale)
        print("[LoRA] Saved checkpoint: \(filename)")
      }
    }
    fflush(stdout)
    return true
  }

  let totalTime = Date().timeIntervalSince(startTime)
  print("Training complete in \(String(format: "%.1f", totalTime))s.")
  print("Final checkpoint: \(output)_\(lastStep)_lora_f32.ckpt")
}

@main
struct DrawThingsCLI: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: CLIIdentity.commandName,
    abstract: "Local inference and training CLI for Draw Things models.",
    discussion: CLIHelpText.root,
    version: CLIIdentity.version,
    subcommands: [Generate.self, Models.self, Train.self, Completion.self]
  )
}

extension DrawThingsCLI {
  struct Generate: ParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Run local inference and save generated output image(s) or video.",
      discussion: CLIHelpText.generate)

    @OptionGroup(title: "Model Resolution") var modelResolution: GenerateModelResolutionOptions
    @OptionGroup(title: "Prompts") var prompts: GeneratePromptOptions
    @OptionGroup(title: "Sampling") var sampling: GenerateSamplingOptions
    @OptionGroup(title: "Configuration Overrides")
    var configurationOverrides: GenerateConfigurationOverrideOptions
    @OptionGroup(title: "Image Input") var imageInput: GenerateImageInputOptions
    @OptionGroup(title: "Output") var output: GenerateOutputOptions
    @OptionGroup(title: "Execution") var execution: GenerateExecutionOptions

    mutating func run() throws {
      NetworkAccessPolicy.offline = execution.offline
      let modelsDirectory = try ModelsDirectoryResolver.resolve(
        path: modelResolution.modelsDirectoryOptions.modelsDir)
      ModelZoo.isExternalUrlsPreferred = true
      ModelZoo.externalUrls = [modelsDirectory]
      try validateVideoOutputOptions(outputPath: output.output, videoFormat: output.videoFormat)

      guard let model = modelResolution.model else {
        printModelResolutionHelp(modelsDirectory: modelsDirectory)
        throw ValidationError("--model is required.")
      }
      guard ModelResolver.resolve(model, modelsDirectory: modelsDirectory) != nil else {
        printModelResolutionHelp(modelsDirectory: modelsDirectory)
        throw unresolvedModelValidationError(model)
      }

      let resolvedConfiguration = try createConfiguration(
        model: model, steps: sampling.steps, cfg: sampling.cfg, width: sampling.width,
        height: sampling.height, frames: sampling.frames,
        seed: sampling.seed, strength: sampling.strength,
        configJSON: configurationOverrides.configJSON,
        configFile: configurationOverrides.configFile,
        modelsDirectory: modelsDirectory)
      let promptValues = try resolvedPrompts(prompts)
      let configuration = resolvedConfiguration.configuration
      let resolvedNegativePrompt =
        promptValues.negative ?? resolvedConfiguration.recommendedNegativePrompt ?? ""

      let files = requiredFiles(for: configuration)
      try ModelDownloader.ensureFiles(
        files, modelsDirectory: modelsDirectory, downloadMissing: execution.downloadMissing)

      let imagePath = try mergedAlias(
        primary: try mergedAlias(
          primary: imageInput.image, alias: imageInput.initImage, primaryFlag: "--image",
          aliasFlag: "--init-image"),
        alias: imageInput.inputImage, primaryFlag: "--image", aliasFlag: "--input-image")
      let inputImageTensor: Tensor<FloatType>? =
        if let imagePath {
          try loadInputImageTensor(
            path: imagePath, imageWidth: Int(configuration.startWidth) * 64,
            imageHeight: Int(configuration.startHeight) * 64)
        } else {
          nil
        }

      let runner = try LocalGenerationRunner()
      let outputPaths = try runner.generate(
        prompt: promptValues.prompt, negativePrompt: resolvedNegativePrompt,
        configuration: configuration, outputPath: output.output, inputImage: inputImageTensor,
        videoFormat: output.videoFormat)
      print("Models directory: \(modelsDirectory.path)")
      for path in outputPaths {
        print("Wrote: \(path)")
      }
    }
  }

  struct Models: ParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Model utilities.",
      discussion: CLIHelpText.models,
      subcommands: [List.self, Ensure.self, Import.self]
    )

    struct List: ParsableCommand {
      static let configuration = CommandConfiguration(
        abstract: "List available local-inference model mappings.",
        discussion: CLIHelpText.modelList)

      @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

      @Flag(name: .long, help: "Show downloaded models only.")
      var downloadedOnly: Bool = false

      @Flag(
        name: .long,
        help: "Disable network access and use cached community model catalogs only.")
      var offline: Bool = false

      mutating func run() throws {
        NetworkAccessPolicy.offline = offline
        let modelsDirectory = try ModelsDirectoryResolver.resolve(
          path: modelsDirectoryOptions.modelsDir)
        ModelZoo.isExternalUrlsPreferred = true
        ModelZoo.externalUrls = [modelsDirectory]
        print("Models directory: \(modelsDirectory.path)")
        printModelList(
          downloadedOnly: downloadedOnly, modelsDirectory: modelsDirectory,
          allowNetwork: !offline)
      }
    }

    struct Ensure: ParsableCommand {
      static let configuration = CommandConfiguration(
        abstract: "Ensure model files exist locally (download if missing).",
        discussion: CLIHelpText.modelEnsure)

      @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

      @Option(name: .shortAndLong, help: modelReferenceHelp)
      var model: String?

      @Flag(name: .long, inversion: .prefixedNo, help: "Include model dependencies.")
      var includeDependencies: Bool = true

      @Flag(
        name: .long,
        help:
          "Disable network access. Uses cached community model catalogs only, and never downloads models."
      )
      var offline: Bool = false

      mutating func run() throws {
        NetworkAccessPolicy.offline = offline
        let modelsDirectory = try ModelsDirectoryResolver.resolve(
          path: modelsDirectoryOptions.modelsDir)
        ModelZoo.isExternalUrlsPreferred = true
        ModelZoo.externalUrls = [modelsDirectory]

        guard let model else {
          printModelResolutionHelp(modelsDirectory: modelsDirectory)
          throw ValidationError("--model is required.")
        }
        guard
          let modelSpecification = ModelResolver.resolve(model, modelsDirectory: modelsDirectory)
        else {
          printModelResolutionHelp(modelsDirectory: modelsDirectory)
          throw unresolvedModelValidationError(model)
        }
        let files =
          includeDependencies
          ? ModelZoo.filesToDownload(modelSpecification).map(\.file)
          : [modelSpecification.file]
        if files.isEmpty {
          throw ValidationError(
            "Model '\(modelSpecification.file)' has no local downloadable files.")
        }
        try ModelDownloader.ensureFiles(
          files, modelsDirectory: modelsDirectory, downloadMissing: true)
        print("Model ready: \(modelSpecification.file)")
      }
    }

    struct Import: ParsableCommand {
      static let configuration = CommandConfiguration(
        abstract: "Import a local checkpoint or safetensors artifact.",
        discussion: CLIHelpText.modelImport)

      @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

      @Argument(help: modelImportArtifactHelp)
      var artifact: String

      @Option(name: .long, help: "Display name for the imported model.")
      var name: String?

      @Option(
        name: .long, help: "Optional trigger word / prefix stored in the model specification.")
      var triggerWord: String?

      @Option(
        name: .long, help: "Optional autoencoder artifact to import alongside the main model.")
      var autoencoder: String?

      @Option(
        name: .long,
        help:
          "Optional text encoder artifact to import when the detected model family supports it.")
      var textEncoder: String?

      @Option(
        name: .customLong("text-encoder-2"),
        help:
          "Optional second text encoder artifact for SDXL Base imports.")
      var textEncoder2: String?

      @Option(name: .long, help: modelImportScaleHelp)
      var scale: Int?

      @Flag(
        name: .long, help: "Inspect and infer the custom model specification without writing files."
      )
      var dryRun: Bool = false

      @Flag(
        name: .long, inversion: .prefixedNo, help: "Auto-download missing companion model files.")
      var downloadMissing: Bool = true

      @Flag(name: .long, help: "Replace existing imported files with the same internal model id.")
      var replace: Bool = false

      @Flag(
        name: .long,
        help:
          "Disable network access. Imports the local artifact only and skips any remote companion downloads."
      )
      var offline: Bool = false

      mutating func run() throws {
        NetworkAccessPolicy.offline = offline
        let modelsDirectory = try ModelsDirectoryResolver.resolve(
          path: modelsDirectoryOptions.modelsDir)
        ModelZoo.isExternalUrlsPreferred = true
        ModelZoo.externalUrls = [modelsDirectory]

        let artifactURL = try resolvedLocalFileURL(artifact)
        let autoencoderURL = try resolvedOptionalLocalFileURL(autoencoder)
        let textEncoderURL = try resolvedOptionalLocalFileURL(textEncoder)
        let textEncoder2URL = try resolvedOptionalLocalFileURL(textEncoder2)

        let displayName =
          name?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
          ? name!.trimmingCharacters(in: .whitespacesAndNewlines)
          : defaultImportedModelDisplayName(for: artifactURL)
        let internalName = Importer.cleanup(
          filename: artifactURL.deletingPathExtension().lastPathComponent)
        guard !internalName.isEmpty else {
          throw ValidationError(
            "Unable to derive a valid internal model id from '\(artifactURL.lastPathComponent)'.")
        }

        let isTextEncoderCustomized = textEncoderURL != nil || textEncoder2URL != nil
        let importer = ModelImporter(
          filePath: artifactURL.path, modelName: internalName,
          isTextEncoderCustomized: isTextEncoderCustomized,
          autoencoderFilePath: autoencoderURL?.path, textEncoderFilePath: textEncoderURL?.path,
          textEncoder2FilePath: textEncoder2URL?.path)

        guard scale == nil || scale! > 0 else {
          throw ValidationError("--scale must be > 0")
        }
        let inspection = try importer.inspect()
        try validateCustomTextEncoderSupport(
          version: inspection.version, textEncoderURL: textEncoderURL,
          textEncoder2URL: textEncoder2URL)
        let expectedFiles = try projectedImportedOutputFiles(
          modelName: internalName, version: inspection.version,
          includeAutoencoder: autoencoderURL != nil,
          includeTextEncoder: textEncoderURL != nil, includeTextEncoder2: textEncoder2URL != nil)
        let existing = existingImportedOutputs(for: expectedFiles)
        if !replace && !existing.isEmpty {
          let lines = existing.map { "  - \($0)" }.joined(separator: "\n")
          throw ValidationError(
            "Refusing to overwrite existing imported files:\n\(lines)\nRe-run with --replace to overwrite them."
          )
        }
        let finetuneScale = UInt16(
          scale
            ?? Int(
              defaultImportScale(
                for: inspection.version, artifactFileName: artifactURL.lastPathComponent)))

        if dryRun {
          let (specification, additionalModels, _) = ModelImporter.inferModelSpecification(
            modelName: displayName,
            fileName: internalName,
            fileNames: expectedFiles,
            modelVersion: inspection.version,
            modifier: inspection.modifier,
            inspectionResult: inspection,
            prefix: importPrefix(triggerWord: triggerWord),
            objective: nil,
            conditioning: nil,
            noiseDiscretization: nil,
            upcastAttention: false,
            finetuneScale: finetuneScale
          )
          print("Dry run: no files written.")
          printImportedModelSummary(
            specification: specification, version: inspection.version,
            modifier: inspection.modifier, importedFiles: expectedFiles,
            dependencyFiles: importedDependencyFiles(
              specification: specification, additionalModels: additionalModels,
              importedFiles: expectedFiles))
          return
        }

        var lastPrintedPercent = -1
        let result = try importer.import { version in
          print(
            "Detected: \(ModelZoo.humanReadableNameForVersion(version)) (\(String(describing: version)))"
          )
        } progress: { progress in
          let percent = min(100, max(0, Int((progress * 100).rounded(.down))))
          if percent >= lastPrintedPercent + 5 || percent == 100 {
            print("Importing: \(percent)%")
            lastPrintedPercent = percent
          }
        }

        let fileNames = result.0.map { URL(fileURLWithPath: $0).lastPathComponent }
        let (specification, additionalModels, _) = ModelImporter.inferModelSpecification(
          modelName: displayName,
          fileName: internalName,
          fileNames: fileNames,
          modelVersion: result.1,
          modifier: result.2,
          inspectionResult: result.3,
          prefix: importPrefix(triggerWord: triggerWord),
          objective: nil,
          conditioning: nil,
          noiseDiscretization: nil,
          upcastAttention: false,
          finetuneScale: finetuneScale
        )
        let dependencyFiles = importedDependencyFiles(
          specification: specification, additionalModels: additionalModels,
          importedFiles: fileNames)

        var dependencyWarning: String?
        do {
          try ModelDownloader.ensureFiles(
            dependencyFiles, modelsDirectory: modelsDirectory, downloadMissing: downloadMissing)
        } catch {
          dependencyWarning = error.localizedDescription
        }

        ModelZoo.appendCustomSpecification(specification)

        printImportedModelSummary(
          specification: specification, version: result.1, modifier: result.2,
          importedFiles: fileNames, dependencyFiles: dependencyFiles)
        if let dependencyWarning {
          print("")
          print("Dependency download warning:")
          print(dependencyWarning)
        } else {
          print("")
          print("Model imported: \(specification.file)")
        }
      }
    }
  }

  struct Train: ParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Training utilities.",
      discussion: CLIHelpText.train,
      subcommands: [LoRA.self]
    )

    struct LoRA: ParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "lora",
        abstract: "Train a LoRA model locally.",
        discussion: CLIHelpText.loraTrain)

      @OptionGroup var options: LoRATrainCommandOptions

      mutating func run() throws {
        NetworkAccessPolicy.offline = options.execution.offline
        try runLoRATraining(options)
      }
    }
  }

  struct Completion: ParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Generate shell completion scripts.",
      discussion: CLIHelpText.completion
    )

    enum Shell: String, ExpressibleByArgument {
      case bash
      case zsh
      case fish

      var completionShell: CompletionShell {
        switch self {
        case .bash:
          return .bash
        case .zsh:
          return .zsh
        case .fish:
          return .fish
        }
      }
    }

    @Argument(help: "Shell to generate completions for.")
    var shell: Shell

    @Option(name: .shortAndLong, help: "Write the completion script to a file instead of stdout.")
    var output: String?

    mutating func run() throws {
      let script = DrawThingsCLI.completionScript(for: shell.completionShell)
      if let output {
        try script.write(toFile: output, atomically: true, encoding: .utf8)
      } else {
        print(script, terminator: "")
      }
    }
  }
}

private func mergedAlias(
  primary: String?, alias: String?, primaryFlag: String, aliasFlag: String
) throws -> String? {
  if let primary, let alias, primary != alias {
    throw ValidationError("Use only one of \(primaryFlag) or \(aliasFlag)")
  }
  return primary ?? alias
}

private func loadTextOption(
  inline: String?, filePath: String?, inlineFlag: String, fileFlag: String
) throws -> String? {
  if inline != nil && filePath != nil {
    throw ValidationError("Use only one of \(inlineFlag) or \(fileFlag)")
  }
  if let inline {
    return inline
  }
  guard let filePath else {
    return nil
  }
  if filePath == "-" {
    let data = FileHandle.standardInput.readDataToEndOfFile()
    guard let text = String(data: data, encoding: .utf8) else {
      throw ValidationError("Failed to read UTF-8 text from stdin for \(fileFlag)")
    }
    return text
  }
  return try String(contentsOfFile: filePath, encoding: .utf8)
}

private func resolvedPrompts(_ options: GeneratePromptOptions) throws -> (
  prompt: String, negative: String?
) {
  if options.promptFile == "-", options.negativePromptFile == "-" {
    throw ValidationError("Use stdin for only one of --prompt-file or --negative-prompt-file")
  }
  let prompt =
    try loadTextOption(
      inline: options.prompt, filePath: options.promptFile, inlineFlag: "--prompt",
      fileFlag: "--prompt-file") ?? ""
  let negative =
    try loadTextOption(
      inline: options.negativePrompt, filePath: options.negativePromptFile,
      inlineFlag: "--negative-prompt", fileFlag: "--negative-prompt-file")
  return (prompt, negative)
}

private func validateVideoOutputOptions(outputPath: String, videoFormat: VideoExportFormat?) throws
{
  guard let videoFormat else { return }
  var url = URL(fileURLWithPath: outputPath)
  if url.pathExtension.isEmpty {
    url = url.appendingPathExtension("png")
  }
  switch url.pathExtension.lowercased() {
  case "mov":
    return
  case "mp4":
    if videoFormat == .prores4444 || videoFormat == .prores422hq {
      throw ValidationError("ProRes video formats require .mov output")
    }
  case "png":
    throw ValidationError("--video-format can only be used with .mov or .mp4 output")
  default:
    return
  }
}

private func loadInputImageTensor(path: String, imageWidth: Int, imageHeight: Int) throws
  -> Tensor<FloatType>
{
  let filePath = URL(fileURLWithPath: path).standardizedFileURL.path
  guard FileManager.default.fileExists(atPath: filePath) else {
    throw DrawThingsCLIError.invalidInputImagePath(filePath)
  }
  guard
    let (tensor, _, _, _) = loadTrainingTensor(
      url: URL(fileURLWithPath: filePath), imageWidth: imageWidth, imageHeight: imageHeight)
  else {
    throw DrawThingsCLIError.invalidInputImage(filePath)
  }
  return tensor
}

private enum LoRATrainConfigLoader {
  static func load(configJSON: String?) throws -> (LoRATrainingConfiguration, [String: Any]) {
    let base = LoRATrainingConfiguration.default
    guard let configJSON, !configJSON.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      return (base, [:])
    }
    guard let configData = configJSON.data(using: .utf8),
      let configDictionary = try? JSONSerialization.jsonObject(with: configData) as? [String: Any]
    else {
      throw DrawThingsCLIError.invalidLoRAConfigurationJSON
    }
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    guard let baseData = try? encoder.encode(base),
      var mergedDictionary = try? JSONSerialization.jsonObject(with: baseData) as? [String: Any]
    else {
      throw DrawThingsCLIError.invalidLoRAConfigurationJSON
    }
    for (key, value) in configDictionary {
      mergedDictionary[key] = value
    }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let mergedData = try? JSONSerialization.data(withJSONObject: mergedDictionary),
      let configuration = try? decoder.decode(LoRATrainingConfiguration.self, from: mergedData)
    else {
      throw DrawThingsCLIError.invalidLoRAConfigurationJSON
    }
    return (configuration, configDictionary)
  }

  static func memorySaverValue(from configDictionary: [String: Any]) -> Any? {
    configDictionary["memory_saver"] ?? configDictionary["memorySaver"]
  }

  static func weightsMemoryValue(from configDictionary: [String: Any]) -> Any? {
    configDictionary["weights_memory_management"] ?? configDictionary["weightsMemoryManagement"]
      ?? configDictionary["weights_memory"]
  }
}

private struct LoRATrainerTokenizers {
  var tokenizerV1: TextualInversionAttentionCLIPTokenizer
  var tokenizerV2: TextualInversionAttentionCLIPTokenizer
  var tokenizerT5: SentencePieceTokenizer
  var tokenizerChatGLM3: SentencePieceTokenizer
  var tokenizerQwen25: TiktokenTokenizer
}

private func createLoRATrainerTokenizers() -> LoRATrainerTokenizers {
  let tokenizerV1 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_json,
    merges: BinaryResources.merges_txt,
    textualInversions: [])
  let tokenizerV2 = TextualInversionAttentionCLIPTokenizer(
    vocabulary: BinaryResources.vocab_16e6_json,
    merges: BinaryResources.bpe_simple_vocab_16e6_txt,
    textualInversions: [])
  let tokenizerT5 = SentencePieceTokenizer(
    data: BinaryResources.t5_spiece_model, startToken: nil,
    endToken: 1, tokenShift: 0)
  let tokenizerChatGLM3 = SentencePieceTokenizer(
    data: BinaryResources.chatglm3_spiece_model, startToken: nil,
    endToken: nil, tokenShift: 0)
  let tokenizerQwen25 = TiktokenTokenizer(
    vocabulary: BinaryResources.vocab_qwen2_5_json, merges: BinaryResources.merges_qwen2_5_txt,
    specialTokens: [
      "</tool_call>": 151658, "<tool_call>": 151657, "<|box_end|>": 151649, "<|box_start|>": 151648,
      "<|quad_end|>": 151651, "<|quad_start|>": 151650, "<|vision_end|>": 151653,
      "<|vision_pad|>": 151655, "<|vision_start|>": 151652, "<|image_pad|>": 151655,
      "<|object_ref_end|>": 151647, "<|object_ref_start|>": 151646, "<|im_end|>": 151645,
      "<|im_start|>": 151644, "<|endoftext|>": 151643,
    ])
  return LoRATrainerTokenizers(
    tokenizerV1: tokenizerV1, tokenizerV2: tokenizerV2, tokenizerT5: tokenizerT5,
    tokenizerChatGLM3: tokenizerChatGLM3, tokenizerQwen25: tokenizerQwen25)
}

private func inspectTrainingImage(url: URL) -> (width: Int, height: Int)? {
  if let image = try? PNG.Data.Rectangular.decompress(path: url.path) {
    return (image.size.x, image.size.y)
  }
  #if canImport(ImageIO)
    guard
      let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      return nil
    }
    return (cgImage.width, cgImage.height)
  #else
    return nil
  #endif
}

private func loadTrainingTensor(
  url: URL, imageWidth: Int, imageHeight: Int
) -> (
  Tensor<FloatType>, (width: Int, height: Int), (top: Int, left: Int), (width: Int, height: Int)
)? {
  if let image = try? PNG.Data.Rectangular.decompress(path: url.path) {
    let rgba: [PNG.RGBA<UInt8>] = image.unpack(as: PNG.RGBA<UInt8>.self)
    let sourceWidth = image.size.x
    let sourceHeight = image.size.y

    var scaledWidth: Int
    var scaledHeight: Int
    if sourceWidth * imageHeight > sourceHeight * imageWidth {
      scaledHeight = imageHeight
      scaledWidth = Int(
        (Double(sourceWidth) * Double(imageHeight) / Double(sourceHeight)).rounded())
    } else {
      scaledWidth = imageWidth
      scaledHeight = Int(
        (Double(sourceHeight) * Double(imageWidth) / Double(sourceWidth)).rounded())
    }

    let offsetX = (scaledWidth - imageWidth) / 2
    let offsetY = (scaledHeight - imageHeight) / 2
    var tensor = Tensor<FloatType>(.CPU, .NHWC(1, imageHeight, imageWidth, 3))
    for y in 0..<imageHeight {
      for x in 0..<imageWidth {
        let srcX = Int(Double(x + offsetX) * Double(sourceWidth) / Double(scaledWidth))
        let srcY = Int(Double(y + offsetY) * Double(sourceHeight) / Double(scaledHeight))
        let clampedX = min(max(srcX, 0), sourceWidth - 1)
        let clampedY = min(max(srcY, 0), sourceHeight - 1)
        let pixel = rgba[clampedY * sourceWidth + clampedX]
        tensor[0, y, x, 0] = FloatType(Float(pixel.r) / 127.5 - 1)
        tensor[0, y, x, 1] = FloatType(Float(pixel.g) / 127.5 - 1)
        tensor[0, y, x, 2] = FloatType(Float(pixel.b) / 127.5 - 1)
      }
    }

    let top = Int(
      (Double(max(scaledWidth - imageWidth, 0))
        * (Double(sourceWidth) / Double(max(scaledWidth, 1)))
        / 2).rounded())
    let left = Int(
      (Double(max(scaledHeight - imageHeight, 0))
        * (Double(sourceHeight) / Double(max(scaledHeight, 1)))
        / 2).rounded())
    let originalWidth = Int(
      (Double(imageWidth) * Double(sourceWidth) / Double(max(scaledWidth, 1))).rounded())
    let originalHeight = Int(
      (Double(imageHeight) * Double(sourceHeight) / Double(max(scaledHeight, 1))).rounded())
    return (
      tensor, (width: originalWidth, height: originalHeight), (top: top, left: left),
      (width: imageWidth, height: imageHeight)
    )
  }

  #if canImport(ImageIO) && canImport(CoreGraphics)
    guard
      let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      return nil
    }

    let cgImageWidth = cgImage.width
    let cgImageHeight = cgImage.height

    var scaledWidth: Int
    var scaledHeight: Int
    if cgImageWidth * imageHeight > cgImageHeight * imageWidth {
      scaledHeight = imageHeight
      scaledWidth = Int(
        (Double(cgImageWidth) * Double(imageHeight) / Double(cgImageHeight)).rounded())
    } else {
      scaledWidth = imageWidth
      scaledHeight = Int(
        (Double(cgImageHeight) * Double(imageWidth) / Double(cgImageWidth)).rounded())
    }

    guard
      let bitmapContext = CGContext(
        data: nil, width: imageWidth, height: imageHeight, bitsPerComponent: 8,
        bytesPerRow: imageWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
          | CGImageAlphaInfo.premultipliedLast.rawValue)
    else {
      return nil
    }
    bitmapContext.interpolationQuality = .high
    bitmapContext.draw(
      cgImage,
      in: CGRect(
        x: (imageWidth - scaledWidth) / 2, y: (imageHeight - scaledHeight) / 2, width: scaledWidth,
        height: scaledHeight))

    guard let data = bitmapContext.data else { return nil }
    let rgba = data.assumingMemoryBound(to: UInt8.self)
    var tensor = Tensor<FloatType>(.CPU, .NHWC(1, imageHeight, imageWidth, 3))
    for y in 0..<imageHeight {
      for x in 0..<imageWidth {
        let i = (y * imageWidth + x) * 4
        tensor[0, y, x, 0] = FloatType(Float(rgba[i]) / 127.5 - 1)
        tensor[0, y, x, 1] = FloatType(Float(rgba[i + 1]) / 127.5 - 1)
        tensor[0, y, x, 2] = FloatType(Float(rgba[i + 2]) / 127.5 - 1)
      }
    }

    let top = Int(
      (Double(max(scaledWidth - imageWidth, 0))
        * (Double(cgImageWidth) / Double(max(scaledWidth, 1)))
        / 2).rounded())
    let left = Int(
      (Double(max(scaledHeight - imageHeight, 0))
        * (Double(cgImageHeight) / Double(max(scaledHeight, 1)))
        / 2).rounded())
    let originalWidth = Int(
      (Double(imageWidth) * Double(cgImageWidth) / Double(max(scaledWidth, 1))).rounded())
    let originalHeight = Int(
      (Double(imageHeight) * Double(cgImageHeight) / Double(max(scaledHeight, 1))).rounded())
    return (
      tensor, (width: originalWidth, height: originalHeight), (top: top, left: left),
      (width: imageWidth, height: imageHeight)
    )
  #else
    return nil
  #endif
}

private func loadTrainingDataset(from directory: String) throws -> [LoRATrainer.Input] {
  let fileManager = FileManager.default
  let directoryURL = URL(fileURLWithPath: directory)
  let contents = try fileManager.contentsOfDirectory(
    at: directoryURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
  let imageExtensions = Set([
    "png", "jpg", "jpeg", "webp", "heic", "heif", "avif", "bmp", "tif", "tiff",
  ])

  var inputs: [LoRATrainer.Input] = []
  for url in contents.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
    let ext = url.pathExtension.lowercased()
    guard imageExtensions.contains(ext) else { continue }
    let baseName = url.deletingPathExtension().lastPathComponent

    let captionURL = directoryURL.appendingPathComponent("\(baseName).txt")
    var caption = ""
    if fileManager.fileExists(atPath: captionURL.path),
      let text = try? String(contentsOf: captionURL, encoding: .utf8)
    {
      caption = text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    inputs.append(
      LoRATrainer.Input(
        identifier: baseName,
        imageUrl: url,
        caption: caption)
    )
  }
  return inputs
}

private func flux1TrainableKeys() -> [String] {
  var keys = [String]()
  keys.append("x_embedder-")
  keys.append("context_embedder-")
  keys.append("linear-0-")
  let layerKeys = [
    "x_q", "x_k", "x_v", "c_q", "c_k", "c_v", "x_o", "c_o", "x_linear1", "x_out_proj",
    "c_linear1", "c_out_proj",
  ]
  for layerIndex in 0..<(19 + 38) {
    keys.append(contentsOf: layerKeys.map { "\($0)-\(layerIndex)-" })
  }
  return keys
}

private func parseMemorySaver(_ value: String) throws -> LoRATrainer.MemorySaver {
  switch value.lowercased() {
  case "minimal":
    return .minimal
  case "balanced":
    return .balanced
  case "speed":
    return .speed
  case "turbo":
    return .turbo
  default:
    throw ValidationError(
      "Invalid memory saver mode '\(value)'. Use one of: minimal, balanced, speed, turbo")
  }
}

private func parseWeightsMemory(_ value: String) throws -> LoRATrainer.WeightsMemoryManagement {
  switch value.lowercased() {
  case "cached":
    return .cached
  case "justintime":
    return .justInTime
  default:
    throw ValidationError("Invalid weights memory mode '\(value)'. Use one of: cached, justInTime")
  }
}

private func parseLearningRateRange(_ value: String) throws -> ClosedRange<Float> {
  let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
  guard !trimmed.isEmpty else {
    throw ValidationError("Invalid --learning-rate value. Expected float or [low,high].")
  }
  let normalized: String
  if trimmed.hasPrefix("[") && trimmed.hasSuffix("]") && trimmed.count > 2 {
    normalized = String(trimmed.dropFirst().dropLast())
  } else {
    normalized = trimmed
  }

  func parsePair(separator: Character) -> ClosedRange<Float>? {
    let parts = normalized.split(separator: separator, omittingEmptySubsequences: true).map {
      $0.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    guard parts.count == 2, let first = Float(parts[0]), let second = Float(parts[1]) else {
      return nil
    }
    return min(first, second)...max(first, second)
  }

  let parsedRange =
    parsePair(separator: ",")
    ?? parsePair(separator: ":")
    ?? Float(normalized).map { $0...$0 }
  guard let parsedRange else {
    throw ValidationError(
      "Invalid --learning-rate value '\(value)'. Use float (e.g. 1e-4) or range (e.g. [5e-5,1e-4])."
    )
  }
  guard parsedRange.lowerBound >= 0 else {
    throw ValidationError("--learning-rate must be >= 0")
  }
  return parsedRange
}

private func parseMemorySaverValue(_ value: Any?) throws -> LoRATrainer.MemorySaver? {
  guard let value else { return nil }
  if let numeric = value as? NSNumber,
    let mode = LoRATrainer.MemorySaver(rawValue: numeric.intValue)
  {
    return mode
  }
  if let string = value as? String {
    return try parseMemorySaver(string)
  }
  throw ValidationError(
    "Invalid memory_saver value in --config-json. Use enum raw value or one of: minimal, balanced, speed, turbo."
  )
}

private func parseWeightsMemoryValue(_ value: Any?) throws -> LoRATrainer.WeightsMemoryManagement? {
  guard let value else { return nil }
  if let numeric = value as? NSNumber,
    let mode = LoRATrainer.WeightsMemoryManagement(rawValue: numeric.intValue)
  {
    return mode
  }
  if let string = value as? String {
    return try parseWeightsMemory(string)
  }
  throw ValidationError(
    "Invalid weights_memory_management value in --config-json. Use enum raw value or one of: cached, justInTime."
  )
}
