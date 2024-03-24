import Diffusion
import Foundation

public struct LoRAZoo: DownloadZoo {
  public struct Specification: Codable {
    public var name: String
    public var file: String
    public var prefix: String
    public var version: ModelVersion
    public var isConsistencyModel: Bool? = nil
    public var isLoHa: Bool? = nil
    public var modifier: SamplerModifier? = nil
    public init(
      name: String, file: String, prefix: String, version: ModelVersion,
      isConsistencyModel: Bool? = nil, isLoHa: Bool? = nil, modifier: SamplerModifier? = nil
    ) {
      self.name = name
      self.file = file
      self.prefix = prefix
      self.version = version
      self.isConsistencyModel = isConsistencyModel
      self.isLoHa = isLoHa
      self.modifier = modifier
    }
  }

  private static let fileSHA256: [String: String] = [
    "openjourney_v1_lora_f16.ckpt":
      "82b8d1f442c80c60edc506adb56cda128742e4661511a873c0b97b26b584dddb",
    "moxin_v1.0_lora_f16.ckpt":
      "f892698f3dc00abdb91d8d3022857e56010a3f7eb42387aa20b35d987ae4bed7",
    "moxin_shukezouma_v1.1_lora_f16.ckpt":
      "1c733b99209620174195c9c1a2dc2b49ded9945361322775b5f415b70cdef6bf",
    "adams_artwork_style_v0.1_lora_f16.ckpt":
      "17590d5d9977c64164cd78da00f4728791fd7ce7d4979e74438066d1c9134665",
    "cyberpunk_2007_concept_art_nightcity_v1.15_lora_f16.ckpt":
      "b990e92854c0a2516cdba6191a456f18c32ef53b883cc81689dbe725c5a28abf",
    "analog_diffusion_v1_lora_f16.ckpt":
      "cf550a13b9a4f3f1f5831146f71321ec3c642cac91907f2b7f7bb465144a7c40",
    "epi_noiseoffset_v2_lora_f16.ckpt":
      "392cd181d644d0f28d083efb28846a23f8056b83900e727efb85514dee42444d",
    "theovercomer8s_contrast_fix_lora_f16.ckpt":
      "41a41d498871e50257c90630f59cc4802e1cabecd9067ff8494deda314ccfd40",
    "anime_lineart_style_v2.0_lora_f16.ckpt":
      "49d7ee07de9827876cff0cd80b1c4d287a69b8c08c91d827df44d43d3cca4a83",
    "to8s_high_key_lora_f16.ckpt":
      "d47ab2871fdcd59eaba765274e3ba30af822edd82ac3a9c02218e51995ab8990",
    "arcane_style_lora_f16.ckpt":
      "d02a3f195b756c00d35f693cdea9f8ff19f6cb498469997504363214826983b7",
    "crazy_expressions_lora_f16.ckpt":
      "ec6866a9b4707f2b41d866221f4bdcaf1cafd9835fe3b1ce1c38abea255bd2b3",
    "hipoly_3d_model_lora_f16.ckpt":
      "e770caebfdcbb991e7b15e17a6cc6ef6f323fb6107399cfdf0414703f2a9442e",
    "haute_couture_or_gowns_v1.0_lora_f16.ckpt":
      "81bb127631eeb0616d96bad9e1cfca341331b67c77974d91ffa0c27d8c39c7e7",
    "theovercomer8s_contrast_fix_sd_v2.x_lora_f16.ckpt":
      "5e3e4cb74d0bac795082ced3f91fb7045ab5d51b8b0b3d3dfcba601a51048216",
    "to8s_high_key_sd_v2.x_lora_f16.ckpt":
      "68f77630cc4c28907e0af3913a8a0e51a630640cdeee536fce51dd72e5adda24",
    "lcm_sd_v1.5_lora_f16.ckpt": "072ef94e15252e963a0bc77702f8db329ef2ce0e2245ed487ee61aeca1cdb69d",
    "lcm_sd_xl_base_1.0_lora_f16.ckpt":
      "7fdd9718855ca59ce40788389c1545eb98e01909256425a5190e18142b8c09b3",
    "lcm_ssd_1b_lora_f16.ckpt": "28cdb66f0257326ffb6f4b1d6996f8fad9378aeee081529e31817aea81dfeb9f",
    "sdxl_offset_v1.0_lora_f16.ckpt":
      "0c5a8ac02c5751cfdcde375b997ab87ba8c809f9f15417c67a9005d74314f256",
    "fooocus_inpaint_v2.6_lora_f16.ckpt":
      "895bad779b596f1dabceb3a54a7f6a319bc3a87db7352418d625e1d44dceb2c6",
    "fooocus_inpaint_v2.6_lora_q8p.ckpt":
      "81ac3f4750dce8bd1be3895ab06acb3e1ec996987d928a247983476b2fb64c6c",
    "tcd_sd_v1.5_lora_f16.ckpt": "2b1f9acfa0a794cd733d667cd4e9c01d971f6c21e055b487a67345f049aeccec",
    "tcd_sd_xl_base_1.0_lora_f16.ckpt":
      "368c22ba70d2cd6984234bc4b2fb34d61cd60e84df618fe405eed8aa9e84fc9e",
  ]

  public static let builtinSpecifications: [Specification] = [
    Specification(
      name: "SDXL Offset (1.0)", file: "sdxl_offset_v1.0_lora_f16.ckpt",
      prefix: "", version: .sdxlBase),
    Specification(
      name: "TCD Stable Diffusion v1.5", file: "tcd_sd_v1.5_lora_f16.ckpt",
      prefix: "", version: .v1, isConsistencyModel: true),
    Specification(
      name: "TCD SDXL Base (1.0)", file: "tcd_sd_xl_base_1.0_lora_f16.ckpt",
      prefix: "", version: .sdxlBase, isConsistencyModel: true),
    Specification(
      name: "LCM Stable Diffusion v1.5", file: "lcm_sd_v1.5_lora_f16.ckpt",
      prefix: "", version: .v1, isConsistencyModel: true),
    Specification(
      name: "LCM SDXL Base (1.0)", file: "lcm_sd_xl_base_1.0_lora_f16.ckpt",
      prefix: "", version: .sdxlBase, isConsistencyModel: true),
    Specification(
      name: "LCM SDXL Refiner (1.0)", file: "lcm_sd_xl_refiner_1.0_lora_f16.ckpt",
      prefix: "", version: .sdxlRefiner, isConsistencyModel: true),
    Specification(
      name: "LCM SSD 1B (Segmind)", file: "lcm_ssd_1b_lora_f16.ckpt",
      prefix: "", version: .ssd1b, isConsistencyModel: true),
    Specification(
      name: "Fooocus Inpaint v2.6", file: "fooocus_inpaint_v2.6_lora_f16.ckpt",
      prefix: "", version: .sdxlBase, modifier: .inpainting),
    Specification(
      name: "Fooocus Inpaint v2.6 (8-bit)", file: "fooocus_inpaint_v2.6_lora_q8p.ckpt",
      prefix: "", version: .sdxlBase, modifier: .inpainting),
    Specification(
      name: "Moxin v1.0", file: "moxin_v1.0_lora_f16.ckpt", prefix: "shuimobysim ", version: .v1),
    Specification(
      name: "Moxin Shukezouma v1.1", file: "moxin_shukezouma_v1.1_lora_f16.ckpt",
      prefix: "shukezouma ", version: .v1),
    Specification(
      name: "Openjourney v1.0", file: "openjourney_v1_lora_f16.ckpt",
      prefix: "mdjrny-v4 ", version: .v1),
    Specification(
      name: "Analog Diffusion v1.0", file: "analog_diffusion_v1_lora_f16.ckpt",
      prefix: "analog ", version: .v1),
    Specification(
      name: "Adam's Artwork Style v.1", file: "adams_artwork_style_v0.1_lora_f16.ckpt",
      prefix: "ajaws ", version: .v1),
    Specification(
      name: "Cyberpunk 2077 Nightcity v1.15",
      file: "cyberpunk_2007_concept_art_nightcity_v1.15_lora_f16.ckpt",
      prefix: "", version: .v1),
    Specification(
      name: "Epi Noise Offset v2", file: "epi_noiseoffset_v2_lora_f16.ckpt",
      prefix: "", version: .v1),
    Specification(
      name: "Anime LineArt Style v2.0", file: "anime_lineart_style_v2.0_lora_f16.ckpt",
      prefix: "", version: .v1),
    Specification(
      name: "Theovercomer8's Contrast Fix",
      file: "theovercomer8s_contrast_fix_lora_f16.ckpt",
      prefix: "to8contrast style ", version: .v1),
    Specification(
      name: "Theovercomer8's Contrast Fix",
      file: "theovercomer8s_contrast_fix_sd_v2.x_lora_f16.ckpt",
      prefix: "to8contrast style ", version: .v2),
    Specification(
      name: "Arcane Style", file: "arcane_style_lora_f16.ckpt",
      prefix: "arcane style ", version: .v1),
    Specification(
      name: "Crazy Expressions", file: "crazy_expressions_lora_f16.ckpt",
      prefix: "crazy face ", version: .v1),
    Specification(
      name: "TO8's High Key", file: "to8s_high_key_lora_f16.ckpt",
      prefix: "to8highkey ", version: .v1),
    Specification(
      name: "TO8's High Key", file: "to8s_high_key_sd_v2.x_lora_f16.ckpt",
      prefix: "to8highkey ", version: .v2),
    Specification(
      name: "Hipoly 3D Model", file: "hipoly_3d_model_lora_f16.ckpt",
      prefix: "hiqcgbody ", version: .v1),
    Specification(
      name: "Haute Couture or Gowns v1.0",
      file: "haute_couture_or_gowns_v1.0_lora_f16.ckpt",
      prefix: "hc_gown ", version: .v1),
  ]

  private static let builtinModels: Set<String> = Set(builtinSpecifications.map { $0.file })

  public static func isBuiltinLoRA(_ name: String) -> Bool {
    return builtinModels.contains(name)
  }

  public static var availableSpecifications: [Specification] = {
    let jsonFile = filePathForModelDownloaded("custom_lora.json")
    guard let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) else {
      return builtinSpecifications
    }

    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    guard let jsonSpecification = try? jsonDecoder.decode([Specification].self, from: jsonData)
    else {
      return builtinSpecifications
    }

    return builtinSpecifications + jsonSpecification
  }()

  private static var specificationMapping: [String: Specification] = {
    var mapping = [String: Specification]()
    for specification in availableSpecifications {
      mapping[specification.file] = specification
    }
    return mapping
  }()

  public static func availableSpecificationForTriggerWord(_ triggerWord: String) -> Specification? {
    let lowerTriggerWord = triggerWord.lowercased()
    for specification in availableSpecifications {
      if specification.name.lowercased().contains(lowerTriggerWord)
        || specification.prefix.lowercased().contains(lowerTriggerWord)
      {
        return specification
      }
    }
    return nil
  }

  public static func filePathForModelDownloaded(_ name: String) -> String {
    return ModelZoo.filePathForModelDownloaded(name)
  }

  public static func appendCustomSpecification(_ specification: Specification) {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [Specification]()
    let jsonFile = filePathForModelDownloaded("custom_lora.json")
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      if let jsonSpecification = try? jsonDecoder.decode([Specification].self, from: jsonData) {
        customSpecifications.append(contentsOf: jsonSpecification)
      }
    }
    if let firstIndex = (customSpecifications.firstIndex { $0.name == specification.name }) {
      customSpecifications[firstIndex] = specification
    } else if let firstIndex = (customSpecifications.firstIndex { $0.file == specification.file }) {
      customSpecifications[firstIndex] = specification
    } else {
      customSpecifications.append(specification)
    }
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    guard let jsonData = try? jsonEncoder.encode(customSpecifications) else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)
    // Modify these two are not thread safe. availableSpecifications are OK. specificationMapping is particularly problematic (as it is access on both main thread and a background thread).
    if let firstIndex = (availableSpecifications.firstIndex { $0.name == specification.name }) {
      availableSpecifications[firstIndex] = specification
    } else if let firstIndex =
      (availableSpecifications.firstIndex { $0.file == specification.file })
    {
      availableSpecifications[firstIndex] = specification
    } else {
      availableSpecifications.append(specification)
    }
    specificationMapping[specification.file] = specification
  }

  public static func sortCustomSpecifications() {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [Specification]()
    let jsonFile = filePathForModelDownloaded("custom_lora.json")
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      if let jsonSpecification = try? jsonDecoder.decode([Specification].self, from: jsonData) {
        customSpecifications.append(contentsOf: jsonSpecification)
      }
    }
    customSpecifications = customSpecifications.sorted(by: {
      $0.name.localizedStandardCompare($1.name) == .orderedAscending
    })

    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    guard let jsonData = try? jsonEncoder.encode(customSpecifications) else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)

    availableSpecifications = builtinSpecifications + customSpecifications
  }

  public static func isModelDownloaded(_ name: String) -> Bool {
    return ModelZoo.isModelDownloaded(name)
  }

  public static func humanReadableNameForModel(_ name: String) -> String {
    guard let specification = specificationMapping[name] else { return name }
    return specification.name
  }

  public static func specificationForModel(_ name: String) -> Specification? {
    return specificationMapping[name]
  }

  public static func textPrefixForModel(_ name: String) -> String {
    guard let specification = specificationMapping[name] else { return "" }
    return specification.prefix
  }

  public static func versionForModel(_ name: String) -> ModelVersion {
    guard let specification = specificationMapping[name] else { return .v1 }
    return specification.version
  }

  public static func modifierForModel(_ name: String) -> SamplerModifier {
    guard let specification = specificationMapping[name] else { return .none }
    return specification.modifier ?? .none
  }

  public static func isConsistencyModelForModel(_ name: String) -> Bool {
    guard let specification = specificationMapping[name] else { return false }
    return specification.isConsistencyModel ?? false
  }

  public static func isLoHaForModel(_ name: String) -> Bool {
    guard let specification = specificationMapping[name] else { return false }
    return specification.isLoHa ?? false
  }

  public static func fileSHA256ForModelDownloaded(_ name: String) -> String? {
    return fileSHA256[name]
  }

  public static func modelVersionSuffixFromLoRAVersion(_ version: ModelVersion) -> String {
    switch version {
    case .v1:
      return " (SD v1.x)"
    case .v2:
      return " (SD v2.x)"
    case .sdxlBase:
      return " (SDXL Base)"
    case .sdxlRefiner:
      return " (SDXL Refiner)"
    case .svdI2v:
      return " (SVD I2V)"
    case .ssd1b:
      return " (SSD 1B)"
    case .kandinsky21:
      return " (Kandinsky v2.1)"
    case .wurstchenStageC, .wurstchenStageB:
      return " (Stable Cascade, Wurstchen v3.0)"
    }
  }

  public static func availableFiles(excluding file: String?) -> Set<String> {
    var files = Set<String>()
    for specification in availableSpecifications {
      guard specification.file != file, LoRAZoo.isModelDownloaded(specification.file) else {
        continue
      }
      files.insert(specification.file)
    }
    return files
  }
}