import DataModels
import Diffusion
import Foundation

public struct FailableDecodable<T: Decodable>: Decodable {
  public let value: T?
  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    value = try? container.decode(T.self)
  }
}

public struct ModelZoo: DownloadZoo {
  public static func humanReadableNameForVersion(_ version: ModelVersion) -> String {
    switch version {
    case .v1:
      return "Stable Diffusion v1"
    case .v2:
      return "Stable Diffusion v2"
    case .kandinsky21:
      return "Kandinsky v2.1"
    case .sdxlBase:
      return "Stable Diffusion XL Base"
    case .sdxlRefiner:
      return "Stable Diffusion XL Refiner"
    case .ssd1b:
      return "Segmind Stable Diffusion XL 1B"
    case .svdI2v:
      return "Stable Video Diffusion"
    case .wurstchenStageC, .wurstchenStageB:
      return "Stable Cascade (Würstchen v3.0)"
    case .sd3:
      return "Stable Diffusion 3 Medium"
    case .sd3Large:
      return "Stable Diffusion 3 Large"
    case .pixart:
      return "PixArt Sigma"
    case .auraflow:
      return "AuraFlow"
    case .flux1:
      return "FLUX.1"
    case .hunyuanVideo:
      return "Hunyuan Video"
    case .wan21_1_3b:
      return "Wan v2.1 1.3B"
    case .wan21_14b:
      return "Wan v2.x 14B"
    case .hiDreamI1:
      return "HiDream I1"
    case .qwenImage:
      return "Qwen Image"
    }
  }

  public enum NoiseDiscretization: Codable {
    case edm(Denoiser.Parameterization.EDM)
    case ddpm(Denoiser.Parameterization.DDPM)
    case rf(Denoiser.Parameterization.RF)
  }

  public struct Specification: Codable {
    public struct MMDiT: Codable {
      public var qkNorm: Bool
      public var dualAttentionLayers: [Int]
      public var distilledGuidanceLayers: Int?
      public var activationFfnScaling: [Int: Int]?
      public init(
        qkNorm: Bool, dualAttentionLayers: [Int], distilledGuidanceLayers: Int? = nil,
        activationFfnScaling: [Int: Int]? = nil
      ) {
        self.qkNorm = qkNorm
        self.dualAttentionLayers = dualAttentionLayers
        self.distilledGuidanceLayers = distilledGuidanceLayers
        self.activationFfnScaling = activationFfnScaling
      }
    }

    public struct RemoteApiModelConfig: Codable {
      public enum JSONNumber: Codable {
        case integer(Int)
        case fraction(Double)

        public init(from decoder: Decoder) throws {
          let container = try decoder.singleValueContainer()
          if let doubleValue = try? container.decode(Double.self) {
            self = .fraction(doubleValue)
          } else if let integerValue = try? container.decode(Int.self) {
            self = .integer(integerValue)
          } else {
            throw DecodingError.dataCorruptedError(
              in: container, debugDescription: "Cannot decode value")
          }
        }

        public func encode(to encoder: Encoder) throws {
          var container = encoder.singleValueContainer()
          switch self {
          case .integer(let value): try container.encode(value)
          case .fraction(let value): try container.encode(value)
          }
        }
      }

      public enum JSONValue: Codable {
        case string(String)
        case number(JSONNumber)
        case bool(Bool)

        public init(from decoder: Decoder) throws {
          let container = try decoder.singleValueContainer()
          if let stringValue = try? container.decode(String.self) {
            self = .string(stringValue)
          } else if let numberValue = try? container.decode(JSONNumber.self) {
            self = .number(numberValue)
          } else if let boolValue = try? container.decode(Bool.self) {
            self = .bool(boolValue)
          } else {
            throw DecodingError.dataCorruptedError(
              in: container, debugDescription: "Cannot decode value")
          }
        }

        public func encode(to encoder: Encoder) throws {
          var container = encoder.singleValueContainer()
          switch self {
          case .string(let value): try container.encode(value)
          case .number(let value): try container.encode(value)
          case .bool(let value): try container.encode(value)
          }
        }
        public var value: Any {
          switch self {
          case .string(let value): return value
          case .number(let value): return value
          case .bool(let value): return value
          }
        }
      }
      public indirect enum CustomRequestBodyValue: Codable {
        case string(String)
        case dictionary([String: CustomRequestBodyValue])

        public init(from decoder: Decoder) throws {
          let container = try decoder.singleValueContainer()
          if let stringValue = try? container.decode(String.self) {
            self = .string(stringValue)
          } else if let dictValue = try? container.decode([String: CustomRequestBodyValue].self) {
            self = .dictionary(dictValue)
          } else {
            throw DecodingError.dataCorruptedError(
              in: container, debugDescription: "Cannot decode CustomRequestBodyValue")
          }
        }

        public func encode(to encoder: Encoder) throws {
          var container = encoder.singleValueContainer()
          switch self {
          case .string(let value): try container.encode(value)
          case .dictionary(let value): try container.encode(value)
          }
        }
      }

      public enum ApiFileFormat: String, Codable {
        case image = "image"
        case video = "video"
      }

      public enum ResultPath: Codable {
        case base64(path: String)
        case url(path: String)

        public var value: String {
          switch self {
          case .base64(let value): return value
          case .url(let value): return value
          }
        }
      }

      public var endpoint: String
      public var url: String
      public var remoteApiModelConfigMapping: [String: String]

      public var ephemeralApiSecret: Bool
      public var requestType: String
      public var taskIdPath: String
      public var statusUrlTemplate: String
      public var resultPath: ResultPath
      public var statusPath: String
      public var successStatus: String
      public var failureStatus: String
      public var pendingStatuses: [String]
      public var errorMsgPath: String?
      public var apiKey: String?
      public var apiSecret: String?
      public var apiFileFormat: ApiFileFormat
      public var pollingInterval: TimeInterval
      public var passthroughConfigs: [String: JSONValue]?
      public var settingsSections: [String]
      public var customImageSizeRatios: [String]?
      public var tokenConfig: [String: String]?
      public var customRequestBody: [String: CustomRequestBodyValue]?
      public var downloadUrlSuffix: String?
      public init(
        endpoint: String,
        url: String,
        remoteApiModelConfigMapping: [String: String],
        ephemeralApiSecret: Bool,
        requestType: String,
        taskIdPath: String,
        statusUrlTemplate: String,
        resultPath: ResultPath,
        statusPath: String,
        successStatus: String,
        failureStatus: String,
        pendingStatuses: [String],
        errorMsgPath: String? = nil,
        apiKey: String? = nil,
        apiSecret: String? = nil,
        apiFileFormat: ApiFileFormat,
        pollingInterval: TimeInterval = 5.0,
        passthroughConfigs: [String: JSONValue]? = nil,
        settingsSections: [String],
        customImageSizeRatios: [String]? = nil,
        tokenConfig: [String: String]? = nil,
        customRequestBody: [String: CustomRequestBodyValue]? = nil,
        downloadUrlSuffix: String? = nil
      ) {
        self.endpoint = endpoint
        self.url = url
        self.remoteApiModelConfigMapping = remoteApiModelConfigMapping
        self.ephemeralApiSecret = ephemeralApiSecret
        self.requestType = requestType
        self.taskIdPath = taskIdPath
        self.statusUrlTemplate = statusUrlTemplate
        self.resultPath = resultPath
        self.statusPath = statusPath
        self.successStatus = successStatus
        self.failureStatus = failureStatus
        self.pendingStatuses = pendingStatuses
        self.errorMsgPath = errorMsgPath
        self.apiSecret = apiSecret
        self.apiKey = apiKey
        self.apiFileFormat = apiFileFormat
        self.pollingInterval = pollingInterval
        self.settingsSections = settingsSections
        self.passthroughConfigs = passthroughConfigs
        self.customImageSizeRatios = customImageSizeRatios
        self.tokenConfig = tokenConfig
        self.customRequestBody = customRequestBody
        self.downloadUrlSuffix = downloadUrlSuffix
      }
    }
    public var name: String
    public var file: String
    public var prefix: String
    public var version: ModelVersion
    public var upcastAttention: Bool
    public var defaultScale: UInt16
    public var textEncoder: String?
    public var autoencoder: String?
    public var modifier: SamplerModifier?
    public var deprecated: Bool?
    public var imageEncoder: String?
    public var clipEncoder: String?
    public var additionalClipEncoders: [String]?
    public var t5Encoder: String?
    public var diffusionMapping: String?
    public var highPrecisionAutoencoder: Bool?
    public var defaultRefiner: String?
    public var isConsistencyModel: Bool?
    public var conditioning: Denoiser.Conditioning?
    public var objective: Denoiser.Objective?
    public var noiseDiscretization: NoiseDiscretization?
    public var latentsMean: [Float]?
    public var latentsStd: [Float]?
    public var latentsScalingFactor: Float?
    public var stageModels: [String]?
    public var textEncoderVersion: TextEncoderVersion?
    public var guidanceEmbed: Bool?
    public var paddedTextEncodingLength: Int?
    public var hiresFixScale: UInt16?
    public var mmdit: MMDiT?
    public var builtinLora: Bool?
    public var teaCacheCoefficients: [Float]?
    public var remoteApiModelConfig: RemoteApiModelConfig?
    public var framesPerSecond: Double?
    public var isBf16: Bool?
    public var note: String?
    public init(
      name: String, file: String, prefix: String, version: ModelVersion,
      upcastAttention: Bool = false, defaultScale: UInt16 = 8, textEncoder: String? = nil,
      autoencoder: String? = nil, modifier: SamplerModifier? = nil, deprecated: Bool? = nil,
      imageEncoder: String? = nil, clipEncoder: String? = nil,
      additionalClipEncoders: [String]? = nil, t5Encoder: String? = nil,
      diffusionMapping: String? = nil, highPrecisionAutoencoder: Bool? = nil,
      defaultRefiner: String? = nil, isConsistencyModel: Bool? = nil,
      conditioning: Denoiser.Conditioning? = nil, objective: Denoiser.Objective? = nil,
      noiseDiscretization: NoiseDiscretization? = nil, latentsMean: [Float]? = nil,
      latentsStd: [Float]? = nil, latentsScalingFactor: Float? = nil, stageModels: [String]? = nil,
      textEncoderVersion: TextEncoderVersion? = nil, guidanceEmbed: Bool? = nil,
      paddedTextEncodingLength: Int? = nil, hiresFixScale: UInt16? = nil, mmdit: MMDiT? = nil,
      builtinLora: Bool? = nil, teaCacheCoefficients: [Float]? = nil,
      framesPerSecond: Double? = nil, isBf16: Bool? = nil,
      remoteApiModelConfig: RemoteApiModelConfig? = nil, note: String? = nil
    ) {
      self.name = name
      self.file = file
      self.prefix = prefix
      self.version = version
      self.upcastAttention = upcastAttention
      self.defaultScale = defaultScale
      self.textEncoder = textEncoder
      self.autoencoder = autoencoder
      self.modifier = modifier
      self.deprecated = deprecated
      self.imageEncoder = imageEncoder
      self.clipEncoder = clipEncoder
      self.additionalClipEncoders = additionalClipEncoders
      self.t5Encoder = t5Encoder
      self.diffusionMapping = diffusionMapping
      self.highPrecisionAutoencoder = highPrecisionAutoencoder
      self.defaultRefiner = defaultRefiner
      self.isConsistencyModel = isConsistencyModel
      self.conditioning = conditioning
      self.objective = objective
      self.noiseDiscretization = noiseDiscretization
      self.latentsMean = latentsMean
      self.latentsStd = latentsStd
      self.latentsScalingFactor = latentsScalingFactor
      self.stageModels = stageModels
      self.textEncoderVersion = textEncoderVersion
      self.guidanceEmbed = guidanceEmbed
      self.paddedTextEncodingLength = paddedTextEncodingLength
      self.hiresFixScale = hiresFixScale
      self.mmdit = mmdit
      self.builtinLora = builtinLora
      self.teaCacheCoefficients = teaCacheCoefficients
      self.remoteApiModelConfig = remoteApiModelConfig
      self.framesPerSecond = framesPerSecond
      self.isBf16 = isBf16
      self.note = note
    }
    fileprivate var predictV: Bool? = nil
  }

  private static var fileSHA256: [String: String] = [
    "clip_vit_l14_f16.ckpt": "809bfd12c8d4b3d79c14e850b99130a70854f6fd8dedcacdf429417c02fa3007",
    "open_clip_vit_h14_f16.ckpt":
      "cdaa1b93cb099d4aff8831ba248780cebbb54bcd2810dd242513c4a8c70ba577",
    "sd_v1.4_f16.ckpt": "0e0d62f677aba5aae59d977e8a48b2ad87b6d47d519e92f11b7f988c882e5910",
    "vae_ft_mse_840000_f16.ckpt":
      "3b35514e11dd2b913e0579089babc1dfbd36589a77044c2e9b8065187e2f4154",
    "sd_v1.5_f16.ckpt": "bf867591702e4c5d86cb126a3601d7e494180cce956b8dfaf90e5093d2e7c0f6",
    "sd_v1.5_inpainting_f16.ckpt":
      "4e935e18e3d1be94378d96e0d9cb347fcd75de4821ff1d142c60640313b60ab2",
    "sd_v2.0_f16.ckpt": "73cbc76b4ecc4a8c33bf4c452d396c86c42c2f50361745bd649a98e9ea269a3b",
    "sd_v2.1_f16.ckpt": "2d9a7302668bacf3b801327bc23b116f24a441e6229cc4a4b7c39aaa4bf3c9f7",
    "sd_v2.0_inpainting_f16.ckpt":
      "d42b44d3614a0e22195aa5e4f94f417c7c755a99c463e8730ad8f7071c2c5a92",
    "sd_v2.0_depth_f16.ckpt": "64f907b7bf40954477439dda42dcf2cf864526b4c498279cd4274bce12fe896d",
    "sd_v2.0_768_v_f16.ckpt": "992be2b0b34e0a591b043a07b4fc32bf04210424872230a15169e68ef45cde43",
    "sd_v2.1_768_v_f16.ckpt": "04378818798ab37ce9adc189ea28c342d9edde8511194bf5a205f56bb38cf05c",
    "minisd_v1.4_f16.ckpt": "7aed73bf40b49083be32791de39e192f6ac4aa20fbc98e13d4cdca7b5bdd07bf",
    "wd_v1.3_f16.ckpt": "b6862eec82ec14cdb754c5df5c131631bae5e4664b5622a615629c42e7a43c05",
    "classicanim_v1_f16.ckpt": "168799472175b77492814ae3cf5e9f793a3d3d009592a9e5b03781626ea25590",
    "modi_v1_f16.ckpt": "ca76d84c1783ef367201e4eac2e1dddbce0c40afc6de62a229b80cb04ae7c4f0",
    "arcane_v3_f16.ckpt": "4c55d2239e1f0ff40cc6e1ae518737735f6d1b7613f8f7aca9239205f0be729a",
    "cyberpunk_anime_f16.ckpt": "df55b6c66704b51921e31711adaab9e37bd78fc10733fcd89e6f86426230ef41",
    "redshift_v1_f16.ckpt": "a7fc94bac178414d7caf844787afcaf8c6c273ebf9011fed75703de7839fc257",
    "redshift_768_v_f16.ckpt": "aa6520ae1fc447082230b2eb646c40e6f776f257c453134d0f064a89ac1de751",
    "redshift_768_v_open_clip_vit_h14_f16.ckpt":
      "9c7f1a65fe890f288c2d2ff7cef11b502bf10965a7eaa7d0d43362cab9f90eca",
    "dnd_30000_f16.ckpt": "3de9309cf4541168fb39d96eaba760146b42e7e9870a3096eb4cd097384ea1d9",
    "trnlgcy_f16.ckpt": "3ed86762dda66f5dc728ee1f67085d2ba9f3e3ea1b5b3464b8f3a791954cfa3c",
    "classicanim_v1_clip_vit_l14_f16.ckpt":
      "77cfbb6054a2a5581873c3b1be8c6457bed526d1f15d6cffb6e381401692a488",
    "modi_v1_clip_vit_l14_f16.ckpt":
      "e7907cbb2f7656bb2f6fb4ead4fcb030721e4218ca2a105976b88bce852f2860",
    "arcane_v3_clip_vit_l14_f16.ckpt":
      "954f1e1fb690dcb1820adaf83099b39057e2b1bcbbdc12ecfe37ac17bcad6fa7",
    "cyberpunk_anime_clip_vit_l14_f16.ckpt":
      "d62bb1de4b579d73111b3355cad72b1d8f3bf22519c4bfd1a224bdd952cd0279",
    "redshift_v1_clip_vit_l14_f16.ckpt":
      "95532a3275a81d909d657c98b73ef576809254e29052aaa809d9336c13f182a1",
    "dnd_30000_clip_vit_l14_f16.ckpt":
      "96c75d1c11030a51aa8dac5410cc6fa98b071b52f0f79a07097df022b20754dc",
    "trnlgcy_clip_vit_l14_f16.ckpt":
      "a99adbecbed4e370abcffc2574fac8e664a2530531fdc89b71d2f15711f40545",
    "mdjrny_v4_f16.ckpt": "a0d976948c18943f1281268cc3edbe1d1fa2a4098b5a290d9947a1a299976699",
    "mdjrny_v4_clip_vit_l14_f16.ckpt":
      "ad4e3d64c0a5e81d529c984dcfbdc6858d73e14ebe8788975e6b8c4fbfc17629",
    "nitro_v1_f16.ckpt": "2549d7220cce7f53311fe145878e1af8bcd52efaf15bcb81a2681c0abcddd6c3",
    "nitro_v1_clip_vit_l14_f16.ckpt":
      "2b5424697630a50ed2d1b8c2449e3fb5f613a6569d72d16dc59d1e28a8a0c07d",
    "anything_v3_f16.ckpt": "f4354727512d6b6a2d5e4cf783fdc8475e7981c50b9f387bc93317c22299e505",
    "anything_v3_clip_vit_l14_f16.ckpt":
      "5f1311561bdac6d43e4b3bacbee8c257bf788e6c86b3c69c68247a9abab1050d",
    "anything_v3_vae_f16.ckpt": "3b7d16260a7d211416739285f97d53354b332cfceadb2b7191817f4e1cfb5d57",
    "hassanblend_v1.4_f16.ckpt": "e3566b98cfa81660cd4833c01cd9a05a853e367726d840a5eb16098b53c042ae",
    "lvngvncnt_v2_f16.ckpt": "dbacd01fb82501895afde1bbcf3f16eefeea8043fa3463de640c09a9315460be",
    "lvngvncnt_v2_clip_vit_l14_f16.ckpt":
      "cbdaae485f60c7cb395e5725dd16e769816274b74498d0c45048962a49cc4a06",
    "spiderverse_v1_f16.ckpt": "8c8c80add2d663732e314c3a2fb49c1f2bd98f48190b79227d660ce687516b2d",
    "spiderverse_v1_clip_vit_l14_f16.ckpt":
      "bab7fcf0e615154ff91c88a8fbf9b18a548e8ba0a338fb030a3fedf17ce0602d",
    "eldenring_v3_f16.ckpt": "c6b79886e426d654c9e84cf53a7dd572fbb9e7083c47384a76d02702c54c50c3",
    "eldenring_v3_clip_vit_l14_f16.ckpt":
      "dcd2234e90f8df2c4eb706f665fa860ad54df2ae109cfcd8b235c1c420bd2d4d",
    "papercut_v1_f16.ckpt": "7b1d14757e1c58b1bef55220d0fd10ab4ad8e2670bb4e065e4b6c4e0b6a6395e",
    "papercut_v1_clip_vit_l14_f16.ckpt":
      "bc0c471e51bbe0649922dad862019b96e68d4abf724998bbfa9495e70bd2023d",
    "voxelart_v1_f16.ckpt": "e771d7acd484162377c62a6033b632ea290d4477bf3cb017a750f15ab5350ca7",
    "voxelart_v1_clip_vit_l14_f16.ckpt":
      "ebd3ce92b9ec831a6f217c42be0b8da887484867583156d0e2ceb3e48bae3be8",
    "balloonart_v1_f16.ckpt": "f73bcbd3a6db0dca10afb081a2066a7aea5b117962bd26efc37320dfc3b9b759",
    "balloonart_v1_clip_vit_l14_f16.ckpt":
      "6be250d1c38325f7ee80f3fcd99e1a490f28deb29a8f78188813e8157f1949b3",
    "f222_f16.ckpt": "ae19854df68e232c3bbda8b9322e9f56ccd2d96517a31a82694c40382003f8ae",
    "supermarionation_v2_f16.ckpt":
      "70e13769ee9c8b8c4d4b8240f23b8d8fcef154325fd9162174b75f67c5629440",
    "supermarionation_v2_clip_vit_l14_f16.ckpt":
      "e2da78a79ee90fe352e465326e2dc0c055888c27a84d465cfd9ea2987a83a131",
    "inkpunk_v2_f16.ckpt": "8957387975caf8c56caa6c4c2b9d8fff07bda7a8a2aadec840be3fd623d1d2fe",
    "inkpunk_v2_clip_vit_l14_f16.ckpt":
      "569d9796b5f3b33ed1ce65b27fa3fb4dfdb8ef2440555fa33f30fa8d118cc293",
    "samdoesart_v3_f16.ckpt": "5a55df0470437ac0f3f0c05d77098c6eb8577c61ce0e1b2dc898240fb49fd10e",
    "samdoesart_v3_clip_vit_l14_f16.ckpt":
      "6d84e79c05f9c89172f4b82821a7c8223d3bd6bacfd80934dd85dce71a8f2519",
    "ghibli_v1_f16.ckpt": "dfcf9358528e8892f82b4ba3d0c9245be928e2e920e746383bdaf1b9a3a93151",
    "ghibli_v1_clip_vit_l14_f16.ckpt":
      "bf7c353e5b2b34bff2216742e114ee707f0ad023cc0bfd5ebde779b3b3162a02",
    "analog_v1_f16.ckpt": "ffed9bb928a20f90f9881ac0d51e918c1580562f099fdd45c061c292dec63ab5",
    "analog_v1_clip_vit_l14_f16.ckpt":
      "f144ac4ad344c82c3b1dc69e46aba8d9c6bc20d24de9e48105a3db3e4437108d",
    "dnd_classes_and_species_f16.ckpt":
      "a6059246c1c06edc73646c77a1aa819ca641e0d8ceba0e25365938ab71311174",
    "dnd_classes_and_species_clip_vit_l14_f16.ckpt":
      "09fdf2d991591947e2743e8431e9d6eaf99fe2f524de9c752ebb7a4289225b02",
    "aloeveras_simpmaker_3k1_f16.ckpt":
      "562db3b5ca4961eed207e308073d99293d27f23f90e09dba58f2eb828a2f8e0c",
    "hna_3dkx_1.0b_f16.ckpt":
      "5e9246ff45380d6e0bd22506d559e2d6616b7aa0e42052a92c0270b89de2defa",
    "hna_3dkx_1.0b_clip_vit_l14_f16.ckpt":
      "7317f067a71f1e2a2a886c60574bb391bf31a559b4daa4901c45d1d5d2acc7d6",
    "seek_art_mega_v1_f16.ckpt":
      "0f10cfa16950fc5bb0a31b9974275c256c1a11f26f92ac26be6f7ea91e7019ac",
    "seek_art_mega_v1_clip_vit_l14_f16.ckpt":
      "9dd3af747d71b10d318b876a9285f8cc7c350806585146a3eaa660bcaf54bc7e",
    "instruct_pix2pix_22000_f16.ckpt":
      "ffe6548ff4e803c64f8ca2b84024058e88494329acff29583fbb9f45305dd410",
    "hassanblend_v1.5.1.2_f16.ckpt":
      "e5eb4e11fa1f882dc084a0e061abf6b7f5e7dd11c416ff14842c049b9727c5d1",
    "hassanblend_v1.5.1.2_clip_vit_l14_f16.ckpt":
      "0d572f5e379c48c88aa7ca1d6aff095d94cacaf8b90f6444f4af46a7d3d18f33",
    "hna_3dkx_1.1_f16.ckpt":
      "9e333094d9b73db3e0438f7520c0cd5deb2f0f6b3aa890ce464050cc7dd8d693",
    "hna_3dkx_1.1_clip_vit_l14_f16.ckpt":
      "5ce38e05ada7ec4488c600bc026db1386fb4cdca2882fe51561c49a1bc70da4d",
    "kandinsky_f16.ckpt":
      "563cbf6dd08c81063c45310a7a420b75004d6f226eb7e045f167d03d485fc36a",
    "kandinsky_diffusion_mapping_f16.ckpt":
      "6467fd6ac08bc4d851ed09286f2273f134fe5d6763086ef06551f1285de059f0",
    "kandinsky_movq_f16.ckpt":
      "f7ac86bd2f1b3bb7487a064df64e39fbf92905e40ebfbe943c3859ff05204071",
    "xlm_roberta_f16.ckpt":
      "772cd148b7254d16cd934aad380362cde8869edb34f787eb7cc4776a64e3d5a2",
    "image_vit_l14_f16.ckpt":
      "f75c2ac4b5f8e0c59001ce05ecf5b11ee893f7687b2154075c2ddd7c11fe9b32",
    "deliberate_v2_q6p_q8p.ckpt":
      "4441ea31f748a5af97021747bc457e78ae0c8d632f819a26cb8019610972c0f0",
    "deliberate_v2_clip_vit_l14_f16.ckpt":
      "79dc846fe47f4bd5188bce108c9791f36cc2927bed6f96c8dc7369b345539d81",
    "disney_pixar_cartoon_type_b_q6p_q8p.ckpt":
      "31f38b788e1acdde65288f1e3780c64df9c98cd5fa7fa38bce5bce085f633d95",
    "disney_pixar_cartoon_type_b_clip_vit_l14_f16.ckpt":
      "0401d93b66dff9de82521765bbcb2292904a247e22155158d91f83aef4b4d351",
    "realistic_vision_v3.0_q6p_q8p.ckpt":
      "6a4294760fb82295522cd7d610c95269070c403e5a4b41f67ce2db93fd93ee3a",
    "realistic_vision_v3.0_clip_vit_l14_f16.ckpt":
      "71f1c7726f842d72fe04a7e17ee468c32752d097b89e9114708c0dc13a0060a2",
    "dreamshaper_v6.31_q6p_q8p.ckpt":
      "14a9c0e4a5ebb4a66d4fd882135e60a3951e5d1d96e802cbf2106e91427e349f",
    "dreamshaper_v6.31_clip_vit_l14_f16.ckpt":
      "7384f31ea620891a7bca84c3b537beda7dbc5473873c90810b797e14ab263fc4",
    "open_clip_vit_bigg14_f16.ckpt":
      "1bc61283f12c3b923f4366a27d316742c0610aa934803481f0b5277124b9a8f4",
    "sd_xl_base_0.9_f16.ckpt": "e7613b7593f8f48b3799b3b9d2308ec2e4327cdd5f4904717d10846e0d13e661",
    "sd_xl_refiner_0.9_f16.ckpt":
      "b6e830f2d2084ca078178178aa67b31d85b17a772304e2ed39927e2f39487277",
    "sdxl_vae_f16.ckpt": "275decbdbe986f55bb20018bd636e3b0a8b0a6a3b8c28754262dcb84f33a62d7",
    "sdxl_vae_v1.0_f16.ckpt": "8ceb1b62fc9b88c20a171452fef55e3a5546cc621c943c78188f648351b4d7e4",
    "sd_xl_base_1.0_f16.ckpt": "741f813f9f7f17bf9e284885fa73b5098a30dc6db336179116e8749da10657a3",
    "sd_xl_refiner_1.0_f16.ckpt":
      "73abf6538793530fe3a2358a5789b7906db4e6dc30ce8d9d34b87a506fa2e34c",
    "sd_xl_base_1.0_q6p_q8p.ckpt":
      "796210c27eec08fd7ea01ad42eaf93efac5783b3f25df60906601a0a293a8f45",
    "sd_xl_refiner_1.0_q6p_q8p.ckpt":
      "be4f78ff34302d1cfbc91c1e83945e798bc58b0bc35ac08209d8d5a66b30c214",
    "ssd_1b_f16.ckpt": "8fed449f74cefadf9f10300eaa704d2fa0601bf087c1196560ce861aa6ab3d68",
    "ssd_1b_q6p_q8p.ckpt": "a4096821ac5fbc9c34be2fe86ca5b0e9d2f0cc64fd9c3ba47e1efe02cec5da09",
    "lcm_sd_xl_base_1.0_f16.ckpt":
      "937a0851d1c3fbb7b546d94edfad014db854721c596e0694d9e4ca7d6e8cd8de",
    "lcm_sd_xl_base_1.0_q6p_q8p.ckpt":
      "0830466d22f5304f415e2d96ab16244f21a2257d5e906ed63a467a393a38c250",
    "lcm_ssd_1b_f16.ckpt": "e1156cc6e6927a462102629d030e3d6377e373664201dad79fb1ff4928bb85b0",
    "lcm_ssd_1b_q6p_q8p.ckpt": "959d09951bdba0a73fafb6a69fed83b21012cbc4df78991463bbd84e283cc6fe",
    "sd_xl_turbo_f16.ckpt": "c85ea750f1ff5d17c032465c07f854eaf5f1551e27bd85dbe9c2d1025a41e004",
    "sd_xl_turbo_q6p_q8p.ckpt": "a8072ace4eb3d6590db8abe8fda6c0c22f4c3e68efb86f0e58a27dc4f68731ef",
    "open_clip_vit_h14_vision_model_f16.ckpt":
      "87b70da1898b05bfc7b53e3587fb5c0383fe4f48ccae286e8dde217bbee1e80d",
    "svd_i2v_1.0_f16.ckpt": "5751756a84bd9b6c91d2d6df7393d066d046e8ca939a8b8fa4ac358a07acaf94",
    "svd_i2v_1.0_q6p_q8p.ckpt": "5c8e4c1a1291456c5516c4c66d094eada0e11660c7b474cc39e45c9ceff27309",
    "svd_i2v_xt_1.0_f16.ckpt": "e5fd1a2f5fb7f1a13424e577a13c04dfd873b1cc6e3cdebc4c797d97d21a6865",
    "svd_i2v_xt_1.0_q6p_q8p.ckpt":
      "f3c4a06c1a1cb71a6b032e2ceb2d04e1d9c8457c455f8984f5324bbd8ba6d2e2",
    "fooocus_inpaint_sd_xl_v2.6_f16.ckpt":
      "f93886d787043cab976d31376b072bdc320185606331349ace9b48c41eeda867",
    "fooocus_inpaint_sd_xl_v2.6_q6p_q8p.ckpt":
      "f299e673da2d0da8ffccd6a01e9901261a9091a278032316c3218598ee9b5f2d",
    "svd_i2v_xt_1.1_f16.ckpt": "cd4d0c43c6cd3a3af51e35d465e2cec5292778f9cd12c92b64873f59de6ef314",
    "svd_i2v_xt_1.1_q6p_q8p.ckpt":
      "61c6fe0cce4d91fc1b83dd65f956624dc2c996fb21fdc4fa847fbf4bc97e0030",
    "wurstchen_3.0_stage_a_hq_f16.ckpt":
      "ad9d2b43ceb68f9bb9d269a6a5fd345a5f177a0f695189be82219cb4d2740277",
    "wurstchen_3.0_stage_b_q6p_q8p.ckpt":
      "b0611225cf2f2a7b9109ae18eaf12bfe04ae60010ac5ea715440d79708e578b8",
    "wurstchen_3.0_stage_c_f32_q6p_q8p.ckpt":
      "0e57d6f6c7749a34ea362a115558aeeb209da82e54b06e3b97433ed64b244439",
    "wurstchen_3.0_stage_b_f16.ckpt":
      "a541358038cb86064a4d43bd0b6dab1cb95129520fca67eb178bce3baccc1d02",
    "wurstchen_3.0_stage_c_f32_f16.ckpt":
      "aa05651d1920d1fd0b70d06397548bf9e77fac93ff4b4bc9bc98cea749e5a8db",
    "playground_v2.5_f16.ckpt": "9a8e167526a65d5caebfd6d5163705672cfd4d201cb273d11c174e46af041b4a",
    "playground_v2.5_q6p_q8p.ckpt":
      "18ddd151c7ae188b6a0036c72bf8b7cd395479472400a3ed4d1eb8e5e65b36e3",
    "open_clip_vit_h14_visual_proj_f16.ckpt":
      "ef03b8ac7805d5a862db048c452c4dbbd295bd95fed0bf5dae50a6e98815d30f",
    "wurstchen_3.0_stage_c_effnet_previewer_f32_f16.ckpt":
      "fd1c698895afc14e68a0d7690c787400796114d2bfac0df254598d8207f93f0f",
    "t5_xxl_encoder_q6p.ckpt":
      "8f8fa0acc618df6f225122b3d03b6f60034490d9f9fd3b8799e7faa3e08943b7",
    "sd3_vae_f16.ckpt":
      "51d42d4745456396f0cbb034f4eb9d6495cc2a552ca4af7ba80d64fdba5f9678",
    "sd3_medium_q8p.ckpt":
      "a313371538d8018ee6f3f3b6aa3e08bff64bfa3a56c1f777b90973f71138b3a2",
    "sd3_medium_f16.ckpt":
      "9ee38fee52867678c21afffd7c176443a61e30eed728c1a28e2ff4982fe89bee",
    "pixart_sigma_xl_2_1024_ms_f16.ckpt":
      "b78f0f8d4988b6edf38eeff8c1d33d2b4ffca1fa79c4b45f51b8647aa3b625a0",
    "pixart_sigma_xl_2_1024_ms_q8p.ckpt":
      "d5379d9f7ad18e3dd8b6f4b564df1f03bc1e8377e1486bb793467fc4fab6ae5c",
    "pile_t5_xl_encoder_q8p.ckpt":
      "ef8b228e915bb21101c4c34e89039e2c42ddba843dae4b1e4f813a4785b1df1b",
    "auraflow_v0.1_q8p.ckpt": "30ebb3796987ff2f79cb67b16e72f4ba5e31dd4706af0e8d0d91fb16165c71ee",
    "auraflow_v0.1_f16.ckpt": "8c5e7ba677ccd11f899f2fb4092ab7cc4ad7686d01be172497e55eeba01c5bb0",
    "auraflow_v0.1_q5p.ckpt": "9aab1942ea2f025846d5d1dcd2ae5df762c1a10887c807b603c223ba8e5e5ad7",
    "auraflow_v0.2_q8p.ckpt": "cda840bce05ada4c97d95080160f18dc594b6c5f2d4da45c33db51c37c070170",
    "auraflow_v0.2_f16.ckpt": "727622af19710b8014da024c7294573c02fefb7be83e178fa4c2b50a9d2bc922",
    "auraflow_v0.2_q5p.ckpt": "b3d4a2c3be69e285028de0d61a17a5fbe34b9e1504725f32f52e75b8d9d8a2cc",
    "flux_1_vae_f16.ckpt": "453d09645419d1ffc2f641e4a4c6ccf75f69c7215938a285e474ece3762fe293",
    "flux_1_schnell_q8p.ckpt": "26a38212290a928aad21d4d9a6e534cca6c06ddb7ce0a926ac31533500e39f64",
    "flux_1_schnell_f16.ckpt": "6fad328261de43847bf6a53a075445e90c5fd90f65c4a68dc538fcb7aa5f13a2",
    "flux_1_schnell_q5p.ckpt": "37a28dcba93e23e4433b64d621b5352f4651d46eaf40251351d8a553642b907b",
    "wan_v2.1_1.3b_480p_f16.ckpt":
      "d8f0e77085890f86b2d61095d1340bb94bbf876edac5a472693e7f047e5ac4a6",
    "wan_v2.1_1.3b_480p_q8p.ckpt":
      "2aa588d0c001d190e8a3a6b176835e89c97800479b8fe5d1837fe4514f9bf805",
    "umt5_xxl_encoder_q8p.ckpt": "72ef62d22c09a3b764ac9e6fe0100f4029619fb3ff8ccec3432e509487b29831",
    "wan_v2.1_video_vae_f16.ckpt":
      "4c518b128b3c1f2ea164aa46269d8875b4be3661d1fc0fba2709d03fa94e418b",
    "wan_v2.1_14b_720p_q8p.ckpt":
      "b21d70e196e5dfd4c3238607c9c3a13150d4aae04848245ed57241b83ee586bd",
    "wan_v2.1_14b_720p_q6p_svd.ckpt":
      "bc421931cd177c25d419123ca5f569b45d6942716e867520481a97f6bb988896",
    "open_clip_xlm_roberta_large_vit_h14_f16.ckpt":
      "362c9940a36acce5a4e13b9167d5daebd005ac026443cd37b7955ac0acd72083",
    "wan_v2.1_14b_i2v_480p_q8p.ckpt":
      "ad9ba7c4db022abd89e9d9f02061f65335b6c222b075cb67831ea5118ecf480a",
    "wan_v2.1_14b_i2v_720p_q8p.ckpt":
      "eacd5ab2d91f982e68c5a786735a88f504c303e34907604c89a01f2083654a6d",
    "wan_v2.1_14b_i2v_480p_q6p_svd.ckpt":
      "dba1a4fe5c29eb33479759b00ed309c32cc63d92663c7bd74c3d0aedd2dbd0b9",
    "wan_v2.1_14b_i2v_720p_q6p_svd.ckpt":
      "16bc54134e4e16998df12713722d8cd1038f2cdd0955023835d2801bae720c54",
    "long_clip_vit_l14_f16.ckpt":
      "82031eaa248d543a072af378ccd6280cd3be1d07f8733c5d15f9ec4feb82501a",
    "long_open_clip_vit_bigg14_f16.ckpt":
      "6beca0db6c1f84b84b6facb0c2ce4abe56fb220be978ee1438064797861f949b",
    "llama_3.1_8b_instruct_q8p.ckpt":
      "9b0a80a78041ea4ad3c608f7255ec2186afb3ce5d504f955cfd821afc590da57",
    "hidream_i1_fast_q8p.ckpt": "a5f17f1a86a903b8ce8a4fe147ddd82d3c166842b5c67116eb7bf4692c22b2e8",
    "hidream_i1_dev_q8p.ckpt": "1ff76a095b8f75e3047409e1704d1fbbb6c923853a5c59cc699a7b94a5b2c83e",
    "hidream_i1_full_q8p.ckpt": "24c76c58d296f467e458a5bec4edd512ddf697ceb6739239e8a2494e5a50cf4e",
    "hidream_i1_fast_q5p.ckpt": "ce7254c72257edd78b821881c72e9a91afe2752a187968fc4eb3a1648ec35053",
    "hidream_i1_dev_q5p.ckpt": "9779d730f8f2258cdf721a770c91ade978836f11510d9173075f89bc8f8be3e3",
    "hidream_i1_full_q5p.ckpt": "a1f5371896c93c7fb55328331c226eb9bcaabf804a142f6c85f5684d5bd4c3ae",
    "hidream_e1_full_q8p.ckpt": "0f24a6f94cd7105a1bfea195e4ce3e064427833e97390fc1ce6bc6945c8cb93f",
    "hidream_e1_full_q5p.ckpt": "63ff5d43c474937b3f83fe9220a2dbbea7d215fc0585db0fcf12e37ac87dc60c",
    "hidream_e1_1_q8p.ckpt": "b1fe02e5992e6696947f76d5afe0ea1aef7908f9947ae2d665b1d21661aeaf46",
    "hidream_e1_1_q5p.ckpt": "c50ddbdccc8e59ea03f1d382dbdb79c3ad9ad1ec8f543173241469324a4396dd",
    "wan_v2.2_a14b_hne_t2v_q8p.ckpt":
      "63b3dc2eaac38d1019017c4745ea763e9894b5aa55a97fb52dfe26985abc42e6",
    "wan_v2.2_a14b_hne_t2v_q6p_svd.ckpt":
      "f3d3adb772520896bc5d016d00f9159c8595fbdec2ecc33dff2572bffb39cadd",
    "wan_v2.2_a14b_lne_t2v_q8p.ckpt":
      "5fe77d9998141a667071353658d71acef656389210327391c832d39b4ee39671",
    "wan_v2.2_a14b_lne_t2v_q6p_svd.ckpt":
      "861aef3cb11c3dd44052e79a5cbd4740550dc05a7d119cf3069de152958c9e1e",
    "wan_v2.2_a14b_hne_i2v_q8p.ckpt":
      "bf44ed82e723cf3f2469c9a89fd5a98296c924d489b883e7dadef4c122ad295d",
    "wan_v2.2_a14b_hne_i2v_q6p_svd.ckpt":
      "565f535dc5264a9264ec17e2e8396d9a23509be92d71a4930c40fafa16b29685",
    "wan_v2.2_a14b_lne_i2v_q8p.ckpt":
      "51486a592ba0190c333ba4e071aa9ed2b502c8aa5b79cb5d38325e399ac6c129",
    "wan_v2.2_a14b_lne_i2v_q6p_svd.ckpt":
      "ed0c61db1dccce4beaa2a805792de43c77c9b87d876e3633ef60e4cb716e153a",
    "qwen_2.5_vl_7b_q8p.ckpt": "513b759b24619946d3ca13c0bf57464a32098593bbf8342001358fbfa51f78b1",
    "qwen_image_vae_f16.ckpt": "701e7c46ed6c2fa8036543780317e06d264374f9b4dbfc22f27c0b3181bb988a",
    "qwen_image_1.0_q8p.ckpt": "34e25c219945f5887bbab5ed0be39db78b8635f4986f6715b80eda2a3d581081",
    "qwen_image_1.0_q6p.ckpt": "51bf057484c66cb7c4e6e3bab80b99e99aa20af043f42086ce5080dc01986b62",
    "qwen_2.5_vl_7b_vit_f16.ckpt":
      "0917609453586573befadff7aa4228868b76f613ff5d5f5d43d8fcbce44c3708",
    "qwen_image_edit_1.0_q8p.ckpt":
      "06810ad2fa2f95835ee9b9b579f9a84435c8477c37614f00d814f066ec6d4539",
    "qwen_image_edit_1.0_q6p.ckpt":
      "ca7cc367447d95b36d601c954158a488f53b5a67cca454aef79bcfef80b0785d",
  ]

  public static let defaultSpecification: Specification = builtinSpecifications[0]

  public static let builtinSpecifications: [Specification] = [
    Specification(
      name: "Qwen Image 1.0", file: "qwen_image_1.0_q8p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      note:
        "[Qwen Image](https://huggingface.co/Qwen/Qwen-Image) is a state-of-the-art open-source image generation model known for its exceptional text layout and prompt adherence across a wide range of styles, including photorealistic, cartoon, and artistic. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended."
    ),
    Specification(
      name: "Qwen Image 1.0 (6-bit)", file: "qwen_image_1.0_q6p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      note:
        "[Qwen Image](https://huggingface.co/Qwen/Qwen-Image) is a state-of-the-art open-source image generation model known for its exceptional text layout and prompt adherence across a wide range of styles, including photorealistic, cartoon, and artistic. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended."
    ), /*
    Specification(
      name: "Qwen Image Edit 1.0", file: "qwen_image_edit_1.0_q8p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(qkNorm: true, dualAttentionLayers: [], activationFfnScaling: [59: 2]),
      note:
        "[Qwen Image Edit](https://huggingface.co/Qwen/Qwen-Image) is a state-of-the-art open-source image edit model excels at image edit tasks such as background alternation, style transfer, object removal etc. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended."
    ),
    Specification(
      name: "Qwen Image Edit 1.0 (6-bit)", file: "qwen_image_edit_1.0_q6p.ckpt", prefix: "",
      version: .qwenImage, defaultScale: 16, textEncoder: "qwen_2.5_vl_7b_q8p.ckpt",
      autoencoder: "qwen_image_vae_f16.ckpt", modifier: .kontext,
      clipEncoder: "qwen_2.5_vl_7b_vit_f16.ckpt", objective: .u(conditionScale: 1000),
      hiresFixScale: 24,
      mmdit: .init(qkNorm: true, dualAttentionLayers: [], activationFfnScaling: [59: 2]),
      note:
        "[Qwen Image Edit](https://huggingface.co/Qwen/Qwen-Image) is a state-of-the-art open-source image edit model excels at image edit tasks such as background alternation, style transfer, object removal etc. It is Apache 2.0-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended."
    ),*/
    Specification(
      name: "HiDream I1 [fast]", file: "hidream_i1_fast_q8p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      isConsistencyModel: true, objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128,
      hiresFixScale: 24,
      note:
        "[HiDream-I1 [fast]](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) is a state-of-the-art open-source image generation model known for its strong prompt adherence across diverse styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 10–20 sampling steps recommended. Text guidance is not effective for this model."
    ),
    Specification(
      name: "HiDream I1 [fast] (5-bit)", file: "hidream_i1_fast_q5p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      isConsistencyModel: true, objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128,
      hiresFixScale: 24,
      note:
        "[HiDream-I1 [fast]](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) is a state-of-the-art open-source image generation model known for its strong prompt adherence across diverse styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 10–20 sampling steps recommended. Text guidance is not effective for this model."
    ),
    Specification(
      name: "HiDream I1 [dev]", file: "hidream_i1_dev_q8p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), guidanceEmbed: true, paddedTextEncodingLength: 128,
      hiresFixScale: 24,
      note:
        "[HiDream-I1 [dev]](https://huggingface.co/HiDream-ai/HiDream-I1-Dev) is a state-of-the-art open-source image generation model known for its strong prompt adherence across diverse styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 20–30 sampling steps recommended. Text guidance is not effective for this model."
    ),
    Specification(
      name: "HiDream I1 [dev] (5-bit)", file: "hidream_i1_dev_q5p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), guidanceEmbed: true, paddedTextEncodingLength: 128,
      hiresFixScale: 24,
      note:
        "[HiDream-I1 [dev]](https://huggingface.co/HiDream-ai/HiDream-I1-Dev) is a state-of-the-art open-source image generation model known for its strong prompt adherence across diverse styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 20–30 sampling steps recommended. Text guidance is not effective for this model."
    ),
    Specification(
      name: "HiDream I1 [full]", file: "hidream_i1_full_q8p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-I1 [full]](https://huggingface.co/HiDream-ai/HiDream-I1-Full) is a state-of-the-art open-source image generation model known for its exceptional prompt adherence across a wide range of styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended."
    ),
    Specification(
      name: "HiDream I1 [full] (5-bit)", file: "hidream_i1_full_q5p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-I1 [full]](https://huggingface.co/HiDream-ai/HiDream-I1-Full) is a state-of-the-art open-source image generation model known for its exceptional prompt adherence across a wide range of styles, including photorealistic, cartoon, and artistic. It is MIT-licensed and commercially friendly. The model is trained at multiple resolutions using a Flow Matching objective; trailing samplers yield the best results, with 30–50 sampling steps recommended."
    ),
    Specification(
      name: "HiDream E1-1", file: "hidream_e1_1_q8p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", modifier: .editing,
      clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) is an image editing model built on HiDream-I1. It is MIT-licensed and commercially friendly. Trained with dynamic resolutions (around 1MP) using a Flow Matching objective, the model performs best with trailing samplers and 30–50 sampling steps."
    ),
    Specification(
      name: "HiDream E1-1 (5-bit)", file: "hidream_e1_1_q5p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 16, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", modifier: .editing,
      clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) is an image editing model built on HiDream-I1. It is MIT-licensed and commercially friendly. Trained with dynamic resolutions (around 1MP) using a Flow Matching objective, the model performs best with trailing samplers and 30–50 sampling steps."
    ),
    Specification(
      name: "HiDream E1 [full]", file: "hidream_e1_full_q8p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 12, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", modifier: .editing,
      clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-E1 [full]](https://huggingface.co/HiDream-ai/HiDream-E1-Full) is an image editing model built on HiDream-I1. It is MIT-licensed and commercially friendly. Trained at 768×768 resolution using a Flow Matching objective, the model performs best with trailing samplers and 30–50 sampling steps. For optimal results, ensure the width is set to 768 and use the following prompt format: Editing Instruction: {}. Target Image Description: {}."
    ),
    Specification(
      name: "HiDream E1 [full] (5-bit)", file: "hidream_e1_full_q5p.ckpt", prefix: "",
      version: .hiDreamI1, defaultScale: 12, textEncoder: "llama_3.1_8b_instruct_q8p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", modifier: .editing,
      clipEncoder: "long_clip_vit_l14_f16.ckpt",
      additionalClipEncoders: ["long_open_clip_vit_bigg14_f16.ckpt"],
      t5Encoder: "t5_xxl_encoder_q6p.ckpt", highPrecisionAutoencoder: true,
      objective: .u(conditionScale: 1000), paddedTextEncodingLength: 128, hiresFixScale: 24,
      note:
        "[HiDream-E1 [full]](https://huggingface.co/HiDream-ai/HiDream-E1-Full) is an image editing model built on HiDream-I1. It is MIT-licensed and commercially friendly. Trained at 768×768 resolution using a Flow Matching objective, the model performs best with trailing samplers and 30–50 sampling steps. For optimal results, ensure the width is set to 768 and use the following prompt format: Editing Instruction: {}. Target Image Description: {}."
    ),
    Specification(
      name: "Wan 2.2 High Noise Expert T2V A14B", file: "wan_v2.2_a14b_hne_t2v_q8p.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.2 High Noise Expert T2V A14B (6-bit, SVDQuant)",
      file: "wan_v2.2_a14b_hne_t2v_q6p_svd.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16, builtinLora: true,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.2 Low Noise Expert T2V A14B", file: "wan_v2.2_a14b_lne_t2v_q8p.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.2 Low Noise Expert T2V A14B (6-bit, SVDQuant)",
      file: "wan_v2.2_a14b_lne_t2v_q6p_svd.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16, builtinLora: true,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.2 High Noise Expert I2V A14B", file: "wan_v2.2_a14b_hne_i2v_q8p.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting, hiresFixScale: 16,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.2 High Noise Expert I2V A14B (6-bit, SVDQuant)",
      file: "wan_v2.2_a14b_hne_i2v_q6p_svd.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting, hiresFixScale: 16,
      builtinLora: true,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.2 Low Noise Expert I2V A14B", file: "wan_v2.2_a14b_lne_i2v_q8p.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting, hiresFixScale: 16,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.2 Low Noise Expert I2V A14B (6-bit, SVDQuant)",
      file: "wan_v2.2_a14b_lne_i2v_q6p_svd.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting, hiresFixScale: 16,
      builtinLora: true,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.1 T2V 1.3B", file: "wan_v2.1_1.3b_480p_f16.ckpt", prefix: "",
      version: .wan21_1_3b, defaultScale: 8, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 12,
      teaCacheCoefficients: [
        -5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 T2V 1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 832×480. The model supports up to 81 frames, with a recommended shift value of 6.0. For best results, set Text Guidance above 5.0. Wan2.1 is trained with a Flow Matching objective, and trailing samplers will produce the best outputs."
    ),
    Specification(
      name: "Wan 2.1 T2V 1.3B (8-bit)", file: "wan_v2.1_1.3b_480p_q8p.ckpt", prefix: "",
      version: .wan21_1_3b, defaultScale: 8, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 12,
      teaCacheCoefficients: [
        -5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 T2V 1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 832×480. The model supports up to 81 frames, with a recommended shift value of 6.0. For best results, set Text Guidance above 5.0. Wan2.1 is trained with a Flow Matching objective, and trailing samplers will produce the best outputs."
    ),
    Specification(
      name: "Wan 2.1 T2V 14B", file: "wan_v2.1_14b_720p_q8p.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 T2V 14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The recommended resolutions are 832×480. The model supports up to 81 frames, with a recommended shift value of 5.0. For best results, set Text Guidance above 5.0. Wan2.1 is trained with a Flow Matching objective, and trailing samplers will produce the best outputs."
    ),
    Specification(
      name: "Wan 2.1 T2V 14B (6-bit, SVDQuant)", file: "wan_v2.1_14b_720p_q6p_svd.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", hiresFixScale: 16, builtinLora: true,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 T2V 14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.1 T2V 14B (5-bit, SVDQuant)", file: "wan_v2.1_14b_720p_q5p_svd.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", deprecated: true, hiresFixScale: 16,
      builtinLora: true,
      teaCacheCoefficients: [
        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 T2V 14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) is a state-of-the-art text-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.1 I2V 14B 480p", file: "wan_v2.1_14b_i2v_480p_q8p.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 8, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting,
      clipEncoder: "open_clip_xlm_roberta_large_vit_h14_f16.ckpt", hiresFixScale: 12,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 I2V 14B 480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 832×480. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.1 I2V 14B 480p (6-bit, SVDQuant)", file: "wan_v2.1_14b_i2v_480p_q6p_svd.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 8, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting,
      clipEncoder: "open_clip_xlm_roberta_large_vit_h14_f16.ckpt", hiresFixScale: 12,
      builtinLora: true,
      teaCacheCoefficients: [
        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 I2V 14B 480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 832×480. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.1 I2V 14B 720p", file: "wan_v2.1_14b_i2v_720p_q8p.ckpt", prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting,
      clipEncoder: "open_clip_xlm_roberta_large_vit_h14_f16.ckpt", hiresFixScale: 16,
      teaCacheCoefficients: [
        8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 I2V 14B 720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "Wan 2.1 I2V 14B 720p (6-bit, SVDQuant)", file: "wan_v2.1_14b_i2v_720p_q6p_svd.ckpt",
      prefix: "",
      version: .wan21_14b, defaultScale: 12, textEncoder: "umt5_xxl_encoder_q8p.ckpt",
      autoencoder: "wan_v2.1_video_vae_f16.ckpt", modifier: .inpainting,
      clipEncoder: "open_clip_xlm_roberta_large_vit_h14_f16.ckpt", hiresFixScale: 16,
      builtinLora: true,
      teaCacheCoefficients: [
        8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02,
      ], framesPerSecond: 16,
      note:
        "[Wan2.1 I2V 14B 720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) is a state-of-the-art image-to-video model developed by Alibaba. It can generate video clips of up to 4 seconds in length from a given start frame. The recommended resolutions are 1280×720. The model supports up to 81 frames, with a recommended shift value of 5.0."
    ),
    Specification(
      name: "FLUX.1 [schnell]", file: "flux_1_schnell_q8p.ckpt", prefix: "",
      version: .flux1, defaultScale: 16, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt",
      highPrecisionAutoencoder: true, isConsistencyModel: true, objective: .u(conditionScale: 1000),
      paddedTextEncodingLength: 256, hiresFixScale: 24),
    Specification(
      name: "FLUX.1 [schnell] (5-bit)", file: "flux_1_schnell_q5p.ckpt", prefix: "",
      version: .flux1, defaultScale: 16, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "flux_1_vae_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt",
      highPrecisionAutoencoder: true, isConsistencyModel: true, objective: .u(conditionScale: 1000),
      paddedTextEncodingLength: 256, hiresFixScale: 24),
    Specification(
      name: "PixArt Sigma XL 1K", file: "pixart_sigma_xl_2_1024_ms_f16.ckpt", prefix: "",
      version: .pixart, defaultScale: 16, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt"),
    Specification(
      name: "PixArt Sigma XL 1K (8-bit)", file: "pixart_sigma_xl_2_1024_ms_q8p.ckpt", prefix: "",
      version: .pixart, defaultScale: 16, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt"),
    Specification(
      name: "PixArt Sigma XL 512", file: "pixart_sigma_xl_2_512_ms_f16.ckpt", prefix: "",
      version: .pixart, defaultScale: 8, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt"),
    Specification(
      name: "PixArt Sigma XL 512 (8-bit)", file: "pixart_sigma_xl_2_512_ms_q8p.ckpt", prefix: "",
      version: .pixart, defaultScale: 8, textEncoder: "t5_xxl_encoder_q6p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt"),
    Specification(
      name: "AuraFlow v0.1", file: "auraflow_v0.1_q8p.ckpt", prefix: "",
      version: .auraflow, defaultScale: 16, textEncoder: "pile_t5_xl_encoder_q8p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", deprecated: true, objective: .u(conditionScale: 1000)),
    Specification(
      name: "AuraFlow v0.1 (8-bit)", file: "auraflow_v0.1_q5p.ckpt", prefix: "",
      version: .auraflow, defaultScale: 16, textEncoder: "pile_t5_xl_encoder_q8p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", deprecated: true, objective: .u(conditionScale: 1000)),
    Specification(
      name: "AuraFlow v0.2", file: "auraflow_v0.2_q8p.ckpt", prefix: "",
      version: .auraflow, defaultScale: 16, textEncoder: "pile_t5_xl_encoder_q8p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", objective: .u(conditionScale: 1000)),
    Specification(
      name: "AuraFlow v0.2 (8-bit)", file: "auraflow_v0.2_q5p.ckpt", prefix: "",
      version: .auraflow, defaultScale: 16, textEncoder: "pile_t5_xl_encoder_q8p.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", objective: .u(conditionScale: 1000)),
    Specification(
      name: "SDXL Base (v1.0)", file: "sd_xl_base_1.0_f16.ckpt", prefix: "", version: .sdxlBase,
      defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "SDXL Base v1.0 (8-bit)", file: "sd_xl_base_1.0_q6p_q8p.ckpt", prefix: "",
      version: .sdxlBase,
      defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "SDXL Refiner (v1.0)", file: "sd_xl_refiner_1.0_f16.ckpt", prefix: "",
      version: .sdxlRefiner, defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "SDXL Refiner v1.0 (8-bit)", file: "sd_xl_refiner_1.0_q6p_q8p.ckpt", prefix: "",
      version: .sdxlRefiner, defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "SDXL Base (v0.9)", file: "sd_xl_base_0.9_f16.ckpt", prefix: "", version: .sdxlBase,
      defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_f16.ckpt", deprecated: true, clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "SDXL Refiner (v0.9)", file: "sd_xl_refiner_0.9_f16.ckpt", prefix: "",
      version: .sdxlRefiner, defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_f16.ckpt", deprecated: true, clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "Fooocus Inpaint SDXL v2.6", file: "fooocus_inpaint_sd_xl_v2.6_f16.ckpt", prefix: "",
      version: .sdxlBase, defaultScale: 16, textEncoder: "open_clip_vit_bigg14_f16.ckpt",
      autoencoder: "sdxl_vae_v1.0_f16.ckpt", modifier: .inpainting,
      clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "Fooocus Inpaint SDXL v2.6 (8-bit)", file: "fooocus_inpaint_sd_xl_v2.6_q6p_q8p.ckpt",
      prefix: "", version: .sdxlBase, defaultScale: 16,
      textEncoder: "open_clip_vit_bigg14_f16.ckpt", autoencoder: "sdxl_vae_v1.0_f16.ckpt",
      modifier: .inpainting, clipEncoder: "clip_vit_l14_f16.ckpt"),
    Specification(
      name: "Stable Diffusion v1.4", file: "sd_v1.4_f16.ckpt", prefix: "", version: .v1,
      deprecated: true),
    Specification(
      name: "Stable Diffusion v1.5", file: "sd_v1.5_f16.ckpt", prefix: "", version: .v1),
    Specification(
      name: "Stable Diffusion v1.5 Inpainting", file: "sd_v1.5_inpainting_f16.ckpt",
      prefix: "", version: .v1, modifier: .inpainting),
    Specification(
      name: "Stable Diffusion v2.0", file: "sd_v2.0_f16.ckpt", prefix: "", version: .v2,
      textEncoder: "open_clip_vit_h14_f16.ckpt", deprecated: true),
    Specification(
      name: "Stable Diffusion v2.0 768-v", file: "sd_v2.0_768_v_f16.ckpt", prefix: "",
      version: .v2, defaultScale: 12, textEncoder: "open_clip_vit_h14_f16.ckpt", deprecated: true,
      objective: .v),
    Specification(
      name: "Stable Diffusion v2.0 Inpainting", file: "sd_v2.0_inpainting_f16.ckpt",
      prefix: "", version: .v2, textEncoder: "open_clip_vit_h14_f16.ckpt", modifier: .inpainting),
    Specification(
      name: "Stable Diffusion v2.0 Depth", file: "sd_v2.0_depth_f16.ckpt",
      prefix: "", version: .v2, textEncoder: "open_clip_vit_h14_f16.ckpt", modifier: .depth),
    Specification(
      name: "Stable Diffusion v2.1", file: "sd_v2.1_f16.ckpt", prefix: "", version: .v2,
      textEncoder: "open_clip_vit_h14_f16.ckpt"),
    Specification(
      name: "Stable Diffusion v2.1 768-v", file: "sd_v2.1_768_v_f16.ckpt", prefix: "",
      version: .v2, upcastAttention: true, defaultScale: 12,
      textEncoder: "open_clip_vit_h14_f16.ckpt", objective: .v),
    Specification(
      name: "Kandinsky v2.1", file: "kandinsky_f16.ckpt", prefix: "",
      version: .kandinsky21, upcastAttention: false, defaultScale: 12,
      textEncoder: "xlm_roberta_f16.ckpt", autoencoder: "kandinsky_movq_f16.ckpt",
      deprecated: true, imageEncoder: "image_vit_l14_f16.ckpt",
      clipEncoder: "clip_vit_l14_f16.ckpt",
      diffusionMapping: "kandinsky_diffusion_mapping_f16.ckpt"),
    Specification(
      name: "Stable Video Diffusion I2V XT v1.0", file: "svd_i2v_xt_1.0_f16.ckpt", prefix: "",
      version: .svdI2v,
      defaultScale: 8, textEncoder: "open_clip_vit_h14_vision_model_f16.ckpt", deprecated: true,
      clipEncoder: "svd_i2v_xt_1.0_f16.ckpt", conditioning: .noise, objective: .v,
      noiseDiscretization: .edm(.init(sigmaMax: 700.0))),
    Specification(
      name: "Stable Video Diffusion I2V XT 1.0 (8-bit)", file: "svd_i2v_xt_1.0_q6p_q8p.ckpt",
      prefix: "",
      version: .svdI2v,
      defaultScale: 8, textEncoder: "open_clip_vit_h14_vision_model_f16.ckpt", deprecated: true,
      clipEncoder: "svd_i2v_xt_1.0_q6p_q8p.ckpt", conditioning: .noise, objective: .v,
      noiseDiscretization: .edm(.init(sigmaMax: 700.0))),
    Specification(
      name: "MiniSD v1.4", file: "minisd_v1.4_f16.ckpt", prefix: "", version: .v1,
      defaultScale: 4, deprecated: true),
    Specification(
      name: "Instruct Pix2Pix", file: "instruct_pix2pix_22000_f16.ckpt", prefix: "",
      version: .v1, defaultScale: 8, modifier: .editing, deprecated: true),
  ]

  private static let builtinModelsAndAvailableSpecifications: (Set<String>, [Specification]) = {
    let jsonFile = filePathForModelDownloaded("custom.json")
    guard let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) else {
      return (Set(builtinSpecifications.map { $0.file }), builtinSpecifications)
    }
    let jsonDecoder = JSONDecoder()
    jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
    guard
      let jsonSpecifications = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value })
    else {
      return (Set(builtinSpecifications.map { $0.file }), builtinSpecifications)
    }

    var availableSpecifications = builtinSpecifications
    var builtinModels = Set(builtinSpecifications.map { $0.file })
    for specification in jsonSpecifications {
      if builtinModels.contains(specification.file) {
        availableSpecifications = availableSpecifications.filter { $0.file != specification.file }
      }
      availableSpecifications.append(specification)
    }
    return (builtinModels, availableSpecifications)
  }()

  private static let builtinModels: Set<String> = builtinModelsAndAvailableSpecifications.0
  public static var availableSpecifications: [Specification] =
    builtinModelsAndAvailableSpecifications.1

  public static func availableSpecificationForTriggerWord(_ triggerWord: String) -> Specification? {
    let cleanupTriggerWord = String(triggerWord.lowercased().filter { $0.isLetter || $0.isNumber })
    for specification in availableSpecifications {
      if String(specification.name.lowercased().filter { $0.isLetter || $0.isNumber }).contains(
        cleanupTriggerWord)
      {
        return specification
      }
    }
    return nil
  }

  public static func sortCustomSpecifications() {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [Specification]()
    let jsonFile = ModelZoo.filePathForOtherModelDownloaded("custom.json")
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      if let jsonSpecification = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value }) {
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

    // Because this only does sorting, it won't impact the builtinModels set.
    var availableSpecifications = builtinSpecifications
    let builtinModels = Set(builtinSpecifications.map { $0.file })
    for specification in customSpecifications {
      if builtinModels.contains(specification.file) {
        availableSpecifications = availableSpecifications.filter { $0.file != specification.file }
      }
      availableSpecifications.append(specification)
    }
    self.availableSpecifications = availableSpecifications
  }

  public static func appendCustomSpecification(_ specification: Specification) {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [Specification]()
    let jsonFile = ModelZoo.filePathForOtherModelDownloaded("custom.json")
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      if let jsonSpecification = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value }) {
        customSpecifications.append(contentsOf: jsonSpecification)
      }
    }
    customSpecifications = customSpecifications.filter { $0.file != specification.file }
    customSpecifications.append(specification)
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    guard let jsonData = try? jsonEncoder.encode(customSpecifications) else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)

    // Modify these two are not thread safe. availableSpecifications are OK. specificationMapping is particularly problematic (as it is access on both main thread and a background thread).
    var availableSpecifications = availableSpecifications
    availableSpecifications = availableSpecifications.filter { $0.file != specification.file }
    // Still respect the order.
    availableSpecifications.append(specification)
    self.availableSpecifications = availableSpecifications
    specificationMapping[specification.file] = specification
  }

  public static func updateCustomSpecification(_ specification: Specification) {
    dispatchPrecondition(condition: .onQueue(.main))
    var customSpecifications = [Specification]()
    let jsonFile = ModelZoo.filePathForOtherModelDownloaded("custom.json")

    // Load existing specifications from custom.json
    if let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonFile)) {
      let jsonDecoder = JSONDecoder()
      jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
      if let jsonSpecification = try? jsonDecoder.decode(
        [FailableDecodable<Specification>].self, from: jsonData
      ).compactMap({ $0.value }) {
        customSpecifications.append(contentsOf: jsonSpecification)
      }
    }

    // Find the index of the specification with matching file name
    if let index = customSpecifications.firstIndex(where: { $0.name == specification.name }) {
      // Replace the existing specification with the updated one
      customSpecifications[index] = specification
    } else {
      // If no matching specification was found, just append it
      customSpecifications.append(specification)
    }

    // Write updated specifications back to custom.json
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    guard let jsonData = try? jsonEncoder.encode(customSpecifications) else { return }
    try? jsonData.write(to: URL(fileURLWithPath: jsonFile), options: .atomic)

    // Update the in-memory cache
    var availableSpecifications = self.availableSpecifications
    // Replace or add the specification in the array
    if let index = availableSpecifications.firstIndex(where: { $0.file == specification.file }) {
      availableSpecifications[index] = specification
    } else {
      availableSpecifications.append(specification)
    }
    self.availableSpecifications = availableSpecifications

    // Update the mapping dictionary
    overrideMapping[specification.file] = specification
    specificationMapping[specification.file] = specification
  }

  private static var specificationMapping: [String: Specification] = {
    var mapping = [String: Specification]()
    for specification in availableSpecifications {
      mapping[specification.file] = specification
    }
    return mapping
  }()

  public static var anyModelDownloaded: String? {
    let availableSpecifications = availableSpecifications
    for specification in availableSpecifications {
      if isModelDownloaded(specification) {
        return specification.file
      }
    }
    return nil
  }

  private static func filePathForOtherModelDownloaded(_ name: String) -> String {
    let urls = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    let modelZooUrl = urls.first!.appendingPathComponent("Models")
    try? FileManager.default.createDirectory(at: modelZooUrl, withIntermediateDirectories: true)
    return modelZooUrl.appendingPathComponent(name).path
  }

  public static var externalUrls: [URL] = [URL]() {
    didSet {
      #if (os(macOS) || (os(iOS) && targetEnvironment(macCatalyst)))
        guard oldValue != externalUrls else { return }
        for url in oldValue {
          url.stopAccessingSecurityScopedResource()
        }
        for url in externalUrls {
          let _ = url.startAccessingSecurityScopedResource()
        }
      #endif
    }
  }

  public static func specificationForHumanReadableModel(_ name: String) -> Specification? {
    return availableSpecifications.first { $0.name == name }
  }

  // We prefer these if it is a hit.
  public static var overrideMapping: [String: Specification] = [:]

  // These are only the hit if everything else fails.
  public static var fallbackMapping: [String: Specification] = [:]

  public static func specificationForModel(_ name: String) -> Specification? {
    if let override = overrideMapping[name] {
      return override
    }
    return specificationMapping[name] ?? fallbackMapping[name]
  }

  public static func filesToDownload(_ specification: Specification)
    -> [(name: String, subtitle: String, file: String, sha256: String?)]
  {
    guard specification.remoteApiModelConfig == nil else {
      return []
    }
    let model = specification.file
    let name = specification.name
    let version = ModelZoo.humanReadableNameForVersion(specification.version)
    var models: [(name: String, subtitle: String, file: String, sha256: String?)] = [
      (name: name, subtitle: version, file: model, sha256: nil)
    ]
    let textEncoder =
      specification.textEncoder
      ?? (specification.version == .v1 ? "clip_vit_l14_f16.ckpt" : "open_clip_vit_h14_f16.ckpt")
    let autoencoder = specification.autoencoder ?? "vae_ft_mse_840000_f16.ckpt"
    models.append((name: name, subtitle: version, file: textEncoder, sha256: nil))
    models.append((name: name, subtitle: version, file: autoencoder, sha256: nil))
    if let imageEncoder = specification.imageEncoder {
      models.append((name: name, subtitle: version, file: imageEncoder, sha256: nil))
    }
    if let CLIPEncoder = specification.clipEncoder {
      models.append((name: name, subtitle: version, file: CLIPEncoder, sha256: nil))
    }
    specification.additionalClipEncoders?.forEach {
      models.append((name: name, subtitle: version, file: $0, sha256: nil))
    }
    if let t5Encoder = specification.t5Encoder {
      models.append((name: name, subtitle: version, file: t5Encoder, sha256: nil))
    }
    if let diffusionMapping = specification.diffusionMapping {
      models.append((name: name, subtitle: version, file: diffusionMapping, sha256: nil))
    }
    for stageModel in (specification.stageModels ?? []) {
      models.append((name: name, subtitle: version, file: stageModel, sha256: nil))
    }
    if let defaultRefiner = specification.defaultRefiner {
      let name = ModelZoo.humanReadableNameForModel(defaultRefiner)
      let version = ModelZoo.humanReadableNameForVersion(ModelZoo.versionForModel(defaultRefiner))
      models.append((name: name, subtitle: version, file: defaultRefiner, sha256: nil))
      if let textEncoder = ModelZoo.textEncoderForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: textEncoder, sha256: nil))
      }
      if let autoencoder = ModelZoo.autoencoderForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: autoencoder, sha256: nil))
      }
      if let imageEncoder = ModelZoo.imageEncoderForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: imageEncoder, sha256: nil))
      }
      ModelZoo.CLIPEncodersForModel(defaultRefiner).forEach { CLIPEncoder in
        models.append((name: name, subtitle: version, file: CLIPEncoder, sha256: nil))
      }
      if let T5Encoder = ModelZoo.T5EncoderForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: T5Encoder, sha256: nil))
      }
      if let diffusionMapping = ModelZoo.diffusionMappingForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: diffusionMapping, sha256: nil))
      }
      for stageModel in ModelZoo.stageModelsForModel(defaultRefiner) {
        models.append((name: name, subtitle: version, file: stageModel, sha256: nil))
      }
    }
    return models
  }

  public static func internalFilePathForModelDownloaded(_ name: String) -> String {
    return filePathForOtherModelDownloaded(name)
  }

  public static func filePathForModelDownloaded(_ name: String) -> String {
    guard let externalUrl = externalUrls.first else {
      return filePathForOtherModelDownloaded(name)
    }
    // If it exists at internal storage, prefer that.
    let otherFilePath = filePathForOtherModelDownloaded(name)
    let fileManager = FileManager.default
    guard !fileManager.fileExists(atPath: otherFilePath) else {
      return otherFilePath
    }
    for externalUrl in externalUrls {
      if FileManager.default.fileExists(atPath: externalUrl.appendingPathComponent(name).path) {
        return externalUrl.appendingPathComponent(name).path
      }
    }
    // Check external storage, return path at external storage regardless.
    let filePath = externalUrl.appendingPathComponent(name).path
    return filePath
  }

  public static func isModelDownloaded(
    _ specification: Specification, memorizedBy: Set<String> = []
  ) -> Bool {
    if let _ = specification.remoteApiModelConfig {
      return true
    }
    var result =
      isModelDownloaded(specification.file, memorizedBy: memorizedBy)
      && isModelDownloaded(
        specification.autoencoder ?? "vae_ft_mse_840000_f16.ckpt", memorizedBy: memorizedBy)
      && isModelDownloaded(
        specification.textEncoder
          ?? (specification.version == .v1
            ? "clip_vit_l14_f16.ckpt" : "open_clip_vit_h14_f16.ckpt"),
        memorizedBy: memorizedBy
      )
    if let imageEncoder = specification.imageEncoder {
      result = result && isModelDownloaded(imageEncoder, memorizedBy: memorizedBy)
    }
    if let clipEncoder = specification.clipEncoder {
      result = result && isModelDownloaded(clipEncoder, memorizedBy: memorizedBy)
    }
    result =
      result
      && (specification.additionalClipEncoders ?? []).allSatisfy {
        isModelDownloaded($0, memorizedBy: memorizedBy)
      }
    if let t5Encoder = specification.t5Encoder {
      result = result && isModelDownloaded(t5Encoder, memorizedBy: memorizedBy)
    }
    if let diffusionMapping = specification.diffusionMapping {
      result = result && isModelDownloaded(diffusionMapping, memorizedBy: memorizedBy)
    }
    return result
  }

  public static func isModelInExternalUrl(_ name: String) -> Bool {
    guard !externalUrls.isEmpty else {
      return false
    }
    for externalUrl in externalUrls {
      if FileManager.default.fileExists(atPath: externalUrl.appendingPathComponent(name).path) {
        return true
      }
    }
    return false
  }

  public static func isRemoteApiModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    guard let _ = specification.remoteApiModelConfig else {
      return false
    }
    return true
  }

  public static func isModelDownloaded(_ name: String, memorizedBy: Set<String>) -> Bool {
    guard !memorizedBy.contains(name) else {
      return true
    }
    let fileManager = FileManager.default
    for externalUrl in externalUrls {
      let filePath = externalUrl.appendingPathComponent(name).path
      if fileManager.fileExists(atPath: filePath) {
        return true
      }
    }
    let otherModelPath = filePathForOtherModelDownloaded(name)
    if fileManager.fileExists(atPath: otherModelPath) {
      return true
    }
    return false
  }

  public static func isBuiltinModel(_ name: String) -> Bool {
    return builtinModels.contains(name)
  }

  public static func humanReadableNameForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return name }
    return specification.name
  }

  public static func textPrefixForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return "" }
    return specification.prefix
  }

  public static func textEncoderForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.textEncoder
  }

  public static func textEncoderVersionForModel(_ name: String) -> TextEncoderVersion? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.textEncoderVersion
  }

  public static func imageEncoderForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.imageEncoder
  }

  public static func CLIPEncodersForModel(_ name: String) -> [String] {
    guard let specification = specificationForModel(name) else { return [] }
    return (specification.clipEncoder.map { [$0] } ?? [])
      + (specification.additionalClipEncoders ?? [])
  }

  public static func T5EncoderForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.t5Encoder
  }

  public static func diffusionMappingForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.diffusionMapping
  }

  public static func versionForModel(_ name: String) -> ModelVersion {
    guard let specification = specificationForModel(name) else { return .v1 }
    return specification.version
  }

  public static func noteForModel(_ name: String) -> String {
    guard let specification = specificationForModel(name) else { return "" }
    return specification.note ?? ""
  }

  public static func autoencoderForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.autoencoder
  }

  public static func builtinLoRAForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.builtinLora ?? false
  }

  public static func isModelDeprecated(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.deprecated ?? false
  }

  public static func objectiveForModel(_ name: String) -> Denoiser.Objective {
    guard let specification = specificationForModel(name) else { return .epsilon }
    if let objective = specification.objective {
      return objective
    }
    if specification.predictV == true {
      return .v
    }
    switch specification.version {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .pixart:
      return .epsilon
    case .sd3, .sd3Large, .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1,
      .qwenImage:
      return .u(conditionScale: 1000)
    }
  }

  public static func conditioningForModel(_ name: String) -> Denoiser.Conditioning {
    guard let specification = specificationForModel(name) else { return .timestep }
    if let conditioning = specification.conditioning {
      return conditioning
    }
    switch specification.version {
    case .kandinsky21, .sdxlBase, .sdxlRefiner, .v1, .v2, .ssd1b, .wurstchenStageC,
      .wurstchenStageB, .sd3, .sd3Large, .pixart, .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b,
      .wan21_14b, .hiDreamI1, .qwenImage:
      return .timestep
    case .svdI2v:
      return .noise
    }
  }

  public static func noiseDiscretizationForModel(_ name: String) -> NoiseDiscretization {
    guard let specification = specificationForModel(name) else {
      return .ddpm(
        .init(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, linspace: .linearWrtSigma))
    }
    if let noiseDiscretization = specification.noiseDiscretization {
      return noiseDiscretization
    }
    switch specification.version {
    case .kandinsky21:
      return .ddpm(
        .init(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, linspace: .linearWrtBeta))
    case .pixart:
      return .ddpm(
        .init(linearStart: 0.0001, linearEnd: 0.02, timesteps: 1_000, linspace: .linearWrtBeta))
    case .sdxlBase, .sdxlRefiner, .ssd1b, .v1, .v2:
      return .ddpm(
        .init(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, linspace: .linearWrtSigma))
    case .svdI2v:
      return .edm(.init(sigmaMax: 700.0))
    case .wurstchenStageC, .wurstchenStageB:
      return .edm(.init(sigmaMin: 0.01, sigmaMax: 99.995))
    case .sd3, .sd3Large, .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1,
      .qwenImage:
      return .rf(.init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
    }
  }

  public static func paddedTextEncodingLengthForModel(_ name: String) -> Int {
    guard let specification = specificationForModel(name) else { return 0 }
    if let paddedTextEncodingLength = specification.paddedTextEncodingLength {
      return paddedTextEncodingLength
    }
    switch specification.version {
    case .v1, .v2, .svdI2v, .ssd1b, .sdxlBase, .sdxlRefiner, .kandinsky21, .wurstchenStageB,
      .wurstchenStageC, .sd3, .sd3Large:
      return 77
    case .pixart:
      return 0
    case .auraflow:
      return 256
    case .flux1:
      return 512
    case .hiDreamI1:
      return 128
    case .hunyuanVideo, .qwenImage:
      return 0
    case .wan21_1_3b, .wan21_14b:
      return 512
    }
  }

  public static func latentsScalingForModel(_ name: String) -> (
    mean: [Float]?, std: [Float]?, scalingFactor: Float, shiftFactor: Float?
  ) {
    guard let specification = specificationForModel(name) else { return (nil, nil, 1, nil) }
    if let mean = specification.latentsMean, let std = specification.latentsStd,
      let scalingFactor = specification.latentsScalingFactor
    {
      return (mean, std, scalingFactor, nil)
    }
    if let scalingFactor = specification.latentsScalingFactor {
      return (nil, nil, scalingFactor, nil)
    }
    switch specification.version {
    case .v1, .v2, .svdI2v:
      return (nil, nil, 0.18215, nil)
    case .ssd1b, .sdxlBase, .sdxlRefiner, .pixart, .auraflow:
      return (nil, nil, 0.13025, nil)
    case .kandinsky21:
      return (nil, nil, 1, nil)
    case .wurstchenStageC, .wurstchenStageB:
      return (nil, nil, 2.32558139535, nil)
    case .sd3, .sd3Large:
      return (nil, nil, 1.5305, 0.0609)
    case .flux1, .hiDreamI1:
      return (nil, nil, 0.3611, 0.11590)
    case .hunyuanVideo:
      return (nil, nil, 0.476986, nil)
    case .wan21_1_3b, .wan21_14b, .qwenImage:
      return (
        [
          -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
          0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ],
        [
          2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
          3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ], 1, nil
      )
    }
  }

  public static func stageModelsForModel(_ name: String) -> [String] {
    guard let specification = specificationForModel(name) else { return [] }
    return specification.stageModels ?? []
  }

  public static func MMDiTForModel(_ name: String) -> Specification.MMDiT? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.mmdit
  }

  public static func isUpcastAttentionForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.upcastAttention
  }

  public static func isHighPrecisionAutoencoderForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.highPrecisionAutoencoder ?? false
  }

  public static func modifierForModel(_ name: String) -> SamplerModifier {
    guard let specification = specificationForModel(name) else { return .none }
    return specification.modifier ?? .none
  }

  public static func isConsistencyModelForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.isConsistencyModel ?? false
  }

  public static func guidanceEmbedForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.guidanceEmbed ?? false
  }

  public static func framesPerSecondForModel(_ name: String) -> Double {
    guard let specification = specificationForModel(name) else { return 30 }
    if let framesPerSecond = specification.framesPerSecond {
      return framesPerSecond
    }
    switch specification.version {
    case .hunyuanVideo:
      return 30
    case .wan21_1_3b, .wan21_14b, .qwenImage:
      return 16
    case .v1, .v2, .auraflow, .flux1, .hiDreamI1, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase,
      .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageB, .wurstchenStageC:
      return 30
    }
  }

  public static func defaultScaleForModel(_ name: String?) -> UInt16 {
    guard let name = name, let specification = specificationForModel(name) else { return 8 }
    return specification.defaultScale
  }

  public static func hiresFixScaleForModel(_ name: String?) -> UInt16 {
    guard let name = name, let specification = specificationForModel(name) else { return 8 }
    return specification.hiresFixScale ?? specification.defaultScale
  }

  public static func teaCacheCoefficientsForModel(_ name: String) -> (
    Float, Float, Float, Float, Float
  )? {
    guard let specification = specificationForModel(name) else { return nil }
    guard let coefficients = specification.teaCacheCoefficients else {
      switch specification.version {
      case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
        .wurstchenStageB, .sd3, .pixart, .auraflow, .sd3Large, .wan21_1_3b, .wan21_14b, .qwenImage:
        return nil
      case .flux1:
        return (4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01)
      case .hunyuanVideo:
        return (7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02)
      case .hiDreamI1:
        return (-3.13605009e+04, -7.12425503e+02, 4.91363285e+01, 8.26515490e+00, 1.08053901e-01)
      }
    }
    return (coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4])
  }

  public static func isBF16ForModel(_ name: String) -> Bool {
    guard let specification = specificationForModel(name) else { return false }
    return specification.isBf16 ?? false
  }

  public static func defaultRefinerForModel(_ name: String) -> String? {
    guard let specification = specificationForModel(name) else { return nil }
    return specification.defaultRefiner
  }

  public static func mergeFileSHA256(_ sha256: [String: String]) {
    var fileSHA256 = fileSHA256
    for (key, value) in sha256 {
      fileSHA256[key] = value
    }
    self.fileSHA256 = fileSHA256
  }

  public static func fileSHA256ForModelDownloaded(_ name: String) -> String? {
    return fileSHA256[name]
  }

  public static func is8BitModel(_ name: String) -> Bool {
    let filePath = Self.filePathForModelDownloaded(name)
    let fileSize = (try? URL(fileURLWithPath: filePath).resourceValues(forKeys: [.fileSizeKey]))?
      .fileSize
    let externalFileSize =
      (try? URL(fileURLWithPath: TensorData.externalStore(filePath: filePath)).resourceValues(
        forKeys: [.fileSizeKey]))?.fileSize
    if var fileSize = fileSize {
      fileSize += externalFileSize ?? 0
      let version = versionForModel(name)
      switch version {
      case .sdxlBase, .sdxlRefiner:
        return fileSize < 3 * 1_024 * 1_024 * 1_024
      case .ssd1b:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .v1, .v2, .pixart:
        return fileSize < 1_024 * 1_024 * 1_024
      case .kandinsky21:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .svdI2v:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .wurstchenStageC:
        return fileSize < 4 * 1_024 * 1_024 * 1_024
      case .wurstchenStageB:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .sd3:
        return fileSize < 3 * 1_024 * 1_024 * 1_024
      case .sd3Large:
        return fileSize < 7 * 1_024 * 1_024 * 1_024
      case .auraflow:
        return fileSize < 6 * 1_024 * 1_024 * 1_024
      case .flux1:
        return fileSize < 10 * 1_024 * 1_024 * 1_024
      case .hunyuanVideo:
        return fileSize < 11 * 1_024 * 1_024 * 1_024
      case .wan21_1_3b:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .wan21_14b:
        return fileSize < 13 * 1_024 * 1_024 * 1_024 + 512 * 1_024 * 1_024
      case .qwenImage:
        return fileSize < 17 * 1_024 * 1_024 * 1_024 + 512 * 1_024 * 1_024
      case .hiDreamI1:
        return fileSize < 13 * 1_024 * 1_024 * 1_024 + 512 * 1_024 * 1_024
      }
    }
    return false
  }

  public static func isQuantizedModel(_ name: String) -> Bool {
    let filePath = Self.filePathForModelDownloaded(name)
    let fileSize = (try? URL(fileURLWithPath: filePath).resourceValues(forKeys: [.fileSizeKey]))?
      .fileSize
    let externalFileSize =
      (try? URL(fileURLWithPath: TensorData.externalStore(filePath: filePath)).resourceValues(
        forKeys: [.fileSizeKey]))?.fileSize
    if var fileSize = fileSize {
      fileSize += externalFileSize ?? 0
      let version = versionForModel(name)
      switch version {
      case .sdxlBase, .sdxlRefiner:
        return fileSize < 3 * 1_024 * 1_024 * 1_024
      case .ssd1b:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .v1, .v2, .pixart:
        return fileSize < 1_024 * 1_024 * 1_024
      case .kandinsky21:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .svdI2v:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .wurstchenStageC:
        return fileSize < 4 * 1_024 * 1_024 * 1_024
      case .wurstchenStageB:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .sd3:
        return fileSize < 3 * 1_024 * 1_024 * 1_024
      case .sd3Large:
        return fileSize < 10 * 1_024 * 1_024 * 1_024
      case .auraflow:
        return fileSize < 10 * 1_024 * 1_024 * 1_024
      case .flux1:
        return fileSize < 16 * 1_024 * 1_024 * 1_024
      case .hunyuanVideo:
        return fileSize < 20 * 1_024 * 1_024 * 1_024
      case .wan21_1_3b:
        return fileSize < 2 * 1_024 * 1_024 * 1_024
      case .wan21_14b:
        return fileSize < 15 * 1_024 * 1_024 * 1_024
      case .qwenImage:
        return fileSize < 20 * 1_024 * 1_024 * 1_024
      case .hiDreamI1:
        return fileSize < 17 * 1_024 * 1_024 * 1_024
      }
    }
    return false
  }
}

extension ModelZoo {
  public static func ensureDownloadsDirectoryExists() {
    let urls = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    let downloadsUrl = urls.first!.appendingPathComponent("Downloads")
    try? FileManager.default.createDirectory(at: downloadsUrl, withIntermediateDirectories: true)
  }

  public static func allDownloadedFiles(
    _ includesSystemDownloadUrl: Bool = true,
    matchingSuffixes: [String] = [
      ".safetensors", ".ckpt", ".ckpt.zip", ".pt", ".pt.zip", ".pth", ".pth.zip", ".bin", ".patch",
    ]
  ) -> [String] {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .documentDirectory, in: .userDomainMask)
    let downloadsUrl = urls.first!.appendingPathComponent("Downloads")
    let systemDownloadUrl = fileManager.urls(for: .downloadsDirectory, in: .userDomainMask).first
    var fileUrls = [URL]()
    if let urls = try? fileManager.contentsOfDirectory(
      at: downloadsUrl, includingPropertiesForKeys: [.fileSizeKey],
      options: [.skipsHiddenFiles, .skipsPackageDescendants, .skipsSubdirectoryDescendants])
    {
      fileUrls.append(contentsOf: urls)
    }
    if includesSystemDownloadUrl,
      let systemDownloadUrl = systemDownloadUrl?.resolvingSymlinksInPath(),
      let urls = try? fileManager.contentsOfDirectory(
        at: systemDownloadUrl, includingPropertiesForKeys: [.fileSizeKey],
        options: [.skipsHiddenFiles, .skipsPackageDescendants, .skipsSubdirectoryDescendants])
    {
      // From the system download directory, we only include file with ckpt, pt or safetensors suffix.
      fileUrls.append(
        contentsOf: urls.filter {
          let path = $0.path.lowercased()
          return matchingSuffixes.contains { path.hasSuffix($0) }
        })
    }
    return fileUrls.compactMap {
      guard let values = try? $0.resourceValues(forKeys: [.fileSizeKey]) else { return nil }
      guard let fileSize = values.fileSize, fileSize > 0 else { return nil }
      let file = $0.lastPathComponent
      guard !file.hasSuffix(".part") else { return nil }
      return file
    }
  }

  public static func filePathForDownloadedFile(_ file: String) -> String {
    let fileManager = FileManager.default
    let urls = fileManager.urls(for: .documentDirectory, in: .userDomainMask)
    let downloadsUrl = urls.first!.appendingPathComponent("Downloads")
    let filePath = downloadsUrl.appendingPathComponent(file).path
    if !fileManager.fileExists(atPath: filePath),
      // Check if the file exists in system download directory.
      let systemDownloadUrl = fileManager.urls(for: .downloadsDirectory, in: .userDomainMask).first
    {
      let systemFilePath = systemDownloadUrl.appendingPathComponent(file).path
      return fileManager.fileExists(atPath: systemFilePath) ? systemFilePath : filePath
    }
    return filePath
  }

  public static func fileBytesForDownloadedFile(_ file: String) -> Int64 {
    let filePath = Self.filePathForDownloadedFile(file)
    let fileSize = (try? URL(fileURLWithPath: filePath).resourceValues(forKeys: [.fileSizeKey]))?
      .fileSize
    return Int64(fileSize ?? 0)
  }

  public static func availableFiles(excluding file: String?) -> Set<String> {
    var files = Set<String>()
    for specification in availableSpecifications {
      guard specification.file != file, ModelZoo.isModelDownloaded(specification.file) else {
        continue
      }
      files.insert(specification.file)
      let textEncoder = specification.textEncoder ?? "clip_vit_l14_f16.ckpt"
      files.insert(textEncoder)
      let autoencoder = specification.autoencoder ?? "vae_ft_mse_840000_f16.ckpt"
      files.insert(autoencoder)
      if let imageEncoder = specification.imageEncoder {
        files.insert(imageEncoder)
      }
      if let clipEncoder = specification.clipEncoder {
        files.insert(clipEncoder)
      }
      specification.additionalClipEncoders?.forEach {
        files.insert($0)
      }
      if let t5Encoder = specification.t5Encoder {
        files.insert(t5Encoder)
      }
      if let diffusionMapping = specification.diffusionMapping {
        files.insert(diffusionMapping)
      }
      if let stageModels = specification.stageModels {
        files.formUnion(stageModels)
      }
    }
    return files
  }

  public static func isResolutionDependentShiftAvailable(
    _ version: ModelVersion, isConsistencyModel: Bool
  ) -> Bool {
    guard
      version == .flux1 || version == .sd3 || version == .sd3Large || version == .hiDreamI1
        || version == .qwenImage
    else { return false }
    if isConsistencyModel {
      return false
    }
    return true
  }

  public static func shiftFor(_ resolution: (width: UInt16, height: UInt16)) -> Double {
    return exp(
      ((Double(resolution.height) * Double(resolution.width)) * 16 - 256)
        * (1.15 - 0.5) / (4096 - 256) + 0.5)
  }

  public static func isCLIPSkipAvailable(_ version: ModelVersion) -> Bool {
    switch version {
    case .pixart, .auraflow, .wan21_14b, .wan21_1_3b, .qwenImage, .svdI2v:
      return false
    case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .v1, .v2, .flux1, .hunyuanVideo, .hiDreamI1,
      .ssd1b, .kandinsky21, .wurstchenStageB, .wurstchenStageC:
      return true
    }
  }

  public static func isCompatibleSampler(_ model: String, sampler: SamplerType) -> Bool {
    let objective = objectiveForModel(model)
    switch objective {
    case .edm(_), .v, .epsilon:
      return true
    case .u(_):
      switch sampler {
      case .DPMPP2MAYS, .DPMPPSDEAYS, .dDIMTrailing, .dPMPP2MTrailing, .dPMPPSDETrailing,
        .eulerAAYS, .eulerATrailing, .uniPCAYS, .uniPCTrailing:
        return true
      case .DDIM, .PLMS, .LCM, .TCD, .dPMPP2MKarras, .dPMPPSDEKarras, .dPMPPSDESubstep, .eulerA,
        .eulerASubstep, .uniPC:
        return false
      }
    }
  }

  public static func isCompatibleRefiner(_ version: ModelVersion, refinerVersion: ModelVersion)
    -> Bool
  {
    if version == refinerVersion {
      return true
    }
    if [.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
      && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(refinerVersion)
    {
      return true
    }
    // All uses SD3 VAE.
    if [.sd3, .sd3Large].contains(version) && [.sd3, .sd3Large].contains(refinerVersion) {
      return true
    }
    // All uses Wan VAE.
    if [.wan21_1_3b, .wan21_14b].contains(version)
      && [.wan21_1_3b, .wan21_14b].contains(refinerVersion)
    {
      return true
    }
    /*
    // All uses SDXL VAE.
    if [.sdxlBase, .sdxlRefiner, .ssd1b, .auraflow, .pixart].contains(version)
      && [.sdxlBase, .sdxlRefiner, .ssd1b, .auraflow, .pixart].contains(refinerVersion)
    {
      return true
    }
    // All uses FLUX VAE.
    if [.flux1, .hiDreamI1].contains(version) && [.flux1, .hiDreamI1].contains(refinerVersion) {
      return true
    }
    // All uses SD3 VAE.
    if [.sd3, .sd3Large].contains(version) && [.sd3, .sd3Large].contains(refinerVersion) {
      return true
    }
    // All uses Wan VAE.
    if [.wan21_1_3b, .wan21_14b, .qwenImage].contains(version)
      && [.wan21_1_3b, .wan21_14b, .qwenImage].contains(refinerVersion)
    {
      return true
    }
     */
    return false
  }
}
