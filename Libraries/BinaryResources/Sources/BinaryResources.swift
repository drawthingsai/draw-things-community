import C_Resources
import Foundation

public enum BinaryResources {
  public static let bpe_simple_vocab_16e6_txt = Data(
    bytesNoCopy: C_Resources.bpe_simple_vocab_16e6_txt()!,
    count: C_Resources.bpe_simple_vocab_16e6_txt_size, deallocator: .none)
  public static let merges_txt = Data(
    bytesNoCopy: C_Resources.merges_txt()!, count: C_Resources.merges_txt_size, deallocator: .none)
  public static let vocab_json = Data(
    bytesNoCopy: C_Resources.vocab_json()!, count: C_Resources.vocab_json_size, deallocator: .none)
  public static let vocab_16e6_json = Data(
    bytesNoCopy: C_Resources.vocab_16e6_json()!, count: C_Resources.vocab_16e6_json_size,
    deallocator: .none)
  public static let chatglm3_spiece_model = Data(
    bytesNoCopy: C_Resources.chatglm3_spiece_model()!,
    count: C_Resources.chatglm3_spiece_model_size,
    deallocator: .none)
  public static let pile_t5_spiece_model = Data(
    bytesNoCopy: C_Resources.pile_t5_spiece_model()!, count: C_Resources.pile_t5_spiece_model_size,
    deallocator: .none)
  public static let t5_spiece_model = Data(
    bytesNoCopy: C_Resources.t5_spiece_model()!, count: C_Resources.t5_spiece_model_size,
    deallocator: .none)
  public static let xlmroberta_bpe_model = Data(
    bytesNoCopy: C_Resources.xlmroberta_bpe_model()!, count: C_Resources.xlmroberta_bpe_model_size,
    deallocator: .none)
  public static let server_crt_crt = Data(
    bytesNoCopy: C_Resources.server_crt_crt()!, count: C_Resources.server_crt_crt_size,
    deallocator: .none)
  public static let server_key_key = Data(
    bytesNoCopy: C_Resources.server_key_key()!, count: C_Resources.server_key_key_size,
    deallocator: .none)
}
