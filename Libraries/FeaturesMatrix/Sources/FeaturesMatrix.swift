import Diffusion

public enum FeaturesMatrix {
  public static func supportsLoRATraining(_ version: ModelVersion) -> Bool {
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .flux1, .zImage, .qwenImage, .ernieImage,
        .flux2_4b, .flux2_9b, .cosmos2_5_2b:
      return true
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .pixart,
      .auraflow, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .wan22_5b,
      .flux2, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b:
      return false
    }
  }
  public static func supportsExportModel(_ version: ModelVersion) -> Bool {
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b:
      return true
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .pixart,
      .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .qwenImage,
      .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2,
      .ltx2_3, .seedvr2_3b, .seedvr2_7b:
      return false
    }
  }
  public static func supportsExportLoRA(_ version: ModelVersion) -> Bool {
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .flux1:
      return true
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .pixart,
      .auraflow, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .qwenImage, .cosmos2_5_2b,
      .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b,
      .seedvr2_7b:
      return false
    }
  }
}
