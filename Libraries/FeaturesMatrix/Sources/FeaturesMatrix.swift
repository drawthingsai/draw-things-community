import Diffusion

public enum FeaturesMatrix {
  public static func supportsLoRATraining(_ version: ModelVersion) -> Bool {
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .flux1:
      return true
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .pixart,
      .auraflow, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
      return false
    }
  }
  public static func supportsExportModel(_ version: ModelVersion) -> Bool {
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b:
      return true
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .pixart,
      .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
      return false
    }
  }
  public static func supportsExportLoRA(_ version: ModelVersion) -> Bool {
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .flux1:
      return true
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .sd3, .sd3Large, .pixart,
      .auraflow, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1:
      return false
    }
  }
}
