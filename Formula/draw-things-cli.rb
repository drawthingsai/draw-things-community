class DrawThingsCli < Formula
  desc "Local inference and LoRA training CLI for Draw Things"
  homepage "https://github.com/drawthingsai/draw-things-community"
  head "https://github.com/drawthingsai/draw-things-community.git", branch: "main"

  depends_on macos: :ventura

  def install
    scratch_path = buildpath/".build"

    ENV["CLANG_MODULE_CACHE_PATH"] = buildpath/"clang-module-cache"
    system(
      "swift", "build", "--disable-sandbox", "--cache-path", buildpath/"swiftpm-cache",
      "--config-path", buildpath/"swiftpm-config", "--security-path",
      buildpath/"swiftpm-security", "--scratch-path", scratch_path, "-c", "release",
      "--product", "draw-things-cli")

    bin.install scratch_path/"release"/"draw-things-cli"

    (bash_completion/"draw-things-cli").write Utils.safe_popen_read(
      bin/"draw-things-cli", "completion", "bash")
    (zsh_completion/"_draw-things-cli").write Utils.safe_popen_read(
      bin/"draw-things-cli", "completion", "zsh")
    (fish_completion/"draw-things-cli.fish").write Utils.safe_popen_read(
      bin/"draw-things-cli", "completion", "fish")
  end

  test do
    assert_match "draw-things-cli", shell_output("#{bin}/--help")
    assert_match "--model", shell_output("#{bin}/generate --help")
    assert_match "models list", shell_output("#{bin}/models --help")
  end
end
