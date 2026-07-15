import XCTest

@testable import PromptJSONExpansion

final class Ideogram4PromptJSONExpansionTests: XCTestCase {
  func testAspectRatioReduction() {
    XCTAssertEqual(Ideogram4PromptJSONExpander.aspectRatio(width: 1_024, height: 1_024), "1:1")
    XCTAssertEqual(Ideogram4PromptJSONExpander.aspectRatio(width: 1_536, height: 1_024), "3:2")
  }

  func testOfficialUserPrompt() {
    XCTAssertEqual(
      Ideogram4PromptJSONExpander.userPrompt(
        prompt: "a café sign", width: 1_024, height: 1_280),
      "TARGET IMAGE ASPECT RATIO: 4:5 (width:height).\n\nUser idea: a café sign")
  }

  func testThinkingIsDisabled() {
    let prompt = Ideogram4PromptJSONExpander.chatPrompt(
      prompt: "a cube", width: 1_024, height: 1_024)
    XCTAssertTrue(prompt.hasPrefix("<|im_start|>system\nYou convert a natural-language user idea"))
    XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
  }

  func testCompleteJSONObject() {
    XCTAssertTrue(
      Ideogram4PromptJSONExpander.isCompleteJSONObject(
        #"{"aspect_ratio":"1:1","high_level_description":"A café sign.","compositional_deconstruction":{"background":"A wall.","elements":[]}}"#
      ))
    XCTAssertFalse(
      Ideogram4PromptJSONExpander.isCompleteJSONObject(
        #"{"aspect_ratio":"1:1","high_level_description":"unfinished""#))
  }
}
