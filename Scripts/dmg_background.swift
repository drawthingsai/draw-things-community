// Renders the background of Local Code's Developer ID disk image window: a 660x420 pt crop of a
// lunar landscape with a dusk grade below the horizon, a dotted arc from the app to Applications,
// the wordmark and a row of callouts. Writes one TIFF holding the 1x and 2x renderings, which
// Finder picks between.
//
// The landscape is framed for the window: about 660:428, horizon about halfway down, Earth top
// right, open sky top left for the wordmark.
//
//   xcrun swift Scripts/dmg_background.swift <landscape.png> \
//     Apps/LocalCode/Resources/Assets.xcassets/local-code-banner.imageset/local-code.svg \
//     Apps/LocalCode/SupportingFiles/Mac/DiskImageBackground.tiff
//
// Finder draws the icons and their labels over this image at the positions in DiskImage.py, in
// black with Light appearance and white with Dark, so the ground under each label is evened out to
// a middle grey that both read on. Below the labels the grade deepens for the light callout text.

import AppKit
import CoreGraphics
import CoreText
import Foundation
import ImageIO
import UniformTypeIdentifiers

let arguments = CommandLine.arguments
guard arguments.count == 4 else {
  FileHandle.standardError.write(
    "usage: dmg_background.swift <landscape.png> <wordmark.svg> <output.tiff>\n".data(
      using: .utf8)!)
  exit(64)
}

// DiskImage.py opens a 660x452 pt window, 420 pt below macOS 26's 32 pt title bar. The art runs
// 8 pt further so a shorter title bar shows more ground rather than a blank strip.
let width = 660.0
let height = 428.0
let horizon = 205.0

guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: arguments[1]) as CFURL, nil),
  let landscape = CGImageSourceCreateImageAtIndex(source, 0, nil)
else {
  fatalError("cannot read \(arguments[1])")
}
// Fill the window, trimming any aspect mismatch evenly from the sides or from the bottom.
let sceneWidth = min(Double(landscape.width), (Double(landscape.height) * width / height).rounded())
let sceneHeight = (sceneWidth * height / width).rounded()
let scene = landscape.cropping(
  to: CGRect(
    x: ((Double(landscape.width) - sceneWidth) / 2).rounded(), y: 0, width: sceneWidth,
    height: sceneHeight))!

guard let wordmark = NSImage(contentsOf: URL(fileURLWithPath: arguments[2])) else {
  fatalError("cannot read \(arguments[2])")
}
let wordmarkWidth = 200.0
let wordmarkHeight = wordmarkWidth * wordmark.size.height / wordmark.size.width
// Finder centers each 13 pt label under its icon, 128 pt icons centered at y = 250 in DiskImage.py.
let labelCenters = [CGPoint(x: 170, y: 324), CGPoint(x: 490, y: 324)]
let captionFont = CTFontCreateUIFontForLanguage(.system, 13, nil)!
let calloutFont = CTFontCreateUIFontForLanguage(.system, 12, nil)!
let callouts: [(dot: UInt32, text: String)] = [
  (0xE69105, "Fast on Apple silicon"),
  (0x009F71, "Works offline"),
  (0x8F66FF, "Nothing leaves your Mac"),
]

func color(_ hex: UInt32, alpha: Double = 1) -> CGColor {
  CGColor(
    srgbRed: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
    blue: Double(hex & 0xFF) / 255, alpha: alpha)
}

// Draws text with its cap height starting `top` points below the top edge; returns its width.
@discardableResult
func drawText(
  _ text: String, font: CTFont, color: CGColor, x: Double, top: Double, in ctx: CGContext
)
  -> Double
{
  let line = CTLineCreateWithAttributedString(
    NSAttributedString(
      string: text,
      attributes: [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
      ]))
  ctx.textPosition = CGPoint(x: x, y: height - top - CTFontGetCapHeight(font))
  CTLineDraw(line, ctx)
  return CTLineGetTypographicBounds(line, nil, nil, nil)
}

func render(scale: Double) -> CGImage {
  let ctx = CGContext(
    data: nil, width: Int(width * scale), height: Int(height * scale), bitsPerComponent: 8,
    bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
  ctx.scaleBy(x: scale, y: scale)
  ctx.interpolationQuality = .high
  // Core Graphics puts y = 0 at the bottom; `height - y` converts from Finder's top-left points.
  ctx.draw(scene, in: CGRect(x: 0, y: 0, width: width, height: height))

  let grade = CGGradient(
    colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
    colors: [
      color(0, alpha: 0), color(0, alpha: 0.22), color(0, alpha: 0.26), color(0, alpha: 0.6),
    ]
      as CFArray,
    locations: [0, (265 - horizon) / (height - horizon), (345 - horizon) / (height - horizon), 1])!
  ctx.drawLinearGradient(
    grade, start: CGPoint(x: 0, y: height - horizon), end: CGPoint(x: 0, y: 0), options: [])

  let start = CGPoint(x: 252, y: height - 236)
  let control = CGPoint(x: 330, y: height - 152)
  let end = CGPoint(x: 408, y: height - 236)
  ctx.setShadow(offset: CGSize(width: 0, height: -1), blur: 3, color: color(0, alpha: 0.55))
  ctx.setStrokeColor(color(0xF4F5F6, alpha: 0.92))
  ctx.setLineWidth(2.6)
  ctx.setLineCap(.round)
  ctx.setLineJoin(.round)
  ctx.setLineDash(phase: 0, lengths: [0.01, 8.5])
  ctx.move(to: start)
  ctx.addQuadCurve(to: end, control: control)
  ctx.strokePath()
  ctx.setLineDash(phase: 0, lengths: [])
  let angle = atan2(end.y - control.y, end.x - control.x)
  for side in [-1.0, 1.0] {
    let wing = angle + .pi + side * .pi / 6
    ctx.move(to: CGPoint(x: end.x + 12 * cos(wing), y: end.y + 12 * sin(wing)))
    ctx.addLine(to: end)
  }
  ctx.strokePath()
  ctx.setShadow(offset: .zero, blur: 0, color: nil)

  // Soft middle-grey patches even out the regolith under the labels.
  let patch = CGGradient(
    colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
    colors: [color(0x7B7B7B, alpha: 0.95), color(0x7B7B7B, alpha: 0.8), color(0x7B7B7B, alpha: 0)]
      as CFArray, locations: [0, 0.55, 1])!
  for center in labelCenters {
    ctx.saveGState()
    ctx.translateBy(x: center.x, y: height - center.y)
    ctx.scaleBy(x: 1, y: 0.24)
    ctx.drawRadialGradient(
      patch, startCenter: .zero, startRadius: 0, endCenter: .zero, endRadius: 74, options: [])
    ctx.restoreGState()
  }

  NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: false)
  wordmark.draw(
    in: NSRect(x: 40, y: height - 32 - wordmarkHeight, width: wordmarkWidth, height: wordmarkHeight)
  )
  NSGraphicsContext.current = nil
  drawText(
    "Drag Local Code into Applications to install.", font: captionFont, color: color(0xB7BFC7),
    x: 41, top: 82, in: ctx)
  var x = 41.0
  for callout in callouts {
    ctx.setFillColor(color(callout.dot))
    ctx.fillEllipse(in: CGRect(x: x, y: height - 397.5, width: 8, height: 8))
    x += 15
    x += drawText(callout.text, font: calloutFont, color: color(0xECEEF0), x: x, top: 389, in: ctx)
    x += 22
  }
  return ctx.makeImage()!
}

guard
  let output = CGImageDestinationCreateWithURL(
    URL(fileURLWithPath: arguments[3]) as CFURL, UTType.tiff.identifier as CFString, 2, nil)
else {
  fatalError("cannot write \(arguments[3])")
}
for scale in [1.0, 2.0] {
  CGImageDestinationAddImage(
    output, render(scale: scale),
    [
      kCGImagePropertyDPIWidth: 72 * scale, kCGImagePropertyDPIHeight: 72 * scale,
      kCGImagePropertyTIFFDictionary: [kCGImagePropertyTIFFCompression: 5],  // LZW
    ] as CFDictionary)
}
guard CGImageDestinationFinalize(output) else {
  fatalError("cannot write \(arguments[3])")
}
