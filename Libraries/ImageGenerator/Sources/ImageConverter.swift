import CoreGraphics
import DataModels
import Diffusion
import ModelZoo
import NNC
import Scripting
import UniformTypeIdentifiers

#if canImport(UIKit)
  import UIKit
#endif

public enum ImageConverter {
  public static func bitmapContext(from cgImage: CGImage) -> CGContext? {
    guard
      let bitmapContext = CGContext(
        data: nil, width: cgImage.width, height: cgImage.height, bitsPerComponent: 8,
        bytesPerRow: cgImage.width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue
          | CGImageAlphaInfo.premultipliedLast.rawValue, releaseCallback: nil, releaseInfo: nil)
    else {
      return nil
    }
    bitmapContext.draw(
      cgImage,
      in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
    return bitmapContext
  }
  #if canImport(UIKit)
    public static func resize(
      from image: UIImage, imageWidth: Int, imageHeight: Int,
      interpolationQuality: CGInterpolationQuality = .high
    ) -> CGContext? {
      guard let cgImage = image.cgImage,
        let bitmapContext = CGContext(
          data: nil, width: imageWidth, height: imageHeight, bitsPerComponent: 8,
          bytesPerRow: imageWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue
            | CGImageAlphaInfo.premultipliedLast.rawValue, releaseCallback: nil, releaseInfo: nil)
      else {
        return nil
      }
      bitmapContext.interpolationQuality = interpolationQuality
      // Scale to fill.
      var cgImageWidth = cgImage.width
      var cgImageHeight = cgImage.height
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&cgImageWidth, &cgImageHeight)
      }
      var scaledWidth: Int
      var scaledHeight: Int
      if cgImageWidth * imageHeight > cgImageHeight * imageWidth {
        scaledHeight = imageHeight
        scaledWidth = cgImageWidth * imageHeight / cgImageHeight
      } else {
        scaledWidth = imageWidth
        scaledHeight = cgImageHeight * imageWidth / cgImageWidth
      }
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&scaledWidth, &scaledHeight)
      }
      // Set rotation angle with image orientation. Don't deal with mirrored just yet.
      switch image.imageOrientation {
      case .up:
        break
      case .upMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        bitmapContext.scaleBy(x: -1, y: 1)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .down, .downMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .downMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .left, .rightMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .rightMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .right, .leftMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .leftMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi * 3 / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      @unknown default:
        break
      }
      bitmapContext.draw(
        cgImage,
        in: CGRect(
          x: (imageWidth - scaledWidth) / 2, y: (imageHeight - scaledHeight) / 2,
          width: scaledWidth, height: scaledHeight))
      return bitmapContext
    }
    public static func grayscaleTensor(from image: UIImage, imageWidth: Int, imageHeight: Int)
      -> Tensor<
        UInt8
      >?
    {
      guard let cgImage = image.cgImage,
        let bitmapContext = CGContext(
          data: nil, width: imageWidth, height: imageHeight, bitsPerComponent: 8,
          bytesPerRow: imageWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue
            | CGImageAlphaInfo.premultipliedLast.rawValue, releaseCallback: nil, releaseInfo: nil)
      else {
        return nil
      }
      bitmapContext.interpolationQuality = .high
      // Scale to fill.
      var cgImageWidth = cgImage.width
      var cgImageHeight = cgImage.height
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&cgImageWidth, &cgImageHeight)
      }
      var scaledWidth: Int
      var scaledHeight: Int
      if cgImageWidth * imageHeight > cgImageHeight * imageWidth {
        scaledHeight = imageHeight
        scaledWidth = cgImageWidth * imageHeight / cgImageHeight
      } else {
        scaledWidth = imageWidth
        scaledHeight = cgImageHeight * imageWidth / cgImageWidth
      }
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&scaledWidth, &scaledHeight)
      }
      // Set rotation angle with image orientation. Don't deal with mirrored just yet.
      switch image.imageOrientation {
      case .up:
        break
      case .upMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        bitmapContext.scaleBy(x: -1, y: 1)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .down, .downMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .downMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .left, .rightMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .rightMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .right, .leftMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .leftMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi * 3 / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      @unknown default:
        break
      }
      bitmapContext.draw(
        cgImage,
        in: CGRect(
          x: (imageWidth - scaledWidth) / 2, y: (imageHeight - scaledHeight) / 2,
          width: scaledWidth, height: scaledHeight))
      return grayscaleTensor(from: bitmapContext)
    }
  #endif
  static func grayscaleTensor(from bitmapContext: CGContext) -> Tensor<UInt8>? {
    precondition(bitmapContext.bytesPerRow >= bitmapContext.width * 4)
    guard let data = bitmapContext.data else {
      return nil
    }
    let bytes = data.assumingMemoryBound(to: UInt8.self)
    let bytesPerRow = bitmapContext.bytesPerRow
    let imageHeight = bitmapContext.height
    let imageWidth = bitmapContext.width
    var tensor = Tensor<UInt8>(.CPU, .NC(bitmapContext.height, bitmapContext.width))
    tensor.withUnsafeMutableBytes {
      guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      for y in 0..<imageHeight {
        for x in 0..<imageWidth {
          let alpha = bytes[y * bytesPerRow + x * 4 + 3]
          let r = Int32(bytes[y * bytesPerRow + x * 4])
          let g = Int32(bytes[y * bytesPerRow + x * 4 + 1])
          let b = Int32(bytes[y * bytesPerRow + x * 4 + 2])
          if alpha < 128 {
            u8[y * imageWidth + x] = 255  // We treat transparent as white.
          } else {
            u8[y * imageWidth + x] = UInt8((r * 6969 + g * 23434 + b * 2365) >> 15)
          }
        }
      }
    }
    return tensor
  }
  public static func scribbleRGB(from tensor: Tensor<UInt8>) -> Tensor<FloatType> {
    let imageHeight = tensor.shape[0]
    let imageWidth = tensor.shape[1]
    var rgb = Tensor<FloatType>(.CPU, .NHWC(1, imageHeight, imageWidth, 3))
    rgb.withUnsafeMutableBytes {
      guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      tensor.withUnsafeBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for i in 0..<imageHeight * imageWidth {
          if u8[i] > 128 {
            fp16[i * 3] = 0
            fp16[i * 3 + 1] = 0
            fp16[i * 3 + 2] = 0
          } else {
            fp16[i * 3] = 1
            fp16[i * 3 + 1] = 1
            fp16[i * 3 + 2] = 1
          }
        }
      }
    }
    return rgb
  }
  static func scribble(from binaryMask: Tensor<UInt8>) -> Tensor<UInt8>? {
    let maskHeight = binaryMask.shape[0]
    let maskWidth = binaryMask.shape[1]
    var flag = true
    var paintCount = 0
    binaryMask.withUnsafeBytes {
      guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      for y in 0..<maskHeight {
        for x in 0..<maskWidth {
          let byte = u8[y * maskWidth + x]
          if byte == 0 {
            paintCount += 1
          } else if byte == 1 {
            // Empty.
          } else {
            flag = false
            break
          }
        }
        if !flag {
          break
        }
      }
    }
    // If we haven't paint even half of the screen, this must be scribble.
    guard flag && paintCount * 2 < maskHeight * maskWidth else {
      return nil
    }
    var scribble = Tensor<UInt8>(.CPU, .NC(maskHeight, maskWidth))
    scribble.withUnsafeMutableBytes {
      guard let s8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      binaryMask.withUnsafeBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for y in 0..<maskHeight {
          for x in 0..<maskWidth {
            let byte = u8[y * maskWidth + x]
            s8[y * maskWidth + x] = byte == 0 ? 0 : 255
          }
        }
      }
    }
    return scribble
  }
  #if canImport(UIKit)
    public static func grayscaleImage(binaryMask: Tensor<UInt8>) -> UIImage {
      let imageHeight = binaryMask.shape[0]
      let imageWidth = binaryMask.shape[1]
      let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
      binaryMask.withUnsafeBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for i in 0..<imageHeight * imageWidth {
          let byteMask = u8[i]
          if (byteMask & 7) != 0 && (byteMask & 7) != 3 {
            let alpha = 0xff - (byteMask & 0xf8)
            bytes[i * 4] = alpha
            bytes[i * 4 + 1] = alpha
            bytes[i * 4 + 2] = alpha
            bytes[i * 4 + 3] = 255
          } else {
            bytes[i * 4] = 0
            bytes[i * 4 + 1] = 0
            bytes[i * 4 + 2] = 0
            bytes[i * 4 + 3] = 255
          }
        }
      }
      return UIImage(
        cgImage: CGImage(
          width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
          bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo(
            rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue),
          provider: CGDataProvider(
            dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
            releaseData: { _, p, _ in
              p.deallocate()
            })!, decode: nil, shouldInterpolate: false,
          intent: CGColorRenderingIntent.defaultIntent)!)
    }
    public static func grayscaleImage(from tensor: Tensor<UInt8>) -> UIImage {
      let imageHeight = tensor.shape[0]
      let imageWidth = tensor.shape[1]
      let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
      tensor.withUnsafeBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for i in 0..<imageHeight * imageWidth {
          let byte = u8[i]
          bytes[i * 4] = byte
          bytes[i * 4 + 1] = byte
          bytes[i * 4 + 2] = byte
          bytes[i * 4 + 3] = 255
        }
      }
      return UIImage(
        cgImage: CGImage(
          width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
          bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo(
            rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue),
          provider: CGDataProvider(
            dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
            releaseData: { _, p, _ in
              p.deallocate()
            })!, decode: nil, shouldInterpolate: false,
          intent: CGColorRenderingIntent.defaultIntent)!)
    }
    public static func grayscaleImage(from tensor: Tensor<FloatType>) -> UIImage {
      let imageHeight = tensor.shape[1]
      let imageWidth = tensor.shape[2]
      let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
      tensor.withUnsafeBytes {
        guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
        for i in 0..<imageHeight * imageWidth {
          let byte = UInt8(min(max(Int((fp16[i] + 1) * 127.5), 0), 255))
          bytes[i * 4] = byte
          bytes[i * 4 + 1] = byte
          bytes[i * 4 + 2] = byte
          bytes[i * 4 + 3] = 255
        }
      }
      return UIImage(
        cgImage: CGImage(
          width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
          bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo(
            rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue),
          provider: CGDataProvider(
            dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
            releaseData: { _, p, _ in
              p.deallocate()
            })!, decode: nil, shouldInterpolate: false,
          intent: CGColorRenderingIntent.defaultIntent)!)
    }
    public static func positiveRangeTensor(
      from image: UIImage, imageWidth: Int, imageHeight: Int
    ) -> Tensor<
      FloatType
    >? {
      let bitmapInfo =
        CGBitmapInfo.byteOrderDefault.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
      guard let cgImage = image.cgImage,
        let bitmapContext = CGContext(
          data: nil, width: imageWidth, height: imageHeight, bitsPerComponent: 8,
          bytesPerRow: imageWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: bitmapInfo, releaseCallback: nil, releaseInfo: nil)
      else {
        return nil
      }
      bitmapContext.interpolationQuality = .high
      // Scale to fill.
      var cgImageWidth = cgImage.width
      var cgImageHeight = cgImage.height
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&cgImageWidth, &cgImageHeight)
      }
      var scaledWidth: Int
      var scaledHeight: Int
      if cgImageWidth * imageHeight > cgImageHeight * imageWidth {
        scaledHeight = imageHeight
        scaledWidth = cgImageWidth * imageHeight / cgImageHeight
      } else {
        scaledWidth = imageWidth
        scaledHeight = cgImageHeight * imageWidth / cgImageWidth
      }
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&scaledWidth, &scaledHeight)
      }
      // Set rotation angle with image orientation. Don't deal with mirrored just yet.
      switch image.imageOrientation {
      case .up:
        break
      case .upMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        bitmapContext.scaleBy(x: -1, y: 1)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .down, .downMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .downMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .left, .rightMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .rightMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .right, .leftMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .leftMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi * 3 / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      @unknown default:
        break
      }
      bitmapContext.draw(
        cgImage,
        in: CGRect(
          x: (imageWidth - scaledWidth) / 2, y: (imageHeight - scaledHeight) / 2,
          width: scaledWidth, height: scaledHeight))
      return positiveRangeTensor(from: bitmapContext)
    }
  #endif
  static func positiveRangeTensor(from bitmapContext: CGContext) -> Tensor<FloatType>? {
    precondition(bitmapContext.bytesPerRow >= bitmapContext.width * 4)
    guard let data = bitmapContext.data else {
      return nil
    }
    let bytes = data.assumingMemoryBound(to: UInt8.self)
    let bytesPerRow = bitmapContext.bytesPerRow
    let imageHeight = bitmapContext.height
    let imageWidth = bitmapContext.width
    var tensor = Tensor<FloatType>(.CPU, .NHWC(1, imageHeight, imageWidth, 3))
    tensor.withUnsafeMutableBytes {
      guard let f16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      for y in 0..<imageHeight {
        for x in 0..<imageWidth {
          f16[y * imageWidth * 3 + x * 3] = FloatType(Float(bytes[y * bytesPerRow + x * 4]) / 255.0)
          f16[y * imageWidth * 3 + x * 3 + 1] = FloatType(
            Float(bytes[y * bytesPerRow + x * 4 + 1]) / 255.0)
          f16[y * imageWidth * 3 + x * 3 + 2] = FloatType(
            Float(bytes[y * bytesPerRow + x * 4 + 2]) / 255.0)
        }
      }
    }
    return tensor
  }
  public static func imageAndMask(from argbTensor: Tensor<FloatType>) -> (
    Tensor<FloatType>, Tensor<UInt8>?
  ) {
    let shape = argbTensor.shape
    guard shape[3] == 4 else {
      precondition(shape[3] == 3)
      return (argbTensor, nil)
    }
    let rgbTensor = argbTensor[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<shape[3]].copied()
    var binaryMask = Tensor<UInt8>(.CPU, .NC(shape[1], shape[2]))
    let height = shape[1]
    let width = shape[2]
    argbTensor.withUnsafeBytes {
      guard let f16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      binaryMask.withUnsafeMutableBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for y in 0..<height {
          for x in 0..<width {
            let alpha = min(max(Int((f16[y * width * 4 + x * 4] * 255).rounded()), 0), 255)
            if alpha != 255 {
              u8[y * width + x] = (1 | UInt8(alpha & 0xf8))
            } else {
              u8[y * width + x] = 3
            }
          }
        }
      }
    }
    return (rgbTensor, binaryMask)
  }
  #if canImport(UIKit)
    public static func image(
      from tensor: Tensor<FloatType>, scaleFactor: CGFloat, binaryMask: Tensor<UInt8>? = nil,
      only1: Bool = false, overlayRects: [CGRect] = []
    ) -> UIImage {
      let imageHeight = tensor.shape[1]
      let imageWidth = tensor.shape[2]
      let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
      tensor.withUnsafeBytes {
        guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
        for i in 0..<imageHeight * imageWidth {
          bytes[i * 4] = UInt8(min(max(Int((fp16[i * 3] + 1) * 127.5), 0), 255))
          bytes[i * 4 + 1] = UInt8(min(max(Int((fp16[i * 3 + 1] + 1) * 127.5), 0), 255))
          bytes[i * 4 + 2] = UInt8(min(max(Int((fp16[i * 3 + 2] + 1) * 127.5), 0), 255))
          bytes[i * 4 + 3] = 255
        }
      }
      let invScale = 1 / scaleFactor
      if let binaryMask = binaryMask {
        let maskHeight = binaryMask.shape[0]
        let maskWidth = binaryMask.shape[1]
        if only1 {
          let overlayIntegralRects: [(minX: Int, minY: Int, maxX: Int, maxY: Int)] =
            overlayRects.map {
              (
                minX: Int($0.minX.rounded(.up)), minY: Int($0.minY.rounded(.up)),
                maxX: Int(($0.maxX - 1).rounded(.down)), maxY: Int(($0.maxY - 1).rounded(.down))
              )
            }
          binaryMask.withUnsafeBytes {
            guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for i in 0..<imageHeight {
              let ii = max(
                0, min(maskHeight - 1, Int(((CGFloat(i) + 0.5) * invScale - 0.5).rounded())))
              for j in 0..<imageWidth {
                let ij = max(
                  0, min(maskWidth - 1, Int(((CGFloat(j) + 0.5) * invScale - 0.5).rounded())))
                let byteMask = u8[ii * maskWidth + ij]
                let flag = overlayIntegralRects.contains {
                  ii >= $0.minY && ii <= $0.maxY && ij >= $0.minX && ij <= $0.maxX
                }
                // We make it transparent if it is 1 or it is overlapped by the overlay rects.
                if (byteMask & 7) == 1 || (flag && (byteMask & 7) != 0 && (byteMask & 7) != 3) {
                  let alpha = byteMask & 0xf8
                  if alpha > 0 {
                    bytes[i * imageWidth * 4 + j * 4] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 1] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4 + 1]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 2] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4 + 2]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 3] = alpha
                  } else {
                    bytes[i * imageWidth * 4 + j * 4] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 1] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 2] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 3] = 0
                  }
                }
              }
            }
          }
        } else {
          binaryMask.withUnsafeBytes {
            guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for i in 0..<imageHeight {
              let ii = max(
                0, min(maskHeight - 1, Int(((CGFloat(i) + 0.5) * invScale - 0.5).rounded())))
              for j in 0..<imageWidth {
                let ij = max(
                  0, min(maskWidth - 1, Int(((CGFloat(j) + 0.5) * invScale - 0.5).rounded())))
                let byteMask = u8[ii * maskWidth + ij]
                if (byteMask & 7) != 0 && (byteMask & 7) != 3 {
                  let alpha = byteMask & 0xf8
                  if alpha > 0 {
                    bytes[i * imageWidth * 4 + j * 4] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 1] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4 + 1]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 2] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4 + 2]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 3] = alpha
                  } else {
                    bytes[i * imageWidth * 4 + j * 4] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 1] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 2] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 3] = 0
                  }
                }
              }
            }
          }
        }
        return UIImage(
          cgImage: CGImage(
            width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(
              rawValue: CGBitmapInfo.byteOrder32Big.rawValue
                | CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: CGDataProvider(
              dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
              releaseData: { _, p, _ in
                p.deallocate()
              })!, decode: nil, shouldInterpolate: false,
            intent: CGColorRenderingIntent.defaultIntent)!)
      } else {
        return UIImage(
          cgImage: CGImage(
            width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(
              rawValue: CGBitmapInfo.byteOrder32Big.rawValue
                | CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: CGDataProvider(
              dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
              releaseData: { _, p, _ in
                p.deallocate()
              })!, decode: nil, shouldInterpolate: false,
            intent: CGColorRenderingIntent.defaultIntent)!)
      }
    }
  #endif
  @inline(__always)
  static private func OKlabToLinearsRGB(L: Float, a: Float, b: Float) -> (Float, Float, Float) {
    let l_ = L + 0.3963377774 * a + 0.2158037573 * b
    let m_ = L - 0.1055613458 * a - 0.0638541728 * b
    let s_ = L - 0.0894841775 * a - 1.2914855480 * b
    let l = l_ * l_ * l_
    let m = m_ * m_ * m_
    let s = s_ * s_ * s_
    return (
      +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    )
  }
  @inline(__always)
  static private func linearsRGBTosRGB(x: Float) -> Float {
    if x >= 0.04045 {
      return pow((x + 0.055) / (1 + 0.055), 2.4)
    } else {
      return x / 12.92
    }
  }
  #if canImport(UIKit)
    public static func image(fromLatent tensor: Tensor<FloatType>, version: ModelVersion)
      -> UIImage?
    {
      let imageHeight = tensor.shape[1]
      let imageWidth = tensor.shape[2]
      let channels = tensor.shape[3]
      guard channels == 4 || channels == 3 || channels == 16 else { return nil }
      let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
      tensor.withUnsafeBytes {
        guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
        switch version {
        case .v1, .v2, .svdI2v:
          for i in 0..<imageHeight * imageWidth {
            // We need to do some computations from the latent values.
            let (v0, v1, v2, v3) = (fp16[i * 4], fp16[i * 4 + 1], fp16[i * 4 + 2], fp16[i * 4 + 3])
            let r = 49.5210 * v0 + 29.0283 * v1 - 23.9673 * v2 - 39.4981 * v3 + 99.9368
            let g = 41.1373 * v0 + 42.4951 * v1 + 24.7349 * v2 - 50.8279 * v3 + 99.8421
            let b = 40.2919 * v0 + 18.9304 * v1 + 30.0236 * v2 - 81.9976 * v3 + 99.5384
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .sd3:
          for i in 0..<imageHeight * imageWidth {
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) = (
              fp16[i * 16], fp16[i * 16 + 1], fp16[i * 16 + 2], fp16[i * 16 + 3], fp16[i * 16 + 4],
              fp16[i * 16 + 5], fp16[i * 16 + 6], fp16[i * 16 + 7], fp16[i * 16 + 8],
              fp16[i * 16 + 9], fp16[i * 16 + 10], fp16[i * 16 + 11], fp16[i * 16 + 12],
              fp16[i * 16 + 13], fp16[i * 16 + 14], fp16[i * 16 + 15]
            )
            // TBD, I may want to regress these coefficients myself.
            let r: FloatType =
              (-0.0645 * v0 + 0.0028 * v1 + 0.1848 * v2 + 0.0944 * v3 + 0.0897 * v4 - 0.0020 * v5
                + 0.0855 * v6 - 0.0539 * v7 - 0.0057 * v8 - 0.0412 * v9 + 0.1106 * v10 - 0.0248
                * v11 + 0.0815 * v12 - 0.0120 * v13 - 0.0749 * v14 - 0.1418 * v15) * 127.5 + 127.5
            let g: FloatType =
              (0.0177 * v0 + 0.0312 * v1 + 0.0762 * v2 + 0.0360 * v3 + 0.0506 * v4 + 0.1203 * v5
                + 0.0118 * v6 + 0.0658 * v7 + 0.0116 * v8 + 0.0281 * v9 + 0.1171 * v10 + 0.0682
                * v11 + 0.0846 * v12 - 0.0055 * v13 - 0.0634 * v14 - 0.1457 * v15) * 127.5 + 127.5
            let b: FloatType =
              (0.1052 * v0 + 0.0650 * v1 + 0.0360 * v2 + 0.0889 * v3 - 0.0364 * v4 + 0.0284 * v5
                + 0.0283 * v6 + 0.1047 * v7 + 0.0700 * v8 - 0.0039 * v9 + 0.1220 * v10 - 0.0481
                * v11 + 0.1207 * v12 - 0.0867 * v13 - 0.0456 * v14 - 0.1259 * v15) * 127.5 + 127.5
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .sdxlBase, .sdxlRefiner, .ssd1b, .pixart, .auraflow:
          for i in 0..<imageHeight * imageWidth {
            // We need to do some computations from the latent values.
            let (v0, v1, v2, v3) = (fp16[i * 4], fp16[i * 4 + 1], fp16[i * 4 + 2], fp16[i * 4 + 3])
            let r = 47.195 * v0 - 29.114 * v1 + 11.883 * v2 - 38.063 * v3 + 141.64
            let g = 53.237 * v0 - 1.4623 * v1 + 12.991 * v2 - 28.043 * v3 + 127.46
            let b = 58.182 * v0 + 4.3734 * v1 - 3.3735 * v2 - 26.722 * v3 + 114.5
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .kandinsky21:
          for i in 0..<imageHeight * imageWidth {
            // We need to do some computations from the latent values.
            let (v0, v1, v2, v3) = (
              Float(fp16[i * 4]), Float(fp16[i * 4 + 1]), Float(fp16[i * 4 + 2]),
              Float(fp16[i * 4 + 3])
            )
            let L = -0.051509 * v0 + 0.039954 * v1 + 0.039893 * v2 - 0.087302 * v3 + 0.88591
            let a = -0.028686 * v0 - 0.0061331 * v1 - 0.016837 * v2 + 0.016139 * v3 + 0.0018263
            let b = -0.0068242 * v0 + 0.0068562 * v1 - 0.03415 * v2 + 0.00056286 * v3 + 0.0096209
            var (sr, sg, sb) = OKlabToLinearsRGB(L: L, a: a, b: b)
            sr = linearsRGBTosRGB(x: sr) * 255
            sg = linearsRGBTosRGB(x: sg) * 255
            sb = linearsRGBTosRGB(x: sb) * 255
            bytes[i * 4] = UInt8(min(max(Int(sr.isFinite ? sr : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(sg.isFinite ? sg : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(sb.isFinite ? sb : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .wurstchenStageC, .wurstchenStageB:
          if channels == 3 {
            for i in 0..<imageHeight * imageWidth {
              // We need to do some computations from the latent values.
              let (r, g, b) = (fp16[i * 3], fp16[i * 3 + 1], fp16[i * 3 + 2])
              bytes[i * 4] = UInt8(
                min(max(Int(r.isFinite ? (Float(r) * 255).rounded() : 0), 0), 255))
              bytes[i * 4 + 1] = UInt8(
                min(max(Int(g.isFinite ? (Float(g) * 255).rounded() : 0), 0), 255))
              bytes[i * 4 + 2] = UInt8(
                min(max(Int(b.isFinite ? (Float(b) * 255).rounded() : 0), 0), 255))
              bytes[i * 4 + 3] = 255
            }
          } else {
            for i in 0..<imageHeight * imageWidth {
              // We need to do some computations from the latent values.
              let (v0, v1, v2, v3) = (
                fp16[i * 4], fp16[i * 4 + 1], fp16[i * 4 + 2], fp16[i * 4 + 3]
              )
              let r = 10.175 * v0 - 20.807 * v1 - 27.834 * v2 - 2.0577 * v3 + 143.39
              let g = 21.07 * v0 - 4.3022 * v1 - 11.258 * v2 - 18.8 * v3 + 131.53
              let b = 7.8454 * v0 - 2.3713 * v1 - 0.45565 * v2 - 41.648 * v3 + 120.76
              bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
              bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
              bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
              bytes[i * 4 + 3] = 255
            }
          }
        }
      }
      return UIImage(
        cgImage: CGImage(
          width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
          bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo(
            rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue),
          provider: CGDataProvider(
            dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
            releaseData: { _, p, _ in
              p.deallocate()
            })!, decode: nil, shouldInterpolate: false,
          intent: CGColorRenderingIntent.defaultIntent)!)
    }
  #endif
  #if canImport(UIKit)
    public static func imageData(
      from tensor: Tensor<FloatType>, binaryMask: Tensor<UInt8>?,
      configuration: GenerationConfiguration,
      savedPositivePrompt: String, savedNegativePrompt: String
    ) -> Data? {
      return autoreleasepool { () -> Data? in
        guard let cgImage = image(from: tensor, scaleFactor: 1, binaryMask: binaryMask).cgImage
        else { return nil }
        let data = NSMutableData()
        guard
          let imageDestination = CGImageDestinationCreateWithData(
            data as CFMutableData, UTType.png.identifier as CFString, 1, nil)
        else { return nil }
        var description = "\(savedPositivePrompt.trimmingCharacters(in: .whitespacesAndNewlines))\n"
        if savedNegativePrompt.count > 0 {
          description += "-\(savedNegativePrompt.trimmingCharacters(in: .whitespacesAndNewlines))\n"
        }
        let model = configuration.model ?? ModelZoo.defaultSpecification.file
        let modelVersion = ModelZoo.versionForModel(model)
        let modifier = ModelPreloader.modifierForModel(
          model, LoRAs: configuration.loras.compactMap(\.file))
        let seedMode: String
        switch configuration.seedMode {
        case .legacy:
          seedMode = "Legacy"
        case .torchCpuCompatible:
          seedMode = "Torch CPU Compatible"
        case .scaleAlike:
          seedMode = "Scale Alike"
        case .nvidiaGpuCompatible:
          seedMode = "NVIDIA GPU Compatible"
        }
        let sampler = configuration.sampler.description
        description +=
          "Steps: \(configuration.steps), Sampler: \(sampler), Guidance Scale: \(configuration.guidanceScale), Seed: \(configuration.seed), Size: \(configuration.startWidth * 64)x\(configuration.startHeight * 64), Model: \(model), Strength: \(configuration.strength), Seed Mode: \(seedMode)"
        var json = [String: Any]()
        let jsonEncoder = JSONEncoder()
        if let data = try? jsonEncoder.encode(
          JSGenerationConfiguration(configuration: configuration))
        {
          json["v2"] = try? JSONSerialization.jsonObject(
            with: data, options: [.topLevelDictionaryAssumed])
        }
        json["c"] = savedPositivePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        json["uc"] = savedNegativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        json["steps"] = configuration.steps
        json["sampler"] = sampler
        json["scale"] = configuration.guidanceScale
        json["seed"] = configuration.seed
        json["size"] = "\(configuration.startWidth * 64)x\(configuration.startHeight * 64)"
        json["model"] = model
        json["strength"] = configuration.strength
        json["seed_mode"] = seedMode
        if let upscaler = configuration.upscaler, UpscalerZoo.isModelDownloaded(upscaler) {
          json["upscaler"] = upscaler
          description += ", Upscaler: \(upscaler)"
        }
        if configuration.shift != 1 {
          json["shift"] = configuration.shift
          description += ", Shift: \(configuration.shift)"
        }
        if configuration.clipSkip > 1 {
          json["clip_skip"] = configuration.clipSkip
          description += ", CLIP Skip: \(configuration.clipSkip)"
        }
        if configuration.sharpness > 0 {
          json["sharpness"] = configuration.sharpness
          description += ", Sharpness: \(configuration.sharpness)"
        }
        if configuration.maskBlur > 0 {
          json["mask_blur"] = configuration.maskBlur
        }
        if configuration.maskBlurOutset != 0 {
          json["mask_blur_outset"] = configuration.maskBlurOutset
          description += ", Mask Blur Outset: \(configuration.maskBlurOutset)"
        }
        if modifier == .editing {
          json["image_guidance"] = configuration.imageGuidanceScale
          description += ", Image Guidance: \(configuration.imageGuidanceScale)"
        }
        if configuration.hiresFix && modelVersion != .wurstchenStageC {
          description +=
            ", Hires Fix: true, First Stage Size: \(configuration.hiresFixStartWidth * 64)x\(configuration.hiresFixStartHeight * 64), Second Stage Strength: \(configuration.hiresFixStrength)"
          json["hires_fix"] = configuration.hiresFix
          json["first_stage_size"] =
            "\(configuration.hiresFixStartWidth * 64)x\(configuration.hiresFixStartHeight * 64)"
          json["second_stage_strength"] = configuration.hiresFixStrength
        }
        let refinerVersion: ModelVersion? = configuration.refinerModel.flatMap {
          guard $0 != configuration.model, ModelZoo.isModelDownloaded($0) else { return nil }
          let version = ModelZoo.versionForModel($0)
          guard
            version == modelVersion
              || ([.sdxlBase, .sdxlRefiner, .ssd1b].contains(version)
                && [.sdxlBase, .sdxlRefiner, .ssd1b].contains(modelVersion))
          else { return nil }
          return ModelZoo.versionForModel($0)
        }
        let loras: [LoRAConfiguration] = configuration.loras.compactMap {
          guard let file = $0.file, LoRAZoo.isModelDownloaded(file) else { return nil }
          let loraVersion = LoRAZoo.versionForModel(file)
          guard modelVersion == loraVersion || refinerVersion == loraVersion
          else { return nil }
          return LoRAConfiguration(
            file: file, weight: $0.weight, version: modelVersion,
            isLoHa: LoRAZoo.isLoHaForModel(file), modifier: LoRAZoo.modifierForModel(file))
        }
        if modelVersion == .kandinsky21 {
          json["clip_weight"] = configuration.clipWeight
          description +=
            ", CLIP Weight: \(configuration.clipWeight.formatted(.percent.precision(.fractionLength(0))))"
          json["image_prior_steps"] = configuration.imagePriorSteps
          description += ", Image Prior Steps: \(configuration.imagePriorSteps)"
          json["negative_prompt_for_image_prior"] = configuration.negativePromptForImagePrior
          description +=
            ", Negative Prompt for Image Prior: \(configuration.negativePromptForImagePrior ? "true" : "false")"
        }
        if modelVersion == .wurstchenStageC {
          let width = Int(
            ((Double(
              configuration.hiresFixStartWidth > 0 && configuration.hiresFix
                ? configuration.hiresFixStartWidth : configuration.startWidth) * 3) / 2).rounded(
                .up))
          let height = Int(
            ((Double(
              configuration.hiresFixStartHeight > 0 && configuration.hiresFix
                ? configuration.hiresFixStartHeight : configuration.startHeight) * 3) / 2).rounded(
                .up))
          json["stage_1_latents_size"] = "\(width)x\(height)"
          description += ", Stage 1 Latents Size: \(width)x\(height)"
          json["stage_2_steps"] = configuration.stage2Steps
          description += ", Stage 2 Steps: \(configuration.stage2Steps)"
          json["stage_2_guidance"] = configuration.stage2Cfg
          description += ", Stage 2 Guidance: \(configuration.stage2Cfg)"
          if configuration.stage2Shift != 1 {
            json["stage_2_shift"] = configuration.stage2Shift
            description += ", Stage 2 Shift: \(configuration.stage2Shift)"
          }
        }
        if let refiner = configuration.refinerModel {
          json["refiner"] = refiner
          json["refiner_start"] = configuration.refinerStart
          description +=
            ", Refiner: \(refiner), Refiner Start: \(configuration.refinerStart.formatted(.percent.precision(.fractionLength(0))))"
        }
        if modelVersion == .sdxlBase || modelVersion == .sdxlRefiner || modelVersion == .ssd1b {
          json["target_size"] =
            "\(configuration.targetImageWidth)x\(configuration.targetImageHeight)"
          json["crop_top"] = configuration.cropTop
          json["crop_left"] = configuration.cropLeft
          json["original_size"] =
            "\(configuration.originalImageWidth)x\(configuration.originalImageHeight)"
          json["aesthetic_score"] = configuration.aestheticScore
          json["negative_aesthetic_score"] = configuration.negativeAestheticScore
          json["zero_negative_prompt"] = configuration.zeroNegativePrompt
          json["negative_original_size"] =
            "\(configuration.negativeOriginalImageWidth)x\(configuration.negativeOriginalImageHeight)"
          description +=
            ", Target Size: \(configuration.targetImageWidth)x\(configuration.targetImageHeight), Crop: (\(configuration.cropTop), \(configuration.cropLeft)), Original Size: \(configuration.originalImageWidth)x\(configuration.originalImageHeight), Negative Original Size: \(configuration.negativeOriginalImageWidth)x\(configuration.negativeOriginalImageHeight), Aesthetic Score: \(configuration.aestheticScore.formatted(.number.precision(.fractionLength(1)))), Negative Aesthetic Score: \(configuration.negativeAestheticScore.formatted(.number.precision(.fractionLength(1)))), Zero Negative Prompt: \(configuration.zeroNegativePrompt ? "true" : "false")"
        }
        if modelVersion == .svdI2v {
          json["num_frames"] = configuration.numFrames
          json["fps"] = configuration.fpsId
          json["motion_scale"] = configuration.motionBucketId
          json["cond_aug"] = configuration.condAug
          json["min_cfg"] = configuration.startFrameCfg
          description +=
            ", Number of Frames: \(configuration.numFrames), FPS: \(configuration.fpsId), Motion Scale: \(configuration.motionBucketId), Guiding Frame Noise: \(configuration.condAug.formatted(.number.precision(.fractionLength(2)))), Start Frame Guidance: \(configuration.startFrameCfg.formatted(.number.precision(.fractionLength(1))))"
        }
        if configuration.tiledDecoding
          && (configuration.startWidth > configuration.decodingTileWidth
            || configuration.startHeight > configuration.decodingTileHeight)
        {
          json["tiled_decoding"] = configuration.tiledDecoding
          json["decoding_tile_width"] = configuration.decodingTileWidth * 64
          json["decoding_tile_height"] = configuration.decodingTileHeight * 64
          json["decoding_tile_overlap"] = configuration.decodingTileOverlap * 64
          description +=
            ", Tiled Decoding Enabled: \(configuration.decodingTileWidth * 64)x\(configuration.decodingTileHeight * 64) [\(configuration.decodingTileOverlap * 64)]"
        }
        if configuration.tiledDiffusion
          && (configuration.startWidth > configuration.diffusionTileWidth
            || configuration.startHeight > configuration.diffusionTileHeight)
        {
          json["tiled_diffusion"] = configuration.tiledDiffusion
          json["diffusion_tile_width"] = configuration.diffusionTileWidth * 64
          json["diffusion_tile_height"] = configuration.diffusionTileHeight * 64
          json["diffusion_tile_overlap"] = configuration.diffusionTileOverlap * 64
          description +=
            ", Tiled Diffusion Enabled: \(configuration.diffusionTileWidth * 64)x\(configuration.diffusionTileHeight * 64) [\(configuration.diffusionTileOverlap * 64)]"
        }
        if configuration.sampler == .TCD {
          json["stochastic_sampling_gamma"] = configuration.stochasticSamplingGamma
          description += ", Strategic Stochastic Sampling: \(configuration.stochasticSamplingGamma)"
        }
        if loras.count > 0 {
          if let firstLoRA = loras.first, loras.count == 1 {
            description += ", LoRA Model: \(firstLoRA.file), LoRA Weight: \(firstLoRA.weight)"
          } else {
            description +=
              ", \(loras.enumerated().map({ "LoRA \($0 + 1) Model: \($1.file), LoRA \($0 + 1) Weight: \($1.weight)" }).joined(separator: ", "))"
          }
          json["lora"] = loras.map {
            ["model": $0.file, "weight": $0.weight] as [String: Any]
          }
        }
        let (canInjectControls, canInjectT2IAdapters, injectIPAdapterLengths, _) =
          ImageGenerator.canInjectControls(
            hasImage: true, hasDepth: true, hasHints: Set([.scribble, .pose, .color]),
            hasCustom: true, shuffleCount: 1,
            controls: configuration.controls, version: modelVersion)
        if canInjectControls || canInjectT2IAdapters || !injectIPAdapterLengths.isEmpty {
          let controls:
            [(
              file: String, weight: Float, guidanceStart: Float, guidanceEnd: Float, noPrompt: Bool,
              globalAveragePooling: Bool
            )] =
              configuration.controls.compactMap {
                guard let file = $0.file,
                  let specification = ControlNetZoo.specificationForModel(file),
                  ControlNetZoo.isModelDownloaded(specification)
                else { return nil }
                let modifier = ControlNetZoo.modifierForModel(file)
                switch modifier {
                case .canny, .custom, .depth, .scribble, .pose, .color, .normalbae, .lineart,
                  .softedge,
                  .seg,
                  .inpaint, .ip2p, .shuffle, .mlsd, .tile:
                  return (
                    file, $0.weight, $0.guidanceStart, $0.guidanceEnd, $0.noPrompt,
                    $0.globalAveragePooling
                  )
                }
              }
          if controls.count > 1 {
            var jsonControls = [[String: Any]]()
            for (i, control) in controls.enumerated() {
              description +=
                ", Control \(i + 1): \(control.file), Control \(i + 1) No Prompt: \(control.noPrompt ? "true" : "false"), Control \(i + 1) Weight: \(control.weight), Control \(i + 1) Start: \(control.guidanceStart), Control \(i + 1) End: \(control.guidanceEnd)"
              var subjson: [String: Any] = [
                "file": control.file, "weight": control.weight, "no_prompt": control.noPrompt,
                "guidance_start": control.guidanceStart, "guidance_end": control.guidanceEnd,
              ]
              if ControlNetZoo.modifierForModel(control.file) == .shuffle {
                subjson["global_average_pooling"] = control.globalAveragePooling
                description +=
                  ", Control \(i + 1) Global Average Pooling: \(control.globalAveragePooling ? "true" : "false")"
              }
              jsonControls.append(subjson)
            }
            json["control"] = jsonControls
          } else if let control = controls.first {
            description +=
              ", Control: \(control.file), Control No Prompt: \(control.noPrompt ? "true" : "false"), Control Weight: \(control.weight), Control Start: \(control.guidanceStart), Control End: \(control.guidanceEnd)"
            json["control"] = [
              [
                "file": control.file, "weight": control.weight, "no_prompt": control.noPrompt,
                "guidance_start": control.guidanceStart, "guidance_end": control.guidanceEnd,
              ] as [String: Any]
            ]
          }
        }
        var info = [String: Any]()
        info[kCGImagePropertyPNGDescription as String] = description
        if let jsonData = try? JSONSerialization.data(withJSONObject: json, options: .sortedKeys) {
          info[kCGImagePropertyPNGComment as String] = String(data: jsonData, encoding: .utf8)
        }
        info[kCGImagePropertyPNGSoftware as String] = "Draw Things"
        var metadata = [String: Any]()
        metadata[kCGImagePropertyPNGDictionary as String] = info
        CGImageDestinationAddImage(imageDestination, cgImage, metadata as CFDictionary)
        CGImageDestinationFinalize(imageDestination)
        return data as Data
      }
    }
  #endif
  public static func tensor(from bitmapContext: CGContext) -> (
    Tensor<FloatType>?, Tensor<UInt8>?, Bool
  ) {
    precondition(bitmapContext.bytesPerRow >= bitmapContext.width * 4)
    guard let data = bitmapContext.data else {
      return (nil, nil, false)
    }
    let bytes = data.assumingMemoryBound(to: UInt8.self)
    var tensor = Tensor<FloatType>(.CPU, .NHWC(1, bitmapContext.height, bitmapContext.width, 3))
    let bytesPerRow = bitmapContext.bytesPerRow
    var hasTransparent = false
    var hasNonTransparent = true
    let imageHeight = bitmapContext.height
    let imageWidth = bitmapContext.width
    tensor.withUnsafeMutableBytes {
      guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      for y in 0..<imageHeight {
        for x in 0..<imageWidth {
          let alpha = bytes[y * bytesPerRow + x * 4 + 3]
          if alpha != 255 {
            hasTransparent = true
            if alpha == 0 {
              fp16[y * imageWidth * 3 + x * 3] = 0
              fp16[y * imageWidth * 3 + x * 3 + 1] = 0
              fp16[y * imageWidth * 3 + x * 3 + 2] = 0
            } else {
              let invAlpha = 255.0 / FloatType(alpha)
              let r = FloatType(bytes[y * bytesPerRow + x * 4]) * 2 / 255
              let g = FloatType(bytes[y * bytesPerRow + x * 4 + 1]) * 2 / 255
              let b = FloatType(bytes[y * bytesPerRow + x * 4 + 2]) * 2 / 255
              fp16[y * imageWidth * 3 + x * 3] = min(r * invAlpha, 2) - 1
              fp16[y * imageWidth * 3 + x * 3 + 1] = min(g * invAlpha, 2) - 1
              fp16[y * imageWidth * 3 + x * 3 + 2] = min(b * invAlpha, 2) - 1
            }
          } else {
            hasNonTransparent = true
            fp16[y * imageWidth * 3 + x * 3] =
              FloatType(bytes[y * bytesPerRow + x * 4]) * 2 / 255 - 1
            fp16[y * imageWidth * 3 + x * 3 + 1] =
              FloatType(bytes[y * bytesPerRow + x * 4 + 1]) * 2 / 255 - 1
            fp16[y * imageWidth * 3 + x * 3 + 2] =
              FloatType(bytes[y * bytesPerRow + x * 4 + 2]) * 2 / 255 - 1
          }
        }
      }
    }
    var newBinaryMask: Tensor<UInt8>? = nil
    if hasTransparent && hasNonTransparent {
      // Create binaryMask.
      var binaryMask = Tensor<UInt8>(.CPU, .NC(bitmapContext.height, bitmapContext.width))
      binaryMask.withUnsafeMutableBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for y in 0..<imageHeight {
          for x in 0..<imageWidth {
            // 3 is to retain everything, 1 is to not.
            let alpha = bytes[y * bytesPerRow + x * 4 + 3]
            let flag: UInt8 = alpha != 255 ? 1 : 3
            u8[y * imageWidth + x] = (flag | (alpha & 0xf8))
          }
        }
      }
      newBinaryMask = binaryMask
    }
    return (tensor, newBinaryMask, hasTransparent)
  }

  public static func configuration(
    from imageProperties: [String: Any]?, configuration: GenerationConfiguration
  ) -> (
    String?, String?, GenerationConfiguration?
  ) {
    var savedPositivePrompt: String?
    var savedNegativePrompt: String?
    var previousConfiguration: GenerationConfiguration?
    if let imageProperties = imageProperties {
      if let imageData = imageProperties[kCGImagePropertyExifDictionary as String]
        as? [String: Any],
        let userComments = imageData[kCGImagePropertyExifUserComment as String] as? String,
        let commentsJsonData = userComments.data(using: .utf8),
        let commentsJsonDictionary = try? JSONSerialization.jsonObject(with: commentsJsonData)
          as? [String: Any]
      {
        var configurationBuilder = GenerationConfigurationBuilder(from: configuration)

        if let positivePrompt = commentsJsonDictionary["c"] as? String {
          savedPositivePrompt = positivePrompt
        }

        if let negativePrompt = commentsJsonDictionary["uc"] as? String {
          savedNegativePrompt = negativePrompt
        }

        if let steps = commentsJsonDictionary["steps"] as? UInt32 {
          configurationBuilder.steps = steps
        }

        if let sampler = commentsJsonDictionary["sampler"] as? String, !sampler.isEmpty {
          configurationBuilder.sampler = SamplerType(from: sampler)
        }

        if let guidanceScale = commentsJsonDictionary["scale"] as? Float32 {
          configurationBuilder.guidanceScale = guidanceScale
        }

        if let seed = commentsJsonDictionary["seed"] as? Int64 {
          configurationBuilder.seed = UInt32(seed)
        }

        if let size = commentsJsonDictionary["size"] as? String, !size.isEmpty {
          let components = size.split(separator: "x").compactMap { Int($0) }
          if components.count == 2 {
            var width = components[0]
            var height = components[1]

            let maxSupportLength = Int(DeviceCapability.maxSupportScale() * 64)
            if width > maxSupportLength || height > maxSupportLength {
              if width > height {
                height = maxSupportLength * height / width
                width = maxSupportLength
              } else {
                width = maxSupportLength * width / height
                height = maxSupportLength
              }
            }
            configurationBuilder.startWidth = UInt16(width / 64)
            configurationBuilder.startHeight = UInt16(height / 64)
          }
        }

        if let model = commentsJsonDictionary["model"] as? String {
          configurationBuilder.model = model
        }

        if let strength = commentsJsonDictionary["strength"] as? Float {
          configurationBuilder.strength = strength
        }

        if let seedModeString = commentsJsonDictionary["seed_mode"] as? String {
          configurationBuilder.seedMode = SeedMode(from: seedModeString)
        }

        if let upscaler = commentsJsonDictionary["upscaler"] as? String,
          UpscalerZoo.isModelDownloaded(upscaler)
        {
          configurationBuilder.upscaler = upscaler
        }

        if let clipSkip = commentsJsonDictionary["clip_skip"] as? UInt32 {
          configurationBuilder.clipSkip = clipSkip
        }

        if let imageGuidance = commentsJsonDictionary["image_guidance"] as? Float32 {
          configurationBuilder.imageGuidanceScale = imageGuidance
        }

        if let hiresFix = commentsJsonDictionary["hires_fix"] as? Bool {
          configurationBuilder.hiresFix = hiresFix
        }

        if let hiresFixSize = commentsJsonDictionary["first_stage_size"] as? String {
          let components = hiresFixSize.split(separator: "x").compactMap { UInt16($0) }
          if components.count == 2 {
            configurationBuilder.hiresFixStartWidth = components[0] / 64
            configurationBuilder.hiresFixStartHeight = components[1] / 64
          }
        }

        if let hiresFixStrength = commentsJsonDictionary["second_stage_strength"] as? Float32 {
          configurationBuilder.hiresFixStrength = hiresFixStrength
        }

        if let clipWeight = commentsJsonDictionary["clip_weight"] as? Float32 {
          configurationBuilder.clipWeight = clipWeight
        }

        if let imagePriorSteps = commentsJsonDictionary["image_prior_steps"] as? UInt32 {
          configurationBuilder.imagePriorSteps = imagePriorSteps
        }

        if let negativePromptImagePrior = commentsJsonDictionary[
          "negative_prompt_for_image_prior"] as? Bool
        {
          configurationBuilder.negativePromptForImagePrior = negativePromptImagePrior
        }

        if let refinerModel = commentsJsonDictionary["refiner"] as? String, !refinerModel.isEmpty {
          configurationBuilder.refinerModel = refinerModel
        }

        if let refinerStart = commentsJsonDictionary["refiner_start"] as? Float32 {
          configurationBuilder.refinerStart = refinerStart
        }

        if let targetSize = commentsJsonDictionary["target_size"] as? String {
          let components = targetSize.split(separator: "x").compactMap { UInt32($0) }
          if components.count == 2 {
            configurationBuilder.targetImageWidth = components[0]
            configurationBuilder.targetImageHeight = components[1]
          }
        }

        if let cropTop = commentsJsonDictionary["crop_top"] as? Int32 {
          configurationBuilder.cropTop = cropTop
        }

        if let cropLeft = commentsJsonDictionary["crop_left"] as? Int32 {
          configurationBuilder.cropLeft = cropLeft
        }

        if let originalSize = commentsJsonDictionary["original_size"] as? String {
          let components = originalSize.split(separator: "x").compactMap { UInt32($0) }
          if components.count == 2 {
            configurationBuilder.originalImageWidth = components[0]
            configurationBuilder.originalImageHeight = components[1]
          }
        }

        if let aestheticScore = commentsJsonDictionary["aesthetic_score"] as? Int64 {
          configurationBuilder.aestheticScore = Float(aestheticScore)
        }

        if let negtiveAestheticScore = commentsJsonDictionary["negative_aesthetic_score"]
          as? Float
        {
          configurationBuilder.negativeAestheticScore = negtiveAestheticScore
        }

        if let zeroNegativePrompt = commentsJsonDictionary["zero_negative_prompt"] as? Bool {
          configurationBuilder.zeroNegativePrompt = zeroNegativePrompt
        }

        if let negativeOriginalSize = commentsJsonDictionary["negative_original_size"] as? String {
          let components = negativeOriginalSize.split(separator: "x").compactMap { UInt32($0) }
          if components.count == 2 {
            configurationBuilder.negativeOriginalImageWidth = components[0]
            configurationBuilder.negativeOriginalImageHeight = components[1]
          }
        }

        if let loraArray = commentsJsonDictionary["lora"] as? [[String: Any]] {
          let loras = loraArray.compactMap { (lora) -> DataModels.LoRA? in
            if let file = lora["model"] as? String, LoRAZoo.isModelDownloaded(file),
              let weight = lora["weight"] as? Float
            {
              return DataModels.LoRA(file: file, weight: weight)
            }
            return nil
          }
          configurationBuilder.loras = loras
        }

        if let controlsArray = commentsJsonDictionary["control"] as? [[String: Any]] {
          let controls = controlsArray.compactMap { (control) -> DataModels.Control? in
            if let file = control["file"] as? String,
              let specification = ControlNetZoo.specificationForModel(file),
              ControlNetZoo.isModelDownloaded(specification),
              let weight = control["weight"] as? Float32,
              let guidanceStart = control["guidance_start"] as? Float32,
              let guidanceEnd = control["guidance_end"] as? Float32,
              let noPrompt = control["no_prompt"] as? Bool
            {
              var controlObject = DataModels.Control(
                file: file, weight: weight, guidanceStart: guidanceStart,
                guidanceEnd: guidanceEnd, noPrompt: noPrompt)
              if let globalAveragePooling = control["global_average_pooling"] as? Bool {
                controlObject.globalAveragePooling = globalAveragePooling
              }
              return controlObject
            }
            return nil
          }
          configurationBuilder.controls = controls
        }

        previousConfiguration = configurationBuilder.build()
      }
    }
    return (savedPositivePrompt, savedNegativePrompt, previousConfiguration)
  }
}
