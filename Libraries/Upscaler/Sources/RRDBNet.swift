import NNC

func ResidualDenseBlock(prefix: String, numberOfFeatures: Int, numberOfGrowChannels: Int) -> Model {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let x1 = conv1(x).leakyReLU(negativeSlope: 0.2)
  let x01 = Functional.concat(axis: 1, x, x1)
  let conv2 = Convolution(
    groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let x2 = conv2(x01).leakyReLU(negativeSlope: 0.2)
  let x012 = Functional.concat(axis: 1, x01, x2)
  let conv3 = Convolution(
    groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let x3 = conv3(x012).leakyReLU(negativeSlope: 0.2)
  let x0123 = Functional.concat(axis: 1, x012, x3)
  let conv4 = Convolution(
    groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let x4 = conv4(x0123).leakyReLU(negativeSlope: 0.2)
  let x01234 = Functional.concat(axis: 1, x0123, x4)
  let conv5 = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let x5 = conv5(x01234)
  let out = 0.2 * x5 + x
  return Model([x], [out])
}

func RRDB(prefix: String, numberOfFeatures: Int, numberOfGrowChannels: Int) -> Model {
  let x = Input()
  let rdb1 = ResidualDenseBlock(
    prefix: "\(prefix).rdb1", numberOfFeatures: numberOfFeatures,
    numberOfGrowChannels: numberOfGrowChannels)
  var out = rdb1(x)
  let rdb2 = ResidualDenseBlock(
    prefix: "\(prefix).rdb2", numberOfFeatures: numberOfFeatures,
    numberOfGrowChannels: numberOfGrowChannels)
  out = rdb2(out)
  let rdb3 = ResidualDenseBlock(
    prefix: "\(prefix).rdb3", numberOfFeatures: numberOfFeatures,
    numberOfGrowChannels: numberOfGrowChannels)
  out = 0.2 * rdb3(out) + x
  return Model([x], [out])
}

public func RRDBNet(
  numberOfOutputChannels: Int, numberOfFeatures: Int, numberOfBlocks: Int, numberOfGrowChannels: Int
) -> Model {
  let x = Input()
  let convFirst = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = convFirst(x)
  let feat = out
  for i in 0..<numberOfBlocks {
    let rrdb = RRDB(
      prefix: "body.\(i)", numberOfFeatures: numberOfFeatures,
      numberOfGrowChannels: numberOfGrowChannels)
    out = rrdb(out)
  }
  let convBody = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convBody(out)
  out = feat + out
  let convUp1 = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convUp1(Upsample(.nearest, widthScale: 2, heightScale: 2)(out)).leakyReLU(
    negativeSlope: 0.2)
  let convUp2 = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convUp2(Upsample(.nearest, widthScale: 2, heightScale: 2)(out)).leakyReLU(
    negativeSlope: 0.2)
  let convHr = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convHr(out).leakyReLU(negativeSlope: 0.2)
  let convLast = Convolution(
    groups: 1, filters: numberOfOutputChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convLast(out)
  return Model([x], [out])
}
