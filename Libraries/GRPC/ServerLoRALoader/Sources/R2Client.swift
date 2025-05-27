import Crypto  // From swift-crypto package
import Foundation

#if os(Linux)
  import FoundationNetworking
#endif

public struct R2Client {
  public enum Error: Swift.Error {
    case invalidURL
    case missingFileURL
    case invalidHTTPResponse
    case httpError(_: Int)
  }

  final class ObjCResponder: NSObject {
    let progress: (Int64, Int64) -> Void
    init(progress: @escaping (Int64, Int64) -> Void) {
      self.progress = progress
    }
  }

  private let accessKey: String
  private let secret: String
  private let endpoint: String
  private let bucket: String
  private let debug: Bool

  public init(
    accessKey: String, secret: String, endpoint: String, bucket: String, debug: Bool = false
  ) {
    self.accessKey = accessKey
    self.secret = secret
    self.endpoint = endpoint
    self.bucket = bucket
    self.debug = debug
  }

  // Update your R2Client.downloadObject method with better error handling
  public func downloadObject(
    key: String, session: URLSession, progress: @escaping (Int64, Int64) -> Void,
    completion: @escaping (Result<URL, Swift.Error>) -> Void
  )
    -> URLSessionDownloadTask?
  {
    // Create the request URL
    guard let url = URL(string: "\(endpoint)/\(bucket)/\(key)") else {
      completion(.failure(Error.invalidURL))
      return nil
    }
    if self.debug {
      print("Attempting to download from URL: \(url.absoluteString)")
    }

    // Create a signed request
    var request = URLRequest(url: url)
    request.httpMethod = "GET"

    // Increase timeout for larger files
    request.timeoutInterval = 300  // 5 minutes

    // Generate signature (AWS Signature V4)
    signRequest(&request, key: key, date: "")

    // Print request details for debugging
    if self.debug {
      print("Request headers: \(request.allHTTPHeaderFields ?? [:])")
    }

    let objCResponder = ObjCResponder(progress: progress)
    // Execute the download task
    let task = session.downloadTask(with: request) { tempUrl, response, error in
      withExtendedLifetime(objCResponder) {}  // We hold it until the request is done.
      if let error = error {
        completion(.failure(error))
        return
      }

      guard let tempUrl = tempUrl else {
        completion(.failure(Error.missingFileURL))
        return
      }

      guard let httpResponse = response as? HTTPURLResponse else {
        completion(.failure(Error.invalidHTTPResponse))
        return
      }
      if self.debug {
        print("HTTP Status: \(httpResponse.statusCode)")
        print("Response headers: \(httpResponse.allHeaderFields)")
      }

      if !(200...299).contains(httpResponse.statusCode) {
        // Extract error message from response if possible
        var errorMessage = "HTTP Error: \(httpResponse.statusCode)"

        do {
          let errorData = try Data(contentsOf: tempUrl)
          if let responseString = String(data: errorData, encoding: .utf8) {
            if self.debug {
              print("Error response body: \(responseString)")
            }
            errorMessage += " - Response: \(responseString)"
          }
        } catch {
          if self.debug {
            print("Could not read error data from temporary file")
          }
        }

        completion(.failure(Error.httpError(httpResponse.statusCode)))
        return
      }

      // Return the temporary URL for the downloaded file
      completion(.success(tempUrl))
    }
    task.delegate = objCResponder

    task.resume()

    return task
  }

  private func signRequest(_ request: inout URLRequest, key: String, date: String) {
    // Create the current date in UTC
    let currentDate = Date()

    // Create the date stamp in the format YYYYMMDD
    let dateStampFormatter = DateFormatter()
    dateStampFormatter.dateFormat = "yyyyMMdd"
    dateStampFormatter.timeZone = TimeZone(abbreviation: "UTC")
    let dateStamp = dateStampFormatter.string(from: currentDate)

    // Format date as YYYYMMDD'T'HHMMSS'Z' (AWS format without separators)
    let amzDateFormatter = DateFormatter()
    amzDateFormatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
    amzDateFormatter.timeZone = TimeZone(abbreviation: "UTC")
    let amzDate = amzDateFormatter.string(from: currentDate)

    if self.debug {
      print("Using date stamp: \(dateStamp)")
      print("Using amz date: \(amzDate)")
    }

    // Hash empty payload with SHA256
    let emptyData = Data()
    let payloadHash = SHA256.hash(data: emptyData)
    let hashedPayload = payloadHash.map { String(format: "%02x", $0) }.joined()

    // Create canonical request
    let httpMethod = request.httpMethod ?? "GET"
    let canonicalURI = "/\(bucket)/\(key)"
    let host = URL(string: endpoint)!.host!
    let canonicalHeaders =
      "host:\(host)\nx-amz-content-sha256:\(hashedPayload)\nx-amz-date:\(amzDate)\n"
    let signedHeaders = "host;x-amz-content-sha256;x-amz-date"

    let canonicalRequest = [
      httpMethod,
      canonicalURI,
      "",  // canonicalQueryString (empty for this example)
      canonicalHeaders,
      signedHeaders,
      hashedPayload,
    ].joined(separator: "\n")

    // Create string to sign
    let algorithm = "AWS4-HMAC-SHA256"
    let scope = "\(dateStamp)/auto/s3/aws4_request"

    let canonicalRequestData = canonicalRequest.data(using: .utf8)!
    let canonicalRequestHash = SHA256.hash(data: canonicalRequestData)
    let canonicalRequestHashHex = canonicalRequestHash.map { String(format: "%02x", $0) }.joined()

    let stringToSign = [
      algorithm,
      amzDate,
      scope,
      canonicalRequestHashHex,
    ].joined(separator: "\n")

    // Calculate signature
    func hmacSHA256(key: [UInt8], data: [UInt8]) -> [UInt8] {
      let hmac = HMAC<SHA256>.authenticationCode(for: data, using: SymmetricKey(data: key))
      return Array(hmac)
    }

    // Convert strings to byte arrays for HMAC operations
    let kSecretBytes = Array("AWS4\(secret)".utf8)
    let dateStampBytes = Array(dateStamp.utf8)
    let regionBytes = Array("auto".utf8)
    let serviceBytes = Array("s3".utf8)
    let requestBytes = Array("aws4_request".utf8)
    let stringToSignBytes = Array(stringToSign.utf8)

    let kDate = hmacSHA256(key: kSecretBytes, data: dateStampBytes)
    let kRegion = hmacSHA256(key: kDate, data: regionBytes)
    let kService = hmacSHA256(key: kRegion, data: serviceBytes)
    let kSigning = hmacSHA256(key: kService, data: requestBytes)
    let signature = hmacSHA256(key: kSigning, data: stringToSignBytes)
    let signatureHex = signature.map { String(format: "%02x", $0) }.joined()

    // Add authorization header
    let authorization =
      "\(algorithm) Credential=\(accessKey)/\(scope), SignedHeaders=\(signedHeaders), Signature=\(signatureHex)"

    // Update the request with the new headers
    request.setValue(amzDate, forHTTPHeaderField: "x-amz-date")
    request.setValue(hashedPayload, forHTTPHeaderField: "x-amz-content-sha256")
    request.setValue(authorization, forHTTPHeaderField: "Authorization")

    if self.debug {
      print("Final Authorization header: \(authorization)")
    }
  }
}

extension R2Client.ObjCResponder: URLSessionDownloadDelegate {
  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) {
    // Do nothing.
  }
  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
  ) {
    progress(totalBytesWritten, totalBytesExpectedToWrite)
  }
}
