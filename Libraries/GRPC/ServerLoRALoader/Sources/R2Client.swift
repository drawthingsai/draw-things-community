import Crypto  // From swift-crypto package
import Foundation

#if os(Linux)
  import FoundationNetworking
#endif

public struct R2Client: Sendable {
  public enum Error: Swift.Error {
    case invalidURL
    case missingData
    case invalidHTTPResponse
    case httpError(_: Int)
  }

  final class ObjCResponder: NSObject {
    var lastUpdated: Date? = nil
    var task: URLSessionDataTask? = nil
    let taskLock = DispatchQueue(label: "task.lock")
    var downloadTask: DownloadTask? = nil
    var index: Int = 0
    let progress: (Int64, Int64, Int) -> Void
    var totalBytesExpectedToWrite: Int64 = 0
    var data: Data = Data()
    init(
      progress: @escaping (Int64, Int64, Int) -> Void
    ) {
      self.progress = progress
    }
  }

  struct DownloadTask: Sendable {
    let client: R2Client
    let url: URL
    let session: URLSession
    let key: String
    let index: Int
    var completion: (@Sendable (Data?, URLResponse?, (any Swift.Error)?) -> Void)
    private let objCResponder: ObjCResponder
    init(
      client: R2Client, url: URL, session: URLSession, key: String, index: Int,
      objCResponder: ObjCResponder,
      completion: @escaping (@Sendable (Data?, URLResponse?, (any Swift.Error)?) -> Void)
    ) {
      self.client = client
      self.url = url
      self.session = session
      self.key = key
      self.index = index
      self.objCResponder = objCResponder
      self.completion = completion
    }
    func cancel() {
      objCResponder.cancel()
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
  func downloadObject(
    key: String, session: URLSession, objCResponder: ObjCResponder,
    completion: @escaping (Result<Data, Swift.Error>) -> Void
  )
    -> DownloadTask?
  {
    // Create the request URL
    guard let url = URL(string: "\(endpoint)/\(bucket)/\(key)") else {
      completion(.failure(Error.invalidURL))
      return nil
    }
    if self.debug {
      print("Attempting to download from URL: \(url.absoluteString)")
    }

    let downloadTask = DownloadTask(
      client: self, url: url, session: session, key: key, index: objCResponder.index,
      objCResponder: objCResponder
    ) { data, response, error in
      if let error = error {
        completion(.failure(error))
        return
      }

      guard let data = data else {
        completion(.failure(Error.missingData))
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

        let errorData = data
        if let responseString = String(data: errorData, encoding: .utf8) {
          if self.debug {
            print("Error response body: \(responseString)")
          }
          errorMessage += " - Response: \(responseString)"
        }

        completion(.failure(Error.httpError(httpResponse.statusCode)))
        return
      }

      // Return the temporary URL for the downloaded file
      completion(.success(data))
    }

    // Create a signed request
    var request = URLRequest(url: url)
    request.httpMethod = "GET"

    // Increase timeout for larger files
    request.timeoutInterval = 300  // 5 minutes

    // Generate signature (AWS Signature V4)
    signRequest(&request, key: key)

    // Print request details for debugging
    if self.debug {
      print("Request headers: \(request.allHTTPHeaderFields ?? [:])")
    }

    let task = session.dataTask(with: request)
    objCResponder.task = task
    objCResponder.downloadTask = downloadTask
    task.resume()

    return downloadTask
  }

  private func signRequest(_ request: inout URLRequest, key: String) {
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

extension R2Client.DownloadTask {
  func isTransient(_ e: NSError) -> Bool {
    if e.domain == NSURLErrorDomain {
      let code = URLError.Code(rawValue: e.code)
      switch code {
      case .timedOut, .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed,
        .networkConnectionLost, .notConnectedToInternet:
        return true
      default: break
      }
    }
    if e.domain == NSPOSIXErrorDomain, e.code == 54 || e.code == 32 { return true }  // ECONNRESET/EPIPE
    return false
  }

  func resume(data: Data?) {
    var request = URLRequest(url: url)
    request.httpMethod = "GET"

    // Increase timeout for larger files
    request.timeoutInterval = 300  // 5 minutes

    // Generate signature (AWS Signature V4)
    client.signRequest(&request, key: key)

    // Print request details for debugging
    if client.debug {
      print("Request headers: \(request.allHTTPHeaderFields ?? [:])")
    }
    if let data = data {
      request.addValue("bytes=\(data.count)-", forHTTPHeaderField: "Range")
    }
    let task = session.dataTask(with: request)
    // Since cancel can be called from anywhere, need to protect the access.
    objCResponder.taskLock.sync {
      objCResponder.task = task
    }
    objCResponder.data = data ?? Data()
    task.resume()
  }
}

extension R2Client.ObjCResponder {
  func cancel() {
    // Protect access to task, cancel can be called from anywhere.
    taskLock.sync {
      task?.cancel()
      task = nil
    }
  }
}

extension R2Client.ObjCResponder: URLSessionDataDelegate {
  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
    self.data.append(data)
    let date = Date()
    let timeElapsed = lastUpdated.map { date.timeIntervalSince($0) } ?? 1
    guard timeElapsed >= 0.25 else { return }
    progress(Int64(self.data.count), totalBytesExpectedToWrite, index)
    lastUpdated = date
  }

  func urlSession(
    _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
  ) {
    guard let downloadTask = downloadTask else { return }
    guard let httpResponse = response as? HTTPURLResponse else {
      completionHandler(.cancel)
      downloadTask.completion(nil, response, R2Client.Error.invalidHTTPResponse)
      self.downloadTask = nil
      return
    }
    // Check for valid response codes
    switch httpResponse.statusCode {
    case 200, 206:  // 200 OK or 206 Partial Content
      let contentLength =
        httpResponse.value(forHTTPHeaderField: "Content-Length")
        .flatMap { Int64($0) } ?? 0
      totalBytesExpectedToWrite = Int64(data.count) + contentLength
      completionHandler(.allow)
    case 416:  // Range Not Satisfiable - file already complete
      completionHandler(.cancel)
      downloadTask.completion(nil, response, nil)
      self.downloadTask = nil
      return
    case 500...599:
      completionHandler(.cancel)
      downloadTask.completion(nil, response, R2Client.Error.httpError(httpResponse.statusCode))
      self.downloadTask = nil
      return
    default:
      completionHandler(.cancel)
      downloadTask.completion(nil, response, R2Client.Error.httpError(httpResponse.statusCode))
      self.downloadTask = nil
      return
    }
  }

  func urlSession(
    _ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?
  ) {
    guard let downloadTask = downloadTask else { return }
    guard let error = error else { return }
    let isTransientError: Bool
    let nsError = error as NSError
    if downloadTask.isTransient(nsError) {
      isTransientError = true
    } else {
      isTransientError = false
    }
    if !isTransientError {
      lastUpdated = nil
      downloadTask.completion(data, task.response, error)
      self.data = Data()
      self.downloadTask = nil
      return
    }
    // At the end, cancel the download request.
    if isTransientError {
      let resumeData = data
      // Restart in 200ms.
      DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(200)) {
        downloadTask.resume(data: resumeData)
      }
    }
  }
}
