//===----------------------------------------------------------------------===//
//
// This source file is part of the SwiftNIO open source project
//
// Copyright (c) 2017-2021 Apple Inc. and the SwiftNIO project authors
// Licensed under Apache License v2.0
//
// See LICENSE.txt for license information
// See CONTRIBUTORS.txt for the list of SwiftNIO project authors
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#if canImport(Network)
  import Dispatch
  import Foundation
  import Network

  import NIOCore
  import NIOConcurrencyHelpers

  /// An `EventLoop` that interacts with `DispatchQoS` to help schedule upcoming work.
  ///
  /// `EventLoop`s that implement ``QoSEventLoop`` can interact with `Dispatch` to propagate information
  /// about the QoS required for a specific task block. This allows tasks to be dispatched onto an
  /// event loop with a different priority than the majority of tasks on that loop.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  public protocol QoSEventLoop: EventLoop {
    /// Submit a given task to be executed by the `EventLoop` at a given `qos`.
    func execute(qos: DispatchQoS, _ task: @escaping () -> Void)

    /// Schedule a `task` that is executed by this `NIOTSEventLoop` after the given amount of time at the
    /// given `qos`.
    func scheduleTask<T>(in time: TimeAmount, qos: DispatchQoS, _ task: @escaping () throws -> T)
      -> Scheduled<T>
  }

  /// The lifecycle state of a given event loop.
  ///
  /// Event loops have the ability to be shut down, and not restarted. When a loop is active it will accept
  /// new registrations, and new scheduled work items. When a loop is shutting down it will no longer accept
  /// new registrations, but it will continue to accept new scheduled work items. When a loop is closed, it
  /// will accept neither new registrations nor new scheduled work items, but it will continue to process
  /// the queue until it has drained.
  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  fileprivate enum LifecycleState {
    case active
    case closing
    case closed
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  internal class NIOTSEventLoop: QoSEventLoop {
    private let loop: DispatchQueue
    private let taskQueue: DispatchQueue
    private let inQueueKey: DispatchSpecificKey<UUID>
    private let loopID: UUID
    private let defaultQoS: DispatchQoS

    /// All the channels registered to this event loop.
    ///
    /// This array does two jobs. Firstly, it ensures that these channels stay alive for as long as
    /// they are registered: they cannot leak. Secondly, it provides a notification mechanism for
    /// this event loop to deliver them specific kinds of events: in particular, to request that
    /// they quiesce or shut themselves down.
    private var registeredChannels: [ObjectIdentifier: Channel] = [:]

    /// The state of this event loop.
    private var state = LifecycleState.active

    /// Whether this event loop is accepting new channels.
    private var open: Bool {
      return self.state == .active
    }

    /// Returns whether the currently executing code is on the event loop.
    ///
    /// Due to limitations in Dispatch's API, this check is pessimistic: there are circumstances where a perfect
    /// implementation *could* return `true`, but this version will be unable to prove that and will return `false`.
    /// If you need to write an assertion about being in the event loop that must be correct, use SwiftNIO 1.11 or
    /// later and call `preconditionInEventLoop` and `assertInEventLoop`.
    ///
    /// Further detail: The use of `DispatchQueue.sync(execute:)` to submit a block to a queue synchronously has the
    /// effect of creating a state where the currently executing code is on two queues simultaneously - the one which
    /// submitted the block, and the one on which the block runs. If another synchronous block is dispatched to a
    /// third queue, that block is effectively running all three at once. Unfortunately, libdispatch maintains only
    /// one "current" queue at a time as far as `DispatchQueue.getSpecific(key:)` is concerned, and it's always the
    /// one actually executing code at the time. Therefore the queue belonging to the original event loop can't be
    /// detected using its queue-specific data. No alternative API for the purpose exists (aside from assertions via
    /// `dispatchPrecondition(condition:)`). Under these circumstances, `inEventLoop` will incorrectly be `false`,
    /// even though the current code _is_ actually on the loop's queue. The only way to avoid this is to ensure no
    /// callers ever use synchronous dispatch (which is impossible to enforce), or to hope that a future version of
    /// libdispatch will provide a solution.
    public var inEventLoop: Bool {
      return DispatchQueue.getSpecific(key: self.inQueueKey) == self.loopID
    }

    public init(qos: DispatchQoS) {
      self.loop = DispatchQueue(
        label: "nio.transportservices.eventloop.loop", qos: qos, autoreleaseFrequency: .workItem)
      self.taskQueue = DispatchQueue(
        label: "nio.transportservices.eventloop.taskqueue", target: self.loop)
      self.loopID = UUID()
      self.inQueueKey = DispatchSpecificKey()
      self.defaultQoS = qos
      loop.setSpecific(key: inQueueKey, value: self.loopID)
    }

    public func execute(_ task: @escaping () -> Void) {
      self.execute(qos: self.defaultQoS, task)
    }

    public func execute(qos: DispatchQoS, _ task: @escaping () -> Void) {
      // Ideally we'd not accept new work while closed. Sadly, that's not possible with the current APIs for this.
      self.taskQueue.async(qos: qos, execute: task)
    }

    public func scheduleTask<T>(deadline: NIODeadline, _ task: @escaping () throws -> T)
      -> Scheduled<T>
    {
      return self.scheduleTask(deadline: deadline, qos: self.defaultQoS, task)
    }

    public func scheduleTask<T>(
      deadline: NIODeadline, qos: DispatchQoS, _ task: @escaping () throws -> T
    ) -> Scheduled<T> {
      let p: EventLoopPromise<T> = self.makePromise()

      // Dispatch support for cancellation exists at the work-item level, so we explicitly create one here.
      // We set the QoS on this work item and explicitly enforce it when the block runs.
      let timerSource = DispatchSource.makeTimerSource(queue: self.taskQueue)
      timerSource.schedule(deadline: DispatchTime(uptimeNanoseconds: deadline.uptimeNanoseconds))
      timerSource.setEventHandler(qos: qos, flags: .enforceQoS) {
        guard self.state != .closed else {
          p.fail(EventLoopError.shutdown)
          return
        }
        do {
          p.succeed(try task())
        } catch {
          p.fail(error)
        }
      }
      timerSource.resume()

      // Create a retain cycle between the future and the timer source. This will be broken when the promise is
      // completed by the event handler and this callback is run.
      p.futureResult.whenComplete { _ in
        timerSource.cancel()
      }

      return Scheduled(
        promise: p,
        cancellationTask: {
          timerSource.cancel()
        })
    }

    public func scheduleTask<T>(in time: TimeAmount, _ task: @escaping () throws -> T) -> Scheduled<
      T
    > {
      return self.scheduleTask(in: time, qos: self.defaultQoS, task)
    }

    public func scheduleTask<T>(
      in time: TimeAmount, qos: DispatchQoS, _ task: @escaping () throws -> T
    ) -> Scheduled<T> {
      return self.scheduleTask(deadline: NIODeadline.now() + time, qos: qos, task)
    }

    public func shutdownGracefully(queue: DispatchQueue, _ callback: @escaping (Error?) -> Void) {
      self.closeGently().map {
        queue.async { callback(nil) }
      }.whenFailure { error in
        queue.async { callback(error) }
      }
    }

    @inlinable
    public func preconditionInEventLoop(file: StaticString, line: UInt) {
      dispatchPrecondition(condition: .onQueue(self.loop))
    }

    @inlinable
    public func preconditionNotInEventLoop(file: StaticString, line: UInt) {
      dispatchPrecondition(condition: .notOnQueue(self.loop))
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSEventLoop {
    /// Create a `DispatchQueue` to use for events on a given `Channel`.
    ///
    /// This `DispatchQueue` will be guaranteed to execute on this `EventLoop`, and
    /// so is safe to use concurrently with the rest of the event loop.
    internal func channelQueue(label: String, qos: DispatchQoS? = nil) -> DispatchQueue {
      // If a QoS override is not requested, use the default.
      let qos = qos ?? self.defaultQoS
      return DispatchQueue(label: label, qos: qos, target: self.loop)
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSEventLoop {
    internal func closeGently() -> EventLoopFuture<Void> {
      let p: EventLoopPromise<Void> = self.makePromise()
      self.taskQueue.async {
        guard self.open else {
          p.fail(EventLoopError.shutdown)
          return
        }

        // Ok, time to shut down.
        self.state = .closing

        // We need to tell all currently-registered channels to close.
        let futures: [EventLoopFuture<Void>] = self.registeredChannels.map { _, channel in
          channel.close(promise: nil)
          return channel.closeFuture.flatMapErrorThrowing { error in
            if let error = error as? ChannelError, error == .alreadyClosed {
              return ()
            } else {
              throw error
            }
          }
        }

        // The ordering here is important.
        // We must not transition into the closed state until *after* the caller has been notified that the
        // event loop is closed. Otherwise, this future is in real trouble, as if it needs to dispatch onto the
        // event loop it will be forbidden from doing so.
        let completionFuture = EventLoopFuture<Void>.andAllComplete(futures, on: self)
        completionFuture.cascade(to: p)
        completionFuture.whenComplete { (_: Result<Void, Error>) in
          self.state = .closed
        }
      }
      return p.futureResult
    }
  }

  @available(OSX 10.14, iOS 12.0, tvOS 12.0, watchOS 6.0, *)
  extension NIOTSEventLoop {
    /// Record a given channel with this event loop.
    internal func register(_ channel: Channel) throws {
      guard self.open else {
        throw EventLoopError.shutdown
      }

      channel.eventLoop.assertInEventLoop()
      self.registeredChannels[ObjectIdentifier(channel)] = channel
    }

    // We don't allow deregister to fail, as it doesn't make any sense.
    internal func deregister(_ channel: Channel) {
      channel.eventLoop.assertInEventLoop()
      let oldChannel = self.registeredChannels.removeValue(forKey: ObjectIdentifier(channel))
      assert(oldChannel != nil)
    }
  }
#endif
