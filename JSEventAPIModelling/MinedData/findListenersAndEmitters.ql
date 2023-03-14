import javascript

/** Gets the name of a method that registers an event listener. */
string listenMethod() { result = EventEmitter::on() }

/** Gets the name of a method that emits an event. */
string emitMethod() { result = "emit" }

/**
 * Finds the original event emitter on which `call` listens to or emits an event, keeping
 * track of inter-procedural state in `t`.
 */
DataFlow::SourceNode getEmitter(DataFlow::MethodCallNode call, DataFlow::TypeBackTracker t) {
  t.start() and
  call.getMethodName() in [listenMethod(), emitMethod()] and
  result = call.getReceiver().getALocalSource()
  or
  // propagate through chaining methods that return `this`
  exists(DataFlow::MethodCallNode chain |
    chain = getEmitter(call, t.continue()) and
    chain.getMethodName() = EventEmitter::chainableMethod() and
    result = call.getReceiver().getALocalSource()
  )
  or
  exists(DataFlow::TypeBackTracker t2 | result = getEmitter(call, t2).backtrack(t2, t))
}

/**
 * Finds the original event emitter on which `call` listens to or emits an event.
 */
DataFlow::SourceNode getEmitter(DataFlow::MethodCallNode call) {
  result = getEmitter(call, DataFlow::TypeBackTracker::end())
}

/**
 * Gets a method call that registers a listener for `eventName` on the given API node `ap`.
 */
DataFlow::MethodCallNode eventRegistration(API::Node ap, string eventName) {
  result.getMethodName() = listenMethod() and
  result.getArgument(0).mayHaveStringValue(eventName) and
  getEmitter(result) = ap.getAnImmediateUse()
}

/**
 * Gets a method call that emits event `eventName` on the given API node `ap`.
 */
DataFlow::MethodCallNode eventEmission(API::Node ap, string eventName) {
  result.getMethodName() = emitMethod() and
  result.getArgument(0).mayHaveStringValue(eventName) and
  getEmitter(result) = ap.getAnImmediateUse()
}

from string type, API::Node nd, string ap, string pkgName, string eventName, int cnt
where
  (
    type = "listen" and cnt = strictcount(eventRegistration(nd, eventName))
    or
    type = "emit" and cnt = strictcount(eventEmission(nd, eventName))
  ) and
  ap = nd.getPath() and
  pkgName = ap.regexpCapture(".*\\(module ([^)]+)\\).*", 1)
select type, ap, pkgName, eventName, cnt
