import javascript
import externalData


/** Holds if `pkg` exports a function `fn` returning type `pkg.tp`. */
predicate exportedFn(string pkg, string fn, string tp) {
  pkg = "fs" and fn = "createReadStream" and tp = "ReadStream"
  or
  pkg = "fs" and fn = "createWriteStream" and tp = "WriteStream"
  or
  pkg = "net" and fn = "createServer" and tp = "Server"
  or
  pkg = "net" and fn = "connect" and tp = "Socket"
  or
  pkg = "net" and fn = "Socket" and tp = "Socket"
  or
  pkg = "child_process" and fn = "exec" and tp = "ChildProcess"
  or
  pkg = "child_process" and fn = "spawn" and tp = "ChildProcess"
  or
  pkg = "child_process" and fn = "fork" and tp = "ChildProcess"
  or
  pkg = "socket.io" and fn = "listen" and tp = "Server"
  or
  pkg = "tls" and fn = "connect" and tp = "TLSSocket"
  or
  pkg = "tls" and fn = "createServer" and tp = "Server"
  or
  pkg = "cluster" and fn = "fork" and tp = "Worker"
  or
  pkg = "readline" and fn = "createInterface" and tp = "Interface" // always make with createInterface
  or
  pkg = "http2" and fn = "createServer" and tp = "Http2Server"
  or
  pkg = "http2" and fn = "createSecureServer" and tp = "Http2SecureServer"
  or
  pkg = "http2" and fn = "connect" and tp = "ClientHttp2Session"
  or
  pkg = "repl" and fn = "start" and tp = "REPLServer"
  or
  // many classes can be invoked without `new`
  exportedClass(pkg, fn) and tp = fn
}

/** Holds if `pkg` exports a class `cls`. */
predicate exportedClass(string pkg, string cls) {
  pkg = "fs" and cls = "ReadStream"
  or
  pkg = "fs" and cls = "WriteStream"
  or
  pkg = "net" and cls = "Socket"
  or
  (pkg = "http" or pkg = "https") and cls = "ClientRequest"
  or
  (pkg = "http" or pkg = "https") and cls = "IncomingMessage"
  or
  pkg = "socket.io-client" and cls = "Manager"
  or
  pkg = "ws" and cls = "Server" // we can have websocket.Server, new websocket.Server
  or
  pkg = "tls" and cls = "Server" 
  or
  pkg = "tls" and cls = "TLSSocket"
  or
  pkg = "cluster" and cls = "Worker"
}

/** Holds if type `decl` has a field `f` of type `tp`. */
predicate field(string decl, string f, string tp) {
  decl = "http.IncomingMessage" and f = "socket" and tp = "net.Socket"
  or
  decl = "http.ServerResponse" and f = "socket" and tp = "net.Socket"
  or
  decl = "http.ClientRequest" and f = "socket" and tp = "net.Socket"
  or
  decl = "http.ClientRequest" and f = "connection" and tp = "net.Socket"
  or 
  decl = "child_process.ChildProcess" and f = "stderr" and tp = "stream.Readable"
  or
  decl = "child_process.ChildProcess" and f = "stdout" and tp = "stream.Readable"
  or
  decl = "child_process.ChildProcess" and f = "stdin" and tp = "stream.Writable"
  or
  decl = "socket.io.Server" and f = "sockets" and tp = "socket.io.Namespace"
  or
  decl = "process.Process" and f = "stderr" and tp = "net.Socket"
  or
  decl = "process.Process" and f = "stdin" and tp = "net.Socket" 
  or 
  decl = "process.Process" and f = "stdout" and tp = "net.Socket"
  or
  decl = "cluster.Cluster" and f = "worker" and tp = "cluster.Worker"
  or
  decl = "cluster.Worker" and f = "process" and tp = "child_process.ChildProcess"
  or
  decl = "send.SendStream" and f = "res" and tp = "http.ServerResponse"
  or
  decl = "send.SendStream" and f = "req" and tp = "http.ClientRequest"
  or
  (decl = "http2.Http2Session" or decl = "http2.Http2ServerRequest" or decl = "http2.Http2ServerResponse") and f = "socket" and (tp = "net.Socket" or tp = "tls.TLSSocket")
  or
  decl = "http2.Http2ServerResponse" and f = "connection" and (tp = "net.Socket" or tp = "tls.TLSSocket")
  or
  (decl = "http2.Http2Stream" or decl = "http2.ClientHttp2Stream" or decl = "http2.ServerHttp2Stream") and f = "session" and tp = "http2.Http2Session"
  or 
  (decl = "http2.Http2ServerRequest" or decl = "http2.Http2ServerResponse") and f = "stream" and tp = "http2.Http2Stream"
}

/** Holds if type `decl` has a method `m` returning values of type `tp`. */
predicate method(string decl, string m, string tp) {
  decl = "socket.io.Server" and
  (
    m = "serveClient" or
    m = "path" or
    m = "adapter" or
    m = "origins" or
    m = "attach" or
    m = "listen" or
    m = "bind" or
    m = "onconnection"
  ) and
  tp = decl
  or
  decl = "socket.io-client.Manager" and
  (
    m = "reconnection" or
    m = "reconnectionAttempt" or
    m = "reconnectionDelay" or
    m = "reconnectionDelayMax" or
    m = "timeout" or
    m = "open" or
    m = "connect"
  ) and
  tp = decl
  or
  decl = "socket.io.Server" and m = "of" and tp = "socket.io.Namespace"
  or
  decl = "socket.io.Namespace" and
  (m = "to" or m = "in") and
  tp = decl
  or
  decl = "socket.io.Socket" and
  (m = "send" or m = "emit") and
  tp = decl
  or
  hasEvent(decl, _) and
  (
    m = "on" or
    m = "once" or
    m = "removeListener" or
    m = "removeAllListeners" or
    m = "off" or
    m = "removeEventListener"
  ) and
  tp = decl
  or
  decl = "socket.io-client.Manager" and
  (
    m = "reconnection" or
    m = "reconnectionAttempt" or
    m = "reconnectionDelay" or
    m = "reconnectionDelayMax" or
    m = "timeout" or
    m = "open" or
    m = "connect"
  ) and
  tp = decl
  or
  decl = "socket.io-client.Manager" and
  m = "socket" and
  tp = "socket.io-client.Socket"
  or
  decl = "socket.io-client.Socket" and
  (m = "open" or m = "connect" or m = "send" or m = "emit") and
  tp = decl
  or
  decl = "ws.WebSocket" and
  (
    m = "close" or
    m = "handleUpgrade" or
    m = "shouldHandle"
  ) and
  tp = decl
  or
  decl = "ws.WebSocket" and
  (m = "ping" or m = "pong" or m = "send") and
  tp = decl
  or
  decl = "ws.WebSocket" and
  m = "createWebSocketStream" and
  tp = "ws.Duplex"
  or
  decl = "process.Process" and
  m = "dlopen" and
  tp = decl
  or
  (decl = "net.Server" or decl = "tls.Server") and 
  (
  	m = "listen" or
  	m = "close" or 
  	m = "ref" or
  	m = "unref" 
  ) and 
  tp = decl
  or
  (decl = "net.Socket" or decl = "tls.TLSSocket") and
  (
  	m = "connect" or
  	m = "destroy" or
  	m = "end" or
  	m = "pause" or
  	m = "ref" or
  	m = "resume" or
  	m = "setEncoding" or
  	m = "setKeepAlive" or 
  	m = "setNoDelay" or
  	m = "setTimeout" or 
  	m = "unref"
  ) and
  tp = decl
  or
  decl = "cluster.Worker" and 
  m = "disconnect" and
  tp = decl
  or
  decl = "cluster.Cluster" and
  m = "fork" and
  tp = "cluster.Worker" 
  or
  decl = "http2.ClientHttp2Session" and
  m = "request" and
  tp = "http2.ClientHttp2Stream"
  or
  (
    decl = "http2.Http2Server" or 
    decl = "http2.Http2SecureServer" or
    decl = "http2.Http2ServerRequest" or
    decl = "http2.Http2ServerResponse"
  ) and 
  m = "setTimeout" and
  tp = decl
  or
  decl = "http2.Http2ServerResponse" and 
  (m = "end" or m = "writeHead") and
  tp = decl
}

/** Infers the type of `accessPath`. */
pragma[noinline]
// bindingset[accessPath]
string typeOf(ExtAPString accessPath) {
  exists(string pkg, string fn, string tp | exportedFn(pkg, fn, tp) |
    accessPath = "(return (member " + fn + " (member exports (module " + pkg + "))))" and
    result = pkg + "." + tp
  )
  or
  exists(string pkg, string cls | exportedClass(pkg, cls) |
    accessPath = "(instance (member " + cls + " (member exports (module " + pkg + "))))" and
    result = pkg + "." + cls
  )
  or
  exists(string decl, string f, string tp, ExtAPString subPath | field(decl, f, tp) |
    typeOf(subPath) = decl and
    accessPath = "(member " + f + " " + subPath + ")" and
    result = tp //and
    // exists(ExtAPString extAccessPath | extAccessPath.matches("%" + accessPath + "%"))
    // any(DataPoint dp).getAccessPath().matches("%" + accessPath + "%")
  )
  or
  exists(string decl, string m, string tp, ExtAPString subPath | method(decl, m, tp) |
    typeOf(subPath) = decl and
    accessPath = "(return (member " + m + " " + subPath + "))" and
    result = tp //and
    // exists(ExtAPString extAccessPath | extAccessPath.matches("%" + accessPath + "%"))
    // any(DataPoint dp).getAccessPath().matches("%" + accessPath + "%")
  )
  or
  accessPath = "(parameter 0 (parameter 0 (member createServer (member exports (module net)))))" and
  result = "net.Socket"
  or
  accessPath = "(parameter -1 (parameter 1 (member connect (member exports (module net)))))" and 
  result = "net.Socket"
  or 
  exists(string http | http = "http" or http = "https" |
    accessPath = "(return (member get (member exports (module " + http + "))))" and
    result = "http.ClientRequest"
    or
    accessPath = "(return (member request (member exports (module " + http + "))))" and
    result = "http.ClientRequest"
    or
    accessPath = "(parameter 0 (parameter 0 (member createServer (member exports (module "
        + http + ")))))" and
    result = "http.IncomingMessage"
    or
    accessPath = "(parameter 0 (parameter 1 (member request (member exports (module " +
        http + ")))))" and
    result = "http.IncomingMessage"
    or
    accessPath = "(return (member createServer (member exports (module " + http + "))))" and
    result = "http.Server"
    or
    accessPath = "(parameter 0 (parameter 1 (member get (member exports (module " + http
        + ")))))" and
    result = "http.IncomingMessage"
    or
    accessPath = "(instance (member Agent (member exports (module " + http + "))))" and
    result = "http.Agent"
    or
    accessPath = "(return (member Agent (member exports (module " + http + "))))" and
    result = "http.Agent"
    or 
    accessPath = "(parameter -1 (member Server (member exports (module " + http + "))))" and 
    result = "http.Server"
  )
  or
  exists(string m |
    m = "createBrotliCompress" or
    m = "createBrotliDecompress" or
    m = "createDeflate" or
    m = "createDeflateRaw" or
    m = "createGunzip" or
    m = "createGzip" or
    m = "createInflate" or
    m = "createInflateRaw" or
    m = "createUnzip"
  |
    accessPath = "(return (member " + m + " (member exports (module zlib))))" and
    result = "stream.Duplex"
  )
  or
  accessPath = "(return (member exports (module socket.io)))" and
  result = "socket.io.Server"
  or
  accessPath = "(instance (member exports (module socket.io)))" and
  result = "socket.io.Server"
  or
  accessPath = "(instance (member listen (member exports (module socket.io))))" and
  result = "socket.io.Server"
  or
  accessPath = "(return (member exports (module socket.io-client)))" and
  result = "socket.io-client.Socket"
  or
  accessPath = "(return (member exports (module readable-stream)))" and
  result = "stream.Readable"
  or
  accessPath = "(parameter -1 (member pipe (instance (member exports (module readable-stream)))))" and 
  result = "stream.Readable"
  or
  accessPath = "(instance (member exports (module readable-stream)))" and
  result = "stream.Readable"
  or
  exists(string stream, string m, string c |
    (
      m = "Readable" or
      m = "Writable" or
      m = "Duplex"
    ) and
    (stream = "stream" or stream = "readable-stream") and
    (c = "return" or c = "instance")
  |
    accessPath = "(" + c + " (member " + m + " (member exports (module " + stream +
        "))))" and
    result = "stream." + m
  )
  or
  exists(string stream, string c, string m |
    (stream = "stream" or stream = "readable-stream") and
    (c = "instance" or c = "return") and 
    (m = "Transform" or m = "PassThrough")
  |
    accessPath = "(" + c + " (member " + m + " (member exports (module " + stream +
        "))))" and
    result = "stream.Duplex"
  )
  or
  accessPath = "(return (member exports (module ws)))" and 
  result = "ws.WebSocket"
  or
  accessPath = "(instance (member exports (module ws)))" and
  result = "ws.WebSocket"
  or
  exists( string c, string m | 
  	(c = "return" or c = "instance") and 
 	 ( m = "handleUpgrade" or m = "shouldHandle")| 
  	accessPath = "(parameter 0 (member " + m + " (" + c + " (member Server (member exports (module ws))))))" and
  	result = "http.IncomingMessage"
  )
  or
  exists( string c | 
  	c = "return" or c = "instance" | 
  	accessPath = "(parameter 1 (member handleUpgrade (" + c + " (member Server (member exports (module ws))))))" and
  	result = "net.Socket"
  )
  or 
  exists( string c | 
  	c = "return" or c = "instance" | 
  	accessPath = "(parameter 0 (parameter 3 (member handleUpgrade (" + c + " (member Server (member exports (module ws)))))))" and
  	result = "ws.WebSocket"
  )
  or 
  exists( string c | 
  	c = "return" or c = "instance" | 
  	accessPath = "(parameter 0 (member createWebSocketStream (" + c + " (member exports (module ws)))))" and
  	result = "ws.WebSocket"
  )
  or 
  exists( string c | 
  	c = "return" or c = "instance" |
  	accessPath = "(" + c + " (member exports (module process)))" and
  	result = "process.Process"
  )
  or 
  exists( string c | 
    c = "return" or c = "instance" |
    accessPath = "(" + c + " (parameter -1 (member nextTick (member exports (module process)))))" and
    result = "process.Process"
  )
  or 
  accessPath = "(member exports (module process))" and
  result = "process.Process"
  or 
  exists( string m | 
  	m = "stdin" or m = "stderr" or m = "stdout" |
  	accessPath = "(member " + m + " (member exports (module process)))" and
  	result = "net.Socket"
  )
  or
  exists( string c | 
    c = "return" or c = "instance" |
    accessPath = "(" + c + " (member exports (module events)))" and
    result = "events.EventEmitter"
  )
  or
  accessPath = "(parameter -1 (member EventEmitter (member exports (module events))))" and
  result = "events.EventEmitter"
  or
  exists( string c | 
    c = "return" or c = "instance" |
    accessPath = "(" + c + " (member exports (module cluster)))" and
    result = "cluster.Cluster"
  )
  or
  exists( string c | 
    c = "return" or c = "instance" |
    accessPath = "(" + c + " (member Socket (member exports (module socket.io-client))))" and
    result = "net.Socket"
  )
  or
  exists( string c | 
    c = "return" or c = "instance" |
    accessPath = "(" + c + " (member Socket (parameter -1 (member isIP (return (member binding (member exports (module process))))))))" and
    result = "net.Socket"
  ) 
  or
  exists ( string m, string e | 
    (
      m = "clearLine" or 
      m = "clearScreenDown" or
      m = "cursorTo" or 
      m = "moveCursor"
    ) and
    (
      e = " (member exports (module readline))))" or
      e = " (return (member exports (module readline)))))" or
      e = " (instance (member exports (module readline)))))"
    ) |
    accessPath = "(parameter 0 (member " + m + e and 
    result = "stream.Writable"
  )
  or 
  exists ( string e | 
    e = " (member exports (module readline))))" or
    e = " (return (member exports (module readline)))))" or
    e = " (instance (member exports (module readline)))))"
    |
    (accessPath = "(parameter 0 (member emitKeypressEvents" + e and 
    result = "stream.Readable") or
    (accessPath = "(parameter 1 (member emitKeypressEvents" + e and 
    result = "readline.Interface")
  )
  or
  exists ( string e | 
    e = " (member exports (module send))))" or
    e = " (return (member exports (module send)))))" or
    e = " (instance (member exports (module send)))))"
    |
    accessPath = "(parameter 0 (member pipe" + e and 
    result = "http.ServerResponse"
  )
  or
  exists ( string e | 
    e = " (member exports (module send)))" or
    e = " (return (member exports (module send))))" or
    e = " (instance (member exports (module send))))"
    |
    accessPath = "(parameter 0" + e and 
    result = "http.ClientRequest" // this is the same as the first argument passed to http.createServer
  )
  or 
  (
    accessPath = "(member exports (module send))" or
    accessPath = "(return (member exports (module send)))" or
    accessPath = "(instance (member exports (module send)))"
  ) and result = "send.SendStream"
  or
  exists( string m | 
    m = "createServer" or m = "createSecureServer" | 
    accessPath = "(parameter 0 (parameter 0 (member " + m + " (member exports (module http2)))))" and
    result = "http2.Http2ServerRequest"
  )
  or
  exists( string m | 
    m = "createServer" or m = "createSecureServer" | 
    accessPath = "(parameter 1 (parameter 0 (member " + m + " (member exports (module http2)))))" and
    result = "http2.Http2ServerResponse"
  )
  or
  exists( string m | 
    m = "createServer" or m = "createSecureServer" | 
    accessPath = "(parameter 1 (member createPushResponse (parameter 1 (parameter 0 (member " + m + " (member exports (module http2)))))))" and
    result = "http2.ServerHttp2Stream"
  )
}

/** Holds if `type` has an event named `eventName`. */
predicate hasEvent(string type, string eventName) {
  type = "fs.ReadStream" and
  (
    eventName = "close" or
    eventName = "open" or
    eventName = "ready" or
    hasEvent("stream.Readable", eventName)
  )
  or
  type = "fs.WriteStream" and
  (
    eventName = "close" or
    eventName = "open" or
    eventName = "ready" or
    hasEvent("stream.Writable", eventName)
  )
  or
  type = "net.Socket" and
  (
    eventName = "close" or
    eventName = "connect" or
    eventName = "data" or
    eventName = "drain" or
    eventName = "end" or
    eventName = "error" or
    eventName = "lookup" or
    eventName = "ready" or
    eventName = "timeout" or
    hasEvent("stream.Duplex", eventName)
  )
  or
  type = "net.Server" and
  (
    eventName = "close" or
    eventName = "connection" or
    eventName = "error" or
    eventName = "listening" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "http.ClientRequest" and
  (
    eventName = "abort" or
    eventName = "connect" or
    eventName = "continue" or
    eventName = "information" or
    eventName = "response" or
    eventName = "socket" or
    eventName = "timeout" or
    eventName = "upgrade" or
    hasEvent("stream.Writable", eventName)
  )
  or
  type = "http.Server" and
  (
    eventName = "checkContinue" or
    eventName = "checkExpectation" or
    eventName = "clientError" or
    eventName = "close" or
    eventName = "connect" or
    eventName = "connection" or
    eventName = "request" or
    eventName = "upgrade" or
    hasEvent("net.Server", eventName)
  )
  or
  type = "http.ServerResponse" and
  (
    eventName = "close" or
    eventName = "finish" or
    hasEvent("stream.Writable", eventName)
  )
  or
  type = "http.IncomingMessage" and
  (
    eventName = "aborted" or
    eventName = "close" or
    hasEvent("stream.Readable", eventName)
  )
  or
  type = "stream.Readable" and
  (
    eventName = "close" or
    eventName = "data" or
    eventName = "end" or
    eventName = "error" or
    eventName = "pause" or
    eventName = "readable" or
    eventName = "resume" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "stream.Writable" and
  (
    eventName = "close" or
    eventName = "drain" or
    eventName = "error" or
    eventName = "finish" or
    eventName = "pipe" or
    eventName = "unpipe" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "stream.Duplex" and
  (
    hasEvent("stream.Readable", eventName) or
    hasEvent("stream.Writable", eventName)
  )
  or
  type = "child_process.ChildProcess" and
  (
    eventName = "close" or
    eventName = "disconnect" or
    eventName = "error" or
    eventName = "exit" or
    eventName = "message" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "socket.io.Namespace" and
  (
    eventName = "connect" or
    eventName = "connection" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "socket.io.Socket" and
  (
    eventName = "disconnect" or
    eventName = "error" or 
    eventName = "disconnecting" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "socket.io-client.Manager" and
  (
    eventName = "connect_error" or
    eventName = "connect_timeout" or
    eventName = "reconnect" or
    eventName = "reconnect_attempt" or
    eventName = "reconnecting" or
    eventName = "reconnect_error" or
    eventName = "reconnect_failed" or
    eventName = "ping" or
    eventName = "pong" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "socket.io-client.Socket" and
  (
    eventName = "connect" or
    eventName = "connect_error" or
    eventName = "connect_timeout" or
    eventName = "error" or
    eventName = "disconnect" or
    eventName = "reconnect" or
    eventName = "reconnect_attempt" or
    eventName = "reconnecting" or
    eventName = "reconnect_error" or
    eventName = "reconnect_failed" or
    eventName = "ping" or
    eventName = "pong" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "ws.WebSocket" and 
  (
  	eventName = "close" or
  	eventName = "message" or
  	eventName = "open" or
  	eventName = "ping" or
  	eventName = "pong" or
  	eventName = "unexpected-response" or
  	eventName = "upgrade" or
    hasEvent("events.EventEmitter", eventName)
  ) 
  or 
  type = "ws.Duplex" and
  (
  	hasEvent("stream.Duplex", eventName) or
  	hasEvent("ws.WebSocket", eventName)
  ) 
  or
  type = "ws.Server" and
  (
  	eventName = "close" or
  	eventName = "connection" or
  	eventName = "error" or
  	eventName = "headers" or
  	eventName = "listening" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "process.Process" and
  (
  	eventName = "beforeExit" or
  	eventName = "disconnect" or
  	eventName = "message" or
  	eventName = "multipleResolves" or
  	eventName = "rejectionHandled" or
  	eventName = "uncaughtException" or
  	eventName = "unhandledRejection" or
  	eventName = "warning" or
    hasEvent("events.EventEmitter", eventName)
  ) 
  or
  type = "tls.Server" and 
  (
  	eventName = "keylog" or
  	eventName = "newSession" or
  	eventName = "OCSPRequest" or
  	eventName = "resumeSession" or
    eventName = "secureConnection" or
    eventName = "secure" or // this is deprecated but still possible
  	eventName = "tlsClientError" or
  	hasEvent("net.Server", eventName)
  )
  or
  type = "tls.TLSSocket" and 
  (
  	eventName = "keylog" or
  	eventName = "OCSPResponse" or
  	eventName = "secureConnect" or
  	eventName = "session" or
  	hasEvent("net.Socket", eventName)
  )
  or
  // everything is an EventEmitter
  type = "events.EventEmitter" and 
  (
    eventName = "newListener" or
    eventName = "removeListener"
  )
  or
  type = "cluster.Cluster" and
  (
    eventName = "disconnect" or
    eventName = "exit" or
    eventName = "fork" or
    eventName = "listening" or
    eventName = "message" or
    eventName = "online" or
    eventName = "setup" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "cluster.Worker" and
  (
    eventName = "disconnect" or
    eventName = "error" or
    eventName = "exit" or
    eventName = "listening" or 
    eventName = "message" or
    eventName = "online" or 
    hasEvent("events.EventEmitter", eventName)
  )
  or 
  type = "readline.Interface" and
  (
    eventName = "close" or
    eventName = "line" or
    eventName = "pause" or
    eventName = "resume" or 
    eventName = "SIGCONT" or
    eventName = "SIGINT" or
    eventName = "SIGTSTP" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "send.SendStream" and
  (
    eventName = "error" or
    eventName = "directory" or
    eventName = "file" or
    eventName = "headers" or
    eventName = "stream" or
    eventName = "end" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "http2.Http2Session" and
  (
    eventName = "close" or
    eventName = "connect" or
    eventName = "error" or
    eventName = "frameError" or
    eventName = "goaway" or
    eventName = "localSettings" or
    eventName = "ping" or
    eventName = "remoteSettings" or
    eventName = "stream" or
    eventName = "timeout" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "http2.ClientHttp2Session" and
  (
    eventName = "altsvc" or
    eventName = "origin" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "http2.Http2Stream" and
  (
    eventName = "aborted" or
    eventName = "close" or
    eventName = "error" or
    eventName = "frameError" or
    eventName = "timeout" or
    eventName = "trailers" or
    eventName = "wantTrailers" or
    hasEvent("stream.Duplex", eventName)
  )
  or
  type = "http2.ClientHttp2Stream" and
  (
    eventName = "continue" or
    eventName = "headers" or
    eventName = "push" or
    eventName = "response" or
    hasEvent("http2.Http2Stream", eventName)
  )
  or 
  type = "http2.Http2Server" and
  (
    eventName = "checkContinue" or
    eventName = "request" or
    eventName = "session" or
    eventName = "sessionError" or
    eventName = "stream" or 
    eventName = "timeout" or
    hasEvent("net.Server", eventName)
  )
  or
  type = "http2.Http2SecureServer" and 
  (
    eventName = "checkContinue" or
    eventName = "request" or
    eventName = "session" or
    eventName = "sessionError" or
    eventName = "stream" or 
    eventName = "timeout" or
    eventName = "unknownProtocol" or
    hasEvent("tls.Server", eventName)
  )
  // http2 compatibility API
  or
  type = "http2.Http2ServerRequest" and
  (
    eventName = "aborted" or
    eventName = "close" or
    hasEvent("stream.Readable", eventName)
  )
  or
  type = "http2.Http2ServerResponse" and
  (
    eventName = "finish" or
    eventName = "close" or
    hasEvent("events.EventEmitter", eventName)
  )
  or
  type = "repl.REPLServer" and
  (
    eventName = "exit" or
    eventName = "reset" or
    hasEvent("readline.Interface", eventName)
  )
}

predicate doesNotHaveEvent(string type, string eventName) {
  exists(string othertype | type != eventName |
    not hasEvent(type, eventName) and
    hasEvent(othertype, eventName) and
    hasEvent(type, _)
  )
  or
  // add special cases that we know are broken but don't correspond to another type
  type = "http.Agent" and eventName = "free"
  or
  // likely meant to call agent.freeSockets
  type = "http.ClientRequest" and eventName = "complete" or
  type = "http.IncomingMessage" and eventName = "complete" or
  type = "http.IncomingMessage" and eventName = "fd" or
  type = "http.IncomingMessage" and eventName = "exit" or
  type = "process.Process" and 
  ( // reserved signals that can't/shouldnt be listened on
    eventName = "SIGKILL" or 
    eventName = "SIGSTOP" or 
    eventName = "SIGTERM" or 
    eventName = "SIGINT" or 
    eventName = "SIGUSR1" or
    eventName = "SIGBUS" or
    eventName = "SIGFPE" or
    eventName = "SIGSEGV" or
    eventName = "SIGILL"
  ) or
  type = "cluster.Worker" and eventName = "death" // could be a mistake for 'exit', 'error', 'disconnect', or the isDeath() method
  or
  (type = "net.Socket" or type = "ws.WebSocket") and (eventName = "fd" or eventName = "data")
  or
  type = "socket.io-client.Socket" and eventName = "connect_failed"
  or 
  type = "stream.Readable" and eventName = "close"
}

// we can add to this list as we find more examples 
bindingset[accessPath]
predicate accessPathContainsImprecision( string accessPath) {
  exists(accessPath.indexOf("(parameter 0 (member inherits ")) or
  exists(accessPath.indexOf("(return (member bind ")) or
  exists(accessPath.indexOf("(parameter 0 (member call (member ")) or
  exists(accessPath.indexOf("(parameter 0 (member apply (member ")) or
  exists(accessPath.indexOf("(parameter 1 (member on ")) or 
  exists(accessPath.indexOf("(parameter 1 (member once ")) or 
  exists(accessPath.indexOf("(parameter 1 (member addListener ")) or 
  exists(accessPath.indexOf("(parameter 1 (member removeListener ")) or 
  exists(accessPath.indexOf("(parameter 1 (member prependListener ")) or
  exists(accessPath.indexOf("(parameter 1 (member prependOnceListener "))
}

// class listing out the packages whose APIs we modelled: we only want to generate 
// the ground truth for ExtAPStrings rooted in one of these packages
class AnalyzedPackageName extends string {
  AnalyzedPackageName() {
    this = "fs" or
    this = "http" or
    this = "net" or
    this = "child_process" or
    this = "zlib" or 
    this = "https" or 
    this = "socket.io" or 
    this = "socket.io-client" or
    this = "stream" or 
    this = "readable-stream" or 
    this = "ws" or 
    this = "process" or 
    this = "tls" or 
    this = "events" or 
    this = "cluster" or
    this = "readline" or 
    this = "http2" or 
    this = "repl"
  }
}

class AnalyzedExtAPString extends ExtAPString {
  AnalyzedPackageName root;

  AnalyzedExtAPString() {
    this.matches("%(member exports (module " + root + "%")
  }
}

bindingset[accessPath]
bindingset[eventName]
predicate incorrect(ExtAPString accessPath, ExtEventString eventName) {
  // exists(DataPoint p | accessPath = p.getAccessPath() and eventName = p.getEventName() |
    isExternalDatapoint(accessPath, eventName) and 
    (doesNotHaveEvent(typeOf(accessPath), eventName)
    // should i add custom things here? the verdict is: yes
    // these are some paths which get found for http2 which don't match well to specific types in the framework, but are wrong
    or
    ( accessPath = "(return (member request (member exports (module http2))))" and (eventName = "push" or eventName = "response"))
    or
    ( exists( string d | accessPath = "(instance (member Readable " + d + "))" and doesNotHaveEvent("stream.Readable", eventName)))
    or
    ( exists( string d | accessPath = "(return (member createReadStream " + d + "))" and doesNotHaveEvent("stream.Readable", eventName)))
    or
    ( exists( string d | accessPath = "(instance (member Writable " + d + "))" and doesNotHaveEvent("stream.Writable", eventName)))
    or
    ( exists( string d | accessPath = "(return (member createWriteStream " + d + "))" and doesNotHaveEvent("stream.Writable", eventName)))
    or // the socket.io Server 
    ( accessPath = "(return (member exports (module socket.io)))" and (hasEvent("socket.io.Socket", eventName) or hasEvent("socket.io.Namespace", eventName)))
    or
    ( accessPath = "(return (member exports (module socket.io-client)))" and eventName = "connect_failed") // typo probably, for reconnect_failed or connect_erro/timeout
    or
    ( accessPath = "(return (member createGunzip (member exports (module zlib))))" and eventName = "complete")
    or
    ( accessPath = "(return (member createGunzip (member exports (module zlib))))" and eventName = "entry")
    or 
    ( exists( string c | c = "instance" or c = "return" | 
         accessPath = "(" + c + " (member exports (module stream)))" and (hasEvent("stream.Readable", eventName) or hasEvent("stream.Writable", eventName))
                                                                          and not hasEvent("events.EventEmitter", eventName)))
    or (accessPath = "(parameter 0 (parameter 0 (member createServer (member exports (module net)))))" and eventName = "connect"))

    and not correct(accessPath, eventName) 
  // )
}

bindingset[accessPath]
bindingset[eventName]
predicate correct(ExtAPString accessPath, ExtEventString eventName) {
  // exists(DataPoint p | accessPath = p.getAccessPath() and eventName = p.getEventName() |
    isExternalDatapoint(accessPath, eventName) and 
    hasEvent(typeOf(accessPath), eventName)
    or
    ( exists( string d | accessPath = "(instance (member Readable " + d + "))" and hasEvent("stream.Readable", eventName)))
    or
    ( exists( string d | accessPath = "(instance (member Writable " + d + "))" and hasEvent("stream.Writable", eventName)))
  // )
}

bindingset[accessPath]
bindingset[eventName]
predicate knownUnknown(ExtAPString accessPath, ExtEventString eventName) {
  // exists(DataPoint p | accessPath = p.getAccessPath() and eventName = p.getEventName() |
      //not correct( accessPath, eventName) and 
      //not incorrect( accessPath, eventName) and 
      isExternalDatapoint(accessPath, eventName) and 
      accessPathContainsImprecision( accessPath)
    // )
}

