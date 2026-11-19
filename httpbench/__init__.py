"""Sequential HTTP download benchmark, standard library only.

- urllib's request layer hides the socket, so timing cannot be split into
  DNS / TCP / TLS / TTFB / body phases and connections cannot be reused;
  this package therefore talks to ``http.client`` directly;
- a chunked read loop keeps memory usage constant for arbitrarily large
  files and the body is never written to disk;
- one total timeout budget covers the whole request, redirects included;
- failed attempts are reported, the run continues, and statistics use
  successful attempts only; the process exits with status 1 if at least one
  attempt failed, so partial failures are detectable from CI.
"""

__version__ = "2.0.0"
