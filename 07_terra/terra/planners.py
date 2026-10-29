"""Planners on a cost.Graph: Dijkstra, A* (any heuristic array, optional weight), and
D* Lite (Koenig & Likhachev 2002, optimized version with k_m for a moving start).

All are plain Python + heapq in the same style, so latency comparisons are apples to apples.
"expanded" counts non-stale pops (nodes whose g was finalized / updated by the search).
"""
import heapq
import math
import time

INF = math.inf


class Result:
    __slots__ = ("path", "cost", "expanded", "ms", "closed")

    def __init__(self, path, cost, expanded, ms, closed=None):
        self.path, self.cost, self.expanded, self.ms, self.closed = path, cost, expanded, ms, closed

    def __repr__(self):
        return f"Result(cost={self.cost:.3f}, len={len(self.path)}, expanded={self.expanded}, ms={self.ms:.2f})"


def _reconstruct(parent, s, g):
    path = [g]
    while path[-1] != s:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def astar(graph, s, g, h=None, weight=1.0, keep_closed=False):
    """A* from s to g. h: flat list/array of heuristic values to g (len n), or None (Dijkstra).
    weight > 1 gives weighted A* (f = g + w*h). Ties broken toward larger g (deeper)."""
    t0 = time.perf_counter()
    adj = graph.adj
    n = graph.n
    gs = [INF] * n
    parent = [-1] * n
    closed = bytearray(n)
    gs[s] = 0.0
    if h is None:
        hv = None
        pq = [(0.0, 0.0, s)]
    else:
        hv = h.tolist() if hasattr(h, "tolist") else h
        pq = [(weight * hv[s], 0.0, s)]
    expanded = 0
    push, pop = heapq.heappush, heapq.heappop
    found = False
    while pq:
        f, ng, u = pop(pq)
        if closed[u]:
            continue
        closed[u] = 1
        expanded += 1
        if u == g:
            found = True
            break
        gu = gs[u]
        for v, c in adj[u]:
            nv = gu + c
            if nv < gs[v]:
                gs[v] = nv
                parent[v] = u
                if hv is None:
                    push(pq, (nv, -nv, v))
                else:
                    push(pq, (nv + weight * hv[v], -nv, v))
    ms = (time.perf_counter() - t0) * 1e3
    cl = closed if keep_closed else None
    if not found:
        return Result([], INF, expanded, ms, cl)
    return Result(_reconstruct(parent, s, g), gs[g], expanded, ms, cl)


def dijkstra(graph, s, g, keep_closed=False):
    return astar(graph, s, g, None, keep_closed=keep_closed)


def dijkstra_all(graph, src):
    """Full cost-to-come field from src (== cost-to-go to src, costs are symmetric)."""
    adj = graph.adj
    dist = [INF] * graph.n
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, c in adj[u]:
            nd = d + c
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


class DStarLite:
    """D* Lite (Koenig & Likhachev 2002, Fig. 3). Searches backward from goal; g/rhs are
    cost-to-goal estimates. Heuristic is octile distance to the *current start*.

    Usage:
        d = DStarLite(graph, start, goal); d.compute()   # initial plan
        d.move_to(new_start)                              # agent moved
        changed = graph.set_cells(...); d.update_vertices(changed); d.compute()
        d.path()
    """

    def __init__(self, graph, start, goal):
        self.G = graph
        n = graph.n
        self.s_start = start
        self.s_last = start
        self.goal = goal
        self.km = 0.0
        self.g = [INF] * n
        self.rhs = [INF] * n
        self.rhs[goal] = 0.0
        self.inU = [None] * n  # current key in the queue, or None
        self.U = []
        self.expanded_total = 0
        self.trace = None  # set to a list to record expanded nodes (for figures)
        self.w = graph.w
        self._rows = [u // self.w for u in range(n)]  # cached coordinates for the heuristic
        self._cols = [u % self.w for u in range(n)]
        self._sr, self._sc = divmod(start, self.w)
        self._insert(goal, self._key(goal))

    # --- helpers -----------------------------------------------------------------------
    def _h(self, s):
        dr = self._rows[s] - self._sr
        dc = self._cols[s] - self._sc
        if dr < 0:
            dr = -dr
        if dc < 0:
            dc = -dc
        return (dr + 0.41421356237309515 * dc) if dr > dc else (dc + 0.41421356237309515 * dr)

    def _key(self, s):
        m = self.g[s] if self.g[s] < self.rhs[s] else self.rhs[s]
        return (m + self._h(s) + self.km, m)

    def _insert(self, s, k):
        self.inU[s] = k
        heapq.heappush(self.U, (k[0], k[1], s))

    def _top(self):
        U, inU = self.U, self.inU
        while U:
            k0, k1, s = U[0]
            cur = inU[s]
            if cur is not None and cur[0] == k0 and cur[1] == k1:
                return (k0, k1), s
            heapq.heappop(U)  # stale
        return (INF, INF), -1

    def _update_vertex(self, u):
        g = self.g[u]
        rhs = self.rhs[u]
        if g != rhs:
            m = g if g < rhs else rhs
            k = (m + self._h(u) + self.km, m)
            if self.inU[u] != k:  # insert or update (old heap entry becomes stale)
                self.inU[u] = k
                heapq.heappush(self.U, (k[0], k[1], u))
        elif self.inU[u] is not None:
            self.inU[u] = None  # remove (lazy)

    def _min_succ(self, u):
        best = INF
        g = self.g
        for v, c in self.G.adj[u]:
            x = c + g[v]
            if x < best:
                best = x
        return best

    # --- API ---------------------------------------------------------------------------
    def compute(self):
        """ComputeShortestPath. Returns (expanded, ms)."""
        t0 = time.perf_counter()
        g, rhs, adj, inU = self.g, self.rhs, self.G.adj, self.inU
        s_start, goal = self.s_start, self.goal
        expanded = 0
        while True:
            k_old, u = self._top()
            k_start = self._key(s_start)
            if not (k_old < k_start or rhs[s_start] > g[s_start]):
                break
            if u < 0:
                break  # queue empty: start unreachable (g = rhs = inf)
            expanded += 1
            if self.trace is not None:
                self.trace.append(u)
            k_new = self._key(u)
            if k_old < k_new:
                self._insert(u, k_new)
            elif g[u] > rhs[u]:
                g[u] = rhs[u]
                inU[u] = None
                gu = g[u]
                for s, c in adj[u]:  # pred == succ (symmetric costs)
                    if s != goal:
                        x = c + gu
                        if x < rhs[s]:
                            rhs[s] = x
                    self._update_vertex(s)
            else:
                g_old = g[u]
                g[u] = INF
                for s, c in adj[u]:
                    if s != goal and rhs[s] == c + g_old:
                        rhs[s] = self._min_succ(s)
                    self._update_vertex(s)
                if u != goal and rhs[u] == g_old:
                    rhs[u] = self._min_succ(u)
                self._update_vertex(u)
        self.expanded_total += expanded
        return expanded, (time.perf_counter() - t0) * 1e3

    def move_to(self, s):
        """Agent moved to s. k_m is updated lazily when edge changes are applied."""
        self.s_start = s
        self._sr, self._sc = divmod(s, self.w)

    def update_vertices(self, nodes):
        """Edge costs around `nodes` changed (graph adjacency already rebuilt). Because every
        changed edge has both endpoints in `nodes`, recomputing rhs for them is exact."""
        # k_m += h(s_last, s_start), with h measured from the new start
        self.km += self._h(self.s_last)
        self.s_last = self.s_start
        g, rhs = self.g, self.rhs
        for u in nodes:
            if u != self.goal:
                rhs[u] = self._min_succ(u)
            self._update_vertex(u)

    def path(self, max_len=None):
        """Greedy descent on c + g from the current start."""
        if self.g[self.s_start] == INF and self.rhs[self.s_start] == INF:
            return []
        s = self.s_start
        out = [s]
        limit = max_len or self.G.n
        g = self.g
        while s != self.goal and len(out) <= limit:
            best, bv = INF, -1
            for v, c in self.G.adj[s]:
                x = c + g[v]
                if x < best:
                    best, bv = x, v
            if bv < 0 or best == INF:
                return []
            s = bv
            out.append(s)
        return out

    @property
    def cost_to_goal(self):
        return self.rhs[self.s_start] if self.rhs[self.s_start] < self.g[self.s_start] else self.g[self.s_start]
