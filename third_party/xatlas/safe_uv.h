// Winding-independent UV validation and graph parameterization.
// No CUDA, Python, or external linear algebra dependency.
#pragma once
#include "xatlas.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cumesh_xatlas { namespace safeuv {

using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

inline int orient(const float *a, const float *b, const float *c) {
    const double ax = double(a[0]) - c[0], ay = double(a[1]) - c[1];
    const double bx = double(b[0]) - c[0], by = double(b[1]) - c[1];
    const double left = ax * by, right = ay * bx, det = left - right;
    const double bound = 8 * std::numeric_limits<double>::epsilon() *
        (std::abs(left) + std::abs(right));
    if (std::abs(det) > bound) return det > 0 ? 1 : -1;
    // Float32 coordinates promoted to long double provide a cheap fallback
    // for the cancellation cases that are ambiguous in double precision.
    const long double lax = (long double)a[0] - c[0], lay = (long double)a[1] - c[1];
    const long double lbx = (long double)b[0] - c[0], lby = (long double)b[1] - c[1];
    const long double precise = lax * lby - lay * lbx;
    return precise == 0 ? 0 : (precise > 0 ? 1 : -1);
}

struct Validation {
    // 0 valid, 1 nonfinite, 2 degenerate, 3 overlap, 4 invalid index.
    int issue = 0;
    int64_t face0 = -1, face1 = -1;
    uint64_t candidates = 0;
    double elapsed = 0;
};

struct Box {
    float lo[2] = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    float hi[2] = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    void add(const float *p) {
        for (int d = 0; d < 2; ++d) { lo[d] = std::min(lo[d], p[d]); hi[d] = std::max(hi[d], p[d]); }
    }
    void add(const Box &b) { add(b.lo); add(b.hi); }
    bool intersects(const Box &b) const {
        return lo[0] < b.hi[0] && b.lo[0] < hi[0] && lo[1] < b.hi[1] && b.lo[1] < hi[1];
    }
};

class Validator {
    struct Node { Box box; uint32_t begin, end, left = 0, right = 0, parent = UINT32_MAX; };
    const float *uv;
    const int32_t *faces;
    std::vector<Box> boxes;
    std::vector<uint32_t> order;
    std::vector<Node> nodes;
    std::vector<uint32_t> faceLeaf;
    uint32_t build(uint32_t begin, uint32_t end) {
        Node n; n.begin = begin; n.end = end;
        for (uint32_t i = begin; i < end; ++i) n.box.add(boxes[order[i]]);
        uint32_t id = uint32_t(nodes.size()); nodes.push_back(n);
        if (end - begin > 8) {
            int axis = double(n.box.hi[0]) - n.box.lo[0] > double(n.box.hi[1]) - n.box.lo[1] ? 0 : 1;
            uint32_t middle = begin + (end - begin) / 2;
            std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
                [&](uint32_t a, uint32_t b) {
                    const double ca = double(boxes[a].lo[axis]) + boxes[a].hi[axis];
                    const double cb = double(boxes[b].lo[axis]) + boxes[b].hi[axis];
                    return ca == cb ? a < b : ca < cb;
                });
            nodes[id].left = build(begin, middle);
            nodes[id].right = build(middle, end);
            nodes[nodes[id].left].parent = id;
            nodes[nodes[id].right].parent = id;
        } else {
            for (uint32_t i = begin; i < end; ++i) faceLeaf[order[i]] = id;
        }
        return id;
    }
    bool overlap(uint32_t a, uint32_t b) const {
        // Separating-axis test, including containment and adjacent faces.
        // Shared edges/vertices alone do not have positive intersection area.
        for (int pass = 0; pass < 2; ++pass) {
            uint32_t t = pass ? b : a, other = pass ? a : b;
            for (int k = 0; k < 3; ++k) {
                const float *p = uv + 2 * size_t(faces[3 * size_t(t) + k]);
                const float *q = uv + 2 * size_t(faces[3 * size_t(t) + (k + 1) % 3]);
                const float *r = uv + 2 * size_t(faces[3 * size_t(t) + (k + 2) % 3]);
                int side = orient(p, q, r);
                bool separated = true;
                for (int j = 0; j < 3; ++j)
                    if (side * orient(p, q, uv + 2 * size_t(faces[3 * size_t(other) + j])) > 0)
                        separated = false;
                if (separated) return false;
            }
        }
        return true;
    }
    bool visit(uint32_t a, uint32_t b, Validation &result) const {
        const Node &x = nodes[a], &y = nodes[b];
        if (!x.box.intersects(y.box)) return false;
        if (a == b && x.left) {
            return visit(x.left, x.left, result) || visit(x.left, x.right, result) || visit(x.right, x.right, result);
        }
        if (!x.left && !y.left) {
            for (uint32_t i = x.begin; i < x.end; ++i)
                for (uint32_t j = a == b ? i + 1 : y.begin; j < y.end; ++j) {
                    uint32_t f = order[i], g = order[j];
                    if (!boxes[f].intersects(boxes[g])) continue;
                    ++result.candidates;
                    if (overlap(f, g)) {
                        result.issue = 3; result.face0 = f; result.face1 = g; return true;
                    }
                }
            return false;
        }
        if (x.left && (!y.left || x.end - x.begin >= y.end - y.begin))
            return visit(x.left, b, result) || visit(x.right, b, result);
        return visit(a, y.left, result) || visit(a, y.right, result);
    }
    bool collectVisit(
        uint32_t a,
        uint32_t b,
        std::vector<std::pair<uint32_t, uint32_t>> &pairs,
        uint64_t &candidates,
        uint64_t maxPairs
    ) const {
        const Node &x = nodes[a], &y = nodes[b];
        if (!x.box.intersects(y.box)) return true;
        if (a == b && x.left) {
            return collectVisit(x.left, x.left, pairs, candidates, maxPairs)
                && collectVisit(x.left, x.right, pairs, candidates, maxPairs)
                && collectVisit(x.right, x.right, pairs, candidates, maxPairs);
        }
        if (!x.left && !y.left) {
            for (uint32_t i = x.begin; i < x.end; ++i)
                for (uint32_t j = a == b ? i + 1 : y.begin; j < y.end; ++j) {
                    uint32_t f = order[i], g = order[j];
                    if (!boxes[f].intersects(boxes[g])) continue;
                    ++candidates;
                    if (overlap(f, g)) {
                        if (pairs.size() >= maxPairs) return false;
                        pairs.emplace_back(f, g);
                    }
                }
            return true;
        }
        if (x.left && (!y.left || x.end - x.begin >= y.end - y.begin))
            return collectVisit(x.left, b, pairs, candidates, maxPairs)
                && collectVisit(x.right, b, pairs, candidates, maxPairs);
        return collectVisit(a, y.left, pairs, candidates, maxPairs)
            && collectVisit(a, y.right, pairs, candidates, maxPairs);
    }
    bool prepare(uint32_t vertexCount, uint32_t faceCount, Validation &result) {
        for (size_t i = 0; i < size_t(vertexCount) * 2; ++i) if (!std::isfinite(uv[i])) {
            result.issue = 1; return false;
        }
        boxes.clear(); boxes.resize(faceCount);
        order.resize(faceCount);
        for (uint32_t f = 0; f < faceCount; ++f) {
            order[f] = f;
            for (int k = 0; k < 3; ++k) {
                int32_t v = faces[size_t(f) * 3 + k];
                if (v < 0 || uint32_t(v) >= vertexCount) {
                    result.issue = 4; result.face0 = f; return false;
                }
                boxes[f].add(uv + size_t(v) * 2);
            }
            if (orient(uv + size_t(faces[size_t(f)*3])*2,
                    uv + size_t(faces[size_t(f)*3+1])*2,
                    uv + size_t(faces[size_t(f)*3+2])*2) == 0) {
                result.issue = 2; result.face0 = f; return false;
            }
        }
        nodes.clear(); faceLeaf.resize(faceCount);
        if (faceCount) { nodes.reserve(faceCount / 2 + 1); build(0, faceCount); }
        return true;
    }
    template<class Accept>
    bool queryNode(uint32_t node, uint32_t face, const Box &box,
                   const std::vector<uint8_t> &excluded, Accept &accept,
                   uint64_t &budget, bool &exhausted) const {
        if (!budget) { exhausted = true; return false; }
        --budget;
        const auto &n = nodes[node];
        if (!box.intersects(n.box)) return true;
        if (n.left) return queryNode(n.left,face,box,excluded,accept,budget,exhausted)
            && queryNode(n.right,face,box,excluded,accept,budget,exhausted);
        for (uint32_t i = n.begin; i < n.end; ++i) {
            if (!budget) { exhausted = true; return false; }
            --budget;
            uint32_t other = order[i];
            if (excluded[other] || !box.intersects(boxes[other])) continue;
            if (overlap(face,other) && !accept(face,other)) return false;
        }
        return true;
    }
public:
    Validator(const float *u, const int32_t *f) : uv(u), faces(f) {}
    bool overlaps(uint32_t a, uint32_t b) const { return overlap(a,b); }
    bool usableTriangle(uint32_t f) const {
        const float *p[3];
        for (int k = 0; k < 3; ++k) {
            p[k] = uv + 2*size_t(faces[3*size_t(f)+k]);
            if (!std::isfinite(p[k][0]) || !std::isfinite(p[k][1])) return false;
        }
        double scale = 0;
        for (int k = 0; k < 3; ++k) {
            double x = double(p[k][0])-p[(k+1)%3][0], y = double(p[k][1])-p[(k+1)%3][1];
            scale = std::max(scale,x*x+y*y);
        }
        double area = (double(p[1][0])-p[0][0])*(double(p[2][1])-p[0][1])
                    - (double(p[1][1])-p[0][1])*(double(p[2][0])-p[0][0]);
        // Reject float32 slivers without imposing an absolute chart scale.
        return orient(p[0],p[1],p[2]) != 0 && std::abs(area) > 1e-8*scale;
    }
    template<class Accept>
    bool queryTriangle(uint32_t f, const std::vector<uint8_t> &excluded,
                       Accept &accept, uint64_t &budget, bool &exhausted) const {
        Box box;
        for (int k = 0; k < 3; ++k) box.add(uv + 2*size_t(faces[3*size_t(f)+k]));
        return nodes.empty() || queryNode(0,f,box,excluded,accept,budget,exhausted);
    }
    void refit(const std::vector<uint32_t> &changed) {
        for (uint32_t f : changed) {
            boxes[f] = Box();
            for (int k = 0; k < 3; ++k) boxes[f].add(uv + 2*size_t(faces[3*size_t(f)+k]));
        }
        for (uint32_t f : changed) {
            uint32_t id = faceLeaf[f];
            nodes[id].box = Box();
            for (uint32_t i = nodes[id].begin; i < nodes[id].end; ++i) nodes[id].box.add(boxes[order[i]]);
            while (nodes[id].parent != UINT32_MAX) {
                id = nodes[id].parent;
                nodes[id].box = Box();
                nodes[id].box.add(nodes[nodes[id].left].box);
                nodes[id].box.add(nodes[nodes[id].right].box);
            }
        }
    }
    Validation run(uint32_t vertexCount, uint32_t faceCount) {
        auto start = Clock::now(); Validation result;
        if (prepare(vertexCount, faceCount, result) && faceCount) visit(0, 0, result);
        result.elapsed = seconds(start); return result;
    }
    bool collectOverlaps(
        uint32_t vertexCount,
        uint32_t faceCount,
        std::vector<std::pair<uint32_t, uint32_t>> &pairs,
        uint64_t &candidates,
        uint64_t maxPairs,
        Validation &result
    ) {
        result = Validation();
        pairs.clear();
        candidates = 0;
        if (!prepare(vertexCount, faceCount, result)) return false;
        if (faceCount) return collectVisit(0, 0, pairs, candidates, maxPairs);
        return true;
    }
};

struct Edge { int a, b; std::vector<int> corners; };
struct Topology {
    std::vector<Edge> edges;
    std::vector<std::vector<int>> neighbors;
    std::vector<std::vector<int>> boundary;
    std::vector<std::vector<std::pair<int, bool>>> dual;
    bool manifold = true;
    Topology(const std::vector<int32_t> &f, int vertexCount) {
        neighbors.resize(vertexCount); boundary.resize(vertexCount); dual.resize(f.size() / 3);
        std::unordered_map<uint64_t, int> table; table.reserve(f.size());
        for (int c = 0; c < int(f.size()); ++c) {
            int a = f[c], b = f[c / 3 * 3 + (c + 1) % 3];
            uint64_t key = (uint64_t(uint32_t(std::min(a,b))) << 32) | uint32_t(std::max(a,b));
            auto it = table.emplace(key, int(edges.size()));
            if (it.second) edges.push_back({std::min(a,b), std::max(a,b), {}});
            edges[it.first->second].corners.push_back(c);
        }
        for (const Edge &e : edges) {
            neighbors[e.a].push_back(e.b); neighbors[e.b].push_back(e.a);
            if (e.corners.size() == 1) { boundary[e.a].push_back(e.b); boundary[e.b].push_back(e.a); }
            else if (e.corners.size() == 2) {
                int c = e.corners[0], d = e.corners[1];
                bool same = f[c] == f[d];
                dual[c/3].push_back({d/3, same}); dual[d/3].push_back({c/3, same});
            } else manifold = false;
        }
    }
    bool orderFaces(std::vector<int32_t> &faces, std::vector<int> &flips) const {
        if (!manifold) return false;
        flips.assign(dual.size(), -1);
        for (int seed = 0; seed < int(dual.size()); ++seed) if (flips[seed] == -1) {
            std::vector<int> q{seed}; flips[seed] = 0;
            for (size_t i = 0; i < q.size(); ++i) for (auto e : dual[q[i]]) {
                int wanted = flips[q[i]] ^ int(e.second);
                if (flips[e.first] == -1) { flips[e.first] = wanted; q.push_back(e.first); }
                else if (flips[e.first] != wanted) return false;
            }
        }
        for (int f = 0; f < int(flips.size()); ++f) if (flips[f]) std::swap(faces[f*3+1], faces[f*3+2]);
        return true;
    }
    bool disk(std::vector<int> &loop) const {
        if (!manifold || dual.empty()) return false;
        int used = 0, boundaryCount = 0, seed = -1;
        for (int v = 0; v < int(neighbors.size()); ++v) {
            if (!neighbors[v].empty()) ++used;
            if (!boundary[v].empty()) {
                if (boundary[v].size() != 2) return false;
                ++boundaryCount; seed = v;
            }
        }
        if (seed < 0 || used - int(edges.size()) + int(dual.size()) != 1) return false;
        std::vector<char> seen(dual.size(), false); std::vector<int> q{0}; seen[0] = true;
        for (size_t i=0; i<q.size(); ++i) for (auto e : dual[q[i]]) if (!seen[e.first]) {
            seen[e.first] = true; q.push_back(e.first);
        }
        if (q.size() != dual.size()) return false;
        int prev = -1, v = seed;
        do {
            if (loop.size() >= size_t(boundaryCount)) return false;
            loop.push_back(v);
            int next = boundary[v][0] == prev ? boundary[v][1] : boundary[v][0];
            prev = v; v = next;
        } while (v != seed);
        // Detect pinched vertex fans as well as disconnected boundaries.
        return loop.size() == size_t(boundaryCount) && loop.size() >= 3;
    }
};

struct UnionFind {
    std::vector<int> p;
    explicit UnionFind(int n) : p(n, -1) {}
    int root(int a) { int r=a; while(p[r]>=0) r=p[r]; while(a!=r) { int next=p[a]; p[a]=r; a=next; } return r; }
    void join(int a, int b) { a=root(a); b=root(b); if(a==b)return; if(p[a]>p[b])std::swap(a,b); p[a]+=p[b]; p[b]=a; }
};

struct OverlapFix {
    std::vector<int32_t> groups;
    uint32_t chartCount = 1;
    uint32_t overlapFaces = 0;
    uint64_t overlapPairs = 0;
    uint64_t candidates = 0;
    uint32_t inputIslands = 0;
    uint32_t inputUndersized = 0;
    uint32_t outputUndersized = 0;
    uint32_t transferredFaces = 0;
    uint32_t inputSingletons = 0, outputSingletons = 0, blockedSingletons = 0;
    uint64_t remainingPairs = 0, repairAttempts = 0, repairedTriangles = 0, movedVariables = 0;
    uint64_t fanSkips = 0, budgetSkips = 0, repairWork = 0;
    double detectionSeconds = 0, repairSeconds = 0, partitionSeconds = 0;
    bool valid = true;
    bool limited = false;
};

struct UvFaceGraph {
    std::vector<uint32_t> offsets, neighbors;
    std::vector<double> lengths;
};

// Match geometry AND UV endpoints, including aliases with different source
// indices. A UV seam or point contact must not join two surface patches.
inline std::vector<uint32_t> uvVariables(const float *uv, uint32_t vertexCount,
                                         const int64_t *vertexIds) {
    std::vector<uint32_t> vertices(vertexCount), order(vertexCount);
    std::iota(order.begin(), order.end(), 0);
    if (vertexIds) {
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (vertexIds[a] != vertexIds[b]) return vertexIds[a] < vertexIds[b];
            if (uv[2*size_t(a)] != uv[2*size_t(b)]) return uv[2*size_t(a)] < uv[2*size_t(b)];
            if (uv[2*size_t(a)+1] != uv[2*size_t(b)+1]) return uv[2*size_t(a)+1] < uv[2*size_t(b)+1];
            return a < b;
        });
    }
    uint32_t canonical = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        uint32_t v = order[i];
        if (i) {
            uint32_t p = order[i-1];
            if (!vertexIds || vertexIds[p] != vertexIds[v] ||
                uv[2*size_t(p)] != uv[2*size_t(v)] || uv[2*size_t(p)+1] != uv[2*size_t(v)+1]) ++canonical;
        }
        vertices[v] = canonical;
    }
    return vertices;
}

inline UvFaceGraph uvFaceGraph(const float *uv, const int32_t *faces,
                              uint32_t vertexCount, uint32_t faceCount,
                              const int64_t *vertexIds,
                              const std::vector<uint32_t> *fixedVariables = nullptr) {
    const auto vertices = fixedVariables ? *fixedVariables : uvVariables(uv,vertexCount,vertexIds);
    struct EdgeRef { uint64_t key; uint32_t face; double length; };
    std::vector<EdgeRef> edges;
    edges.reserve(size_t(faceCount)*3);
    for (uint32_t f = 0; f < faceCount; ++f) for (uint32_t k = 0; k < 3; ++k) {
        uint32_t a = uint32_t(faces[3*size_t(f)+k]);
        uint32_t b = uint32_t(faces[3*size_t(f)+(k+1)%3]);
        uint32_t x = vertices[a], y = vertices[b];
        if (x == y) continue;
        const uint64_t key = (uint64_t(std::min(x,y)) << 32) | std::max(x,y);
        double dx = double(uv[2*size_t(a)])-uv[2*size_t(b)];
        double dy = double(uv[2*size_t(a)+1])-uv[2*size_t(b)+1];
        edges.push_back({key, f, std::sqrt(dx*dx+dy*dy)});
    }
    std::sort(edges.begin(), edges.end(), [](const EdgeRef &a, const EdgeRef &b) {
        return a.key != b.key ? a.key < b.key : a.face < b.face;
    });
    UvFaceGraph graph;
    graph.offsets.assign(size_t(faceCount)+1, 0);
    // A star keeps non-manifold edges linear in their number of incidences.
    for (size_t start = 0, end; start < edges.size(); start = end) {
        for (end = start+1; end < edges.size() && edges[end].key == edges[start].key; ++end) {}
        for (size_t i = start+1; i < end; ++i) if (edges[i].face != edges[i-1].face) {
            ++graph.offsets[edges[start].face+1]; ++graph.offsets[edges[i].face+1];
        }
    }
    for (size_t i = 1; i < graph.offsets.size(); ++i) graph.offsets[i] += graph.offsets[i-1];
    graph.neighbors.resize(graph.offsets.back());
    graph.lengths.resize(graph.offsets.back());
    auto cursor = graph.offsets;
    for (size_t start = 0, end; start < edges.size(); start = end) {
        for (end = start+1; end < edges.size() && edges[end].key == edges[start].key; ++end) {}
        for (size_t i = start+1; i < end; ++i) if (edges[i].face != edges[i-1].face) {
            const uint32_t a = edges[start].face, b = edges[i].face;
            uint32_t p = cursor[a]++, q = cursor[b]++;
            graph.neighbors[p] = b; graph.neighbors[q] = a;
            graph.lengths[p] = graph.lengths[q] = edges[start].length;
        }
    }
    return graph;
}

// A bounded local edit, using the same BVH and conflict graph as partitioning.
// An accepted edit may only DELETE conflicts, never create a new pair.
class LocalUvUntangler {
    static constexpr uint32_t maxFan = 12;
    float *uv;
    const int32_t *faces;
    const std::vector<uint32_t> &variables;
    const UvFaceGraph &graph;
    Validator &validator;
    OverlapFix &stats;
    const std::vector<std::pair<uint32_t,uint32_t>> &pairs;
    std::vector<std::vector<uint32_t>> aliases, fans;
    std::vector<uint32_t> pairOffsets, pairIds, degree;
    std::vector<uint8_t> live, affected, visits, pending, repaired, moved;
    std::unordered_map<uint64_t,uint32_t> pairIndex;
    std::queue<uint32_t> queue;
    uint64_t workLeft;
    static uint64_t pairKey(uint32_t a, uint32_t b) {
        return (uint64_t(std::min(a,b)) << 32) | std::max(a,b);
    }
    bool existing(uint32_t a, uint32_t b) const {
        auto it = pairIndex.find(pairKey(a,b));
        return it != pairIndex.end() && live[it->second];
    }
    void enqueue(uint32_t f) {
        if (degree[f] && visits[f] < 2 && !pending[f]) { pending[f] = 1; queue.push(f); }
    }
    bool anchored(uint32_t f, int corner) const {
        uint32_t v = variables[uint32_t(faces[3*size_t(f)+corner])];
        uint32_t a = variables[uint32_t(faces[3*size_t(f)+(corner+1)%3])];
        uint32_t b = variables[uint32_t(faces[3*size_t(f)+(corner+2)%3])];
        for (uint32_t i = graph.offsets[f]; i < graph.offsets[f+1]; ++i) {
            bool hasA = false, hasB = false, hasV = false;
            for (int k = 0; k < 3; ++k) {
                uint32_t x = variables[uint32_t(faces[3*size_t(graph.neighbors[i])+k])];
                hasA |= x == a; hasB |= x == b; hasV |= x == v;
            }
            if (hasA && hasB && !hasV) return true;
        }
        return false;
    }
    bool tryCorner(uint32_t seed, int corner) {
        uint32_t v = variables[uint32_t(faces[3*size_t(seed)+corner])];
        if (fans[v].size() > maxFan) { ++stats.fanSkips; return false; }
        if (!anchored(seed,corner)) return false;
        const float *a = uv + 2*size_t(faces[3*size_t(seed)+(corner+1)%3]);
        const float *b = uv + 2*size_t(faces[3*size_t(seed)+(corner+2)%3]);
        const uint32_t representative = aliases[v][0];
        const std::array<float,2> old{{uv[2*size_t(representative)],uv[2*size_t(representative)+1]}};
        const double dx = double(b[0])-a[0], dy = double(b[1])-a[1];
        const double length2 = dx*dx+dy*dy;
        if (!(length2 > 0)) return false;
        const double t = ((double(old[0])-a[0])*dx+(double(old[1])-a[1])*dy)/length2;
        const double foot[2] = {a[0]+t*dx,a[1]+t*dy};
        auto set = [&](float x,float y) {
            for (uint32_t alias : aliases[v]) { uv[2*size_t(alias)] = x; uv[2*size_t(alias)+1] = y; }
        };
        for (uint32_t f : fans[v]) affected[f] = 1;
        bool accepted = false;
        for (double height : {1.0,0.5}) {
            if (!workLeft) { ++stats.budgetSkips; break; }
            ++stats.repairAttempts;
            // First reflect the apex, then try a shorter reflected altitude.
            // Cast BEFORE predicates: the stored float32 layout is what matters.
            set(float(foot[0]+height*(foot[0]-old[0])),float(foot[1]+height*(foot[1]-old[1])));
            bool good = true, exhausted = false;
            for (uint32_t f : fans[v]) if (!validator.usableTriangle(f)) { good = false; break; }
            auto allow = [&](uint32_t f,uint32_t g) {
                return f != seed && g != seed && existing(f,g);
            };
            for (size_t i = 0; good && i < fans[v].size(); ++i) {
                uint32_t f = fans[v][i];
                good = validator.queryTriangle(f,affected,allow,workLeft,exhausted);
                for (size_t j = i+1; good && j < fans[v].size(); ++j) {
                    if (!workLeft) { exhausted = true; good = false; break; }
                    --workLeft;
                    uint32_t g = fans[v][j];
                    if (validator.overlaps(f,g) && !allow(f,g)) good = false;
                }
            }
            std::vector<uint32_t> resolved;
            // Charge graph maintenance to the same budget before committing.
            // Dense conflict fans cannot bypass the spatial-query work cap.
            for (uint32_t f : fans[v]) {
                if (!good) break;
                for (uint32_t i = pairOffsets[f]; i < pairOffsets[f+1]; ++i) {
                    if (!workLeft) { exhausted = true; good = false; break; }
                    --workLeft;
                    uint32_t id = pairIds[i];
                    if (live[id] && !validator.overlaps(pairs[id].first,pairs[id].second)) resolved.push_back(id);
                }
            }
            if (exhausted) ++stats.budgetSkips;
            if (!good) { set(old[0],old[1]); continue; }
            // Every overlap of seed has gone, so the active graph strictly shrinks.
            for (uint32_t id : resolved) if (live[id]) {
                uint32_t x = pairs[id].first, y = pairs[id].second;
                live[id] = 0; --degree[x]; --degree[y];
                for (uint32_t f : {x,y}) {
                    if (!degree[f] && !repaired[f]) { repaired[f] = 1; ++stats.repairedTriangles; }
                    enqueue(f);
                }
            }
            validator.refit(fans[v]);
            if (!moved[v]) { moved[v] = 1; ++stats.movedVariables; }
            for (uint32_t f : fans[v]) {
                enqueue(f);
                for (uint32_t i = graph.offsets[f]; i < graph.offsets[f+1]; ++i) enqueue(graph.neighbors[i]);
            }
            accepted = true; break;
        }
        for (uint32_t f : fans[v]) affected[f] = 0;
        return accepted;
    }
public:
    LocalUvUntangler(float *coordinates, const int32_t *triangles,
                     const std::vector<uint32_t> &vertexVariables, const UvFaceGraph &adjacency,
                     Validator &index, const std::vector<std::pair<uint32_t,uint32_t>> &overlaps,
                     uint32_t faceCount, OverlapFix &result)
        : uv(coordinates), faces(triangles), variables(vertexVariables), graph(adjacency),
          validator(index), stats(result), pairs(overlaps), degree(faceCount,0),
          live(overlaps.size(),1), affected(faceCount,0), visits(faceCount,0),
          pending(faceCount,0), repaired(faceCount,0), workLeft(0) {
        uint32_t count = variables.empty() ? 0 : *std::max_element(variables.begin(),variables.end())+1;
        aliases.resize(count); fans.resize(count); moved.assign(count,0);
        for (uint32_t v = 0; v < variables.size(); ++v) aliases[variables[v]].push_back(v);
        for (uint32_t f = 0; f < faceCount; ++f)
            for (int k = 0; k < 3; ++k) fans[variables[uint32_t(faces[3*size_t(f)+k])]].push_back(f);
        pairOffsets.assign(size_t(faceCount)+1,0);
        pairIndex.reserve(pairs.size());
        for (uint32_t id = 0; id < pairs.size(); ++id) {
            const auto &p = pairs[id];
            ++degree[p.first]; ++degree[p.second];
            ++pairOffsets[p.first+1]; ++pairOffsets[p.second+1];
            pairIndex.emplace(pairKey(p.first,p.second),id);
        }
        for (size_t i = 1; i < pairOffsets.size(); ++i) pairOffsets[i] += pairOffsets[i-1];
        auto cursor = pairOffsets; pairIds.resize(pairs.size()*2);
        for (uint32_t id = 0; id < pairs.size(); ++id) {
            pairIds[cursor[pairs[id].first]++] = id; pairIds[cursor[pairs[id].second]++] = id;
        }
    }
    std::vector<std::pair<uint32_t,uint32_t>> run(uint64_t budgetLimit = 8'000'000) {
        std::vector<uint32_t> seeds;
        for (uint32_t f = 0; f < degree.size(); ++f) if (degree[f]) seeds.push_back(f);
        auto smallestFan = [&](uint32_t f) {
            size_t n = std::numeric_limits<size_t>::max();
            for (int k = 0; k < 3; ++k) n = std::min(n,fans[variables[uint32_t(faces[3*size_t(f)+k])]].size());
            return n;
        };
        std::sort(seeds.begin(),seeds.end(),[&](uint32_t a,uint32_t b) {
            size_t x = smallestFan(a), y = smallestFan(b);
            if (x != y) return x < y;
            return degree[a] != degree[b] ? degree[a] < degree[b] : a < b;
        });
        // Bound query work per chart as well as positions/visits per face.
        const uint64_t budget = std::min<uint64_t>(budgetLimit,4096+512*uint64_t(seeds.size()));
        workLeft = budget;
        for (uint32_t f : seeds) enqueue(f);
        while (!queue.empty()) {
            uint32_t f = queue.front(); queue.pop(); pending[f] = 0;
            if (!degree[f]) continue;
            if (!workLeft) { ++stats.budgetSkips; continue; }
            ++visits[f];
            std::array<int,3> corners{{0,1,2}};
            std::stable_sort(corners.begin(),corners.end(),[&](int a,int b) {
                return fans[variables[uint32_t(faces[3*size_t(f)+a])]].size()
                     < fans[variables[uint32_t(faces[3*size_t(f)+b])]].size();
            });
            for (int corner : corners) if (tryCorner(f,corner)) break;
        }
        stats.repairWork = budget-workLeft;
        std::vector<std::pair<uint32_t,uint32_t>> remaining;
        for (size_t i = 0; i < pairs.size(); ++i) if (live[i]) remaining.push_back(pairs[i]);
        return remaining;
    }
};

// One partitioner for all flatteners: simultaneous connected growth followed
// by a single, leaf-only boundary-balancing pass. Coordinates never change.
class ConnectedUvPatches {
    struct Frontier {
        double support;
        uint32_t face, parent, remaining;
        bool operator<(const Frontier &b) const {
            if (remaining != b.remaining) return remaining > b.remaining;
            if (support != b.support) return support < b.support;
            if (face != b.face) return face > b.face;
            return parent > b.parent;
        }
    };
    struct Patch {
        uint32_t size = 0;
        uint64_t version = 0;
        std::priority_queue<Frontier> frontier;
    };
    struct Scheduled {
        bool small;
        uint32_t size, group;
        double support;
        uint64_t version;
        bool operator<(const Scheduled &b) const {
            if (small != b.small) return small < b.small;
            if (small && size != b.size) return size > b.size;
            if (support != b.support) return support < b.support;
            return group > b.group;
        }
    };
    const UvFaceGraph &graph;
    uint32_t minimum;
    std::vector<uint32_t> conflictOffsets, conflicts, children, unassignedNeighbors;
    std::vector<uint8_t> compatibleEdge;
    std::vector<int32_t> labels, parent;
    std::vector<Patch> patches;
    std::priority_queue<Scheduled> scheduled;
    // Counts permit O(1) eligibility checks, including after boundary moves.
    std::unordered_map<uint64_t, uint32_t> blocked;
    std::unordered_map<uint64_t, double> support;
    std::unordered_map<uint64_t, Frontier> deferred;
    std::vector<std::vector<uint64_t>> wakeOnAssignment;
    static uint64_t key(uint32_t face, uint32_t group) { return (uint64_t(face) << 32) | group; }
    bool isBlocked(uint32_t f, uint32_t g) const { return blocked.find(key(f,g)) != blocked.end(); }
    uint32_t strandedNeighbor(uint32_t f, uint32_t g) const {
        for (uint32_t i = graph.offsets[f]; i < graph.offsets[f+1]; ++i) {
            uint32_t n = graph.neighbors[i];
            if (compatibleEdge[i] && labels[n] < 0 && unassignedNeighbors[n] == 1 && isBlocked(n,g))
                return n;
        }
        return UINT32_MAX;
    }
    void updateBlocks(uint32_t f, uint32_t g, bool add) {
        for (uint32_t i = conflictOffsets[f]; i < conflictOffsets[f+1]; ++i) {
            const uint64_t k = key(conflicts[i],g);
            if (add) ++blocked[k];
            else {
                auto it = blocked.find(k);
                if (it != blocked.end() && --it->second == 0) blocked.erase(it);
            }
        }
    }
    void assign(uint32_t f, uint32_t g, int32_t from) {
        labels[f] = int32_t(g); parent[f] = from;
        if (from >= 0) ++children[uint32_t(from)];
        ++patches[g].size;
        updateBlocks(f,g,true);
        for (uint32_t i = graph.offsets[f]; i < graph.offsets[f+1]; ++i) {
            uint32_t n = graph.neighbors[i];
            if (labels[n] < 0 && compatibleEdge[i]) --unassignedNeighbors[n];
            if (labels[n] >= 0 || isBlocked(n,g)) continue;
            double weight = (support[key(n,g)] += graph.lengths[i]);
            patches[g].frontier.push({weight,n,f,unassignedNeighbors[n]});
        }
        // A reserved connector may become eligible when its blocked neighbor
        // joins another patch. Wake it locally instead of seeding a new island.
        for (uint64_t k : wakeOnAssignment[f]) {
            auto it = deferred.find(k);
            if (it == deferred.end()) continue;
            auto entry = it->second; uint32_t receiver = uint32_t(k);
            deferred.erase(it);
            if (labels[entry.face] >= 0 || isBlocked(entry.face,receiver)) continue;
            entry.support = support.at(k); entry.remaining = unassignedNeighbors[entry.face];
            patches[receiver].frontier.push(entry); schedule(receiver);
        }
        wakeOnAssignment[f].clear();
    }
    void clean(uint32_t g) {
        auto &frontier = patches[g].frontier;
        while (!frontier.empty()) {
            auto entry = frontier.top();
            if (labels[entry.face] < 0 && !isBlocked(entry.face,g) &&
                support.at(key(entry.face,g)) == entry.support) {
                uint32_t stranded = strandedNeighbor(entry.face,g);
                if (stranded != UINT32_MAX) {
                    uint64_t k = key(entry.face,g);
                    if (deferred.emplace(k,entry).second) wakeOnAssignment[stranded].push_back(k);
                    frontier.pop(); continue;
                }
                if (entry.remaining == unassignedNeighbors[entry.face]) break;
                frontier.pop(); entry.remaining = unassignedNeighbors[entry.face]; frontier.push(entry);
                continue;
            }
            frontier.pop();
        }
    }
    void schedule(uint32_t g) {
        clean(g);
        auto &patch = patches[g];
        ++patch.version;
        if (!patch.frontier.empty()) scheduled.push({patch.size < minimum, patch.size,
            g, patch.frontier.top().support, patch.version});
    }
    uint32_t seed(uint32_t f) {
        uint32_t g = uint32_t(patches.size());
        patches.emplace_back();
        assign(f,g,-1);
        schedule(g);
        return g;
    }
    void grow() {
        while (!scheduled.empty()) {
            const auto item = scheduled.top(); scheduled.pop();
            if (item.version != patches[item.group].version) continue;
            clean(item.group);
            auto &frontier = patches[item.group].frontier;
            if (frontier.empty()) continue;
            if (frontier.top().support != item.support) { schedule(item.group); continue; }
            auto entry = frontier.top(); frontier.pop();
            assign(entry.face,item.group,int32_t(entry.parent));
            schedule(item.group);
        }
    }
    uint32_t balance() {
        if (minimum <= 1) return 0;
        std::queue<uint32_t> leaves;
        std::vector<uint8_t> moved(labels.size(), 0);
        for (uint32_t f = 0; f < labels.size(); ++f)
            if (!children[f] && parent[f] >= 0 && patches[uint32_t(labels[f])].size > minimum) leaves.push(f);
        uint32_t transfers = 0;
        while (!leaves.empty()) {
            uint32_t f = leaves.front(); leaves.pop();
            uint32_t donor = uint32_t(labels[f]);
            if (moved[f] || children[f] || parent[f] < 0 || patches[donor].size <= minimum) continue;
            int32_t receiver = -1, attach = -1;
            double best = -1;
            for (uint32_t i = graph.offsets[f]; i < graph.offsets[f+1]; ++i) {
                uint32_t n = graph.neighbors[i], g = uint32_t(labels[n]);
                if (g == donor || patches[g].size >= minimum || isBlocked(f,g)) continue;
                if (receiver < 0 || patches[g].size < patches[uint32_t(receiver)].size ||
                    (patches[g].size == patches[uint32_t(receiver)].size && graph.lengths[i] > best)) {
                    receiver = int32_t(g); attach = int32_t(n); best = graph.lengths[i];
                }
            }
            if (receiver < 0) continue;
            // A leaf can be removed without disconnecting its donor. Attaching
            // it through a shared edge preserves the receiver's spanning tree.
            int32_t oldParent = parent[f];
            if (oldParent >= 0 && --children[uint32_t(oldParent)] == 0) leaves.push(uint32_t(oldParent));
            updateBlocks(f,donor,false);
            updateBlocks(f,uint32_t(receiver),true);
            --patches[donor].size; ++patches[uint32_t(receiver)].size;
            labels[f] = receiver; parent[f] = attach; ++children[uint32_t(attach)];
            moved[f] = 1; ++transfers;
        }
        return transfers;
    }
public:
    ConnectedUvPatches(const UvFaceGraph &adjacency, uint32_t faceCount,
                       const std::vector<std::pair<uint32_t,uint32_t>> &pairs,
                       uint32_t target)
        : graph(adjacency), minimum(std::max(1u,target)), children(faceCount,0),
          labels(faceCount,-1), parent(faceCount,-1), wakeOnAssignment(faceCount) {
        conflictOffsets.assign(size_t(faceCount)+1,0);
        for (const auto &pair : pairs) { ++conflictOffsets[pair.first+1]; ++conflictOffsets[pair.second+1]; }
        for (size_t i = 1; i < conflictOffsets.size(); ++i) conflictOffsets[i] += conflictOffsets[i-1];
        auto cursor = conflictOffsets;
        conflicts.resize(pairs.size()*2);
        for (const auto &pair : pairs) {
            conflicts[cursor[pair.first]++] = pair.second;
            conflicts[cursor[pair.second]++] = pair.first;
        }
        // Sorted conflict rows make compatibility tests cheap without a second
        // hash table proportional to all overlap pairs.
        for (uint32_t f = 0; f < faceCount; ++f)
            std::sort(conflicts.begin()+conflictOffsets[f],conflicts.begin()+conflictOffsets[f+1]);
        unassignedNeighbors.assign(faceCount,0); compatibleEdge.resize(graph.neighbors.size());
        for (uint32_t f = 0; f < faceCount; ++f)
            for (uint32_t i = graph.offsets[f]; i < graph.offsets[f+1]; ++i) {
                bool compatible = !std::binary_search(conflicts.begin()+conflictOffsets[f],
                    conflicts.begin()+conflictOffsets[f+1],graph.neighbors[i]);
                compatibleEdge[i] = uint8_t(compatible);
                unassignedNeighbors[f] += uint32_t(compatible);
            }
    }
    void run(OverlapFix &result, bool limited = false) {
        const uint32_t count = uint32_t(labels.size());
        UnionFind connected{int(count)};
        for (uint32_t f = 0; f < count; ++f)
            for (uint32_t i = graph.offsets[f]; i < graph.offsets[f+1]; ++i) connected.join(int(f),int(graph.neighbors[i]));
        std::vector<int32_t> componentId(count,-1);
        std::vector<std::vector<uint32_t>> components;
        for (uint32_t f = 0; f < count; ++f) {
            int root = connected.root(int(f));
            if (componentId[uint32_t(root)] < 0) {
                componentId[uint32_t(root)] = int32_t(components.size()); components.emplace_back();
            }
            components[size_t(componentId[uint32_t(root)])].push_back(f);
        }
        result.inputIslands = uint32_t(components.size());
        for (const auto &component : components) if (component.size() == 1) ++result.inputSingletons;
        for (const auto &component : components) if (component.size() < minimum) ++result.inputUndersized;
        if (limited) {
            // Retain the existing dense-graph safety limit. Allocate before
            // writing: the previous safeguard indexed an empty groups vector.
            result.groups.resize(count);
            std::iota(result.groups.begin(),result.groups.end(),0);
            result.chartCount = count;
            result.outputSingletons = count;
            result.outputUndersized = minimum > 1 ? count : 0;
            return;
        }
        auto degree = [&](uint32_t f) { return conflictOffsets[f+1]-conflictOffsets[f]; };
        for (auto &component : components) {
            uint32_t first = component[0];
            for (uint32_t f : component) if (degree(f) > degree(first)) first = f;
            int32_t second = -1;
            // Cross-component overlap pairs need no cut: the surface islands
            // already become separate patches.
            for (uint32_t f : component) for (uint32_t i = conflictOffsets[f]; i < conflictOffsets[f+1]; ++i) {
                uint32_t n = conflicts[i];
                if (connected.root(int(f)) != connected.root(int(n))) continue;
                if (second < 0 || degree(f) > degree(first) ||
                    (degree(f) == degree(first) && degree(n) > degree(uint32_t(second)))) {
                    first = f; second = int32_t(n);
                }
            }
            if (second < 0) {
                // Whole-component fast path. No queues or sparse blocking
                // entries are needed for an overlap-free surface island.
                uint32_t g = uint32_t(patches.size()); patches.emplace_back();
                patches[g].size = uint32_t(component.size());
                for (uint32_t f : component) labels[f] = int32_t(g);
                continue;
            }
            std::sort(component.begin(),component.end(),[&](uint32_t a,uint32_t b) {
                return degree(a) != degree(b) ? degree(a) > degree(b) : a < b;
            });
            seed(first); seed(uint32_t(second));
            grow();
            for (uint32_t f : component) if (labels[f] < 0) { seed(f); grow(); }
        }
        result.transferredFaces = balance();
        result.groups = std::move(labels);
        result.chartCount = uint32_t(patches.size());
        for (const auto &patch : patches) if (patch.size < minimum) ++result.outputUndersized;
        for (uint32_t f = 0; f < count; ++f) if (patches[uint32_t(result.groups[f])].size == 1) {
            ++result.outputSingletons;
            bool compatible = false;
            for (uint32_t i = graph.offsets[f]; i < graph.offsets[f+1]; ++i) compatible |= bool(compatibleEdge[i]);
            if (!compatible) ++result.blockedSingletons;
        }
    }
};

inline OverlapFix fixOverlappingUvFaces(
    float *uv,
    const int32_t *faces,
    uint32_t vertexCount,
    uint32_t faceCount,
    const int64_t *vertexIds = nullptr,
    uint32_t minimum = 0
) {
    OverlapFix result;
    if (!faceCount) return result;

    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    // A dense overlap graph is not useful to materialize without a guard.
    // The conservative fallback below gives every face its own chart, which
    // guarantees separation while keeping normal charts allocation-light.
    constexpr uint64_t maxPairs = 4'000'000;
    Validation validation;
    Validator validator(uv, faces);
    auto started = Clock::now();
    const bool complete = validator.collectOverlaps(
        vertexCount, faceCount, pairs, result.candidates, maxPairs, validation
    );
    result.detectionSeconds = seconds(started);
    if (!validation.issue && !complete) {
        result.limited = true;
        result.overlapPairs = pairs.size();
        result.overlapFaces = faceCount;
    }
    if (validation.issue) {
        result.valid = false;
        return result;
    }
    std::vector<uint32_t> degree(faceCount, 0);
    for (const auto &pair : pairs) {
        ++degree[pair.first]; ++degree[pair.second];
    }
    result.overlapPairs = pairs.size();
    if (!result.limited) for (uint32_t f = 0; f < faceCount; ++f) if (degree[f]) ++result.overlapFaces;
    const auto variables = uvVariables(uv,vertexCount,vertexIds);
    auto graph = uvFaceGraph(uv,faces,vertexCount,faceCount,vertexIds,&variables);
    started = Clock::now();
    if (!pairs.empty() && !result.limited) {
        LocalUvUntangler repair(uv,faces,variables,graph,validator,pairs,faceCount,result);
        pairs = repair.run();
        graph = uvFaceGraph(uv,faces,vertexCount,faceCount,vertexIds,&variables);
    }
    result.repairSeconds = seconds(started);
    result.remainingPairs = pairs.size();
    started = Clock::now();
    ConnectedUvPatches partitioner(graph,faceCount,pairs,minimum);
    partitioner.run(result,result.limited);
    result.partitionSeconds = seconds(started);
    return result;
}

// Glue a dual spanning tree, opening complex topology into disks. Faces keep
// their original corner ordering. No position-based welding is performed.
inline void cutToDisks(std::vector<int32_t> &faces, std::vector<int32_t> &vmap,
                       const Topology &topology, int &seams) {
    const std::vector<int32_t> originalFaces = faces;
    UnionFind components(int(faces.size()/3)), corners(int(faces.size()));
    // Keep a dual spanning tree glued. Cutting every other interior edge opens
    // handles and holes while keeping the result connected. The former
    // implementation glued the cotree edges, which could leave the topology
    // unchanged or produce a disconnected fan.
    for (const auto &e : topology.edges) if (e.corners.size() == 2) {
        int c=e.corners[0], d=e.corners[1];
        if (components.root(c/3) == components.root(d/3)) { ++seams; continue; }
        components.join(c/3,d/3);
        for (int v : {e.a,e.b}) {
            int x=c/3*3, y=d/3*3;
            while(x<c/3*3+3 && originalFaces[x]!=v)++x;
            while(y<d/3*3+3 && originalFaces[y]!=v)++y;
            if (x>=c/3*3+3 || y>=d/3*3+3) continue;
            corners.join(x,y);
        }
    }
    std::vector<int> ids(faces.size(), -1); std::vector<int32_t> nextMap;
    for (int c=0;c<int(faces.size());++c) {
        int r=corners.root(c);
        if(ids[r]<0) { ids[r]=int(nextMap.size()); nextMap.push_back(vmap[originalFaces[c]]); }
        faces[c]=ids[r];
    }
    vmap.swap(nextMap);
}

// Jacobi-preconditioned CG for the symmetric uniform-weight Dirichlet graph
// Laplacian. Reuse the graph for both coordinate solves, in double precision.
inline bool tutte(const Topology &top, std::vector<float> &uv,
                  std::vector<int> *boundaryLoop = nullptr,
                  const std::vector<float> *fixedBoundaryUv = nullptr) {
    std::vector<int> loop;
    if (boundaryLoop) boundaryLoop->clear();
    if (!top.disk(loop)) return false;
    if (fixedBoundaryUv && fixedBoundaryUv->size() != loop.size() * 2)
        return false;
    const size_t n=top.neighbors.size();
    std::vector<char> fixed(n,false); std::vector<double> xy(n*2,0);
    for(size_t k=0;k<loop.size();++k) {
        double angle=6.2831853071795864769*double(k)/double(loop.size());
        fixed[loop[k]]=true; xy[2*loop[k]]=std::cos(angle); xy[2*loop[k]+1]=std::sin(angle);
        if (fixedBoundaryUv) {
            const float u = (*fixedBoundaryUv)[2 * k];
            const float v = (*fixedBoundaryUv)[2 * k + 1];
            if (!std::isfinite(u) || !std::isfinite(v)) return false;
            xy[2 * loop[k]] = u; xy[2 * loop[k] + 1] = v;
        }
    }
    std::vector<double> x(n),b(n),r(n),z(n),p(n),ap(n);
    for(int axis=0;axis<2;++axis) {
        std::fill(x.begin(),x.end(),0); std::fill(b.begin(),b.end(),0);
        double norm=0,rz=0;
        for(size_t v=0;v<n;++v) if(!fixed[v] && !top.neighbors[v].empty()) {
            for(int w:top.neighbors[v]) if(fixed[w]) b[v]+=xy[2*w+axis];
            norm+=b[v]*b[v];
        }
        for(size_t v=0;v<n;++v) {
            r[v]=b[v]; z[v]=top.neighbors[v].empty()?0:r[v]/double(top.neighbors[v].size()); p[v]=z[v]; rz+=r[v]*z[v];
        }
        double residual=norm, tolerance=std::max(1e-28,norm*1e-12);
        size_t limit=std::min<size_t>(10000,std::max<size_t>(64,n*5));
        for(size_t it=0;residual>tolerance && it<limit;++it) {
            if (std::abs(rz) <= std::numeric_limits<double>::min()) return false;
            double pap=0;
            for(size_t v=0;v<n;++v) {
                ap[v]=0;
                if(!fixed[v]) {
                    ap[v]=double(top.neighbors[v].size())*p[v];
                    for(int w:top.neighbors[v]) if(!fixed[w]) ap[v]-=p[w];
                }
                pap+=p[v]*ap[v];
            }
            if(!(pap>0) || !std::isfinite(pap)) return false;
            double alpha=rz/pap,next=0; residual=0;
            for(size_t v=0;v<n;++v) {
                x[v]+=alpha*p[v]; r[v]-=alpha*ap[v]; residual+=r[v]*r[v];
                z[v]=top.neighbors[v].empty()?0:r[v]/double(top.neighbors[v].size()); next+=r[v]*z[v];
            }
            double beta=next/rz; rz=next;
            for(size_t v=0;v<n;++v)p[v]=z[v]+beta*p[v];
        }
        if(!std::isfinite(residual) || residual>tolerance) return false;
        for(size_t v=0;v<n;++v) if(!fixed[v]) xy[2*v+axis]=x[v];
    }
    uv.resize(n*2);
    for(size_t i=0;i<xy.size();++i)uv[i]=float(xy[i]);
    if (boundaryLoop) *boundaryLoop = loop;
    return true;
}

struct RepairPiece {
    std::vector<int32_t> faces, vmap, faceIds;
    std::vector<int> boundary;
    std::vector<float> uv;
};

struct ProjectedBoundaryPoint {
    double x = 0, y = 0;
};

inline double cross2(const ProjectedBoundaryPoint &a,
                     const ProjectedBoundaryPoint &b,
                     const ProjectedBoundaryPoint &c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// Project the chart boundary to a local plane and sample its convex hull in
// boundary order. Every original boundary vertex remains represented, but
// the fixed polygon is no longer forced to be a circle. The result is scaled
// to the same area as the unit circle so the subsequent solve keeps a useful
// numerical scale. Returning false simply selects the old circular fallback.
inline bool build_tutte_convex_boundary(
    const float *positions,
    const std::vector<int32_t> &vmap,
    const std::vector<int> &boundary,
    std::vector<float> &boundaryUv) {
    boundaryUv.clear();
    if (!positions || boundary.size() < 3) return false;

    double center[3] = {0, 0, 0};
    for (int localVertex : boundary) {
        if (localVertex < 0 || size_t(localVertex) >= vmap.size() ||
            vmap[localVertex] < 0)
            return false;
        const float *p = positions + size_t(vmap[localVertex]) * 3;
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
            return false;
        center[0] += p[0]; center[1] += p[1]; center[2] += p[2];
    }
    const double inverseBoundaryCount = 1.0 / double(boundary.size());
    center[0] *= inverseBoundaryCount;
    center[1] *= inverseBoundaryCount;
    center[2] *= inverseBoundaryCount;

    // Newell's normal is a cheap plane estimate for an ordered polygon.
    double normal[3] = {0, 0, 0};
    for (size_t k = 0; k < boundary.size(); ++k) {
        const float *p = positions + size_t(vmap[boundary[k]]) * 3;
        const float *q = positions + size_t(vmap[boundary[(k + 1) % boundary.size()]]) * 3;
        const double px = double(p[0]) - center[0], py = double(p[1]) - center[1];
        const double pz = double(p[2]) - center[2];
        const double qx = double(q[0]) - center[0], qy = double(q[1]) - center[1];
        const double qz = double(q[2]) - center[2];
        normal[0] += (py - qy) * (pz + qz);
        normal[1] += (pz - qz) * (px + qx);
        normal[2] += (px - qx) * (py + qy);
    }
    double normalLength2 = normal[0] * normal[0] + normal[1] * normal[1] +
        normal[2] * normal[2];
    if (!(normalLength2 > 1e-28) || !std::isfinite(normalLength2)) {
        for (size_t k = 0; k < boundary.size(); ++k) {
            const float *p0 = positions + size_t(vmap[boundary[k]]) * 3;
            const float *p1 = positions + size_t(vmap[boundary[(k + 1) % boundary.size()]]) * 3;
            const float *p2 = positions + size_t(vmap[boundary[(k + 2) % boundary.size()]]) * 3;
            const double ax = double(p1[0]) - p0[0], ay = double(p1[1]) - p0[1];
            const double az = double(p1[2]) - p0[2];
            const double bx = double(p2[0]) - p0[0], by = double(p2[1]) - p0[1];
            const double bz = double(p2[2]) - p0[2];
            normal[0] = ay * bz - az * by;
            normal[1] = az * bx - ax * bz;
            normal[2] = ax * by - ay * bx;
            normalLength2 = normal[0] * normal[0] + normal[1] * normal[1] +
                normal[2] * normal[2];
            if (normalLength2 > 1e-28 && std::isfinite(normalLength2)) break;
        }
    }
    if (!(normalLength2 > 1e-28) || !std::isfinite(normalLength2)) return false;
    const double inverseNormalLength = 1.0 / std::sqrt(normalLength2);
    normal[0] *= inverseNormalLength;
    normal[1] *= inverseNormalLength;
    normal[2] *= inverseNormalLength;

    const double absNormal[3] = {
        std::abs(normal[0]), std::abs(normal[1]), std::abs(normal[2])};
    const int referenceAxis = absNormal[0] <= absNormal[1] && absNormal[0] <= absNormal[2]
        ? 0 : (absNormal[1] <= absNormal[2] ? 1 : 2);
    const double reference[3] = {
        referenceAxis == 0 ? 1.0 : 0.0,
        referenceAxis == 1 ? 1.0 : 0.0,
        referenceAxis == 2 ? 1.0 : 0.0};
    double tangent[3] = {
        reference[1] * normal[2] - reference[2] * normal[1],
        reference[2] * normal[0] - reference[0] * normal[2],
        reference[0] * normal[1] - reference[1] * normal[0]};
    const double tangentLength = std::sqrt(
        tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]);
    if (!(tangentLength > 1e-14) || !std::isfinite(tangentLength)) return false;
    tangent[0] /= tangentLength; tangent[1] /= tangentLength; tangent[2] /= tangentLength;
    double bitangent[3] = {
        normal[1] * tangent[2] - normal[2] * tangent[1],
        normal[2] * tangent[0] - normal[0] * tangent[2],
        normal[0] * tangent[1] - normal[1] * tangent[0]};

    std::vector<ProjectedBoundaryPoint> projected(boundary.size());
    for (size_t k = 0; k < boundary.size(); ++k) {
        const float *p = positions + size_t(vmap[boundary[k]]) * 3;
        const double dx = double(p[0]) - center[0];
        const double dy = double(p[1]) - center[1];
        const double dz = double(p[2]) - center[2];
        projected[k].x = dx * tangent[0] + dy * tangent[1] + dz * tangent[2];
        projected[k].y = dx * bitangent[0] + dy * bitangent[1] + dz * bitangent[2];
        if (!std::isfinite(projected[k].x) || !std::isfinite(projected[k].y)) return false;
    }

    // Make the projected boundary and the monotone-chain hull use the same
    // winding as the loop order. This keeps the fixed UV polygon positive.
    double boundaryArea2 = 0;
    for (size_t k = 0; k < projected.size(); ++k) {
        const ProjectedBoundaryPoint &a = projected[k];
        const ProjectedBoundaryPoint &b = projected[(k + 1) % projected.size()];
        boundaryArea2 += a.x * b.y - b.x * a.y;
    }
    if (boundaryArea2 < 0) {
        for (ProjectedBoundaryPoint &point : projected) point.y = -point.y;
    }

    double minX = projected[0].x, maxX = projected[0].x;
    double minY = projected[0].y, maxY = projected[0].y;
    for (const ProjectedBoundaryPoint &point : projected) {
        minX = std::min(minX, point.x); maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y); maxY = std::max(maxY, point.y);
    }
    const double extent = std::max(maxX - minX, maxY - minY);
    if (!(extent > 1e-14) || !std::isfinite(extent)) return false;
    const double hullEpsilon = 1e-12 * std::max(1e-24, extent * extent);

    std::vector<int> order(projected.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (projected[a].x != projected[b].x) return projected[a].x < projected[b].x;
        if (projected[a].y != projected[b].y) return projected[a].y < projected[b].y;
        return a < b;
    });

    std::vector<int> hull;
    hull.reserve(order.size() * 2);
    for (int point : order) {
        while (hull.size() >= 2 &&
               cross2(projected[hull[hull.size() - 2]], projected[hull.back()],
                      projected[point]) <= hullEpsilon)
            hull.pop_back();
        hull.push_back(point);
    }
    const size_t lowerSize = hull.size();
    if (order.size() >= 2) {
        for (int i = int(order.size()) - 2; i >= 0; --i) {
            while (hull.size() > lowerSize &&
                   cross2(projected[hull[hull.size() - 2]], projected[hull.back()],
                          projected[order[size_t(i)]]) <= hullEpsilon)
                hull.pop_back();
            hull.push_back(order[size_t(i)]);
        }
    }
    if (hull.size() > 1) hull.pop_back();
    if (hull.size() < 3) return false;

    std::vector<double> hullDistance(hull.size() + 1, 0.0);
    double hullArea2 = 0;
    double hullCenterX = 0, hullCenterY = 0;
    for (size_t i = 0; i < hull.size(); ++i) {
        const ProjectedBoundaryPoint &a = projected[hull[i]];
        const ProjectedBoundaryPoint &b = projected[hull[(i + 1) % hull.size()]];
        const double edgeLength = std::hypot(b.x - a.x, b.y - a.y);
        if (!(edgeLength > 1e-20) || !std::isfinite(edgeLength)) return false;
        hullDistance[i + 1] = hullDistance[i] + edgeLength;
        hullArea2 += a.x * b.y - b.x * a.y;
        hullCenterX += a.x; hullCenterY += a.y;
    }
    const double hullPerimeter = hullDistance.back();
    if (!(hullArea2 > extent * extent * 1e-12) ||
        !(hullPerimeter > 1e-14) || !std::isfinite(hullArea2))
        return false;
    hullCenterX /= double(hull.size());
    hullCenterY /= double(hull.size());
    const double uvScale = std::sqrt(3.14159265358979323846 / (0.5 * hullArea2));
    if (!(uvScale > 0) || !std::isfinite(uvScale)) return false;

    // Use physical boundary edge lengths to sample the hull. This avoids
    // over-weighting dense triangulation along one part of the outline.
    std::vector<double> sourceDistance(boundary.size() + 1, 0.0);
    for (size_t k = 0; k < boundary.size(); ++k) {
        const float *p = positions + size_t(vmap[boundary[k]]) * 3;
        const float *q = positions + size_t(vmap[boundary[(k + 1) % boundary.size()]]) * 3;
        const double dx = double(q[0]) - p[0], dy = double(q[1]) - p[1];
        const double dz = double(q[2]) - p[2];
        const double edgeLength = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (!(edgeLength > 1e-20) || !std::isfinite(edgeLength)) return false;
        sourceDistance[k + 1] = sourceDistance[k] + edgeLength;
    }
    const double sourcePerimeter = sourceDistance.back();
    if (!(sourcePerimeter > 1e-14) || !std::isfinite(sourcePerimeter)) return false;

    // Align the first source boundary vertex with the closest point on the
    // hull. This preserves the correspondence for already-convex charts.
    double startDistance = 0;
    double closestDistance2 = std::numeric_limits<double>::infinity();
    const ProjectedBoundaryPoint &sourceStart = projected[0];
    for (size_t i = 0; i < hull.size(); ++i) {
        const ProjectedBoundaryPoint &a = projected[hull[i]];
        const ProjectedBoundaryPoint &b = projected[hull[(i + 1) % hull.size()]];
        const double dx = b.x - a.x, dy = b.y - a.y;
        const double length2 = dx * dx + dy * dy;
        const double length = std::sqrt(length2);
        double t = ((sourceStart.x - a.x) * dx + (sourceStart.y - a.y) * dy) / length2;
        t = std::max(0.0, std::min(1.0, t));
        const double closestX = a.x + t * dx, closestY = a.y + t * dy;
        const double distance2 = (sourceStart.x - closestX) * (sourceStart.x - closestX) +
            (sourceStart.y - closestY) * (sourceStart.y - closestY);
        if (distance2 < closestDistance2) {
            closestDistance2 = distance2;
            startDistance = hullDistance[i] + t * length;
        }
    }
    if (startDistance >= hullPerimeter) startDistance = 0;

    boundaryUv.resize(boundary.size() * 2);
    size_t hullSegment = 0;
    while (hullSegment + 1 < hull.size() &&
           hullDistance[hullSegment + 1] <= startDistance)
        ++hullSegment;
    double previousDistance = startDistance;
    for (size_t k = 0; k < boundary.size(); ++k) {
        double targetDistance = startDistance +
            sourceDistance[k] / sourcePerimeter * hullPerimeter;
        if (targetDistance >= hullPerimeter) targetDistance -= hullPerimeter;
        if (k > 0 && targetDistance < previousDistance) hullSegment = 0;
        while (hullSegment + 1 < hull.size() &&
               hullDistance[hullSegment + 1] <= targetDistance)
            ++hullSegment;
        if (hullSegment >= hull.size()) hullSegment = hull.size() - 1;
        const ProjectedBoundaryPoint &a = projected[hull[hullSegment]];
        const ProjectedBoundaryPoint &b = projected[hull[(hullSegment + 1) % hull.size()]];
        const double segmentStart = hullDistance[hullSegment];
        const double segmentLength = hullDistance[hullSegment + 1] - segmentStart;
        if (!(segmentLength > 1e-20)) return false;
        const double t = (targetDistance - segmentStart) / segmentLength;
        const double x = a.x + t * (b.x - a.x);
        const double y = a.y + t * (b.y - a.y);
        boundaryUv[2 * k] = float((x - hullCenterX) * uvScale);
        boundaryUv[2 * k + 1] = float((y - hullCenterY) * uvScale);
        previousDistance = targetDistance;
    }
    return true;
}

// A compact local metric for one triangle.  The 3D triangle is expressed in
// an orthonormal 2D basis once, so the nonlinear iterations only touch UVs.
struct DistortionTriangle {
    int32_t a = 0, b = 0, c = 0;
    double x1 = 0, x2 = 0, y2 = 0;
    double weight = 0;
    double orientation = 1;
    double minCross = 0;
};

inline bool distortion_candidate_is_valid(
    const std::vector<float> &uv,
    const std::vector<DistortionTriangle> &triangles) {
    for (const DistortionTriangle &triangle : triangles) {
        const float *p0 = uv.data() + size_t(triangle.a) * 2;
        const float *p1 = uv.data() + size_t(triangle.b) * 2;
        const float *p2 = uv.data() + size_t(triangle.c) * 2;
        const double cross =
            (double(p1[0]) - p0[0]) * (double(p2[1]) - p0[1]) -
            (double(p1[1]) - p0[1]) * (double(p2[0]) - p0[0]);
        if (!std::isfinite(cross) ||
            triangle.orientation * cross <= triangle.minCross)
            return false;
    }
    return true;
}

// Minimize the scale-invariant symmetric Dirichlet distortion
//     (||J||^2) / det(J)
// for one already valid UV chart.  In 2D this is the sum of singular-value
// ratios, so it penalizes both stretch and compression without imposing a
// target UV scale.  The boundary is intentionally unconstrained.  A short
// backtracking line search accepts only orientation-preserving candidates.
//
// ``skipTolerance`` is relative to the theoretical per-area minimum (2.0).
// It lets callers avoid spending time on charts that are already close to
// isometric/conformal.
inline bool optimize_distortion(
    const float *positions,
    const std::vector<int32_t> &faces,
    const std::vector<int32_t> &vmap,
    std::vector<float> &uv,
    int maxIterations,
    int maxLineSearchSteps,
    double relativeImprovement,
    double skipTolerance) {

    if (faces.empty() || faces.size() % 3 != 0 ||
        uv.size() != size_t(vmap.size()) * 2)
        return false;

    std::vector<DistortionTriangle> triangles;
    triangles.reserve(faces.size() / 3);
    double referenceArea = 0.0;
    for (size_t face = 0; face < faces.size(); face += 3) {
        const int32_t a = faces[face], b = faces[face + 1], c = faces[face + 2];
        if (a < 0 || b < 0 || c < 0 ||
            size_t(a) >= vmap.size() || size_t(b) >= vmap.size() || size_t(c) >= vmap.size())
            return false;

        const float *p0 = positions + size_t(vmap[a]) * 3;
        const float *p1 = positions + size_t(vmap[b]) * 3;
        const float *p2 = positions + size_t(vmap[c]) * 3;
        const double e1x = double(p1[0]) - p0[0];
        const double e1y = double(p1[1]) - p0[1];
        const double e1z = double(p1[2]) - p0[2];
        const double e2x = double(p2[0]) - p0[0];
        const double e2y = double(p2[1]) - p0[1];
        const double e2z = double(p2[2]) - p0[2];
        const double x1 = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
        if (!(x1 > 1e-20) || !std::isfinite(x1)) return false;
        const double x2 = (e1x * e2x + e1y * e2y + e1z * e2z) / x1;
        const double e2Length2 = e2x * e2x + e2y * e2y + e2z * e2z;
        const double y2Squared = e2Length2 - x2 * x2;
        if (!(y2Squared > 1e-40) || !std::isfinite(y2Squared)) return false;
        const double y2 = std::sqrt(y2Squared);

        const float *uv0 = uv.data() + size_t(a) * 2;
        const float *uv1 = uv.data() + size_t(b) * 2;
        const float *uv2 = uv.data() + size_t(c) * 2;
        const double cross =
            (double(uv1[0]) - uv0[0]) * (double(uv2[1]) - uv0[1]) -
            (double(uv1[1]) - uv0[1]) * (double(uv2[0]) - uv0[0]);
        if (!std::isfinite(cross) || cross == 0) return false;

        DistortionTriangle triangle;
        triangle.a = a; triangle.b = b; triangle.c = c;
        triangle.x1 = x1; triangle.x2 = x2; triangle.y2 = y2;
        triangle.weight = 0.5 * x1 * y2;
        triangle.orientation = cross > 0 ? 1.0 : -1.0;
        triangle.minCross = std::max(1e-16, std::abs(cross) * 1e-8);
        referenceArea += triangle.weight;
        triangles.push_back(triangle);
    }
    if (triangles.empty() || !distortion_candidate_is_valid(uv, triangles)) return false;

    // These buffers are allocated only if the initial energy is high enough
    // to justify an optimization pass.
    std::vector<double> diagonal;
    std::vector<double> gradient;
    auto energy_and_gradient = [&](bool withGradient) {
        if (withGradient) std::fill(gradient.begin(), gradient.end(), 0.0);
        double energy = 0.0;
        for (const DistortionTriangle &triangle : triangles) {
            const float *uv0 = uv.data() + size_t(triangle.a) * 2;
            const float *uv1 = uv.data() + size_t(triangle.b) * 2;
            const float *uv2 = uv.data() + size_t(triangle.c) * 2;
            const double du1x = double(uv1[0]) - uv0[0];
            const double du1y = double(uv1[1]) - uv0[1];
            const double du2x = double(uv2[0]) - uv0[0];
            const double du2y = double(uv2[1]) - uv0[1];
            const double a = du1x / triangle.x1;
            const double c = du1y / triangle.x1;
            const double b = (du2x - a * triangle.x2) / triangle.y2;
            const double d = (du2y - c * triangle.x2) / triangle.y2;
            const double det = a * d - b * c;
            const double positiveDet = triangle.orientation * det;
            const double squaredNorm = a * a + b * b + c * c + d * d;
            if (!(positiveDet > triangle.minCross /
                    (triangle.x1 * triangle.y2)) ||
                !std::isfinite(positiveDet) || !std::isfinite(squaredNorm))
                return std::numeric_limits<double>::infinity();
            energy += triangle.weight * squaredNorm / positiveDet;

            if (withGradient) {
                const double inverseDet2 = 1.0 / (positiveDet * positiveDet);
                const double factor = triangle.weight;
                const double ga = factor * (2.0 * a / positiveDet -
                    squaredNorm * triangle.orientation * d * inverseDet2);
                const double gb = factor * (2.0 * b / positiveDet +
                    squaredNorm * triangle.orientation * c * inverseDet2);
                const double gc = factor * (2.0 * c / positiveDet +
                    squaredNorm * triangle.orientation * b * inverseDet2);
                const double gd = factor * (2.0 * d / positiveDet -
                    squaredNorm * triangle.orientation * a * inverseDet2);

                const double gdu1x = ga / triangle.x1 -
                    gb * triangle.x2 / (triangle.x1 * triangle.y2);
                const double gdu1y = gc / triangle.x1 -
                    gd * triangle.x2 / (triangle.x1 * triangle.y2);
                const double gdu2x = gb / triangle.y2;
                const double gdu2y = gd / triangle.y2;
                gradient[size_t(triangle.a) * 2] -= gdu1x + gdu2x;
                gradient[size_t(triangle.a) * 2 + 1] -= gdu1y + gdu2y;
                gradient[size_t(triangle.b) * 2] += gdu1x;
                gradient[size_t(triangle.b) * 2 + 1] += gdu1y;
                gradient[size_t(triangle.c) * 2] += gdu2x;
                gradient[size_t(triangle.c) * 2 + 1] += gdu2y;
            }
        }
        return std::isfinite(energy) ? energy : std::numeric_limits<double>::infinity();
    };

    auto compute_direction = [&]() {
        const double value = energy_and_gradient(true);
        if (std::isfinite(value)) {
            for (size_t vertex = 0; vertex < vmap.size(); ++vertex) {
                const double scale = std::max(1e-12, diagonal[vertex]);
                gradient[2 * vertex] /= scale;
                gradient[2 * vertex + 1] /= scale;
            }
        }
        return value;
    };

    const double initialEnergy = energy_and_gradient(false);
    if (!std::isfinite(initialEnergy) || !(referenceArea > 0.0)) return false;
    if (skipTolerance > 0.0 &&
        initialEnergy <= 2.0 * referenceArea * (1.0 + skipTolerance))
        return false;

    // Approximate the diagonal of the local UV Hessian. Dividing by it keeps
    // high-valence or very small triangles from setting the step size for the
    // entire chart, while adding only one cheap pass over the triangles.
    diagonal.assign(vmap.size(), 0.0);
    for (const DistortionTriangle &triangle : triangles) {
        const double invX1 = 1.0 / triangle.x1;
        const double invY2 = 1.0 / triangle.y2;
        const double edge1 = triangle.weight * invX1 * invX1 *
            (1.0 + triangle.x2 * triangle.x2 * invY2 * invY2);
        const double edge2 = triangle.weight * invY2 * invY2;
        diagonal[triangle.a] += edge1 + edge2;
        diagonal[triangle.b] += edge1;
        diagonal[triangle.c] += edge2;
    }
    gradient.resize(uv.size());

    double energy = compute_direction();
    if (!std::isfinite(energy)) return false;
    bool changed = false;
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        double maxDirection = 0.0, minUv = std::numeric_limits<double>::infinity();
        double maxUv = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < uv.size(); ++i) {
            maxDirection = std::max(maxDirection, std::abs(gradient[i]));
            minUv = std::min(minUv, double(uv[i]));
            maxUv = std::max(maxUv, double(uv[i]));
        }
        if (!(maxDirection > 1e-12) || !std::isfinite(maxDirection)) return changed;
        const double uvScale = std::max(1e-6, maxUv - minUv);
        // A larger trial step makes boundary vertices leave the circular
        // initialization within the same small iteration budget. The line
        // search still reduces it whenever this would threaten validity.
        double step = 0.25 * uvScale / maxDirection;
        bool accepted = false;
        for (int lineSearch = 0; lineSearch < maxLineSearchSteps; ++lineSearch) {
            for (size_t i = 0; i < uv.size(); ++i)
                uv[i] = float(double(uv[i]) - step * gradient[i]);
            if (distortion_candidate_is_valid(uv, triangles)) {
                const double trialEnergy = energy_and_gradient(false);
                const double improvement = energy - trialEnergy;
                if (std::isfinite(trialEnergy) &&
                    improvement > std::max(1e-9, std::abs(energy) * relativeImprovement)) {
                    energy = trialEnergy;
                    accepted = true;
                    changed = true;
                    break;
                }
            }
            for (size_t i = 0; i < uv.size(); ++i)
                uv[i] = float(double(uv[i]) + step * gradient[i]);
            step *= 0.5;
        }
        if (!accepted) return changed;
        // Recompute the descent direction only after accepting a step.
        energy = compute_direction();
        if (!std::isfinite(energy)) return changed;
    }
    return changed;
}

// Keep the original Tutte-only entry point for the earlier experiment.  The
// final post-flatten pass below uses a smaller budget and applies to all
// flatteners.
inline void optimize_tutte_distortion(
    const float *positions,
    const std::vector<int32_t> &faces,
    const std::vector<int32_t> &vmap,
    std::vector<float> &uv) {
    (void) optimize_distortion(
        positions, faces, vmap, uv, 8, 8, 2e-6, 0.0);
}

inline bool optimize_uv_distortion(
    const float *positions,
    const std::vector<int32_t> &faces,
    const std::vector<int32_t> &vmap,
    std::vector<float> &uv) {
    // Four nonlinear steps are enough to remove the largest local stretch
    // while keeping this pass cheap for the many charts in a mesh.
    return optimize_distortion(
        positions, faces, vmap, uv, 4, 6, 1e-5, 0.03);
}

// Keep the nonlinear pass disabled while comparing the projected convex-hull
// Tutte initialization. Re-enable this switch after the boundary experiment.
constexpr bool kEnableTutteDistortionOptimizer = false;

inline Validation validate(const std::vector<float> &uv, const std::vector<int32_t> &faces,
                           LscmChartResult &stats) {
    Validation v = Validator(uv.data(), faces.data()).run(uint32_t(uv.size()/2), uint32_t(faces.size()/3));
    stats.validationSeconds += v.elapsed; stats.validationCandidates += v.candidates;
    return v;
}

inline bool repair(const RepairPiece &input, std::vector<RepairPiece> &pieces,
                   LscmChartResult &stats, bool flattenTutte = true,
                   const float *positions = nullptr) {
    if (!flattenTutte) return false;
    // Recovery keeps a chart whole whenever possible. Splitting is reserved
    // for a disconnected input component; arbitrary bisection creates many
    // seams without establishing a valid parameterization.
    RepairPiece candidate = input;
    Topology top(candidate.faces, int(candidate.vmap.size()));
    std::vector<int> loop;
    if (!top.disk(loop)) {
        cutToDisks(candidate.faces, candidate.vmap, top, stats.topologyCuts);
        top = Topology(candidate.faces, int(candidate.vmap.size()));
        loop.clear();
        top.disk(loop);
    }
    std::vector<float> boundaryUv;
    const bool hasConvexBoundary = positions && !loop.empty() &&
        build_tutte_convex_boundary(positions, candidate.vmap, loop, boundaryUv);
    if (hasConvexBoundary &&
        tutte(top, candidate.uv, &candidate.boundary, &boundaryUv) &&
        !validate(candidate.uv, candidate.faces, stats).issue) {
        pieces.push_back(std::move(candidate)); return true;
    }
    // Keep the original circular initialization as a cheap safety fallback
    // for unusual projections, hull degeneracies, or numerical failures.
    if (tutte(top, candidate.uv, &candidate.boundary) &&
        !validate(candidate.uv, candidate.faces, stats).issue) {
        pieces.push_back(std::move(candidate)); return true;
    }
    return false;
}

inline bool assemble(const float *positions, uint32_t faceCount, std::vector<RepairPiece> &pieces,
                     LscmChartResult &stats, std::vector<float> &uv,
                     std::vector<int32_t> &faces, std::vector<int32_t> &vmap) {
    uv.clear(); vmap.clear(); faces.assign(size_t(faceCount)*3,-1);
    double cursor=0;
    for(auto &part:pieces) {
        if (kEnableTutteDistortionOptimizer)
            optimize_tutte_distortion(positions, part.faces, part.vmap, part.uv);
        double area3=0,area2=0;
        for(size_t f=0;f<part.faceIds.size();++f) {
            const float *p[3], *t[3];
            for(int k=0;k<3;++k) {
                int v=part.faces[3*f+k]; p[k]=positions+size_t(part.vmap[v])*3; t[k]=part.uv.data()+size_t(v)*2;
            }
            double a[3],b[3],c[3];
            for(int k=0;k<3;++k){a[k]=double(p[1][k])-p[0][k];b[k]=double(p[2][k])-p[0][k];}
            for(int k=0;k<3;++k)c[k]=a[(k+1)%3]*b[(k+2)%3]-a[(k+2)%3]*b[(k+1)%3];
            area3+=std::sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]);
            area2+=std::abs((double(t[1][0])-t[0][0])*(double(t[2][1])-t[0][1])-
                           (double(t[1][1])-t[0][1])*(double(t[2][0])-t[0][0]));
        }
        if(!(area3>0) || !(area2>0))return false;
        double scale=std::sqrt(area3/area2);
        int offset=int(vmap.size());
        for(size_t v=0;v<part.vmap.size();++v) {
            uv.push_back(float(cursor+(double(part.uv[2*v])+1)*scale));
            uv.push_back(float((double(part.uv[2*v+1])+1)*scale));
            vmap.push_back(part.vmap[v]);
        }
        for(size_t f=0;f<part.faceIds.size();++f)for(int k=0;k<3;++k)
            faces[size_t(part.faceIds[f])*3+k]=offset+part.faces[3*f+k];
        cursor+=2.1*scale;
    }
    return !validate(uv,faces,stats).issue;
}

}} // namespace cumesh_xatlas::safeuv
