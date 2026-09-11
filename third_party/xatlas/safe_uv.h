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
#include <unordered_map>
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
    struct Node { Box box; uint32_t begin, end, left = 0, right = 0; };
    const float *uv;
    const int32_t *faces;
    std::vector<Box> boxes;
    std::vector<uint32_t> order;
    std::vector<Node> nodes;
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
public:
    Validator(const float *u, const int32_t *f) : uv(u), faces(f) {}
    Validation run(uint32_t vertexCount, uint32_t faceCount) {
        auto start = Clock::now(); Validation result;
        for (size_t i = 0; i < size_t(vertexCount) * 2; ++i) if (!std::isfinite(uv[i])) {
            result.issue = 1; result.elapsed = seconds(start); return result;
        }
        boxes.resize(faceCount); order.resize(faceCount);
        for (uint32_t f = 0; f < faceCount; ++f) {
            order[f] = f;
            for (int k = 0; k < 3; ++k) {
                int32_t v = faces[size_t(f) * 3 + k];
                if (v < 0 || uint32_t(v) >= vertexCount) { result.issue = 4; result.face0 = f; break; }
                boxes[f].add(uv + size_t(v) * 2);
            }
            if (!result.issue && orient(uv + size_t(faces[size_t(f)*3])*2,
                    uv + size_t(faces[size_t(f)*3+1])*2, uv + size_t(faces[size_t(f)*3+2])*2) == 0) {
                result.issue = 2; result.face0 = f;
            }
            if (result.issue) { result.elapsed = seconds(start); return result; }
        }
        if (faceCount) { nodes.reserve(faceCount / 2 + 1); build(0, faceCount); visit(0, 0, result); }
        result.elapsed = seconds(start); return result;
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
inline bool tutte(const Topology &top, std::vector<float> &uv) {
    std::vector<int> loop;
    if (!top.disk(loop)) return false;
    const size_t n=top.neighbors.size();
    std::vector<char> fixed(n,false); std::vector<double> xy(n*2,0);
    for(size_t k=0;k<loop.size();++k) {
        double angle=6.2831853071795864769*double(k)/double(loop.size());
        fixed[loop[k]]=true; xy[2*loop[k]]=std::cos(angle); xy[2*loop[k]+1]=std::sin(angle);
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
    return true;
}

struct RepairPiece {
    std::vector<int32_t> faces, vmap, faceIds;
    std::vector<float> uv;
};

inline Validation validate(const std::vector<float> &uv, const std::vector<int32_t> &faces,
                           LscmChartResult &stats) {
    Validation v = Validator(uv.data(), faces.data()).run(uint32_t(uv.size()/2), uint32_t(faces.size()/3));
    stats.validationSeconds += v.elapsed; stats.validationCandidates += v.candidates;
    return v;
}

inline bool repair(RepairPiece input, std::vector<RepairPiece> &pieces,
                   LscmChartResult &stats) {
    // Recovery keeps a chart whole whenever possible. Splitting is reserved
    // for a disconnected input component; arbitrary bisection creates many
    // seams without establishing a valid parameterization.
    RepairPiece candidate = input;
    Topology top(candidate.faces, int(candidate.vmap.size()));
    std::vector<int> loop;
    if (!top.disk(loop)) {
        cutToDisks(candidate.faces, candidate.vmap, top, stats.topologyCuts);
        top = Topology(candidate.faces, int(candidate.vmap.size()));
    }
    if (tutte(top, candidate.uv) && !validate(candidate.uv, candidate.faces, stats).issue) {
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
