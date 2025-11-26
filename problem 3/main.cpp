#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <random>
using namespace std;
struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 cross(const Vec3& b) const {
        return Vec3(y * b.z - z * b.y,
                    z * b.x - x * b.z,
                    x * b.y - y * b.x);
    }
    double dot(const Vec3& b) const {
        return x * b.x + y * b.y + z * b.z;
    }
    double length() const {
        return  sqrt(x * x + y * y + z * z);
    }
    Vec3 normalized() const {
        double len = length();
        if (len < 1e-15) return Vec3(0, 0, 0);
        return Vec3(x / len, y / len, z / len);
    }
};

struct Triangle {
     array<int, 3> v;
    Triangle(int a, int b, int c) : v{{a, b, c}} {}
};

bool rayIntersectsTriangle(const Vec3& orig, const Vec3& dir,
                           const Vec3& v0, const Vec3& v1, const Vec3& v2,
                           double& t) {
    const double EPSILON = 1e-8;
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 h = dir.cross(edge2);
    double a = edge1.dot(h);
    if ( abs(a) < EPSILON) return false;  // Ray parallel to triangle
    double f = 1.0 / a;
    Vec3 s = orig - v0;
    double u = f * s.dot(h);
    if (u < 0.0 || u > 1.0) return false;
    Vec3 q = s.cross(edge1);
    double v = f * dir.dot(q);
    if (v < 0.0 || u + v > 1.0) return false;
    t = f * edge2.dot(q);
    if (t > EPSILON) return true;
    return false;
}

Vec3 computeNormal(const Vec3& a, const Vec3& b, const Vec3& c) {
    return (b - a).cross(c - a).normalized();
}

bool readOFF(const  string& filename,
              vector<Vec3>& vertices,
              vector<Triangle>& triangles) {
     ifstream fin(filename);
    if (!fin) {
         cerr << "Cannot open " << filename << "\n";
        return false;
    }
     string header;
    fin >> header;
    if (header != "OFF") {
         cerr << "Not an OFF file\n";
        return false;
    }
    int nVertices, nTriangles, nEdges;
    fin >> nVertices >> nTriangles >> nEdges;
    vertices.resize(nVertices);
    for (int i = 0; i < nVertices; ++i)
        fin >> vertices[i].x >> vertices[i].y >> vertices[i].z;
    triangles.reserve(nTriangles);
    for (int i = 0; i < nTriangles; ++i) {
        int n, a, b, c;
        fin >> n >> a >> b >> c;
        if (n != 3) {
             cerr << "Non-triangular face found\n";
            return false;
        }
        triangles.emplace_back(a, b, c);
    }
    return true;
}

bool writeOFF(const  string& filename,
              const  vector<Vec3>& vertices,
              const  vector<Triangle>& triangles) {

     ofstream fout(filename);
    if (!fout) {
         cerr << "Cannot write to " << filename << "\n";
        return false;
    }
    fout.precision(10);
    fout << "OFF\n";
    fout << vertices.size() << " " << triangles.size() << " 0\n";
    for (const auto& v : vertices)
        fout << v.x << " " << v.y << " " << v.z << "\n";
    for (const auto& t : triangles)
        fout << "3 " << t.v[0] << " " << t.v[1] << " " << t.v[2] << "\n";
    return true;
}

Vec3 randomDeviation() {
    static  mt19937 gen(42);
    static  uniform_real_distribution<double> dist(-1e-2, 1e-2);
    return Vec3(dist(gen), dist(gen), dist(gen));
}

void fixTriangleOrientations( vector<Vec3>& vertices,
                              vector<Triangle>& triangles) {
    for (auto& tri : triangles) {
        Vec3 a = vertices[tri.v[0]];
        Vec3 b = vertices[tri.v[1]];
        Vec3 c = vertices[tri.v[2]];
        // Compute normal vector
        Vec3 normal = computeNormal(a, b, c);
        // Compute center of triangle
        Vec3 center = Vec3((a.x + b.x + c.x) / 3.0,
                           (a.y + b.y + c.y) / 3.0,
                           (a.z + b.z + c.z) / 3.0);
        // Create ray direction with slight deviation
        Vec3 rayDir = (normal + randomDeviation()).normalized();
        int countIntersections = 0;
        for (const auto& otherTri : triangles) {
            Vec3 v0 = vertices[otherTri.v[0]];
            Vec3 v1 = vertices[otherTri.v[1]];
            Vec3 v2 = vertices[otherTri.v[2]];
            double t;
            if (rayIntersectsTriangle(center, rayDir, v0, v1, v2, t)) {
                ++countIntersections;
            }
        }
        // If odd number of intersections, flip orientation
        if (countIntersections % 2 == 1) {
             swap(tri.v[1], tri.v[2]);
        }
    }
}

int main(int argc, char* argv[]) {
     vector<Vec3> vertices;
     vector<Triangle> triangles;
    if (!readOFF("test.off", vertices, triangles)) return 1;
    fixTriangleOrientations(vertices, triangles);
    if (!writeOFF("test_fixed.off", vertices, triangles)) return 1;
     cout << "Orientation fix done.\n";
    return 0;
}
