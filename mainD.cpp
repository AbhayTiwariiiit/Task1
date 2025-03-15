
// Headers
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <limits>
#include "json.hpp"

using namespace std;
using json = nlohmann::json;
//

//Strucures
struct IMUEvent {
    string timestamp;
    double heading;
    string state;
    long long time_num;
};

struct SensorEvent {
    string timestamp;
    string sensor_id;
    double x;
    double y;
    long long time_num;
};
struct Cluster {
    string f_id;
    double last_x;
    double last_y;
    long long last_seen;
};
struct pair_hash {
    size_t operator()(const pair<int, int>& p) const {
        auto h1 = hash<int>{}(p.first);
        auto h2 = hash<int>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};
//


// Union-Find
class UnionFind {
public:
    UnionFind(int n) : parent(n) {
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }
    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }
    void unite(int x, int y) {
        int fx = find(x), fy = find(y);
        if (fx != fy)
            parent[fx] = fy;
    }
private:
    vector<int> parent;
};

//


// Simple Kalman Filter for heading smoothing
class KalmanFilter {
private:
    double theta = 0.0;
    double theta_dot = 0.0;
    double P[2][2] = { {1.0, 0.0}, {0.0, 1.0} };
    const double Q[2][2] = { {0.01, 0.0}, {0.0, 0.01} };
    const double R = 0.1;
    long long last_time = -1;
public:
    void predict(double delta_t) {
        double F[2][2] = { {1, delta_t}, {0, 1} };
        double new_theta = theta + theta_dot * delta_t;
        double P00 = F[0][0] * P[0][0] + F[0][1] * P[1][0];
        double P01 = F[0][0] * P[0][1] + F[0][1] * P[1][1];
        double P10 = F[1][0] * P[0][0] + F[1][1] * P[1][0];
        double P11 = F[1][0] * P[0][1] + F[1][1] * P[1][1];
        P[0][0] = P00 + Q[0][0];
        P[0][1] = P01 + Q[0][1];
        P[1][0] = P10 + Q[1][0];
        P[1][1] = P11 + Q[1][1];
        theta = new_theta;
    }
    void update(double measurement, long long current_time) {
        if (last_time != -1) {
            double delta_t = (current_time - last_time) / 1000.0;
            predict(delta_t);
        }
        double S = P[0][0] + R;
        double K[2] = { P[0][0] / S, P[1][0] / S };
        double y = measurement - theta;
        theta += K[0] * y;
        theta_dot += K[1] * y;
        double P00 = P[0][0] - K[0] * P[0][0];
        double P01 = P[0][1] - K[0] * P[0][1];
        P[1][0] = P[1][0] - K[1] * P[0][0];
        P[1][1] = P[1][1] - K[1] * P[0][1];
        P[0][0] = P00;
        P[0][1] = P01;
        last_time = current_time;
    }
    double get_heading() const { return theta; }
};

// Modified parse_timestamp to handle UTC explicitly (avoids local timezone issues)
long long parse_timestamp(const string &timestamp) {
    tm tm = {};
    string buffer;
    int ms = 0;
    size_t t_pos = timestamp.find('T');
    string time_part = timestamp;
    if (t_pos != string::npos)
        time_part[t_pos] = ' ';
    size_t dot_pos = time_part.find('.');
    if (dot_pos != string::npos) {
        buffer = time_part.substr(0, dot_pos);
        string ms_str = time_part.substr(dot_pos + 1, 3);
        ms = stoi(ms_str);
    } else {
        buffer = time_part;
    }
    istringstream ss(buffer);
    if (buffer.find(':') != string::npos)
        ss >> get_time(&tm, "%Y-%m-%d %H:%M:%S");
    else
        ss >> get_time(&tm, "%Y-%m-%d %H-%M-%S");
    if (ss.fail())
        throw runtime_error("Failed to parse timestamp: " + timestamp);
    tm.tm_isdst = -1; 
    time_t tt = mktime(&tm);
    tt -= timezone;
    auto time_point = chrono::system_clock::from_time_t(tt);
    auto since_epoch = time_point.time_since_epoch();
    auto millis = chrono::duration_cast<chrono::milliseconds>(since_epoch) + chrono::milliseconds(ms);
    return millis.count();
}


int main() {
    try {

        ifstream imu_file("imu_data.csv");
        if (!imu_file.is_open())
            throw runtime_error("Failed to open IMU file");


        string imu_header;
        getline(imu_file, imu_header);


        string imu_line;
        IMUEvent current_imu;
        bool imu_valid = false;
        if (getline(imu_file, imu_line)) {
            replace(imu_line.begin(), imu_line.end(), ',', ' ');
            istringstream imu_ss(imu_line);
            imu_ss >> current_imu.timestamp >> current_imu.heading >> current_imu.state;
            current_imu.time_num = parse_timestamp(current_imu.timestamp);
            imu_valid = true;
        }

        ifstream sensor_file("sensor.json");
        if (!sensor_file.is_open())
            throw runtime_error("Failed to open sensor file");
            
        json sensor_data = json::parse(sensor_file);
        vector<SensorEvent> sensor_events;
        for (const auto &entry : sensor_data) {
            for (auto it = entry.begin(); it != entry.end(); ++it) {
                string sensor_id = it.key();
                auto obj = it.value();
                string timestamp = obj["timestamp"].get<string>();
                auto positions = obj["object_positions_x_y"];
                for (const auto &pos : positions) {
                    SensorEvent se;
                    se.sensor_id = sensor_id;
                    se.timestamp = timestamp;
                    se.x = pos[0];
                    se.y = pos[1];
                    se.time_num = parse_timestamp(timestamp);
                    sensor_events.push_back(se);
                }
            }
        }
        size_t sensor_index = 0;

        ofstream output("fused_data.csv");
        if (!output.is_open())
            throw runtime_error("Failed to open output file");
        output << "f_timestamp,f_id,cluster_data,heading,status\n";

        KalmanFilter kf;
        double current_heading = 0.0;
        string current_status = "stationary";
        vector<Cluster> active_clusters;
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> id_dist(1000, 9999);

        int fusedRowsCount = 0;
        int totalClusteredCount = 0;
        int multiCamClusterCount = 0; 

        while (imu_valid || sensor_index < sensor_events.size()) {
            
            long long imu_time = imu_valid ? current_imu.time_num : numeric_limits<long long>::max();
            long long sensor_time = (sensor_index < sensor_events.size()) ? sensor_events[sensor_index].time_num : numeric_limits<long long>::max();

            if (imu_time <= sensor_time) {
                kf.update(current_imu.heading, current_imu.time_num);
                current_heading = kf.get_heading();
                current_status = current_imu.state;
                if (getline(imu_file, imu_line)) {
                    replace(imu_line.begin(), imu_line.end(), ',', ' ');
                    istringstream imu_ss(imu_line);
                    imu_ss >> current_imu.timestamp >> current_imu.heading >> current_imu.state;
                    current_imu.time_num = parse_timestamp(current_imu.timestamp);
                } else {
                    imu_valid = false;
                }
            } else {
                long long cluster_time = sensor_events[sensor_index].time_num;
                vector<SensorEvent> current_sensor_events;
                while (sensor_index < sensor_events.size() &&sensor_events[sensor_index].time_num == cluster_time) {
                    current_sensor_events.push_back(sensor_events[sensor_index]);
                    sensor_index++;
                }

                unordered_map<pair<int, int>, vector<int>, pair_hash> grid;
                for (int j = 0; j < current_sensor_events.size(); ++j) {
                    int cell_x = static_cast<int>(floor(current_sensor_events[j].x / 2.0));
                    int cell_y = static_cast<int>(floor(current_sensor_events[j].y / 2.0));
                    grid[{cell_x, cell_y}].push_back(j);
                }

                UnionFind uf(current_sensor_events.size());
                for (int j = 0; j < current_sensor_events.size(); ++j) {
                    const auto &s = current_sensor_events[j];
                    int cell_x = static_cast<int>(floor(s.x / 2.0));
                    int cell_y = static_cast<int>(floor(s.y / 2.0));
                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            auto key = make_pair(cell_x + dx, cell_y + dy);
                            if (grid.find(key) != grid.end()) {
                                for (int k : grid[key]) {
                                    if (k > j) {
                                        const auto &other = current_sensor_events[k];
                                        double dist = hypot(s.x - other.x, s.y - other.y);
                                        if (dist <= 2.0)
                                            uf.unite(j, k);
                                    }
                                }
                            }
                        }
                    }
                }

                unordered_map<int, vector<SensorEvent>> clusters;
                for (int j = 0; j < current_sensor_events.size(); ++j) {
                    clusters[uf.find(j)].push_back(current_sensor_events[j]);
                }

                for (const auto &cluster_pair : clusters) {
                    const auto &members = cluster_pair.second;
                    double avg_x = 0.0, avg_y = 0.0;
                    for (const auto &m : members) {
                        avg_x += m.x;
                        avg_y += m.y;
                    }
                    avg_x /= members.size();
                    avg_y /= members.size();

                    unordered_set<string> uniqueCams;
                    for (const auto &m : members) {
                        uniqueCams.insert(m.sensor_id);
                    }
                    if (uniqueCams.size() >= 2) {
                        multiCamClusterCount++;
                    }

                    string f_id;
                    auto closest = active_clusters.end();
                    double min_dist = 2.0;
                    for (auto it = active_clusters.begin(); it != active_clusters.end(); ++it) {
                        double dist = hypot(avg_x - it->last_x, avg_y - it->last_y);
                        if (dist < min_dist) {
                            min_dist = dist;
                            closest = it;
                        }
                    }
                    if (closest != active_clusters.end()) {
                        f_id = closest->f_id;
                        closest->last_x = avg_x;
                        closest->last_y = avg_y;
                        closest->last_seen = cluster_time;
                    } else {
                        f_id = "f_" + to_string(id_dist(gen));
                        active_clusters.push_back({ f_id, avg_x, avg_y, cluster_time });
                    }

                    fusedRowsCount++;
                    totalClusteredCount += members.size();

                    stringstream cluster_data;
                    cluster_data << "\"[";
                    for (size_t m = 0; m < members.size(); ++m) {
                        if (m > 0) cluster_data << ",";cluster_data << "[" << members[m].x << "," << members[m].y<< ",\"" << members[m].sensor_id << "\"]";
                    }
                    cluster_data << "]\"";
                    output << cluster_time << "," << f_id << "," << cluster_data.str() << ","<< fixed << setprecision(3) << current_heading << "," << current_status << "\n";
                }
                active_clusters.erase(
                    remove_if(active_clusters.begin(), active_clusters.end(),
                        [cluster_time](const Cluster &c) { return (cluster_time - c.last_seen) > 60000; }),
                    active_clusters.end());
            }
        }
        cout << "Total fused output rows: " << fusedRowsCount << "\n";
        cout << "Total sensor events clustered: " << totalClusteredCount << "\n";
        cout << "Total clusters with >= 2 cams: " << multiCamClusterCount << "\n";

        return 0;
    }
    catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
