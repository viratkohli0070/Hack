#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void sequential_lr(const vector<double> &x, const vector<double> &y, double &m, double &c, double &time)
{
    int n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;

    double start = omp_get_wtime();

    for (int i = 0; i < n; ++i)
    {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    c = (sum_y - m * sum_x) / n;

    double end = omp_get_wtime();
    time = end - start;
}

void parallel_lr(const vector<double> &x, const vector<double> &y, double &m, double &c, double &time)
{
    int n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+ : sum_x, sum_y, sum_xy, sum_x2)
    for (int i = 0; i < n; ++i)
    {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    c = (sum_y - m * sum_x) / n;

    double end = omp_get_wtime();
    time = end - start;
}

int main() {
    int n;
    cout << "Enter the number of data points: ";
    cin >> n;

    vector<double> x(n), y(n);

    for (int i = 0; i < n; ++i) {
        x[i] = rand() % 10000;
        y[i] = 3 * x[i] + 7 + rand() % 50;
    }

    int threads = omp_get_max_threads();
    cout << "System will use up to " << threads << " threads.\n";

    double m_seq, c_seq, time_seq;
    double m_par, c_par, time_par;

    sequential_lr(x, y, m_seq, c_seq, time_seq);
    parallel_lr(x, y, m_par, c_par, time_par);

    cout << "\nSequential Linear Regression:\n";
    cout << "  m = " << m_seq << ", c = " << c_seq << ", Time = " << time_seq << " seconds\n";

    cout << "Parallel Linear Regression:\n";
    cout << "  m = " << m_par << ", c = " << c_par << ", Time = " << time_par << " seconds\n";

    if (time_par > 0)
        cout << "Speedup: " << time_seq / time_par << endl;
    else
        cout << "Speedup: N/A (parallel time too small to measure reliably)\n";

    return 0;
   }

