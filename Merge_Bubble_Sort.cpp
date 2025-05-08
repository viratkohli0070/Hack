#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

// Sequential Bubble Sort
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
}

// Parallel Bubble Sort (Odd-Even Transposition Sort)
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        int start = i % 2;
        #pragma omp parallel for
        for (int j = start; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Merge function
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;
    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int i = 0; i < n2; i++) R[i] = arr[m + 1 + i];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

// Sequential Merge Sort
void mergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort using OpenMP
void parallelMergeSort(vector<int>& arr, int l, int r, int depth = 0) {
    if (l < r) {
        int m = (l + r) / 2;
        if (depth < 4) {
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelMergeSort(arr, l, m, depth + 1);

                #pragma omp section
                parallelMergeSort(arr, m + 1, r, depth + 1);
            }
        } else {
            mergeSort(arr, l, m);
            mergeSort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}

// Helper to print first 20 elements
void printFirst20(const vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++)
        cout << arr[i] << " ";
    cout << "\n";
}

// Helper to measure execution time
template<typename Func>
long long measureTime(Func func, vector<int>& arr) {
    auto start = high_resolution_clock::now();
    func(arr);
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}

int main() {
    const int SIZE = 10;
    vector<int> arr(SIZE);

    // Fill with random values
    for (int i = 0; i < SIZE; i++)
        arr[i] = rand() % 10;

    cout << "Original Array (first 20): ";
    printFirst20(arr);

    vector<int> a1 = arr, a2 = arr, a3 = arr, a4 = arr;

    cout << "\nSorting " << SIZE << " elements...\n\n";

    // Sequential Bubble Sort
    long long t1 = measureTime(bubbleSort, a1);
    cout << "Sequential Bubble Sort (first 20): ";
    printFirst20(a1);
    cout << "Time: " << t1 << " ms\n\n";

    // Parallel Bubble Sort
    long long t2 = measureTime(parallelBubbleSort, a2);
    cout << "Parallel Bubble Sort (first 20):   ";
    printFirst20(a2);
    cout << "Time: " << t2 << " ms\n\n";

    // Sequential Merge Sort
    auto start3 = high_resolution_clock::now();
    mergeSort(a3, 0, a3.size() - 1);
    auto end3 = high_resolution_clock::now();
    cout << "Sequential Merge Sort (first 20):  ";
    printFirst20(a3);
    cout << "Time: " << duration_cast<milliseconds>(end3 - start3).count() << " ms\n\n";

    // Parallel Merge Sort
    auto start4 = high_resolution_clock::now();
    parallelMergeSort(a4, 0, a4.size() - 1);
    auto end4 = high_resolution_clock::now();
    cout << "Parallel Merge Sort (first 20):    ";
    printFirst20(a4);
    cout << "Time: " << duration_cast<milliseconds>(end4 - start4).count() << " ms\n";

    return 0;
}
