#include <iostream>
#include <omp.h>
#include <chrono> // For measuring time

using namespace std;

// Sequential Min function
int minval_sequential(int arr[], int n) {
    int minval = arr[0];
    for (int i = 0; i < n; i++) {
        if (arr[i] < minval) 
            minval = arr[i];
    }
    return minval;
}

// Sequential Max function
int maxval_sequential(int arr[], int n) {
    int maxval = arr[0];
    for (int i = 0; i < n; i++) {
        if (arr[i] > maxval) 
            maxval = arr[i];
    }
    return maxval;
}

// Sequential Sum function
int sum_sequential(int arr[], int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// Sequential Average function
double average_sequential(int arr[], int n) {
    return (double)sum_sequential(arr, n) / n;
}

// Parallel Min function using OpenMP
int minval_parallel(int arr[], int n) {
    int minval = arr[0];
    #pragma omp parallel for reduction(min : minval)
    for (int i = 0; i < n; i++) {
        if (arr[i] < minval) 
            minval = arr[i];
    }
    return minval;
}

// Parallel Max function using OpenMP
int maxval_parallel(int arr[], int n) {
    int maxval = arr[0];
    #pragma omp parallel for reduction(max : maxval)
    for (int i = 0; i < n; i++) {
        if (arr[i] > maxval) 
            maxval = arr[i];
    }
    return maxval;
}

// Parallel Sum function using OpenMP
int sum_parallel(int arr[], int n) {
    int sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// Parallel Average function using OpenMP
double average_parallel(int arr[], int n) {
    return (double)sum_parallel(arr, n) / n;
}

int main() {
    int n = 100;  // Size of the array
    int* arr = new int[n];  // Dynamic memory allocation
    
    // Fill the array with some values (e.g., 1 to n)
    for (int i = 0; i < n; i++) {
        arr[i] = i + 1;
    }

    // Sequential execution time measurement
    auto start = chrono::high_resolution_clock::now();

    cout << "Sequential results:\n";
    cout << "The minimum value is: " << minval_sequential(arr, n) << '\n';
    cout << "The maximum value is: " << maxval_sequential(arr, n) << '\n';
    cout << "The summation is: " << sum_sequential(arr, n) << '\n';
    cout << "The average is: " << average_sequential(arr, n) << '\n';

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_duration = end - start;
    cout << "Time taken for sequential execution: " << seq_duration.count() << " seconds\n\n";

    // Parallel execution time measurement
    start = chrono::high_resolution_clock::now();

    cout << "Parallel results:\n";
    cout << "The minimum value is: " << minval_parallel(arr, n) << '\n';
    cout << "The maximum value is: " << maxval_parallel(arr, n) << '\n';
    cout << "The summation is: " << sum_parallel(arr, n) << '\n';
    cout << "The average is: " << average_parallel(arr, n) << '\n';

    end = chrono::high_resolution_clock::now();
    chrono::duration<double> par_duration = end - start;
    cout << "Time taken for parallel execution: " << par_duration.count() << " seconds\n";

    double speedup_time = seq_duration/par_duration;
    cout<<"Speed Up Time :"<<speedup_time;

    delete[] arr; // Free heap memory
    return 0;
}
