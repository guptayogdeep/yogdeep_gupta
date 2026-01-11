import numpy as np
from scipy import stats

def event_study_analysis():
    """
    Event Study Analysis Code
    Calculates AR (Abnormal Returns) and t-statistics for an event window
    """

    # Get user inputs
    print("=" * 60)
    print("EVENT STUDY ANALYSIS")
    print("=" * 60)

    # Input: Size of arrays
    n = int(input("\nEnter the size of arrays (n): "))

    # Input: Target index returns
    print(f"\nEnter {n} values for TARGET_INDEX returns:")
    target_index = []
    for i in range(n):
        val = float(input(f"  Target_index[{i}]: "))
        target_index.append(val)
    target_index = np.array(target_index)

    # Input: Reference index returns
    print(f"\nEnter {n} values for REFERENCE_INDEX returns:")
    reference_index = []
    for i in range(n):
        val = float(input(f"  Reference_index[{i}]: "))
        reference_index.append(val)
    reference_index = np.array(reference_index)

    # Input: Event parameters
    print("\n" + "=" * 60)
    a = int(input("Enter event pointer 'a' (event day index): "))
    b = int(input("Enter window size 'b': "))
    print("=" * 60)

    # Define windows
    event_window_start = a - b
    event_window_end = a + b
    estimation_window_start = a + b + 1
    estimation_window_end = a + b + 30

    print(f"\nEvent Window: [{event_window_start}, {event_window_end}]")
    print(f"Estimation Window: [{estimation_window_start}, {estimation_window_end}]")

    # Validate indices
    if event_window_start < 0 or estimation_window_end >= n:
        print(f"\nERROR: Invalid window bounds. Ensure 0 <= {event_window_start} and {estimation_window_end} < {n}")
        return None

    # Extract estimation window data
    est_target = target_index[estimation_window_start:estimation_window_end + 1]
    est_reference = reference_index[estimation_window_start:estimation_window_end + 1]

    # Perform Linear Regression (OLS) on estimation window
    # target = intercept + slope * reference
    slope, intercept, r_value, p_value, std_err = stats.linregress(est_reference, est_target)
    r_squared = r_value ** 2

    print("\n" + "=" * 60)
    print("REGRESSION RESULTS (Estimation Window)")
    print("=" * 60)
    print(f"Intercept (α): {intercept:.6f}")
    print(f"Slope (β): {slope:.6f}")
    print(f"Standard Error: {std_err:.6f}")
    print(f"R-squared: {r_squared:.6f}")
    print("=" * 60)

    # Calculate AR and t-stat for each point in event window
    print("\n" + "=" * 60)
    print("ABNORMAL RETURNS (AR) AND T-STATISTICS")
    print("=" * 60)
    print(f"{'Index':<8} {'Target':<12} {'Reference':<12} {'Expected':<12} {'AR':<12} {'t-stat':<12}")
    print("-" * 80)

    ar_results = []

    for i in range(event_window_start, event_window_end + 1):
        target_return = target_index[i]
        reference_return = reference_index[i]

        # Expected return based on regression model
        expected_return = intercept + (slope * reference_return)

        # Abnormal Return
        ar = target_return - expected_return

        # t-statistic
        t_stat = ar / std_err if std_err != 0 else 0

        ar_results.append({
            'index': i,
            'target': target_return,
            'reference': reference_return,
            'expected': expected_return,
            'ar': ar,
            't_stat': t_stat
        })

        print(f"{i:<8} {target_return:<12.6f} {reference_return:<12.6f} {expected_return:<12.6f} {ar:<12.6f} {t_stat:<12.6f}")

    print("=" * 60)

    # Calculate Cumulative Abnormal Return (CAR)
    total_ar = sum([result['ar'] for result in ar_results])
    avg_ar = total_ar / len(ar_results)

    print(f"\nCumulative Abnormal Return (CAR): {total_ar:.6f}")
    print(f"Average Abnormal Return (AAR): {avg_ar:.6f}")
    print("=" * 60)

    return ar_results, intercept, slope, std_err, r_squared


# Run the analysis
if __name__ == "__main__":
    results = event_study_analysis()

    # Optional: Save results to file
    if results is not None:
        save = input("\nDo you want to save results to CSV? (y/n): ")
        if save.lower() == 'y':
            import pandas as pd
            ar_results, intercept, slope, std_err, r_squared = results
            df = pd.DataFrame(ar_results)
            filename = "event_study_results.csv"
            df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")
