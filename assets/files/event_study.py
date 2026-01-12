import numpy as np
import pandas as pd
from scipy import stats

def event_study_analysis_iterative():
    """
    Iterative Event Study Analysis Code
    Reads data from data.xlsx and calculates AR for multiple event pointers
    """

    print("=" * 60)
    print("ITERATIVE EVENT STUDY ANALYSIS")
    print("=" * 60)

    # Read data from Excel file
    try:
        df = pd.read_excel('data.xlsx')
        print(f"\n✓ Successfully loaded data.xlsx")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("\nERROR: data.xlsx not found in the current directory!")
        print("Please ensure data.xlsx is in the same folder as this script.")
        return None
    except Exception as e:
        print(f"\nERROR: Could not read data.xlsx - {e}")
        return None

    # Extract arrays (remove NaN values if any)
    target_index = df['target_index'].dropna().values
    reference_index = df['reference_index'].dropna().values

    # Ensure both arrays have the same length
    n = min(len(target_index), len(reference_index))
    target_index = target_index[:n]
    reference_index = reference_index[:n]

    print(f"\n  Valid data points: {n}")

    # Input: Parameters
    print("\n" + "=" * 60)
    b = int(input("Enter window size 'b': "))
    c = int(input("Enter number of iterations 'c': "))
    print("=" * 60)

    # Calculate valid range for 'a'
    min_a = b  # a - b >= 0
    max_a = n - b - 31  # a + b + 30 < n

    if min_a > max_a:
        print(f"\nERROR: No valid range for 'a' with b={b}")
        print(f"       Data size too small or b too large.")
        return None

    print(f"\nValid range for 'a': [{min_a}, {max_a}]")
    print(f"Starting 'a' at: {min_a}")
    print(f"Number of iterations: {c}")
    print(f"Ending 'a' at: {min_a + c - 1}")

    if min_a + c - 1 > max_a:
        print(f"\nWARNING: Some iterations will exceed valid range!")
        print(f"         Only {max_a - min_a + 1} iterations possible.")
        c = max_a - min_a + 1
        print(f"         Adjusted to {c} iterations.")

    # Store all results
    all_results = []
    all_regression_params = []

    # Iterate through different values of 'a'
    for iteration in range(c):
        a = min_a + iteration

        print("\n" + "=" * 80)
        print(f"ITERATION {iteration + 1}/{c} | Event Pointer a = {a}")
        print("=" * 80)

        # Define windows
        event_window_start = a - b
        event_window_end = a + b
        estimation_window_start = a + b + 1
        estimation_window_end = a + b + 30

        print(f"Event Window: [{event_window_start}, {event_window_end}]")
        print(f"Estimation Window: [{estimation_window_start}, {estimation_window_end}]")

        # Extract estimation window data
        est_target = target_index[estimation_window_start:estimation_window_end + 1]
        est_reference = reference_index[estimation_window_start:estimation_window_end + 1]

        # Perform Linear Regression (OLS) on estimation window
        slope, intercept, r_value, p_value, std_err = stats.linregress(est_reference, est_target)
        r_squared = r_value ** 2

        print(f"\nRegression: α={intercept:.6f}, β={slope:.6f}, SE={std_err:.6f}, R²={r_squared:.6f}")

        # Store regression parameters
        all_regression_params.append({
            'iteration': iteration + 1,
            'a': a,
            'b': b,
            'intercept': intercept,
            'slope': slope,
            'std_err': std_err,
            'r_squared': r_squared
        })

        # Calculate AR and t-stat for each point in event window
        print(f"\n{'Index':<8} {'Target':<12} {'Reference':<12} {'Expected':<12} {'AR':<12} {'t-stat':<12}")
        print("-" * 80)

        ar_list = []

        for i in range(event_window_start, event_window_end + 1):
            target_return = target_index[i]
            reference_return = reference_index[i]

            # Expected return based on regression model
            expected_return = intercept + (slope * reference_return)

            # Abnormal Return
            ar = target_return - expected_return

            # t-statistic
            t_stat = ar / std_err if std_err != 0 else 0

            all_results.append({
                'iteration': iteration + 1,
                'a': a,
                'b': b,
                'index': i,
                'target': target_return,
                'reference': reference_return,
                'expected': expected_return,
                'ar': ar,
                't_stat': t_stat
            })

            ar_list.append(ar)

            print(f"{i:<8} {target_return:<12.6f} {reference_return:<12.6f} {expected_return:<12.6f} {ar:<12.6f} {t_stat:<12.6f}")

        # Calculate CAR and AAR
        car = sum(ar_list)
        aar = car / len(ar_list)

        print(f"\nCAR: {car:.6f} | AAR: {aar:.6f}")

    print("\n" + "=" * 80)
    print("ALL ITERATIONS COMPLETED")
    print("=" * 80)

    return all_results, all_regression_params


# Run the analysis
if __name__ == "__main__":
    results = event_study_analysis_iterative()

    # Save results to file
    if results is not None:
        all_results, all_regression_params = results

        # Create DataFrames
        df_results = pd.DataFrame(all_results)
        df_params = pd.DataFrame(all_regression_params)

        # Save to Excel with multiple sheets
        filename = "event_study_results_iterative.xlsx"
        with pd.ExcelWriter(filename) as writer:
            df_results.to_excel(writer, sheet_name='AR_Results', index=False)
            df_params.to_excel(writer, sheet_name='Regression_Params', index=False)

        print(f"\n✓ Results saved to {filename}")
        print(f"  - Sheet 'AR_Results': All AR and t-statistics for all iterations")
        print(f"  - Sheet 'Regression_Params': Regression parameters for each iteration")
        print(f"  - Total iterations: {len(all_regression_params)}")
        print(f"  - Total AR calculations: {len(all_results)}")
