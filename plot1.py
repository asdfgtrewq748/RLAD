import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_resistance_with_anomalies(csv_filepath, output_directory):
    """
    Reads hydraulic support data with anomaly predictions and plots the resistance,
    highlighting anomalous regions in red.

    Args:
        csv_filepath (str): Path to the CSV file.
        output_directory (str): Directory to save the plot.
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: File not found at {csv_filepath}")
        return

    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # --- Try to determine the resistance column name ---
    resistance_col_name = None
    for col in df.columns:
        if '#' in col and col not in ['Date', 'is_anomaly_predicted']:
            resistance_col_name = col
            break
    
    if resistance_col_name is None:
        if 'is_anomaly_predicted' in df.columns:
            try:
                anomaly_col_index = df.columns.get_loc('is_anomaly_predicted')
                if anomaly_col_index > 0:
                    potential_col = df.columns[anomaly_col_index - 1]
                    if potential_col != 'Date':
                        resistance_col_name = potential_col
            except KeyError:
                pass 

    if resistance_col_name is None:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col != 'is_anomaly_predicted':
                resistance_col_name = col
                break
    
    if resistance_col_name is None:
        print("Error: Could not automatically determine the resistance column.")
        print("Please ensure your CSV has a clear resistance column (e.g., '100#') or modify the script.")
        return
        
    anomaly_col_name = 'is_anomaly_predicted'
    if anomaly_col_name not in df.columns:
        print(f"Error: Anomaly column '{anomaly_col_name}' not found in the CSV.")
        return

    print(f"Using resistance column: '{resistance_col_name}'")
    print(f"Using anomaly column: '{anomaly_col_name}'")

    x_data = df.index
    y_data = df[resistance_col_name]
    anomalies = df[anomaly_col_name]

    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False 
        plt.rcParams['font.family'] = 'Times New Roman' 
        fig_test_font = plt.figure(figsize=(1,1)) 
        plt.title("测试", fontname='SimHei')
        plt.close(fig_test_font)
        print("SimHei font found and set.")
        plot_font = 'SimHei'
    except Exception:
        print("SimHei font not found, using Times New Roman.")
        plot_font = 'Times New Roman'
    
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['font.size'] = 12

    plt.figure(figsize=(18, 8))
    plt.plot(x_data, y_data, label=f'{resistance_col_name} 阻力 (Resistance)', color='dodgerblue', linewidth=1.0, alpha=0.8)
    y_anomalous = y_data.copy()
    y_anomalous[anomalies == 0] = np.nan  
    plt.plot(x_data, y_anomalous, color='red', linewidth=1.5, label='异常值 (Anomaly)')

    title_text = f'{resistance_col_name} 液压支架工作阻力及异常检测结果'
    xlabel_text = '数据点步长 (Time Step)'
    ylabel_text = f'{resistance_col_name} 阻力值 (Resistance Value)'
    
    plt.title(title_text, fontsize=16, fontname=plot_font)
    plt.xlabel(xlabel_text, fontsize=14, fontname=plot_font)
    plt.ylabel(ylabel_text, fontsize=14, fontname=plot_font)

    plt.legend(fontsize=12, prop={'family': plot_font})
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() 

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Construct the full path for the output file
    base_filename = f"resistance_plot_with_anomalies_{resistance_col_name.replace('#', '')}.png"
    output_plot_filepath = os.path.join(output_directory, base_filename)
    
    plt.savefig(output_plot_filepath, dpi=300)
    print(f"Plot saved as {output_plot_filepath}")

    plt.show()


if __name__ == "__main__":
    csv_file_path = r"C:\Users\Liu HaoTian\Desktop\Python files\deeplearning\example\timeseries\examples\RLAD\output_modified\enhanced_rlad_results_100__20250616_191323\all_data_with_point_predictions.csv"
    
    # --- Updated output directory ---
    output_save_directory = r"C:\Users\Liu HaoTian\Desktop\Python files\deeplearning\example\timeseries\examples\RLAD\output"
    
    plot_resistance_with_anomalies(csv_file_path, output_save_directory)