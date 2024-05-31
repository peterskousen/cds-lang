import os
import matplotlib.pyplot as plt
import pandas as pd

def calculate_subtask_emissions(data, csv, output_dir):
    plt.figure()
    plt.bar(data['task_name'], data['emissions'])
    plt.xlabel('Task Name')
    plt.ylabel('Emissions (CO₂eq)')
    plt.title(f'Approx. emissions per task for {csv}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subtask_emissions.png'))
    plt.close()

def calculate_subtask_durations(data, csv, output_dir):
    plt.figure()
    plt.bar(data['task_name'], data['duration'])
    plt.xlabel('Task Name')
    plt.ylabel('Duration (s)')
    plt.title(f'Duration of tasks for {csv}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subtask_durations.png'))
    plt.close()

def calculate_total_emissions(data, csv, total_emissions, output_dir):
    plt.figure()
    plt.bar(['Total Emissions'], [total_emissions], width=0.3)
    plt.ylabel('Emissions (CO₂eq)')
    plt.title(f'Total Emissions for {csv}: {total_emissions:.6f} kg CO₂eq')
    plt.xticks()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_emissions.png'))
    plt.close()

def plot_all_total_emissions(total_emissions_dict, output_path):
    sorted_emissions = dict(sorted(total_emissions_dict.items(), key=lambda item: item[1]))

    plt.figure()
    plt.bar(sorted_emissions.keys(), sorted_emissions.values(), width=0.3)
    plt.xlabel('CSV Files')
    plt.ylabel('Total Emissions (CO₂eq)')
    plt.title('Total Emissions for All CSV Files')
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'all_total_emissions.png'))
    plt.close()

def main():
    input_path = "in"
    output_path = "out"
    total_emissions_dict = {}

    for csv in sorted(os.listdir(input_path)):
        data = pd.read_csv(os.path.join(input_path, csv))
        total_emissions = data['emissions'].sum()
        data["total_emissions"] = total_emissions

        output_dir = os.path.join(output_path, os.path.splitext(csv)[0])
        os.makedirs(output_dir, exist_ok=True)

        calculate_subtask_emissions(data, csv, output_dir)
        calculate_subtask_durations(data, csv, output_dir)
        calculate_total_emissions(data, csv, total_emissions, output_dir)
        
        total_emissions_dict[csv] = total_emissions
        print(f'Plots successfully saved to {os.path.join(output_path, csv)}')

    plot_all_total_emissions(total_emissions_dict, output_path)
    print(f'All total emissions plot saved to {output_path}')

if __name__ == "__main__":
    main()
