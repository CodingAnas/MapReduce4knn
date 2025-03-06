import os
import subprocess
import csv

# Define the range of k values you want to test
k_values = [3, 5, 7, 9]

# Path to your k-NN MapReduce script
mapreduce_script = "MapReduce.py"

# Paths to your datasets
train_file = "/home/sonu/Documents/MapReduce4knn/train.csv"
test_file = "test.csv"

# Output CSV file to store results
output_csv = "knn_results.csv"

# Store results
results = []

for k in k_values:
    print(f"\nRunning k-NN MapReduce for k={k}...\n")

    # Run the MapReduce job
    command = f"python {mapreduce_script} {test_file} --train {train_file} --k {k}"
    os.environ["KNN_K"] = str(k)  # Set k as an environment variable for the script
    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error executing for k={k}: {stderr}")
        continue

    # Extract accuracy from output
    accuracy = None
    for line in stdout.split("\n"):
        if "Final Accuracy" in line:
            accuracy = float(line.split()[-1])  # Extract accuracy value
    
    if accuracy is not None:
        results.append((k, accuracy))
        print(f"✅ k={k}, Accuracy={accuracy:.2f}%")
    else:
        print(f"❌ Could not extract accuracy for k={k}")

# Write results to a CSV file
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["k", "Accuracy"])
    writer.writerows(results)

print(f"\n📊 Experiment results saved to {output_csv}")
