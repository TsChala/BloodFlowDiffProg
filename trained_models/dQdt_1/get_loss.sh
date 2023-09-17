#!/bin/bash

# Check if the input file exists
input_file="./log_dQdt.out"
if [ ! -f "$input_file" ]; then
  echo "Input file not found: $input_file"
  exit 1
fi

# Extract mean train, interpolation, and extrapolation losses using grep
mean_train_loss=$(grep -oP '(?<=mean train loss:)[0-9.]+' "$input_file")
mean_interpolation_loss=$(grep -oP '(?<=mean test loss - interpolation:)[0-9.]+' "$input_file")
mean_extrapolation_loss=$(grep -oP '(?<=mean test loss - extrapolation:)[0-9.]+' "$input_file")

# Check if the output file exists and delete it if it does
output_file="./new_loss.txt"
if [ -f "$output_file" ]; then
  rm "$output_file"
fi

# Save the results into the output file with three columns
#echo "Mean Train Loss   Interpolation Loss   Extrapolation Loss" >> "$output_file"
echo "$mean_train_loss" >> "$output_file"
echo "$mean_interpolation_loss" >> "$output_file"
echo "$mean_extrapolation_loss" >> "$output_file"
echo "Mean train, interpolation, and extrapolation losses saved to $output_file in three columns."
