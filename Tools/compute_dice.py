import csv

def calculate_averages(file_path):
    total_dice = 0
    total_jaccard = 0
    valid_entries = 0
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            try:
                dice = float(row[1])
                jaccard = float(row[2])
                if dice > 0 and jaccard > 0:
                    total_dice += dice
                    total_jaccard += jaccard
                    valid_entries += 1
            except ValueError:
                continue

    if valid_entries > 0:
        average_dice = total_dice / valid_entries
        average_jaccard = total_jaccard / valid_entries
        return average_dice, average_jaccard
    else:
        return 0, 0


file_path = '../Data/results/unet/mish/split_1/dice_jacc_scores.csv'
average_dice, average_jaccard = calculate_averages(file_path)
print(f"Average Dice Coefficient: {average_dice}")
print(f"Average Jaccard Index: {average_jaccard}")