def calculate_average(nums):
    # A simple bug for the AI to find pull
    sum_nums = sum(nums)
    count = len(nums)
    return sum_nums / count if count > 0 else 0
# Triggering NeuroTidy
print(calculate_average([10, 20]))
print(calculate_average([]))
