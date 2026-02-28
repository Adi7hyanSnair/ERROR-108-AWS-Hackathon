def calculate_average(nums):
    # A simple bug for the AI to find
    sum_nums = sum(nums)
    count = len(nums)
    return sum_nums / count if count > 0 else 0

print(calculate_average([10, 20]))
print(calculate_average([]))
