import poker

#poker.run_simulation(4, "AdKd", "QdJdTd", 1)

#equity = poker.run_simulation(4, "6d3s", "", 1000000)

#print(f"My equity is {equity}")

test_array = poker.take_screenshot()

print(test_array.shape)
print(test_array.dtype)
print(test_array)

print(test_array[49][19])
print(test_array[48][18])


