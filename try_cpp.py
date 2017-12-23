import poker

#poker.run_simulation(4, "AdKd", "QdJdTd", 1)

equity = poker.run_simulation(4, "6d3s", "", 1000000)

print(f"My equity is {equity}")