import pandas as pd

df = pd.DataFrame({
    'grade' : [80,85,90,99,100,79,89],
    'name' : ["abdi", "kb", "abr", "aben", "zak", "matew", "eric"]
})

print(df)

df['grade'] = df['grade'] / 100
print("fraction of the grade 0 to 1")
print(df)