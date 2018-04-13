import pandas as pd
demo_df = pd. DataFrame({'numeric feature':[0,1,2,1],
                         'categrical feature':['book','pen','book','box']})

print(demo_df)

print(pd.get_dummies(demo_df))