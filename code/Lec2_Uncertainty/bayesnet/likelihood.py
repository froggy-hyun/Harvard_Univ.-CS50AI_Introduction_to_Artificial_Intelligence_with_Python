from model import model

# Calculate probability for a given observation
# No rain, no track maintenance, my train is on time, and I'm able to attend the meeting.
probability = model.probability([["none", "no", "on time", "attend"]])
# probability = model.probability([["none", "no", "on time", "miss"]]) 

print(probability)
