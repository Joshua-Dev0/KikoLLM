# Byte Pair Encoding training is a CPU task

# Placeholder
text = "My very first large language model"
space_char = "_"  # Might change depending on training example

class Tokenizer:
  def encode(self, text):
    return 0
  
  def decode(self, result):
    return 0
  
  def learn(self, data):
    temp = []
    modified_text = text.replace(" ", "_")
    tokens = list(modified_text)
    
    for i in range(len(tokens) - 1):
      array = [tokens[i], tokens[i + 1]]
      temp.append(array)

    print(temp)
    # print(tokens[0], tokens[1])
    # print(tokens)
    # print(len(tokens))
    
    counts = {}

    for array in temp:
      key = tuple(array)  # Convert list to tuple so it can be a dictionary key

      if key in counts:
        counts[key] += 1
      else:
        counts[key] = 1

    for key, count in counts.items():
      if count > 1:
        print(list(key), "appears", count, "times")
    return 0


tokenizer = Tokenizer()
tokenizer.learn(text)
